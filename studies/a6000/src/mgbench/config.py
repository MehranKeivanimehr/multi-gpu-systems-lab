"""Load YAML configs and expand an experiment file into an ordered list of run specs."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
TRAIN_FRAMEWORKS = ("ddp", "fsdp", "zero1", "zero2", "zero3")
MAX_MEM_FRACTION = 0.95  # hard ceiling for a per-run torch memory-cap override (leaves >= 2.4 GiB per A6000)


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def cluster_config() -> dict:
    return load_yaml(ROOT / "configs/cluster.yaml")


def model_registry() -> dict:
    return load_yaml(ROOT / "configs/models.yaml")


def framework_config(name: str) -> dict:
    """Framework settings: YAML for ddp/fsdp/vllm, a DeepSpeed JSON for zero{1,2,3}."""
    if name.startswith("zero"):
        with open(ROOT / f"configs/frameworks/ds_{name}.json") as f:
            return {"name": name, "launcher": "torchrun", "deepspeed": json.load(f)}
    return load_yaml(ROOT / f"configs/frameworks/{name}.yaml")


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def resolved_framework_config(spec: dict) -> dict:
    """Framework config with the workload's per-framework overrides applied (spec['fw_overrides'][framework])."""
    return deep_merge(framework_config(spec["framework"]), (spec.get("fw_overrides") or {}).get(spec["framework"]))


def validate_gpus(gpus: list[int], cluster: dict) -> None:
    """Refuse any GPU outside the allowlist. This is the main guard against touching other people's GPUs."""
    allowed = set(cluster["allowed_gpus"])
    if not gpus:
        raise ValueError("empty GPU list")
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"duplicate GPUs in {gpus}")
    bad = [g for g in gpus if g not in allowed]
    if bad:
        raise ValueError(f"GPUs {bad} are not in allowed_gpus={sorted(allowed)} (configs/cluster.yaml)")


def gpus_for_count(n: int, cluster: dict) -> list[int]:
    sets = {int(k): v for k, v in cluster["gpu_sets"].items()}
    if n not in sets:
        raise ValueError(f"no gpu_set for {n} GPUs in configs/cluster.yaml (have {sorted(sets)})")
    gpus = list(sets[n])
    validate_gpus(gpus, cluster)
    return gpus


def _train_runs(exp: dict, cluster: dict) -> list[dict]:
    runs = []
    defaults = exp.get("defaults", {})
    for n in exp["gpu_counts"]:
        gpus = gpus_for_count(n, cluster)
        max_rep = max([exp.get("repeats", 1)] + [wl.get("repeats", 0) for wl in exp["workloads"]])
        for rep in range(max_rep):
            for wl in exp["workloads"]:
                if rep >= wl.get("repeats", exp.get("repeats", 1)):  # optional per-workload repeat count
                    continue
                for fw in wl["frameworks"]:
                    if fw not in TRAIN_FRAMEWORKS:
                        raise ValueError(f"unknown framework {fw}")
                    spec = copy.deepcopy(defaults)
                    spec.update({k: v for k, v in wl.items() if k not in ("frameworks", "repeats")})
                    spec.update(
                        kind="train", experiment=exp["name"], framework=fw,
                        n_gpus=n, gpus=gpus, repeat=rep,
                    )
                    variant = f"-{spec['variant']}" if spec.get("variant") else ""
                    spec["run_id"] = f"{spec['model']}{variant}__{fw}__g{n}__r{rep}"
                    if not 0 < spec.get("mem_fraction", MAX_MEM_FRACTION) <= MAX_MEM_FRACTION:
                        raise ValueError(f"mem_fraction must be in (0, {MAX_MEM_FRACTION}]: {spec['run_id']}")
                    runs.append(spec)
    return runs


def _vllm_runs(exp: dict, cluster: dict) -> list[dict]:
    runs = []
    defaults = exp.get("defaults", {})
    for rep in range(exp.get("repeats", 1)):
        for wl in exp["workloads"]:
            for layout in wl["layouts"]:
                tp, dp = int(layout["tp"]), int(layout["dp"])
                n = tp * dp
                spec = copy.deepcopy(defaults)
                spec.update({k: v for k, v in wl.items() if k != "layouts"})
                spec.update(
                    kind="vllm", experiment=exp["name"], framework="vllm", tp=tp, dp=dp,
                    n_gpus=n, gpus=gpus_for_count(n, cluster), repeat=rep,
                )
                # Optional per-layout environment (e.g. NCCL transport settings) merged over the workload's env.
                spec["env"] = {**(spec.get("env") or {}), **(layout.get("env") or {})}
                if "max_num_seqs" in layout:  # per-layout scheduler concurrency (default: configs/frameworks/vllm.yaml)
                    spec["max_num_seqs"] = int(layout["max_num_seqs"])
                tag = f"-{layout['tag']}" if layout.get("tag") else ""
                spec["run_id"] = f"{spec['model']}__vllm_tp{tp}_dp{dp}{tag}__g{n}__r{rep}"
                runs.append(spec)
    # Small GPU counts first ("start small"), repeats interleaved.
    runs.sort(key=lambda s: (s["n_gpus"], s["repeat"]))
    return runs


def _nccl_runs(exp: dict, cluster: dict) -> list[dict]:
    runs = []
    for rep in range(exp.get("repeats", 1)):
        for item in exp["gpu_sets"]:
            # item: a GPU list, or {gpus: [...], tag: str, env: {NCCL_...: value}} for diagnostic variants
            item = item if isinstance(item, dict) else {"gpus": item}
            gpus = list(item["gpus"])
            validate_gpus(gpus, cluster)
            spec = copy.deepcopy(exp.get("defaults", {}))
            spec.update(kind="nccl", experiment=exp["name"], framework="nccl", model="allreduce",
                        n_gpus=len(gpus), gpus=gpus, repeat=rep, env=item.get("env", {}), variant=item.get("tag", ""))
            tag = f"-{item['tag']}" if item.get("tag") else ""
            spec["run_id"] = f"allreduce{tag}__gpus{'-'.join(map(str, gpus))}__r{rep}"
            runs.append(spec)
    return runs


def expand_experiment(exp: dict, cluster: dict) -> list[dict]:
    kind = exp["kind"]
    if kind == "train":
        return _train_runs(exp, cluster)
    if kind == "vllm":
        return _vllm_runs(exp, cluster)
    if kind == "nccl":
        return _nccl_runs(exp, cluster)
    raise ValueError(f"unknown experiment kind {kind}")
