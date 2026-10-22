#!/usr/bin/env python
"""Run every configuration of an experiment file one at a time, safely.

For each run:
  1. skip if it already finished OK (resumable), or if the memory estimate says it cannot fit;
  2. wait until its GPUs (and the previous run's GPUs) are idle in every respect: no processes of any user,
     ~0 MiB used, 0 % utilization, idle power, cooled down; host RAM headroom;
  3. snapshot nvidia-smi, launch under a hard timeout in its own process group, pinned to NUMA node 1,
     with explicit CUDA_VISIBLE_DEVICES and CPU-thread caps;
  4. snapshot nvidia-smi again and clean up any leftover process that carries this run's marker.

Stop cleanly at any time with:  touch .cache/STOP

Launch hold (recovery review 20261015r): nothing is launched while configs/cluster.yaml has `launch_hold` active
(exit code 6, no files written). Before every launch the GPU UUIDs are checked against `gpu_uuids` (exit code 7).
An earlier attempt's directory is never written into (evidence preservation).

Usage:
  python scripts/run_matrix.py configs/experiments/train_scaling.yaml [--dry-run] [--gpus 1,2] [--force]
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mgbench.config import cluster_config, expand_experiment, load_yaml, model_registry, validate_gpus  # noqa: E402
from mgbench.models import load_hf_config_dict  # noqa: E402
from mgbench.results import read_json, write_json  # noqa: E402
from mgbench.safety import (PowerWatch, check_feasible, device_identity_problems,  # noqa: E402
                            launch_hold_problems, node_gpu_power, terminate_process_group,
                            terminate_run_processes, wait_for_idle)

VENV_BIN = ROOT / ".venv/bin"
CACHE = ROOT / ".cache"
STOP_FILE = CACHE / "STOP"


class Log:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "a")

    def __call__(self, msg: str):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def nvidia_smi_snapshot(path: Path):
    cmds = [
        ["nvidia-smi"],
        ["nvidia-smi", "--query-gpu=index,pstate,temperature.gpu,power.draw,utilization.gpu,memory.used,"
         "clocks.sm", "--format=csv"],
        ["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,used_memory", "--format=csv"],
    ]
    with open(path, "w") as f:
        for c in cmds:
            try:
                f.write(subprocess.run(c, capture_output=True, text=True, timeout=30).stdout + "\n")
            except Exception as e:  # noqa: BLE001
                f.write(f"{c}: {e}\n")


def run_env(spec: dict, cluster: dict, marker: str) -> dict:
    env = dict(os.environ)
    env.update({
        "PATH": f"{VENV_BIN}:{env.get('PATH', '')}",
        "PYTHONPATH": str(ROOT / "src"),
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, spec["gpus"])),
        "OMP_NUM_THREADS": str(cluster["omp_threads_per_rank"]),
        "MKL_NUM_THREADS": str(cluster["omp_threads_per_rank"]),
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "NCCL_DEBUG": "WARN",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        "TRITON_CACHE_DIR": str(CACHE / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(CACHE / "inductor"),
        "TORCH_EXTENSIONS_DIR": str(CACHE / "torch_extensions"),
        "VLLM_CACHE_ROOT": str(CACHE / "vllm"),
        "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        "VLLM_NO_USAGE_STATS": "1",
        "DO_NOT_TRACK": "1",
        "MGBENCH_RUN_ID": marker,
    })
    # Per-run diagnostic overrides (e.g. NCCL_DEBUG, NCCL_P2P_DISABLE); never CUDA_VISIBLE_DEVICES.
    for k, v in (spec.get("env") or {}).items():
        if k in ("CUDA_VISIBLE_DEVICES", "MGBENCH_RUN_ID"):
            raise ValueError(f"spec env may not override {k}")
        env[k] = str(v)
    return env


def command(spec: dict, cluster: dict, spec_path: Path, out: Path) -> list[str]:
    numa = ["numactl", f"--cpunodebind={cluster['numa_node']}", f"--preferred={cluster['numa_node']}"]
    if not shutil.which("numactl"):
        numa = []
    io = ["--spec", str(spec_path), "--out", str(out)]
    if spec["kind"] in ("train", "nccl"):
        module = "mgbench.train" if spec["kind"] == "train" else "mgbench.comm"
        return numa + [str(VENV_BIN / "torchrun"), "--standalone", "--nnodes=1",
                       f"--nproc-per-node={spec['n_gpus']}", "-m", module] + io
    if spec["kind"] == "vllm":
        return numa + [str(VENV_BIN / "python"), "-m", "mgbench.infer.vllm_bench"] + io + [
            "--timeout", str(cluster["run_timeout_s"] - 100)]
    raise ValueError(spec["kind"])


def launch(cmd, env, log_path: Path, timeout_s: float, grace_s: float, log, guard: dict | None = None):
    """Run in a new session (own process group), polling once per second. Stop the whole group (SIGTERM, then
    SIGKILL after grace) on timeout or when the node power watchdog trips (total GPU draw of the node above
    guard["abort_total_w"] for guard["abort_after_s"] consecutive seconds). After a normal exit, anything still
    alive in the group is terminated too. Returns (exit code or None, timed_out, leftovers_killed, abort_reason,
    node_power_peak_w)."""
    watch = PowerWatch(guard) if guard else None
    with open(log_path, "w") as lf:
        p = subprocess.Popen(cmd, env=env, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        t0 = time.time()
        reason = None
        while True:
            try:
                rc = p.wait(timeout=1.0)
                return rc, False, terminate_process_group(p.pid, grace_s=grace_s, log=log), None, \
                    (watch.peak if watch else None)
            except subprocess.TimeoutExpired:
                pass
            if watch is not None:
                try:
                    total = sum(node_gpu_power().values())
                except Exception as exc:  # NVML failure: stop rather than run blind
                    reason = f"node power unreadable: {exc!r}"
                    break
                if watch.update(total):
                    reason = (f"node power watchdog: all GPUs {total:.0f} W > {watch.limit} W "
                              f"for {watch.over} s")
                    break
            if time.time() - t0 > timeout_s:
                reason = f"timeout after {timeout_s:.0f} s"
                break
        log(f"  STOPPING run: {reason} -> SIGTERM process group {p.pid}")
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            p.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            log(f"  still alive -> SIGKILL process group {p.pid}")
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait(timeout=30)
        terminate_process_group(p.pid, grace_s=grace_s, log=log)
        return None, reason.startswith("timeout"), True, reason, (watch.peak if watch else None)


# Status records that mean "nothing was launched"; only these may be replaced by a later invocation.
NOT_LAUNCHED = ("deferred_safety", "skipped_infeasible", "aborted_not_idle", "aborted_device_identity")


def status_writable(out: Path) -> bool:
    """Evidence preservation: write a status record only into a new or empty run directory, or over an earlier
    not-launched marker. An earlier attempt (e.g. one interrupted by a host failure, with no status.json) is kept
    exactly as it is."""
    if not out.exists():
        return True
    files = list(out.iterdir())
    if not files:
        return True
    if [f.name for f in files] == ["status.json"]:
        try:
            return read_json(out / "status.json").get("status") in NOT_LAUNCHED
        except Exception:  # noqa: BLE001 - unreadable record: keep it
            return False
    return False


def classify(out: Path, rc, timed_out: bool) -> tuple[str, str]:
    res = out / "result.json"
    if res.exists() and read_json(res).get("status") == "ok" and rc == 0:
        return "ok", ""
    if res.exists() and read_json(res).get("status") == "unverified_gradients":
        return "unverified", "gradient observation incomplete on some attempted step; see result.json verification"
    errs = [read_json(f) for f in sorted(out.glob("error_*.json"))]
    if any(e.get("status") == "oom" for e in errs):
        return "oom", next(e["error"][:300] for e in errs if e.get("status") == "oom")
    if timed_out:
        return "timeout", ""
    msg = errs[0].get("error", "")[:300] if errs else ""
    return "failed", msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run configurations that already finished OK")
    ap.add_argument("--gpus", default=None, help="only these GPU counts, e.g. 1,2")
    ap.add_argument("--models", default=None)
    ap.add_argument("--frameworks", default=None)
    ap.add_argument("--repeats", type=int, default=None, help="override the experiment's repeat count")
    args = ap.parse_args()

    cluster = cluster_config()
    if not args.dry_run:
        hold = launch_hold_problems(cluster)
        if hold:  # before any directory, log or lock is created
            print(f"NOT LAUNCHING: {'; '.join(hold)}", file=sys.stderr)
            sys.exit(6)
    exp = load_yaml(args.experiment)
    if args.repeats is not None:
        exp["repeats"] = args.repeats
    specs = expand_experiment(exp, cluster)
    if args.gpus:
        keep = {int(x) for x in args.gpus.split(",")}
        specs = [s for s in specs if s["n_gpus"] in keep]
    if args.models:
        specs = [s for s in specs if s["model"] in args.models.split(",")]
    if args.frameworks:
        specs = [s for s in specs if s["framework"] in args.frameworks.split(",")]

    exp_dir = ROOT / "results/raw" / exp["name"]
    if not exp_dir.resolve().is_relative_to((ROOT / "results/raw").resolve()) or exp_dir.resolve() == (ROOT / "results/raw").resolve():
        ap.error("experiment name must identify a child of results/raw")
    manifest_path = ROOT.parents[1] / "provenance/measurements.sha256.json"
    if manifest_path.exists():
        prefix = exp_dir.relative_to(ROOT.parents[1]).as_posix() + "/"
        frozen = any(item["path"].startswith(prefix) for item in json.loads(manifest_path.read_text())["files"])
        if frozen and not args.dry_run:
            ap.error("Frozen historical experiment: choose a new validation_v2 experiment name; --force cannot overwrite provenance")
    log = print if args.dry_run else Log(exp_dir / "runner.log")
    registry = model_registry()

    CACHE.mkdir(exist_ok=True)
    if not args.dry_run:
        lock = open(CACHE / "runner.lock", "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit("another run_matrix.py is already running (lock .cache/runner.lock)")

    log(f"== {exp['name']}: {len(specs)} runs (dry_run={args.dry_run})")
    prev_gpus: list[int] = []
    consecutive_failures = 0
    counts: dict[str, int] = {}
    for i, spec in enumerate(specs, 1):
        validate_gpus(spec["gpus"], cluster)
        out = exp_dir / spec["run_id"]
        tag = f"[{i}/{len(specs)}] {spec['run_id']} gpus={spec['gpus']}"
        status_path = out / "status.json"
        if status_path.exists() and not args.force and read_json(status_path)["status"] in ("ok", "skipped_infeasible", "oom"):
            log(f"{tag}: already {read_json(status_path)['status']}, skipping")
            continue

        guard = cluster["idle"].get("node_guard")
        if out.exists() and not status_path.exists() and any(out.iterdir()):
            log(f"{tag}: earlier attempt without a status record is preserved unchanged (not re-run, nothing "
                "written); use a fresh experiment name")
            counts["preserved_earlier_attempt"] = counts.get("preserved_earlier_attempt", 0) + 1
            continue
        if guard and len(spec["gpus"]) > guard["max_gpus_per_run"]:
            log(f"{tag}: DEFERRED for safety ({len(spec['gpus'])} GPUs > max_gpus_per_run "
                f"{guard['max_gpus_per_run']}; cross-pair runs held until the 2026-10-14 incident is explained)")
            if not args.dry_run and status_writable(out):
                write_json(status_path, {"status": "deferred_safety", "spec": spec,
                                         "reason": "max_gpus_per_run node guard"})
            counts["deferred_safety"] = counts.get("deferred_safety", 0) + 1
            continue
        entry = registry.get(spec["model"], {})
        hf_cfg = load_hf_config_dict(entry) if entry.get("kind") == "causal_lm" else None
        feasible, est = check_feasible(spec, hf_cfg, cluster)
        if not feasible:
            log(f"{tag}: SKIP infeasible (model states ~{est['state_gib_per_gpu_est']} GiB/GPU > cap "
                f"{est['cap_gib']} GiB)")
            if not args.dry_run and status_writable(out):
                write_json(status_path, {"status": "skipped_infeasible", "spec": spec, "estimate": est})
            counts["skipped_infeasible"] = counts.get("skipped_infeasible", 0) + 1
            continue
        if args.dry_run:
            log(f"{tag}: would run" + (f" (state ~{est['state_gib_per_gpu_est']} GiB/GPU)" if est else ""))
            continue
        if STOP_FILE.exists():
            log("STOP file present -> stopping before the next run")
            break
        hold = launch_hold_problems(cluster_config())       # re-read: the owner may set the hold mid-matrix
        if hold:
            log(f"{tag}: NOT LAUNCHING - {'; '.join(hold)}")
            sys.exit(6)

        gate_gpus = sorted(set(prev_gpus) | set(spec["gpus"]))
        ok, waited, problems = wait_for_idle(gate_gpus, cluster["idle"], log=log)
        if not ok:
            log(f"{tag}: GPUs {gate_gpus} not idle after {waited:.0f} s ({'; '.join(problems)}) -> ABORT matrix")
            if status_writable(out):
                write_json(status_path, {"status": "aborted_not_idle", "spec": spec, "problems": problems})
            sys.exit(2)
        identity = device_identity_problems(spec["gpus"], cluster)
        if identity:
            log(f"{tag}: GPU identity check failed ({'; '.join(identity)}) -> ABORT matrix")
            if status_writable(out):
                write_json(status_path, {"status": "aborted_device_identity", "spec": spec, "problems": identity})
            sys.exit(7)

        out.mkdir(parents=True, exist_ok=True)
        if any(out.iterdir()):
            raise RuntimeError(f"Refusing to erase an earlier attempt: {out}; use a fresh experiment name")
        spec_path = out / "spec.json"
        write_json(spec_path, spec)
        nvidia_smi_snapshot(out / "nvidia_smi_before.txt")
        marker = f"{exp['name']}/{spec['run_id']}"
        cmd = command(spec, cluster, spec_path, out)
        (out / "command.txt").write_text(" ".join(cmd) + "\n")
        log(f"{tag}: start (idle after {waited:.0f} s)")
        t0 = time.time()
        rc, timed_out, group_leftovers, abort_reason, node_peak = launch(
            cmd, run_env(spec, cluster, marker), out / "run.log", cluster["run_timeout_s"], cluster["kill_grace_s"],
            log, guard=cluster["idle"].get("node_guard"))
        duration = time.time() - t0
        leftovers = terminate_run_processes(marker, log=log) + int(group_leftovers)
        nvidia_smi_snapshot(out / "nvidia_smi_after.txt")
        status, msg = classify(out, rc, timed_out)
        if abort_reason and not timed_out:
            status, msg = "aborted_power_guard", abort_reason
        write_json(status_path, {
            "status": status, "rc": rc, "duration_s": duration, "idle_wait_before_s": waited,
            "leftover_processes_killed": leftovers, "estimate": est, "message": msg, "spec": spec,
            "node_gpu_power_peak_w": node_peak,
        })
        counts[status] = counts.get(status, 0) + 1
        summary = ""
        if status == "ok":
            for line in (out / "run.log").read_text().splitlines():
                if line.startswith("[mgbench]"):
                    summary = line[len("[mgbench] "):]
        log(f"{tag}: {status.upper()} in {duration:.0f} s {summary or msg}")
        prev_gpus = spec["gpus"]
        if status == "aborted_power_guard":
            log("node power watchdog tripped -> ABORT matrix (do not keep loading the node)")
            sys.exit(5)

        consecutive_failures = consecutive_failures + 1 if status in ("failed", "timeout") else 0  # "unverified" is not a failure
        if consecutive_failures >= cluster["max_consecutive_failures"]:
            log(f"{consecutive_failures} consecutive failures -> ABORT matrix")
            sys.exit(3)

    if prev_gpus and not args.dry_run:
        ok, waited, _ = wait_for_idle(prev_gpus, cluster["idle"], log=log)
        log(f"final idle check on {prev_gpus}: {'idle' if ok else 'NOT idle'} after {waited:.0f} s")
    log(f"== {exp['name']} done: {counts}")


if __name__ == "__main__":
    main()
