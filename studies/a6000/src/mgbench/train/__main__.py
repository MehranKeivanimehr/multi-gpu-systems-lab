"""Training throughput benchmark. Launched by scripts/run_matrix.py as:

    torchrun --standalone --nproc-per-node N -m mgbench.train --spec spec.json --out results/raw/<exp>/<run_id>

Runs a fixed number of warmup + measured optimizer steps on synthetic data and writes result.json (rank 0).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import timedelta

import torch
import torch.distributed as dist

from ..config import cluster_config, model_registry, resolved_framework_config
from ..data import GlobalSynthetic, SyntheticImages, SyntheticTokens
from ..gpu_monitor import GpuSampler, aggregate_gpu_summary
from ..metrics import GIB, lm_train_flops_per_token, mfu, step_stats
from ..models import load_hf_config_dict, vision_train_flops_per_sample
from ..results import environment, executed_source_record, read_json, write_json
from .frameworks import SETUP
from .boundary_ledger import assess


def _runtime_device_record(stage: str) -> dict:
    """CUDA's own view of the device this process uses (smoke stage: must be exactly the authorized UUID)."""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    uuid = str(getattr(props, "uuid", ""))
    uuid = uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"
    free, total = torch.cuda.mem_get_info()
    dom, bus, dev = (getattr(props, k, None) for k in ("pci_domain_id", "pci_bus_id", "pci_device_id"))
    pci = f"{dom:04x}:{bus:02x}:{dev:02x}.0" if None not in (dom, bus, dev) else None
    return {"stage": stage, "device_count": torch.cuda.device_count(), "current_device": torch.cuda.current_device(),
            "name": props.name, "uuid": uuid, "pci_bus_id": pci, "total_memory": props.total_memory,
            "cuda_mem_get_info_free": free, "cuda_mem_get_info_total": total}


def _precision_record(spec: dict) -> dict:
    b = torch.backends
    rec = {"interface_written": "legacy (allow_tf32, set_float32_matmul_precision, cudnn flags) - not mixed with the new API",
           "legacy": {"cuda_matmul_allow_tf32": b.cuda.matmul.allow_tf32, "cudnn_allow_tf32": b.cudnn.allow_tf32,
                      "float32_matmul_precision": torch.get_float32_matmul_precision(),
                      "cudnn_benchmark": b.cudnn.benchmark, "cudnn_deterministic": b.cudnn.deterministic},
           "sdpa_enabled": {"flash": b.cuda.flash_sdp_enabled(), "mem_efficient": b.cuda.mem_efficient_sdp_enabled(),
                            "math": b.cuda.math_sdp_enabled(), "cudnn": b.cuda.cudnn_sdp_enabled()},
           "env_overrides": {k: os.environ.get(k) for k in ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
                                                            "CUBLAS_WORKSPACE_CONFIG", "TORCH_CUDNN_V8_API_DISABLED")},
           "autocast_bf16": (spec["framework"] == "ddp" and not spec.get("fp32_reference")
                             and spec.get("param_dtype") != "bf16") or spec.get("ds_precision") == "autocast",
           "fp32_reference": bool(spec.get("fp32_reference")), "param_dtype": spec.get("param_dtype"),
           "ds_grad_accum_dtype": spec.get("ds_grad_accum_dtype"), "matched_policy": spec.get("matched_policy"),
           "bf16_reduced_precision_reduction": getattr(b.cuda.matmul, "allow_bf16_reduced_precision_reduction", None),
           "fp16_reduced_precision_reduction": getattr(b.cuda.matmul, "allow_fp16_reduced_precision_reduction", None),
           "kind_of_evidence": "runtime-observed flag values; kernel dispatch is NOT observed by these flags"}
    try:
        rec["new_getters_readback"] = {"generic": b.fp32_precision, "cuda_matmul": b.cuda.matmul.fp32_precision,
                                       "cudnn": b.cudnn.fp32_precision, "cudnn_conv": b.cudnn.conv.fp32_precision,
                                       "cudnn_rnn": b.cudnn.rnn.fp32_precision}
    except Exception as exc:  # noqa: BLE001
        rec["new_getters_readback"] = f"unreadable ({exc!r})"
    return rec


def _identity_chain(spec: dict, device_records: list[dict]) -> dict:
    """Links 1-4 of the device-identity chain (see mgbench.device_identity), checked before any model allocation or
    DeepSpeed initialisation. Only active when the launcher provided an expected identity."""
    from ..device_identity import chain_problems

    uuid, pci, index, n_dev, local = _expected_identity()
    if not (uuid and pci and index):
        return {"checked": False, "problems": []}
    index = int(index)
    out = {"checked": True, "approved": {"uuid": uuid, "pci_bus_id": pci, "nvml_index": index}}
    import pynvml

    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(index)
    nv_uuid = pynvml.nvmlDeviceGetUUID(h)
    nv_pci = pynvml.nvmlDeviceGetPciInfo(h).busId
    out["nvml"] = {"index": index, "uuid": nv_uuid.decode() if isinstance(nv_uuid, bytes) else nv_uuid,
                   "pci_bus_id": nv_pci.decode() if isinstance(nv_pci, bytes) else nv_pci}
    driver_rec = json.loads(os.environ.get("MGBENCH_DRIVER_IDENTITY", "{}"))
    out["driver"] = driver_rec.get("driver")
    out["driver_pre_torch_problems"] = driver_rec.get("problems", ["no pre-torch driver record"])
    if spec["framework"].startswith("zero"):
        import hashlib
        import inspect

        from deepspeed.accelerator import get_accelerator

        acc = get_accelerator()
        mapped = acc._get_nvml_gpu_id(torch.cuda.current_device())
        hm = pynvml.nvmlDeviceGetHandleByIndex(mapped)
        mu, mp = pynvml.nvmlDeviceGetUUID(hm), pynvml.nvmlDeviceGetPciInfo(hm).busId
        src = inspect.getsourcefile(type(acc))
        out["deepspeed"] = {"accelerator": f"{type(acc).__module__}.{type(acc).__name__}", "source_file": src,
                            "source_sha256": hashlib.sha256(open(src, "rb").read()).hexdigest(),
                            "logical": torch.cuda.current_device(), "nvml_index": mapped, "expected_index": index,
                            "uuid": mu.decode() if isinstance(mu, bytes) else mu,
                            "pci_bus_id": mp.decode() if isinstance(mp, bytes) else mp}
    else:
        out["deepspeed"] = "not applicable (framework does not use DeepSpeed)"
    out["torch"] = device_records
    out["problems"] = list(out["driver_pre_torch_problems"]) + chain_problems(
        approved_uuid=uuid, approved_pci=pci, nvml=out["nvml"], driver=out["driver"] if n_dev == 1 else None,
        deepspeed=out["deepspeed"] if isinstance(out["deepspeed"], dict) else None, torch_records=device_records,
        expected_device_count=n_dev, expected_current=local)
    out["declared_devices"], out["local_rank"] = n_dev, local
    if out["driver"] is None:
        out["problems"].append("driver-level identity was not recorded before torch")
    return out


def _expected_identity():
    """(uuid, pci, nvml index, declared device count, local rank) for THIS rank. Candidate 17: the plural variables
    declare an ordered multi-device set (rank r -> physical index r of the list); otherwise the single-device case."""
    if os.environ.get("MGBENCH_EXPECTED_GPU_UUIDS"):
        uuids = os.environ["MGBENCH_EXPECTED_GPU_UUIDS"].split(",")
        pcis = os.environ.get("MGBENCH_EXPECTED_GPU_PCIS", "").split(",")
        idx = os.environ.get("MGBENCH_EXPECTED_GPU_INDICES", "").split(",")
        local = int(os.environ.get("LOCAL_RANK", "-1"))
        if not (len(uuids) == len(pcis) == len(idx) > 1) or not 0 <= local < len(uuids):
            raise RuntimeError("declared multi-device identity malformed for this rank")
        return uuids[local], pcis[local], idx[local], len(uuids), local
    return (os.environ.get("MGBENCH_EXPECTED_GPU_UUID"), os.environ.get("MGBENCH_EXPECTED_GPU_PCI"),
            os.environ.get("MGBENCH_EXPECTED_GPU_INDEX"), 1, 0)


def _identity_checks(records: list[dict]) -> list[str]:
    want, _, _, n_dev, _ = _expected_identity()
    problems = []
    for r in records:
        if os.environ.get("MGBENCH_REQUIRE_SINGLE_DEVICE") == "1" and r["device_count"] != 1:
            problems.append(f"{r['stage']}: {r['device_count']} visible CUDA devices, exactly 1 required")
        if os.environ.get("MGBENCH_REQUIRE_DEVICE_COUNT") and r["device_count"] != int(os.environ["MGBENCH_REQUIRE_DEVICE_COUNT"]):
            problems.append(f"{r['stage']}: {r['device_count']} visible CUDA devices, declared {n_dev}")
        if want and r["uuid"] != want:
            problems.append(f"{r['stage']}: runtime CUDA UUID {r['uuid']} != authorized {want}")
    return problems


def _process_record() -> dict:
    import mgbench

    from ..scopes import own_cgroup_limits
    return {"sys_executable": sys.executable, "mgbench_file": mgbench.__file__, "python_flags_no_user_site":
            bool(sys.flags.no_user_site), "sys_path_head": sys.path[:4], "pid": os.getpid(),
            "cgroup": own_cgroup_limits(), "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"), "torch_num_threads": torch.get_num_threads()}


ALLOC_KEYS = ("num_alloc_retries", "num_ooms", "num_device_alloc", "num_device_free")


def _alloc_counters() -> dict:
    """Cumulative caching-allocator counters (max over ranks is taken later)."""
    st = torch.cuda.memory_stats()
    return {k: st.get(k, 0) for k in ALLOC_KEYS}


def _compare_canonical(canonical: dict, gathered: dict, spec: dict) -> dict:
    """Did the framework's gathered full logical state equal the canonical initial state (per the recipe)?"""
    canon = {k: v for k, v in canonical.items() if k != "param_initial_slices"}
    bf16_weights = (spec["framework"].startswith("zero") and spec.get("ds_precision", "bf16") == "bf16") or \
        (spec["framework"] == "ddp" and spec.get("param_dtype") == "bf16")
    expected = canon["canonical_bf16"] if bf16_weights else canon["canonical_fp32"]
    out = {"canonical": canon, "gathered": gathered,
           "expected_module_state": "canonical_bf16 (bf16 weight recipe)" if bf16_weights else "canonical_fp32",
           "module_state_matches_canonical": gathered["module_state"]["sha256"] == expected["sha256"]}
    if "optimizer_master_fp32_params" in gathered:
        want = canon["params_bf16_as_fp32"] if bf16_weights else canon["params_fp32"]
        out["optimizer_master_matches_canonical"] = gathered["optimizer_master_fp32_params"]["sha256"] == want["sha256"]
        out["expected_optimizer_master"] = "params_bf16_as_fp32" if bf16_weights else "params_fp32"
    return out


def apply_matched_policy(spec: dict) -> None:
    """Candidate 14: one declared attention/TF32/reduced-reduction policy shared by every case of a matched contrast
    (legacy interface only). A forced backend is a mechanistic deviation from the historical recipe runs."""
    pol = spec.get("matched_policy")
    if not pol:
        return
    tf32 = bool(pol["tf32"])
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision("high" if tf32 else "highest")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = bool(pol["bf16_reduced_precision_reduction"])
    if pol["sdpa"] == "math":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    elif pol["sdpa"] != "default":
        raise ValueError(f"unknown sdpa policy {pol['sdpa']!r}")


class SdpaDispatchRecorder:
    """Wraps torch.nn.functional.scaled_dot_product_attention: per call, the input dtypes/shape and the backend flags
    enabled AT CALL TIME. With only math enabled the CUDA selection is constrained to math; with several enabled the
    actual kernel is NOT observed here (hardware-unverified)."""

    def __init__(self):
        import torch.nn.functional as F

        self.F, self.orig, self.calls, self.kinds = F, F.scaled_dot_product_attention, 0, {}

    def __enter__(self):
        def wrapped(q, *a, **k):
            b = torch.backends.cuda
            key = json.dumps({"dtype": str(q.dtype), "device": q.device.type,
                              "enabled": {"flash": b.flash_sdp_enabled(), "mem_efficient": b.mem_efficient_sdp_enabled(),
                                          "math": b.math_sdp_enabled(), "cudnn": b.cudnn_sdp_enabled()},
                              "is_causal": bool(k.get("is_causal", False))}, sort_keys=True)
            self.calls += 1
            self.kinds[key] = self.kinds.get(key, 0) + 1
            return self.orig(q, *a, **k)
        self.F.scaled_dot_product_attention = wrapped
        return self

    def __exit__(self, *exc):
        self.F.scaled_dot_product_attention = self.orig

    def record(self) -> dict:
        kinds = [dict(json.loads(k), calls=v) for k, v in self.kinds.items()]
        only_math = all(x["enabled"] == {"flash": False, "mem_efficient": False, "math": True, "cudnn": False}
                        for x in kinds) and bool(kinds)
        return {"calls": self.calls, "kinds": kinds,
                "dispatch": "constrained to math by flags" if only_math else "NOT observed (several backends enabled)"}


def apply_microbatch_order(mbs: list, order) -> tuple[list, list[int]]:
    """Candidate 15 exploratory X1: the SAME microbatches (input digests are recorded in generation order) accumulated
    in a declared permuted order. Only summation order changes; samples and loss weighting do not."""
    if order is None:
        return mbs, list(range(len(mbs)))
    if not isinstance(order, list) or sorted(order) != list(range(len(mbs))) or any(isinstance(i, bool) for i in order):
        raise RuntimeError(f"microbatch_order {order!r} is not a permutation of {len(mbs)} microbatches")
    return [mbs[i] for i in order], list(order)


def assess_first_boundary(summary: dict, ga: int, unchanged: bool, comparison: dict | None) -> dict:
    """Separate verdicts for the first-boundary result kind. Update/state verification is not_applicable (no update
    was intended or made) - never 'passed'."""
    m, w = summary["measured"], summary["warmup"]
    reasons = []
    if (m["logical_attempts"], m["microbatches"]) != (1, ga) or w["logical_attempts"] != 0:
        reasons.append(f"attempts/microbatches {(m['logical_attempts'], m['microbatches'])}, expected (1, {ga})")
    if m["base_optimizer_calls"] != 0 or m["counter_updates"] != 0:
        reasons.append(f"optimizer calls {m['base_optimizer_calls']} / updates {m['counter_updates']}, expected 0 / 0")
    if not m["microbatch_losses_all_finite"]:
        reasons.append("a microbatch loss is missing or non-finite")
    if not unchanged:
        reasons.append("parameters (or masters) differ from the realized initial state after the boundary")
    cov = m["boundary_coverage"]
    v = (comparison or {}).get("snapshots", {}).get("vs_sealed_r0_grad_b1", {})
    numerics = ("comparable" if v.get("coverage_complete") and v.get("nonzero_certifiable") else
                "not_certifiable (zero vector or incomplete coverage)")
    return {"execution": "ok" if not reasons else "problem_detected", "execution_reasons": reasons,
            "gradient": {"status": "complete_at_the_boundary" if cov == ["complete"] else "incomplete_or_unaccounted",
                         "coverage": cov},
            "update_state": {"status": "not_applicable", "condition": "first-boundary result kind: no optimizer update"},
            "parameters_unchanged": unchanged, "numerical_comparison": numerics,
            "measurement_eligibility": "ineligible: diagnostic, not a performance measurement"}


def rank_payload(rank: int, summary: dict, input_digests: dict, update_state: str, identity_ok: bool,
                 identity_row: dict | None = None) -> dict:
    """Candidate 16: what each rank contributes to the full-control record (gathered to rank 0, bounded size).
    Candidate 17: plus this rank's observed identity row {rank, nvml_index, uuid, pci_bus_id}."""
    m = summary["measured"]
    return {"rank": rank, "update_state": update_state, "identity_ok": bool(identity_ok), "identity_row": identity_row,
            "warmup": {"logical_attempts": summary["warmup"]["logical_attempts"]},
            "measured": {k: m[k] for k in ("logical_attempts", "microbatches", "counter_updates", "base_calls_per_attempt",
                                           "boundary_coverage", "microbatch_losses_all_finite", "microbatch_losses")},
            "digests": input_digests["per_step_microbatch_sha256"], "rows": input_digests["per_step_global_rows"]}


def build_control_record(spec: dict, payloads: list[dict], *, canonical_ok: bool, snapshots: dict | None) -> dict:
    """Rank 0: global sample map (step, rank, microstep -> global row, digest, raw loss) and per-rank evidence. Every
    sample keeps weight 1/(micro_batch * grad_accum * world) in the per-step global mean (computed by the comparator)."""
    ga = spec["grad_accum"]
    gm = []
    for p in sorted(payloads, key=lambda x: x["rank"]):
        losses = p["measured"]["microbatch_losses"]
        for k, (rows, digs) in enumerate(zip(p["rows"], p["digests"])):
            for i, (row, dg) in enumerate(zip(rows, digs)):
                gm.append({"step": k, "rank": p["rank"], "microstep": i, "global_row": row, "digest": dg,
                           "raw_loss": losses[k * ga + i] if k * ga + i < len(losses) else None})
    return {"result_kind": "full_control_5update", "spec": {k: v for k, v in spec.items() if not k.startswith("_")},
            "per_rank": [{k: v for k, v in p.items() if k not in ("digests", "rows")} for p in payloads],
            "global_sample_map": gm, "canonical_state_matches": bool(canonical_ok), "snapshots": snapshots,
            "loss_weighting": "each global sample's mean token loss has weight 1/4 per step; gradients averaged over ranks"}


def finalize_reference(trainer, measured: dict) -> tuple[dict | None, dict | None]:
    """Stage S1a, after the last measured update: final-state snapshot write/compare, scalar reference, and the
    framework's full final checks. `measured` is the ledger's measured-phase summary (raw microbatch losses)."""
    reference_comparison, final_checks = None, None
    if trainer.reference is not None:
        if trainer.final_items is None or trainer.theta0 is None:
            raise RuntimeError("reference mode needs final-state items and the canonical start")
        trainer.reference.final(trainer.final_items(), trainer.theta0)
        reference_comparison = trainer.reference.finish(measured["microbatch_losses"])
        trainer.theta0 = None
    if trainer.final_checks is not None:
        final_checks = trainer.final_checks()
    return reference_comparison, final_checks


def run(spec: dict, out: str) -> dict | None:
    cluster = cluster_config()
    entry = model_registry()[spec["model"]]
    fw = resolved_framework_config(spec)

    local_rank = int(os.environ["LOCAL_RANK"])
    if os.environ.get("MGBENCH_REQUIRE_SINGLE_DEVICE") == "1" and torch.cuda.device_count() != 1:
        raise RuntimeError(f"{torch.cuda.device_count()} CUDA devices visible; exactly one is authorized")
    if os.environ.get("MGBENCH_REQUIRE_DEVICE_COUNT") and torch.cuda.device_count() != int(os.environ["MGBENCH_REQUIRE_DEVICE_COUNT"]):
        raise RuntimeError(f"{torch.cuda.device_count()} CUDA devices visible; the declared set has "
                           f"{os.environ['MGBENCH_REQUIRE_DEVICE_COUNT']}")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    device_records = [_runtime_device_record("after set_device")]
    if _identity_checks(device_records):
        raise RuntimeError("; ".join(_identity_checks(device_records)))
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=5))
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == spec["n_gpus"], f"world size {world} != n_gpus {spec['n_gpus']}"
    # Per-run override (spec "mem_fraction", validated <= MAX_MEM_FRACTION in config.py), else the cluster default.
    mem_fraction = spec.get("mem_fraction", cluster["per_process_mem_fraction"])
    torch.cuda.set_per_process_memory_fraction(mem_fraction, device)
    device_records.append(_runtime_device_record("after set_per_process_memory_fraction"))
    if _identity_checks(device_records):
        raise RuntimeError("; ".join(_identity_checks(device_records)))
    # Identity chain NVML -> driver -> DeepSpeed -> torch, before any model allocation / DeepSpeed initialisation.
    identity_chain = _identity_chain(spec, device_records)
    if identity_chain["problems"]:
        raise RuntimeError("device identity chain: " + "; ".join(identity_chain["problems"]))
    if os.environ.get("MGBENCH_CAPACITY_CONTRACT") is not None:
        # Candidate 18: this rank's ACTUAL capacity at the frozen fraction, before any model allocation; every rank
        # must pass (collective MIN) before anything is allocated or any optimizer step can begin.
        from ..window_contract import all_ranks_agree, worker_capacity_check
        cap = worker_capacity_check(json.loads(os.environ["MGBENCH_CAPACITY_CONTRACT"]), local_rank,
                                    int(torch.cuda.get_device_properties(device).total_memory))
        with open(os.path.join(out, f"rank{rank}_capacity.json"), "x") as f:
            json.dump(cap, f, default=str)
        if not all_ranks_agree(cap["pass"], dist, device):
            raise RuntimeError(f"capacity: not every rank passed (this rank: {cap['problems']}); no allocation, "
                               "no optimizer step")
    torch.manual_seed(spec["seed"])
    # Precision: ONLY the legacy interface the retained recipe uses (torch 2.11 raises if legacy and new
    # fp32_precision settings are mixed); the new getters are read back for the record only.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if spec.get("fp32_reference"):
        # Declared FP32 control: literal fp32 compute (no TF32), fixed cuDNN algorithms, math-only SDPA (its matmuls
        # follow the fp32 matmul setting; flash/cuDNN SDPA do not take fp32 and the memory-efficient kernel is not used).
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    apply_matched_policy(spec)
    precision_record = _precision_record(spec)

    sampler = GpuSampler(spec["gpus"]).start() if rank == 0 else None
    t_setup = time.perf_counter()
    if spec.get("result_kind") == "first_boundary_gradient":
        if (spec.get("warmup_steps"), spec.get("measure_steps")) != (0, 1) or not spec.get("canonical_init"):
            raise RuntimeError("first-boundary result kind needs warmup 0, measure_steps 1 and the canonical start")
        spec["_out_dir"] = out
    trainer = SETUP[spec["framework"]](spec, entry, fw, device)

    ga = spec.get("grad_accum", 1)
    seed = spec["seed"] * 1000 + rank
    if entry["kind"] == "causal_lm":
        hf_cfg = load_hf_config_dict(entry)
        flops_per_sample = lm_train_flops_per_token(hf_cfg, spec["seq_len"]) * spec["seq_len"]
    else:
        flops_per_sample = vision_train_flops_per_sample(entry)
    if spec.get("global_inputs"):
        # Rank-independent global batches (numerical reference diagnostics only).
        data = GlobalSynthetic(entry["kind"], micro_batch=spec["micro_batch"], grad_accum=ga, world=world, rank=rank,
                               device=device, seed=spec["seed"] * 1000, steps=spec["warmup_steps"] + spec["measure_steps"],
                               vocab_size=hf_cfg["vocab_size"] if entry["kind"] == "causal_lm" else None,
                               seq_len=spec.get("seq_len"), image_size=entry.get("image_size"),
                               num_classes=entry.get("num_classes"), dtype=trainer.data_dtype)
    elif entry["kind"] == "causal_lm":
        data = SyntheticTokens(hf_cfg["vocab_size"], spec["seq_len"], spec["micro_batch"], device, seed)
    else:
        data = SyntheticImages(entry["image_size"], entry["num_classes"], spec["micro_batch"], device, seed,
                               dtype=trainer.data_dtype)

    torch.cuda.synchronize()
    setup_s = time.perf_counter() - t_setup
    setup_peak = torch.cuda.max_memory_allocated()

    # Canonical-state control: fingerprint the framework's full logical state after wrapping (collective).
    canonical_check = None
    if spec.get("canonical_init"):
        gathered = trainer.state_fingerprint()
        if rank == 0:
            canonical_check = _compare_canonical(trainer.canonical, gathered, spec)
    torch.cuda.reset_peak_memory_stats()

    alloc_at_train_start = _alloc_counters()
    if spec.get("result_kind") == "first_boundary_gradient":
        return _run_first_boundary(spec, out, trainer, data, ga, gathered, canonical_check, device_records,
                                   identity_chain, precision_record, sampler, alloc_at_train_start, setup_s, setup_peak,
                                   entry, fw, mem_fraction)
    for _ in range(spec["warmup_steps"]):
        if trainer.diag is not None:
            trainer.diag.measure_index = -1
        trainer.step([data.next() for _ in range(ga)])

    times = []
    losses = []
    for key in ("optimizer_calls", "successful_updates", "skipped_updates", "engine_update_boundaries",
                "gradient_checks", "nonfinite_gradient_steps"):
        trainer.update_audit[key] = 0
    trainer.ledger.start_measured()  # warmup attempts stay recorded, separately
    torch.cuda.synchronize()
    dist.barrier()
    alloc_at_measure_start = _alloc_counters()
    if sampler:
        sampler.mark_start()
    t0 = time.perf_counter()
    for step_index in range(spec["measure_steps"]):
        if trainer.diag is not None:
            trainer.diag.measure_index = step_index
        ts = time.perf_counter()
        loss = trainer.step([data.next() for _ in range(ga)])
        losses.append(loss.detach())
        torch.cuda.synchronize()
        times.append(time.perf_counter() - ts)
    dist.barrier()
    window_s = time.perf_counter() - t0
    if sampler:
        sampler.mark_stop()

    loss_values = [float(value.float()) for value in losses]
    audit = dict(trainer.update_audit)
    audit["expected_logical_updates"] = spec["measure_steps"]
    audit["all_updates_observed"] = (audit["successful_updates"] == spec["measure_steps"]
                                     and audit["optimizer_calls"] >= audit["successful_updates"]
                                     and audit["skipped_updates"] == 0)
    audit["losses_all_finite"] = all(bool(torch.isfinite(torch.tensor(value))) for value in loss_values)
    audit["losses"] = loss_values
    # Per attempt and phase: microbatch losses, boundary coverage vs expected units, base calls, update evidence.
    audit["boundary_ledger"] = trainer.ledger.summary(spec["measure_steps"], spec["warmup_steps"], ga)
    # Stage S1a: full-tensor reference write/compare after the timed loop (not part of any timing).
    reference_comparison, final_checks = finalize_reference(trainer, audit["boundary_ledger"]["measured"])
    per_rank_audit = [None] * world
    dist.all_gather_object(per_rank_audit, audit)
    if spec.get("result_kind") == "full_control_5update":      # Candidate 16: rank records + rank-0 control record
        from ..train.boundary_ledger import assess as _assess
        upd = _assess([audit["boundary_ledger"]], checks_requested=True, update_evidence_requested=True,
                      measurement_role="diagnostic", diagnostics_in_timed_path=True)["update_state"]["status"]
        row = {"rank": rank, "nvml_index": (identity_chain.get("nvml") or {}).get("index"),
               "uuid": device_records[0].get("uuid"), "pci_bus_id": device_records[0].get("pci_bus_id")}
        payload = rank_payload(rank, audit["boundary_ledger"], data.input_digests, upd,
                               not _identity_checks(device_records) and not identity_chain.get("problems"), row)
        write_json(os.path.join(out, f"rank{rank}_record.json"), payload)
        payloads = [None] * world
        dist.all_gather_object(payloads, payload)
        if rank == 0:
            snaps = {"writer_rank": 0, "dtheta_after_update": spec["measure_steps"],
                     "files": sorted(p for p in ("grad_b1.f32", "dtheta_final.f32") if os.path.exists(os.path.join(out, p)))}
            canon_ok = bool(canonical_check and canonical_check.get("module_state_matches_canonical"))
            write_json(os.path.join(out, "control_record.json"),
                       build_control_record(spec, payloads, canonical_ok=canon_ok, snapshots=snaps))

    # Numerical diagnostics (all ranks; collective gathers for reference tensors).
    diag_record, reference = None, None
    if trainer.diag is not None:
        if spec.get("reference_capture"):
            trainer.diag.capture_reference_params_final(deepspeed=spec["framework"].startswith("zero"))
        if trainer.extra_diag is not None:
            trainer.diag.record["reduction"].update(trainer.extra_diag())
        diag_record = trainer.diag.record
        reference = trainer.diag.reference
    per_rank_diag = [None] * world
    dist.all_gather_object(per_rank_diag, diag_record)
    legacy_valid = all(item["all_updates_observed"] and item["losses_all_finite"]
                       and not item["nonfinite_gradient_steps"] for item in per_rank_audit)
    assessment = assess([a["boundary_ledger"] for a in per_rank_audit],
                        checks_requested=bool(spec.get("numerical_checks")),
                        update_evidence_requested=bool(spec.get("numerical_checks")),
                        measurement_role=spec.get("measurement_role", "measurement"),
                        diagnostics_in_timed_path=bool(trainer.diag is not None or spec.get("numerical_checks")))
    run_state_reasons = list(assessment["execution_reasons"])
    if not legacy_valid:
        run_state_reasons.append("legacy update audit failed")
    if assessment["execution"] != "ok" or not legacy_valid or assessment["gradient"]["status"] == "nonfinite_detected" \
            or assessment["update_state"]["status"] == "failed":
        run_state = "invalid_numerics_or_updates"
    elif spec.get("numerical_checks") and (assessment["gradient"]["status"] != "complete_at_every_measured_boundary"
                                           or assessment["update_state"]["status"] != "verified_sampled_every_base_call"):
        run_state = "unverified_gradients"
    else:
        run_state = "ok"

    # Per-step time = slowest rank; memory gathered from every rank.
    t = torch.tensor(times, device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    mem = torch.tensor([torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved(), setup_peak],
                       device=device, dtype=torch.float64)
    mem_all = [torch.zeros_like(mem) for _ in range(world)]
    dist.all_gather(mem_all, mem)
    # Allocator pressure: cudaMalloc retries (= cache flushes) and OOMs, during training and during the timed window.
    end = _alloc_counters()
    alloc = torch.tensor([end[k] - alloc_at_train_start[k] for k in ALLOC_KEYS]
                         + [end[k] - alloc_at_measure_start[k] for k in ALLOC_KEYS], device=device, dtype=torch.float64)
    dist.all_reduce(alloc, op=dist.ReduceOp.MAX)
    loss_val = float(loss.float())

    if rank != 0:
        return None
    sampler.stop()
    write_json(os.path.join(out, "telemetry_raw.json"), sampler.raw_record())
    # Global logical loss per measured step: mean over ranks (equal sample counts per rank).
    global_loss = [sum(a["losses"][k] for a in per_rank_audit) / world for k in range(spec["measure_steps"])]
    if reference is not None and spec.get("reference_capture"):
        initial = (trainer.canonical or {}).get("param_initial_slices")
        torch.save({"spec": spec, "global_loss_per_step": global_loss, "param_initial": initial, **reference},
                   os.path.join(out, "reference_tensors.pt"))
    samples_per_step = spec["micro_batch"] * ga * world
    samples_s = samples_per_step * spec["measure_steps"] / window_s
    gpu = aggregate_gpu_summary(sampler.summary())
    env = environment(spec["gpus"])
    gpu_name = env["gpus"][0]["name"]
    result = {
        "status": run_state,
        "status_reasons": run_state_reasons,
        "assessment": assessment,
        "runtime_devices": device_records + [_runtime_device_record("end of run")],
        "identity_chain": identity_chain,
        "precision": precision_record,
        "canonical_start": spec.get("canonical_start"),
        "reference_comparison": reference_comparison,
        "final_checks": final_checks,
        "process": _process_record(),
        "effective_config_sha256": __import__("hashlib").sha256(json.dumps(
            {"spec": spec, "framework_config": fw, "mem_fraction": mem_fraction}, sort_keys=True,
            default=str).encode()).hexdigest(),
        "input_digests": getattr(data, "input_digests", None),
        "torch_memory": {"max_memory_allocated": torch.cuda.max_memory_allocated(),
                         "max_memory_reserved": torch.cuda.max_memory_reserved(),
                         "note": "allocator scope only; CUDA context, NCCL and library memory are outside it"},
        "verification_note": ("per-rank optimizer_update_audit_per_rank[*].verification keeps attempted steps, observed "
                              "gradients, logical updates and base optimizer calls distinct; anything not observed is "
                              "listed as unverified"),
        "protocol_version": "scientific-v2",
        "measurement_kind": "numerical_diagnostic" if spec.get("numerical_checks") else "throughput_with_update_counter",
        "optimizer_update_audit_per_rank": per_rank_audit,
        "global_loss_per_step": global_loss,
        "loss_definition": "mean over micro-batches of the unscaled per-micro-batch loss; global = mean over ranks",
        "numerical_diagnostics_per_rank": per_rank_diag,
        "canonical_state_check": canonical_check,
        "reference_tensors_file": "reference_tensors.pt" if reference is not None and spec.get("reference_capture") else None,
        "spec": spec,
        "framework_config": fw,
        "model_entry": entry,
        "n_params": trainer.n_params,
        "trainer_info": trainer.info,
        "setup_s": setup_s,
        "window_s": window_s,
        "samples_per_step": samples_per_step,
        "throughput_samples_s": samples_s,
        "throughput_tokens_s": samples_s * spec["seq_len"] if entry["kind"] == "causal_lm" else None,
        "throughput_per_gpu_samples_s": samples_s / world,
        "step_time": step_stats(t.tolist()),
        "flops_per_sample": flops_per_sample,
        "tflops_per_gpu": flops_per_sample * samples_s / world / 1e12,
        "mfu": mfu(flops_per_sample, samples_s, world, gpu_name),
        "peak_mem_alloc_gib": max(m[0].item() for m in mem_all) / GIB,
        "peak_mem_reserved_gib": max(m[1].item() for m in mem_all) / GIB,
        "setup_peak_alloc_gib": max(m[2].item() for m in mem_all) / GIB,
        "peak_mem_alloc_gib_per_rank": [m[0].item() / GIB for m in mem_all],
        "mem_fraction": mem_fraction,
        "allocator_train": {k: int(v) for k, v in zip(ALLOC_KEYS, alloc[:len(ALLOC_KEYS)].tolist())},
        "allocator_measured": {k: int(v) for k, v in zip(ALLOC_KEYS, alloc[len(ALLOC_KEYS):].tolist())},
        "final_loss": loss_val,
        "loss_finite": bool(torch.isfinite(torch.tensor(loss_val))),
        "gpu": gpu,
        "gpu_per_device": sampler.summary(),
        "telemetry_complete": set(sampler.summary()) == {str(g) for g in spec["gpus"]},
        "env": env,
        "executed_source": executed_source_record(),
    }
    write_json(os.path.join(out, "result.json"), result)
    print(f"[mgbench] {spec['run_id']}: {samples_s:.1f} samples/s"
          + (f", {result['throughput_tokens_s']:.0f} tok/s" if result["throughput_tokens_s"] else "")
          + f", step {result['step_time']['median_ms']:.1f} ms, peak {result['peak_mem_alloc_gib']:.1f} GiB,"
          f" util {gpu.get('gpu_util_mean_pct', float('nan')):.0f}%, MFU {result['mfu'] or 0:.3f}", flush=True)
    return result


def _run_first_boundary(spec, out, trainer, data, ga, gathered, canonical_check, device_records, identity_chain,
                        precision_record, sampler, alloc0, setup_s, setup_peak, entry, fw, mem_fraction) -> dict:
    """Candidate 14: one accumulation boundary of `ga` real microbatches, gradient extracted at the boundary, NO
    optimizer update; parameters/masters re-fingerprinted afterwards and required to equal the realized start."""
    trainer.ledger.start_measured()
    if trainer.diag is not None:
        trainer.diag.measure_index = 0
    rec = SdpaDispatchRecorder()
    mbs, order = apply_microbatch_order([data.next() for _ in range(ga)], spec.get("microbatch_order"))
    with rec:
        trainer.step(mbs)
    torch.cuda.synchronize()
    summary = trainer.ledger.summary(1, 0, ga)
    after = trainer.state_fingerprint()
    unchanged = after == gathered
    comparison = trainer.reference.finish(summary["measured"]["microbatch_losses"]) if trainer.reference else None
    assessment = assess_first_boundary(summary, ga, unchanged, comparison)
    end = _alloc_counters()
    if sampler:
        sampler.stop()
        write_json(os.path.join(out, "telemetry_raw.json"), sampler.raw_record())
    ok = assessment["execution"] == "ok" and assessment["gradient"]["status"] == "complete_at_the_boundary" \
        and assessment["numerical_comparison"] == "comparable"
    result = {"status": "ok" if ok else "invalid_or_unverified", "result_kind": "first_boundary_gradient",
              "assessment": assessment, "boundary_ledger": summary, "reference_comparison": comparison,
              "parameters_unchanged": unchanged, "state_fingerprint_before": gathered, "state_fingerprint_after": after,
              "microbatch_order": order,
              "canonical_state_check": canonical_check, "canonical_start": spec.get("canonical_start"),
              "input_digests": getattr(data, "input_digests", None), "precision": precision_record,
              "sdpa_dispatch": rec.record(), "runtime_devices": device_records + [_runtime_device_record("end of run")],
              "identity_chain": identity_chain, "process": _process_record(), "update_audit": dict(trainer.update_audit),
              "torch_memory": {"max_memory_allocated": torch.cuda.max_memory_allocated(),
                               "max_memory_reserved": torch.cuda.max_memory_reserved(),
                               "scope": "reset after setup/canonical check; read after extraction and comparison"},
              "setup_s": setup_s, "setup_peak_alloc_gib": setup_peak / GIB, "mem_fraction": mem_fraction,
              "allocator_after_setup": {k: end[k] - alloc0[k] for k in ALLOC_KEYS}, "spec": spec,
              "framework_config": fw, "model_entry": entry, "trainer_info": trainer.info,
              "executed_source": executed_source_record(), "protocol_version": "candidate14-first-boundary"}
    write_json(os.path.join(out, "result.json"), result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spec = read_json(args.spec)
    rank = int(os.environ.get("RANK", 0))
    code = 0
    try:
        result = run(spec, args.out)
        if result is not None and result["status"] != "ok":
            code = 4
    except torch.OutOfMemoryError as e:
        code = 3
        write_json(os.path.join(args.out, f"error_rank{rank}.json"),
                   {"status": "oom", "error": str(e)[:2000],
                    "max_alloc_gib": torch.cuda.max_memory_allocated() / GIB})
        print(f"[mgbench] rank {rank} OOM: {str(e)[:300]}", file=sys.stderr, flush=True)
    except Exception as e:  # noqa: BLE001
        code = 1
        write_json(os.path.join(args.out, f"error_rank{rank}.json"),
                   {"status": "failed", "error": repr(e)[:2000], "traceback": traceback.format_exc()[-6000:]})
        traceback.print_exc()
    finally:
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:  # noqa: BLE001
                pass
    sys.exit(code)


if __name__ == "__main__":
    main()
