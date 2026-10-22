"""Controls for the owner-approved, single-GPU, clean-only smoke test (A6000 stage 20261016s).

Pure functions (testable on CPU with stubs): frozen-setting checks, the pre-launch gate split into "busy" (wait) and
"fatal" (stop the smoke) reasons, the time budget inside an absolute window, and evaluation of a case result against
expectations frozen before any result was observed.
"""
from __future__ import annotations

import os
from pathlib import Path

# Existing refusal rules, copied from configs/cluster.yaml as approved; the supervisor refuses if they differ.
FROZEN_IDLE = {"poll_s": 5, "consecutive_polls": 3, "max_wait_s": 1800, "max_mem_mib": 100, "max_util_pct": 0,
               "max_power_w": 60, "max_temp_c": 50, "min_free_ram_gib": 150}
FROZEN_GUARD = {"planned_w_per_gpu": 300, "start_budget_w": 1500, "abort_total_w": 1700, "abort_after_s": 5,
                "max_load1": 96, "max_gpus_per_run": 2, "node_gpu_count": 8}
FATAL_MARKERS = ("unknown", "unreadable", "malformed", "implausible", "not finite", "missing", "not numeric",
                 "expected 8", "UUID", "MPS", "authorization", "authorized", "candidate", "ceiling", "settings",
                 "power limit", "scope", "enrollment", "host lease")


def settings_problems(cluster: dict) -> list[str]:
    idle = cluster.get("idle") or {}
    guard = idle.get("node_guard") or {}
    out = [f"settings: idle.{k} = {idle.get(k)!r}, approved {v!r}" for k, v in FROZEN_IDLE.items() if idle.get(k) != v]
    out += [f"settings: node_guard.{k} = {guard.get(k)!r}, approved {v!r}" for k, v in FROZEN_GUARD.items()
            if guard.get(k) != v]
    hold = cluster.get("launch_hold")
    if not (isinstance(hold, dict) and hold.get("active") is True):
        out.append("settings: repository launch_hold is not active (it must stay active; the smoke uses an exception)")
    return out


def smoke_config_problems(smoke: dict) -> list[str]:
    from .safety import UUID_RE

    out = []
    if smoke.get("allowed_gpus") != [4]:
        out.append(f"smoke allowlist must be exactly [4], got {smoke.get('allowed_gpus')!r}")
    pins = smoke.get("gpu_uuids") or {}
    if set(map(int, pins)) != {4} or not UUID_RE.match(str(pins.get(4, pins.get("4", "")))):
        out.append(f"smoke UUID binding malformed: {pins!r}")
    if smoke.get("max_cases", 0) > 4 or len(smoke.get("cases", [])) > smoke.get("max_cases", 0):
        out.append("more than four cases configured")
    for c in smoke.get("cases", []):
        if c.get("diagnostic_fault") or c.get("n_gpus", 1) != 1:
            out.append(f"case {c.get('id')} is not a clean single-GPU case")
    lim = smoke.get("limits", {})
    if lim.get("worker_cpu_quota_pct", 0) + lim.get("supervisor_cpu_quota_pct", 0) > 1600:
        out.append("aggregate CPU quota above 16 CPU-equivalents")
    if lim.get("worker_memory_bytes", 0) + lim.get("supervisor_memory_bytes", 0) > 48 * 2**30:
        out.append("aggregate memory above 48 GiB")
    t = smoke.get("timing", {})
    limits = {"window_s": 1800, "startup_s": 120, "gate_s": 60, "child_s": 240, "grace_s": 60, "reap_s": 10,
              "final_reserve_s": 120}
    for k, v in limits.items():
        if not isinstance(t.get(k), (int, float)) or t[k] > v or t[k] <= 0:
            out.append(f"timing.{k} = {t.get(k)!r} outside the approved maximum {v}")
    n = len(smoke.get("cases", []))
    if not out and t["startup_s"] + n * (t["gate_s"] + t["child_s"] + t["grace_s"] + t["reap_s"]) + \
            t["final_reserve_s"] > t["window_s"]:
        out.append("startup + cases x (gate + child + grace + reap) + final reserve exceeds the window")
    order = [c.get("id") for c in smoke.get("cases", [])]
    if order != ["c3_zero2_checked", "c1_ddp_checked", "c2_fsdp_checked", "c4_ddp_unchecked"]:
        out.append(f"case order {order} is not the frozen diagnostic order c3, c1, c2, c4")
    pci = (smoke.get("gpu_pci") or {})
    if set(map(int, pci)) != {4}:
        out.append(f"smoke PCI binding malformed: {pci!r}")
    return out


def classify(problems: list[str]) -> tuple[list[str], list[str]]:
    """(busy, fatal): busy = wait and re-check; fatal = unreadable/unknown/identity/authorization -> stop the smoke."""
    fatal = [p for p in problems if any(m in p for m in FATAL_MARKERS)]
    return [p for p in problems if p not in fatal], fatal


def case_budget(now: float, deadline: float, timing: dict) -> dict:
    """One more case fits only if gate + child + grace + reap AND the final reserve fit before the (monotonic)
    deadline. Phases are never extended; a case that cannot fit is refused."""
    per_case = timing["gate_s"] + timing["child_s"] + timing["grace_s"] + timing["reap_s"]
    remaining = deadline - now
    return {"remaining_s": remaining, "per_case_s": per_case, "fits": remaining >= per_case + timing["final_reserve_s"],
            "gate_deadline": now + timing["gate_s"], "child_s": timing["child_s"], "grace_s": timing["grace_s"],
            "reap_s": timing["reap_s"]}


def gpu_power_limit_problems(snapshot: dict, gpu: int, planned_w: float) -> list[str]:
    g = snapshot.get("gpus", {}).get(gpu, {})
    lim = g.get("power_limit_w")
    if lim is None or abs(lim - planned_w) > 0.5:
        return [f"power limit of gpu{gpu} is {lim!r} W, the guard assumes {planned_w} W -> do not launch"]
    return []


# ------------------------------------------------------------------------------------------ expectations
def _get(d, path, default=None):
    for k in path.split("."):
        if isinstance(d, list):
            d = d[int(k)] if int(k) < len(d) else default
        elif isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d


def evaluate_case(result: dict | None, rc, case: dict, frozen: dict, *, candidate_src: Path,
                  interpreter: str, uuid: str, worker_cpu_max: str, worker_memory_max: str) -> dict:
    """Compare one case result with the expectations frozen before the smoke started."""
    checks = []

    def check(name, ok, observed=None):
        checks.append({"check": name, "pass": bool(ok), "observed": observed})

    check("worker exit code 0", rc == 0, rc)
    check("result.json present", result is not None)
    if result is None:
        return {"pass": False, "checks": checks}
    a = result.get("assessment", {})
    led = _get(result, "optimizer_update_audit_per_rank.0.boundary_ledger", {}) or {}
    w, m = led.get("warmup", {}), led.get("measured", {})
    proc = result.get("process", {})
    check("execution assessment ok", a.get("execution") == "ok", a.get("execution_reasons"))
    check("runtime CUDA devices: exactly one, authorized UUID, every stage",
          all(r.get("device_count") == 1 and r.get("uuid") == uuid and r.get("current_device") == 0
              for r in result.get("runtime_devices", [])) and len(result.get("runtime_devices", [])) >= 3,
          [(r.get("stage"), r.get("device_count"), r.get("uuid"), r.get("current_device"))
           for r in result.get("runtime_devices", [])])
    check("executed mgbench from the candidate", str(proc.get("mgbench_file", "")).startswith(str(candidate_src)),
          proc.get("mgbench_file"))
    check("explicit interpreter", proc.get("sys_executable") == interpreter, proc.get("sys_executable"))
    cg = proc.get("cgroup") or {}                       # keys are cgroup file names such as "cpu.max"
    check("worker cgroup CPU limit", cg.get("cpu.max") == worker_cpu_max, cg.get("cpu.max"))
    check("worker cgroup memory limit", cg.get("memory.max") == worker_memory_max, cg.get("memory.max"))
    check("allocator fraction", result.get("mem_fraction") == case.get("mem_fraction", frozen.get("mem_fraction")),
          result.get("mem_fraction"))
    check(f"warmup: {frozen['warmup_attempts']} logical attempt(s), "
          f"{frozen['warmup_attempts'] * frozen['grad_accum']} microbatches",
          (w.get("logical_attempts"), w.get("microbatches")) == (frozen["warmup_attempts"],
                                                                 frozen["warmup_attempts"] * frozen["grad_accum"]),
          (w.get("logical_attempts"), w.get("microbatches")))
    check(f"measured: {frozen['measured_attempts']} logical attempts, "
          f"{frozen['measured_attempts'] * frozen['grad_accum']} microbatches",
          (m.get("logical_attempts"), m.get("microbatches")) == (frozen["measured_attempts"],
                                                                 frozen["measured_attempts"] * frozen["grad_accum"]),
          (m.get("logical_attempts"), m.get("microbatches")))
    check("every microbatch loss observed and finite",
          w.get("microbatch_losses_all_finite") and m.get("microbatch_losses_all_finite"),
          (w.get("microbatch_losses"), m.get("microbatch_losses")))
    check("counter updates 3, no framework skip",
          (m.get("counter_updates"), m.get("framework_reported_skips")) == (frozen["measured_attempts"], 0),
          (m.get("counter_updates"), m.get("framework_reported_skips")))
    check("base-call multiplicity matches the implementation", m.get("base_call_multiplicity_matches") is True,
          (m.get("base_calls_per_attempt"), m.get("base_call_multiplicity_expected")))
    chain = result.get("identity_chain") or {}
    check("identity chain NVML -> driver -> (DeepSpeed) -> torch checked with no problems before model setup",
          chain.get("checked") is True and chain.get("problems") == [] and chain.get("driver") is not None,
          {k: chain.get(k) for k in ("problems", "nvml", "driver", "deepspeed")})
    if case["framework"].startswith("zero"):
        ds = chain.get("deepspeed") if isinstance(chain.get("deepspeed"), dict) else {}
        check("DeepSpeed maps logical 0 to the approved NVML index", ds.get("logical") == 0 and
              ds.get("nvml_index") == ds.get("expected_index") == 4 and ds.get("uuid") == uuid, ds)
    check("all torch identity records carry the approved UUID and a PCI bus id",
          all(r.get("pci_bus_id") for r in result.get("runtime_devices", [])),
          [(r.get("stage"), r.get("pci_bus_id")) for r in result.get("runtime_devices", [])])
    prefix = frozen.get("eligibility_prefix", "ineligible: smoke")
    check(f"measurement eligibility: {prefix}, not a measurement",
          str(a.get("measurement_eligibility", "")).startswith(prefix), a.get("measurement_eligibility"))
    if case["numerical_checks"]:
        check("gradient coverage complete at every measured boundary",
              _get(a, "gradient.status") == "complete_at_every_measured_boundary"
              and m.get("boundary_coverage") == ["complete"] * frozen["measured_attempts"],
              (_get(a, "gradient.status"), m.get("boundary_coverage")))
        check("sampled AdamW update evidence verified at every base call",
              _get(a, "update_state.status") == "verified_sampled_every_base_call",
              (_get(a, "update_state.status"), m.get("update_evidence_verdicts")))
        check("pre-step sample plan sampled completely at every measured boundary",
              m.get("sample_plan_complete_each_boundary") is True, m.get("sampled_names_per_boundary"))
        check("every base call has samples whose expected change exceeds tolerance (distinguishes a no-op)",
              bool(m.get("discriminating_samples_per_call")) and all((x or 0) > 0 for x in m["discriminating_samples_per_call"]),
              m.get("discriminating_samples_per_call"))
        lay = led.get("layout") or {}
        if case["framework"] in ("ddp", "fsdp") and frozen.get("native_sample_plan_sha256"):
            import hashlib as _h
            import json as _j
            got = _h.sha256(_j.dumps(lay.get("sample_plan"), sort_keys=True).encode()).hexdigest()
            check("runtime sample plan identical to the plan frozen before the window (names, sizes, indices)",
                  got == frozen["native_sample_plan_sha256"], got)
        check("independent model-to-unit map without problems", lay.get("problems") in ([], None) and
              (lay.get("unit_map") or lay.get("model_to_partition_map")), lay.get("problems"))
        check("status ok", result.get("status") == "ok", result.get("status"))
    else:
        check("gradient validity null / not checked",
              _get(a, "gradient.status") == "not_checked" and _get(a, "gradient.gradient_validity") is None,
              a.get("gradient"))
        check("update/state evidence not collected (unverified)", _get(a, "update_state.status") == "not_collected",
              a.get("update_state"))
        check("no boundary inspected", m.get("inspected_boundaries") == 0, m.get("inspected_boundaries"))
    if case["framework"] in ("zero1", "zero2"):
        lay = _get(result, "optimizer_update_audit_per_rank.0.deepspeed_layout", {}) or {}
        check("DeepSpeed optimizer class with identity-verified partition accounting",
              str(lay.get("optimizer_class", "")).endswith("DeepSpeedZeroOptimizer") and lay.get("accounting")
              and lay.get("model_to_partition_map"), (lay.get("optimizer_class"), lay.get("model_to_partition_map")))
        steps = m.get("steps", [])
        check("bf16 module partition equals rounded fp32 master (sampled), every measured attempt",
              len(steps) == frozen["measured_attempts"] and all(_get(s, "module_bf16_vs_master.pass") for s in steps),
              [s.get("module_bf16_vs_master") for s in steps])
    return {"pass": all(c["pass"] for c in checks), "checks": checks}


# ------------------------------------------------------------------- stage S1a: single-GPU numerical diagnostics
DIAG_SCOPE = "train-diagnostics"
DIAG_ORDER = ["r0_fp32_control", "a1_ddp_bf16_autocast", "a2_fsdp2_bf16", "a3_zero1_bf16", "a4_zero2_bf16"]
DIAG_FRAMEWORK = {"r0_fp32_control": "ddp", "a1_ddp_bf16_autocast": "ddp", "a2_fsdp2_bf16": "fsdp",
                  "a3_zero1_bf16": "zero1", "a4_zero2_bf16": "zero2"}
DIAG_TIMING = {"window_s": 2700, "startup_s": 120, "gate_s": 60, "child_s": 300, "grace_s": 60, "reap_s": 10,
               "final_reserve_s": 120}
DIAG_COMMON = {"model": "qwen2.5-1.5b", "seq_len": 1024, "micro_batch": 1, "grad_accum": 4,
               "activation_checkpointing": False, "warmup_steps": 0, "measure_steps": 3, "numerical_checks": True,
               "canonical_init": True, "canonical_start": "bf16_representable", "global_inputs": True,
               "measurement_role": "diagnostic"}
MAX_ALLOCATOR_FRACTION = 0.92
REFERENCE_FILES = ("grad_b1.f32", "grad_b1.f32.index.json", "dtheta_final.f32", "dtheta_final.f32.index.json",
                   "reference_scalars.json")


def diag_config_problems(cfg: dict, feasibility: dict | None = None) -> list[str]:
    """Refusal reasons for the S1a diagnostics configuration (empty = approved shape). With `feasibility`, every
    allocator fraction must equal the frozen feasibility value of an eligible case."""
    from .safety import UUID_RE

    out = []
    if cfg.get("scope") != DIAG_SCOPE:
        out.append(f"diagnostics scope must be {DIAG_SCOPE!r}, got {cfg.get('scope')!r}")
    if cfg.get("allowed_gpus") != [4]:
        out.append(f"diagnostics allowlist must be exactly [4], got {cfg.get('allowed_gpus')!r}")
    pins, pci = cfg.get("gpu_uuids") or {}, cfg.get("gpu_pci") or {}
    if set(map(int, pins)) != {4} or not UUID_RE.match(str(pins.get(4, pins.get("4", "")))):
        out.append(f"diagnostics UUID binding malformed: {pins!r}")
    if set(map(int, pci)) != {4}:
        out.append(f"diagnostics PCI binding malformed: {pci!r}")
    cases = cfg.get("cases", [])
    if not isinstance(cfg.get("max_cases"), int) or cfg["max_cases"] > 5 or len(cases) > cfg["max_cases"]:
        out.append("more than five cases configured")
    order = [c.get("id") for c in cases]
    if order != DIAG_ORDER:
        out.append(f"case order {order} is not the frozen order {DIAG_ORDER}")
    for c in cases:
        cid = c.get("id")
        if c.get("diagnostic_fault") or c.get("n_gpus", 1) != 1:
            out.append(f"case {cid} is not a clean single-GPU case")
        if DIAG_FRAMEWORK.get(cid) != c.get("framework"):
            out.append(f"case {cid}: framework {c.get('framework')!r}, frozen {DIAG_FRAMEWORK.get(cid)!r}")
        f = c.get("mem_fraction")
        if isinstance(f, bool) or not isinstance(f, (int, float)) or not 0 < f <= MAX_ALLOCATOR_FRACTION:
            out.append(f"case {cid}: allocator fraction {f!r} outside (0, {MAX_ALLOCATOR_FRACTION}]")
        if feasibility is not None:
            out += feasibility_problems(feasibility, [c])        # Candidate 14: recomputed, cushion kept, no clamp
        if cid == DIAG_ORDER[0]:
            if not (c.get("fp32_reference") is True and c.get("reference_role") == "write" and not c.get("depends_on")):
                out.append(f"case {cid} must be the independent FP32 control that writes the reference")
        elif c.get("fp32_reference") or c.get("reference_role") != "compare" or c.get("depends_on") != DIAG_ORDER[0]:
            out.append(f"case {cid} must compare against, and depend on, {DIAG_ORDER[0]}")
    common = cfg.get("common") or {}
    for k, v in DIAG_COMMON.items():
        if common.get(k) != v or type(common.get(k)) is not type(v):
            out.append(f"common.{k} = {common.get(k)!r}, frozen {v!r}")
    lim = cfg.get("limits", {})
    if lim.get("worker_cpu_quota_pct", 0) + lim.get("supervisor_cpu_quota_pct", 0) > 1600:
        out.append("aggregate CPU quota above 16 CPU-equivalents")
    if lim.get("worker_memory_bytes", 0) + lim.get("supervisor_memory_bytes", 0) > 48 * 2**30:
        out.append("aggregate memory above 48 GiB")
    t = cfg.get("timing", {})
    for k, v in DIAG_TIMING.items():
        if not isinstance(t.get(k), (int, float)) or t[k] > v or t[k] <= 0:
            out.append(f"timing.{k} = {t.get(k)!r} outside the approved maximum {v}")
    if not any(p.startswith("timing.") for p in out) and t["startup_s"] + len(cases) * (
            t["gate_s"] + t["child_s"] + t["grace_s"] + t["reap_s"]) + t["final_reserve_s"] > t["window_s"]:
        out.append("startup + cases x (gate + child + grace + reap) + final reserve exceeds the window")
    rb, fd = cfg.get("reference_budget_bytes"), cfg.get("min_free_disk_bytes")
    if not isinstance(rb, int) or not 0 < rb <= 13_500_000_000:
        out.append(f"reference_budget_bytes {rb!r} outside (0, 13.5e9]")
    if not isinstance(fd, int) or fd < 100_000_000_000:
        out.append(f"min_free_disk_bytes {fd!r} below 100 GB")
    return out


def diag_experiment(case: dict, cfg: dict, ref_dir: Path) -> dict:
    """The one-run experiment for an S1a case: the frozen common recipe plus the case's framework, allocator fraction
    and reference role (expanded by config.expand_experiment exactly like every other training run)."""
    wl = {"model": cfg["common"]["model"], "frameworks": [case["framework"]], "variant": case["id"],
          "mem_fraction": case["mem_fraction"], "fp32_reference": bool(case.get("fp32_reference")),
          "reference_role": case["reference_role"], "reference_dir": str(ref_dir)}
    return {"name": f"validation_v2_s1a_diag_{case['id']}", "kind": "train", "repeats": 1, "gpu_counts": [1],
            "defaults": dict(cfg["common"]), "workloads": [wl]}


# ---------------------------------------------------------------- Candidate 14: corrected admission predicate
ADMISSION = {"cushion": "1.02", "step": "0.01", "cap": "0.92", "min_headroom_gib": "2"}


def _num(x, name):
    """Strict numeric input: no bool, no string, finite, > 0. Returns Decimal of the literal JSON/YAML value."""
    import math
    from decimal import Decimal

    if isinstance(x, bool) or not isinstance(x, (int, float, Decimal)):
        raise ValueError(f"{name}: {x!r} is not a number")
    d = Decimal(str(x)) if not isinstance(x, Decimal) else x
    if not d.is_finite() or (isinstance(x, float) and not math.isfinite(x)) or d <= 0:
        raise ValueError(f"{name}: {x!r} is not a finite positive number")
    return d


def admission(predicted_reserved_gib, total_gib, non_allocator_gib) -> dict:
    """Allocator fraction that KEEPS the uncertainty cushion: f = ceil_0.01(1.02 x predicted / total). Refused (never
    clamped) when f > 0.92; refused when whole-device headroom total - f x total - non_allocator < 2 GiB."""
    from decimal import Decimal, ROUND_CEILING

    c, step, cap, hr = (Decimal(ADMISSION[k]) for k in ("cushion", "step", "cap", "min_headroom_gib"))
    out = {"eligible": False, "fraction": None, "reasons": []}
    try:
        p, t, na = (_num(predicted_reserved_gib, "predicted_reserved_gib"), _num(total_gib, "total_gib"),
                    _num(non_allocator_gib, "non_allocator_gib"))
    except ValueError as exc:
        out["reasons"].append(str(exc))
        return out
    need = c * p
    f = (need / t / step).to_integral_value(ROUND_CEILING) * step
    out.update(required_allowance_gib=str(need), required_fraction=str(f), cap=str(cap))
    if f > cap:
        out["reasons"].append(f"required fraction {f} (cushioned {need} GiB of {t} GiB) exceeds the cap {cap}: ineligible")
        return out
    headroom = t - f * t - na
    out.update(fraction=float(f), allowance_gib=str(f * t), whole_device_headroom_gib=str(headroom))
    if headroom < hr:
        out["reasons"].append(f"whole-device headroom {headroom} GiB < {hr} GiB: ineligible")
        return out
    out["eligible"] = True
    return out


def feasibility_problems(feasibility, cases: list[dict]) -> list[str]:
    """Admission of configured cases against a feasibility record (recomputed here, not trusted): every case present,
    inputs well formed, recomputed fraction == configured fraction == recorded fraction, recorded eligibility not
    contradicting the recomputation."""
    if not isinstance(feasibility, dict) or not isinstance(feasibility.get("cases"), dict):
        return ["feasibility evidence missing or malformed"]
    out = []
    for c in cases:
        cid = c.get("id")
        rec = feasibility["cases"].get(cid)
        if not isinstance(rec, dict):
            out.append(f"case {cid}: no feasibility record")
            continue
        a = admission(rec.get("predicted_reserved_gib"), feasibility.get("torch_total_memory_gib"),
                      feasibility.get("non_allocator_gib"))
        if not a["eligible"]:
            out.append(f"case {cid}: not admissible ({'; '.join(a['reasons'])})")
            continue
        f = c.get("mem_fraction")
        if isinstance(f, bool) or f != a["fraction"] or rec.get("fraction") != a["fraction"] or rec.get("eligible") is not True:
            out.append(f"case {cid}: configured {f!r} / recorded {rec.get('fraction')!r} (eligible={rec.get('eligible')!r}) "
                       f"!= recomputed {a['fraction']!r}")
    return out


# ------------------------------------------------------------------ Candidate 14: per-case private caches
CACHE_SUBDIRS = ("triton", "triton_home", "inductor", "torch_extensions", "xdg", "mpl")


def private_cache_env(cache: Path) -> dict:
    """Every cache location the stack may write, pointed into one private directory (set BEFORE imports)."""
    cache = Path(cache)
    return {"TRITON_CACHE_DIR": str(cache / "triton"), "TRITON_HOME": str(cache / "triton_home"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"), "TORCH_EXTENSIONS_DIR": str(cache / "torch_extensions"),
            "XDG_CACHE_HOME": str(cache / "xdg"), "MPLCONFIGDIR": str(cache / "mpl")}


def make_case_cache(root: Path, case_id: str, seed_dir: Path | None, seed_sha256: str | None) -> dict:
    """Fresh owner-only cache for ONE case. Only a verified binary seed (one .so, hash-checked before and after the
    copy) is placed in it; nothing from any other case. Returns the provenance and the expected manifest."""
    import hashlib
    import os
    import shutil

    d = Path(root) / case_id
    d.mkdir(mode=0o700)
    os.chmod(d, 0o700)
    for s in CACHE_SUBDIRS:
        (d / s).mkdir(mode=0o700)
    prov = {"case": case_id, "dir": str(d), "seed": None, "problems": []}
    if seed_dir is not None:
        files = [p for p in Path(seed_dir).iterdir()]
        sos = [p for p in files if p.suffix == ".so" and p.is_file() and not p.is_symlink()]
        if len(sos) != 1 or len(files) != 1:
            prov["problems"].append(f"seed {seed_dir} must hold exactly one regular .so file, found {[p.name for p in files]}")
        else:
            got = hashlib.sha256(sos[0].read_bytes()).hexdigest()
            if got != seed_sha256 or Path(seed_dir).stat().st_uid != os.getuid():
                prov["problems"].append(f"seed not verified (sha256 {got}, expected {seed_sha256})")
            else:
                dst = d / "triton" / Path(seed_dir).name
                dst.mkdir(mode=0o700)
                shutil.copy2(sos[0], dst / sos[0].name)
                after = hashlib.sha256((dst / sos[0].name).read_bytes()).hexdigest()
                prov["seed"] = {"source": str(sos[0]), "sha256_source": got, "copied_to": str(dst / sos[0].name),
                                "sha256_copy": after}
                if after != seed_sha256:
                    prov["problems"].append("seed copy hash mismatch")
    prov["expected_manifest"] = cache_manifest(d)
    return prov


def cache_manifest(d: Path) -> dict:
    import hashlib

    out = {}
    for p in sorted(Path(d).rglob("*")):
        rel = p.relative_to(d).as_posix()
        out[rel] = ("dir" if p.is_dir() and not p.is_symlink() else
                    "symlink" if p.is_symlink() else hashlib.sha256(p.read_bytes()).hexdigest())
    return out


def cache_preload_problems(d: Path, expected: dict) -> list[str]:
    """Immediately before launch: the case cache must equal its expected manifest exactly. Any extra entry (a pickle,
    a lock, a foreign .so, a symlink) or changed hash refuses the launch BEFORE anything can load it."""
    got = cache_manifest(d)
    extra = sorted(set(got) - set(expected))
    missing = sorted(set(expected) - set(got))
    changed = sorted(k for k in set(got) & set(expected) if got[k] != expected[k])
    out = []
    if extra:
        out.append(f"case cache holds unexpected entries {extra[:10]} -> do not launch")
    if missing or changed:
        out.append(f"case cache entries missing {missing[:10]} or changed {changed[:10]} -> do not launch")
    return out


# --------------------------------------------- Candidate 14: first-boundary diagnostic scope (proposed; not authorized)
FB_SCOPE = "train-first-boundary"
FB_CASES = {   # id -> framework and the per-case precision fields that must be present (and nothing else)
    "G0_fp32_control": {"framework": "ddp", "fp32_reference": True},
    "G1_ddp_autocast": {"framework": "ddp"},
    "G3_ddp_bf16_params": {"framework": "ddp", "param_dtype": "bf16"},
    "G4_ds_bf16": {"framework": "zero2"},
    # Candidate 15: EXPLORATORY single-GPU order analogue - explicitly NOT the pre-registered control C (which needs
    # world 2 and five updates); it leaves the original validation requirement open.
    "X1_fp32_order_permuted": {"framework": "ddp", "fp32_reference": True, "microbatch_order": [3, 2, 1, 0]},
}
FB_ORDER = list(FB_CASES)
FB_TIMING = {"window_s": 2700, "startup_s": 120, "gate_s": 60, "child_s": 240, "grace_s": 60, "reap_s": 10,
             "kill_wait_s": 10, "metadata_call_s": 3, "telemetry_timeout_s": 5, "post_case_s": 10, "final_s": 30,
             "sup_kill_margin_s": 10}       # Candidate 19: the Candidate 18 timing model's keys (enroll_s must be 0)
FB_COMMON = dict(DIAG_COMMON, measure_steps=1, result_kind="first_boundary_gradient",
                 matched_policy={"sdpa": "math", "tf32": False, "bf16_reduced_precision_reduction": False})
FB_PRECISION_KEYS = ("fp32_reference", "param_dtype", "ds_precision", "ds_grad_accum_dtype", "microbatch_order")


def fb_config_problems(cfg: dict, feasibility) -> list[str]:
    from .safety import UUID_RE

    out = []
    if cfg.get("scope") != FB_SCOPE:
        out.append(f"scope must be {FB_SCOPE!r}")
    if cfg.get("allowed_gpus") != [4]:
        out.append("allowlist must be exactly [4]")
    pins, pci = cfg.get("gpu_uuids") or {}, cfg.get("gpu_pci") or {}
    if set(map(int, pins)) != {4} or not UUID_RE.match(str(pins.get(4, pins.get("4", "")))) or set(map(int, pci)) != {4}:
        out.append("GPU-4 UUID/PCI binding malformed")
    cases = cfg.get("cases", [])
    if [c.get("id") for c in cases] != FB_ORDER or cfg.get("max_cases") != len(FB_ORDER):
        out.append(f"cases must be exactly {FB_ORDER} (max_cases {len(FB_ORDER)})")
    for c in cases:
        want = FB_CASES.get(c.get("id"), {})
        got = {k: c[k] for k in ("framework",) + FB_PRECISION_KEYS if k in c}
        if got != want or c.get("diagnostic_fault") or c.get("n_gpus", 1) != 1:
            out.append(f"case {c.get('id')}: fields {got} differ from the frozen declaration {want}")
    out += feasibility_problems(feasibility, cases)
    common = cfg.get("common") or {}
    for k, v in FB_COMMON.items():
        if common.get(k) != v or type(common.get(k)) is not type(v):
            out.append(f"common.{k} = {common.get(k)!r}, frozen {v!r}")
    lim = cfg.get("limits", {})
    if lim.get("worker_cpu_quota_pct", 0) + lim.get("supervisor_cpu_quota_pct", 0) > 1600 or \
            lim.get("worker_memory_bytes", 0) + lim.get("supervisor_memory_bytes", 0) > 48 * 2**30:
        out.append("aggregate limits above 16 CPU-equivalents / 48 GiB")
    t = cfg.get("timing", {})
    for k, v in FB_TIMING.items():
        if not isinstance(t.get(k), (int, float)) or isinstance(t.get(k), bool) or t[k] > v or t[k] <= 0:
            out.append(f"timing.{k} = {t.get(k)!r} outside (0, {v}]")
    if t.get("enroll_s") != 0 or isinstance(t.get("enroll_s"), bool):
        out.append("timing.enroll_s must be 0 (no enrollment in the first-boundary scope)")
    if not any(p.startswith("timing.") for p in out):
        from .window_contract import timing_model
        inclusive = timing_model(t, n_gpus_watched=1, n_cases=len(cases))["inclusive_window_s"]
        if inclusive > t["window_s"]:
            out.append(f"inclusive budget {inclusive} s exceeds the window {t['window_s']} s")
    if not str(cfg.get("lease_dir", "")).startswith("/"):
        out.append("first-boundary config must name the absolute project-wide host lease directory")
    kt = cfg.get("known_torch_total_bytes") or {}
    if set(map(int, kt)) != {4} or not isinstance(kt.get(4, kt.get("4")), int) or kt.get(4, kt.get("4")) < 2**30:
        out.append("known_torch_total_bytes must give GPU 4's measured Torch total in bytes")
    rb, fd = cfg.get("reference_budget_bytes"), cfg.get("min_free_disk_bytes")
    if not isinstance(rb, int) or isinstance(rb, bool) or not 0 < rb <= 32_000_000_000 or \
            not isinstance(fd, int) or isinstance(fd, bool) or fd < 100_000_000_000:
        out.append("storage budgets malformed")
    return out


def fb_experiment(case: dict, cfg: dict, ref_dir: Path) -> dict:
    wl = {"model": cfg["common"]["model"], "frameworks": [case["framework"]], "variant": case["id"],
          "mem_fraction": case["mem_fraction"], "reference_dir": str(ref_dir),
          **{k: case[k] for k in FB_PRECISION_KEYS if k in case}}
    return {"name": f"validation_c14_fb_{case['id']}", "kind": "train", "repeats": 1, "gpu_counts": [1],
            "defaults": dict(cfg["common"]), "workloads": [wl]}


def evaluate_fb_case(result: dict | None, rc, case: dict, frozen: dict, *, uuid: str, candidate_src: Path,
                     interpreter: str) -> dict:
    """Integrity of one first-boundary case. The numerical comparison is reported, never graded here (no new
    equivalence margin has been justified); zero == zero is not certifiable."""
    checks = []

    def check(name, ok, observed=None):
        checks.append({"check": name, "pass": bool(ok), "observed": observed})

    check("worker exit code 0", rc == 0, rc)
    check("result.json present", result is not None)
    if result is None:
        return {"pass": False, "checks": checks}
    a = result.get("assessment") or {}
    check("result kind first_boundary_gradient", result.get("result_kind") == "first_boundary_gradient")
    check("execution ok: 1 attempt, 4 microbatches, 0 optimizer calls, 0 updates, finite losses",
          a.get("execution") == "ok", a.get("execution_reasons"))
    check("gradient coverage complete at the single boundary", (a.get("gradient") or {}).get("status") ==
          "complete_at_the_boundary", a.get("gradient"))
    check("update/state verification not_applicable (not passed)", (a.get("update_state") or {}).get("status") ==
          "not_applicable", a.get("update_state"))
    check("parameters (and masters) unchanged after the boundary", result.get("parameters_unchanged") is True)
    check("comparison certifiable (complete, nonzero)", a.get("numerical_comparison") == "comparable",
          a.get("numerical_comparison"))
    cs = result.get("canonical_state_check") or {}
    check("realized canonical start", cs.get("module_state_matches_canonical") is True and
          (not case["framework"].startswith("zero") or cs.get("optimizer_master_matches_canonical") is True))
    dig = (result.get("input_digests") or {}).get("per_step_microbatch_sha256")
    check("the 4 inputs equal the first frozen S1a step", dig == frozen["input_per_step_microbatch_sha256"][:1], dig)
    devs = {r.get("stage"): r for r in result.get("runtime_devices", [])}
    for st in ("after set_device", "after set_per_process_memory_fraction", "end of run"):
        r = devs.get(st) or {}
        check(f"identity at '{st}'", r.get("device_count") == 1 and r.get("uuid") == uuid and bool(r.get("pci_bus_id")))
    chain = result.get("identity_chain") or {}
    check("identity chain checked before allocation without problems", chain.get("checked") is True
          and chain.get("problems") == [])
    proc = result.get("process") or {}
    check("executed the candidate with the explicit interpreter", str(proc.get("mgbench_file", "")).startswith(
        str(candidate_src)) and proc.get("sys_executable") == interpreter)
    pr = result.get("precision") or {}
    check("matched policy recorded as declared", pr.get("matched_policy") == FB_COMMON["matched_policy"] and
          (pr.get("legacy") or {}).get("cuda_matmul_allow_tf32") is False and
          (result.get("sdpa_dispatch") or {}).get("dispatch") == "constrained to math by flags", pr.get("matched_policy"))
    check("allocator fraction as admitted", result.get("mem_fraction") == case["mem_fraction"], result.get("mem_fraction"))
    return {"pass": all(c["pass"] for c in checks), "checks": checks}


def diag_budget_seconds(timing: dict, n_cases: int) -> int:
    return timing["startup_s"] + n_cases * (timing["gate_s"] + timing["child_s"] + timing["grace_s"]
                                            + timing["reap_s"]) + timing["final_reserve_s"]


def file_sha256(path: Path, chunk: int = 1 << 24) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def bound_file_problems(bound: dict) -> list[str]:
    """Every file bound into the authorization must still hash to its frozen value."""
    if not isinstance(bound, dict) or not bound:
        return ["authorization binds no design files"]
    out = []
    for path, want in sorted(bound.items()):
        try:
            got = file_sha256(Path(path))
        except OSError as exc:
            out.append(f"bound file unreadable: {path} ({exc!r})")
            continue
        if got != want:
            out.append(f"bound file changed since authorization: {path}")
    return out


def reference_receipt(ref_dir: Path, budget_bytes: int, *, expected_tensors: int, expected_elements: int) -> dict:
    """After the FP32 control: hash every reference file in full (independently of the writer's streaming hash),
    and check presence, budget, and the writer's own index totals. `valid` gates every dependent case."""
    import json

    files, problems = {}, []
    present = sorted(p.name for p in Path(ref_dir).iterdir()) if Path(ref_dir).is_dir() else []
    extra = [n for n in present if n not in REFERENCE_FILES]
    if extra:
        problems.append(f"unexpected files in the reference directory: {extra}")
    for name in REFERENCE_FILES:
        p = Path(ref_dir) / name
        if not p.is_file():
            problems.append(f"reference file missing: {name}")
            continue
        files[name] = {"bytes": p.stat().st_size, "sha256": file_sha256(p)}
    total = sum(v["bytes"] for v in files.values())
    if total > budget_bytes:
        problems.append(f"reference files {total} B exceed the budget {budget_bytes} B")
    for snap in ("grad_b1.f32", "dtheta_final.f32"):
        idx = Path(ref_dir) / f"{snap}.index.json"
        if not idx.is_file() or snap not in files:
            continue
        meta = json.loads(idx.read_text())
        if meta.get("data_sha256") != files[snap]["sha256"]:
            problems.append(f"{snap}: file sha256 differs from the writer's streaming sha256")
        if meta.get("bytes") != files[snap]["bytes"] or meta.get("elements", 0) * 4 != files[snap]["bytes"]:
            problems.append(f"{snap}: size disagrees with its index")
        if meta.get("tensors") != expected_tensors or meta.get("elements") != expected_elements:
            problems.append(f"{snap}: {meta.get('tensors')} tensors / {meta.get('elements')} elements, expected "
                            f"{expected_tensors} / {expected_elements}")
        names = [e["name"] for e in meta.get("index", [])]
        if len(set(names)) != len(names):
            problems.append(f"{snap}: duplicate names in the index")
    return {"files": files, "total_bytes": total, "budget_bytes": budget_bytes, "problems": problems,
            "valid": not problems}


def reference_recheck(ref_dir: Path, receipt: dict) -> list[str]:
    """Before each dependent case: small files re-hashed in full, large data files by size (each tensor's sha256 is
    verified against the hashed index inside the worker when it is read)."""
    out = []
    if not receipt or not receipt.get("valid"):
        return ["FP32 control reference is not valid -> dependent case blocked"]
    for name, rec in receipt["files"].items():
        p = Path(ref_dir) / name
        if not p.is_file() or p.stat().st_size != rec["bytes"]:
            out.append(f"reference file {name} missing or resized since the receipt")
        elif not name.endswith(".f32") and file_sha256(p) != rec["sha256"]:
            out.append(f"reference file {name} changed since the receipt")
    return out


def _within(x, lo, hi) -> bool:
    import math

    return isinstance(x, (int, float)) and math.isfinite(x) and lo <= x <= hi


def evaluate_diag_case(result: dict | None, rc, case: dict, frozen: dict, tolerances: dict, **kw) -> dict:
    """S1a: integrity checks (malfunction class: a failure stops the window), gross-error bounds (also stop) and the
    pre-registered graded metrics (recorded; a graded fail does not stop later independent cases)."""
    case = {**case, "numerical_checks": case.get("numerical_checks", True)}   # S1a: pinned in `common` (validated)
    base = evaluate_case(result, rc, case, frozen, **kw)
    checks = base["checks"]

    def check(name, ok, observed=None):
        checks.append({"check": name, "pass": bool(ok), "observed": observed})

    out = {"integrity_checks": checks, "gross_checks": [], "grades": {}, "category": None, "witnesses": {}}
    if result is None:
        out.update(integrity_pass=False, gross_pass=False, pass_=False, overall_grade="not_evaluated")
        return out
    # Identity: pre-allocation receipt vs. the post-cap and end-of-run rechecks, kept separate.
    devs = {r.get("stage"): r for r in result.get("runtime_devices", [])}
    for stage, label in (("after set_device", "pre-allocation torch identity receipt (after set_device, before any "
                                              "model allocation)"),
                         ("after set_per_process_memory_fraction", "identity recheck after the allocator cap"),
                         ("end of run", "identity recheck at the end of the run")):
        r = devs.get(stage) or {}
        check(label, r.get("device_count") == 1 and r.get("uuid") == kw["uuid"] and bool(r.get("pci_bus_id")),
              {k: r.get(k) for k in ("device_count", "uuid", "pci_bus_id")})
    cs = result.get("canonical_state_check") or {}
    check("declared common representable start realized after cast/wrap (full-state SHA-256)",
          result.get("canonical_start") == "bf16_representable" and cs.get("module_state_matches_canonical") is True
          and (not case["framework"].startswith("zero") or cs.get("optimizer_master_matches_canonical") is True),
          {k: cs.get(k) for k in ("expected_module_state", "module_state_matches_canonical",
                                  "optimizer_master_matches_canonical")})
    dig = (result.get("input_digests") or {}).get("per_step_microbatch_sha256")
    check("every microbatch input equals the frozen input plan", dig == frozen["input_per_step_microbatch_sha256"],
          "match" if dig == frozen["input_per_step_microbatch_sha256"] else dig)
    pr = result.get("precision") or {}
    leg, sd = pr.get("legacy") or {}, pr.get("sdpa_enabled") or {}
    env_clean = all(v is None for v in (pr.get("env_overrides") or {"missing": "x"}).values())
    if case.get("fp32_reference"):
        want = (leg.get("cuda_matmul_allow_tf32") is False and leg.get("cudnn_allow_tf32") is False
                and leg.get("float32_matmul_precision") == "highest" and leg.get("cudnn_deterministic") is True
                and leg.get("cudnn_benchmark") is False and sd.get("math") is True and sd.get("flash") is False
                and sd.get("mem_efficient") is False and sd.get("cudnn") is False and pr.get("autocast_bf16") is False)
    else:
        want = (leg.get("cuda_matmul_allow_tf32") is True and leg.get("cudnn_allow_tf32") is True
                and pr.get("autocast_bf16") is (case["framework"] == "ddp"))
    check("precision state as declared (legacy interface only; no environment override)", want and env_clean,
          {"legacy": leg, "sdpa": sd, "autocast_bf16": pr.get("autocast_bf16"), "env": pr.get("env_overrides")})
    br = _get(result, "optimizer_update_audit_per_rank.0.optimizer_branch") or {}
    check("AdamW execution branch resolved and recorded", br.get("branch") is not None, br.get("branch"))
    rcmp = result.get("reference_comparison") or {}
    n_t, n_e = frozen["schema"]["unique_trainable_tensors"], frozen["schema"]["trainable_elements"]
    snaps = rcmp.get("snapshots") or {}
    if case.get("reference_role") == "write":
        for s in ("grad_b1", "dtheta_final"):
            m = snaps.get(s) or {}
            check(f"reference snapshot {s} written in full ({n_t} tensors, {n_e} elements)",
                  m.get("tensors") == n_t and m.get("elements") == n_e and m.get("bytes") == 4 * n_e,
                  {k: m.get(k) for k in ("tensors", "elements", "bytes", "data_sha256")})
        check("reference scalars written", rcmp.get("role") == "write" and rcmp.get("scalars_written") is True,
              rcmp.get("role"))
        integrity = all(c["pass"] for c in checks)
        out.update(integrity_pass=integrity, gross_pass=integrity, pass_=integrity,
                   overall_grade="reference" if integrity else "invalid_reference")
        return out
    # ---- comparison cases
    for s in ("grad_b1", "dtheta_final"):
        m = snaps.get(s) or {}
        check(f"{s}: full coverage ({n_t} tensors, {n_e} elements, each tensor SHA-verified, no duplicate/unexpected)",
              rcmp.get("role") == "compare" and m.get("coverage_complete") is True and m.get("compared_tensors") == n_t
              and m.get("compared_elements") == n_e and not m.get("problems"),
              {k: m.get(k) for k in ("coverage_complete", "compared_tensors", "compared_elements", "n_missing",
                                     "problems")})
    lr = rcmp.get("loss_rel") or []
    check("all 12 raw microbatch losses compared", rcmp.get("loss_count_match") is True and len(lr) == 12
          and all(x is not None for x in lr), len(lr))
    gn = rcmp.get("grad_norm_ratios") or []
    check("gradient norms compared at all 3 boundaries, every tensor",
          len(gn) == 3 and all(g.get("tensors") == n_t and not g.get("missing") for g in gn),
          [(g.get("tensors"), g.get("missing")) for g in gn])
    mr = rcmp.get("moment_norm_ratios") or {}
    check("moment norms compared for every tensor", mr.get("tensors") == n_t and not mr.get("missing"),
          (mr.get("tensors"), mr.get("missing")))
    if case["framework"].startswith("zero"):
        fc = (result.get("final_checks") or {}).get("module_bf16_vs_master_full") or {}
        out["gross_checks"].append({"check": "DeepSpeed bf16 module equals bf16(fp32 master), every element",
                                    "pass": fc.get("pass") is True and fc.get("compared") == n_e, "observed": fc})
    integrity = all(c["pass"] for c in checks)
    gb = tolerances["gross_error_bounds"]
    g1, dt_ = snaps.get("grad_b1") or {}, snaps.get("dtheta_final") or {}
    lo, hi = gb["dtheta_final"]["global_norm_ratio"]
    out["gross_checks"] += [
        {"check": f"dtheta_final global cosine >= {gb['dtheta_final']['global_cosine_min']}",
         "pass": _within(dt_.get("global_cosine"), gb["dtheta_final"]["global_cosine_min"], 1.0 + 1e-9),
         "observed": dt_.get("global_cosine")},
        {"check": f"dtheta_final global norm ratio in [{lo}, {hi}]", "pass": _within(dt_.get("global_norm_ratio"), lo, hi),
         "observed": dt_.get("global_norm_ratio")}]
    lo, hi = gb["grad_norm_global_ratio_boundaries_2_3"]
    for g in gn[1:]:
        out["gross_checks"].append({"check": f"gradient global norm ratio at boundary {g.get('boundary', 0) + 1} in "
                                             f"[{lo}, {hi}]", "pass": _within(g.get("global_norm_ratio"), lo, hi),
                                    "observed": g.get("global_norm_ratio")})
    for k in ("m", "v"):
        lo, hi = gb["moment_norm_global_ratio"][k]
        x = (mr.get("global") or {}).get(k)
        out["gross_checks"].append({"check": f"Adam {k} global norm ratio in [{lo}, {hi}]", "pass": _within(x, lo, hi),
                                    "observed": x})
    gross = all(c["pass"] for c in out["gross_checks"])
    cat = tolerances["categories"].get(case["id"])
    gm = tolerances["graded_metrics"]
    from .train.reference import grade

    values = {"grad_b1_global_rel_l2": g1.get("global_rel_l2"), "grad_b1_global_cosine": g1.get("global_cosine"),
              "loss_update1_max_rel": max(lr[:4]) if len(lr) >= 4 and None not in lr[:4] else None,
              "loss_all_max_rel": max(lr) if lr and None not in lr else None}
    for k, v in values.items():
        lim = gm[k][cat]
        out["grades"][k] = {"value": v, "pass_limit": lim["pass"], "marginal_limit": lim["marginal"],
                            "grade": grade(v, lim["pass"], lim["marginal"], higher_is_better=k.endswith("cosine"))}
    order = ["pass", "marginal", "fail", "missing"]
    overall = max((g["grade"] for g in out["grades"].values()), key=order.index)
    out["witnesses"] = {"grad_b1_worst_tensor": {k: (g1.get("worst_tensor_rel_l2") or {}).get(k)
                                                 for k in ("name", "numel", "rel_l2", "cosine", "max_abs_diff")},
                        "grad_b1_max_abs_diff": g1.get("max_abs_diff"),
                        "dtheta_worst_tensor": {k: (dt_.get("worst_tensor_rel_l2") or {}).get(k)
                                                for k in ("name", "numel", "rel_l2", "cosine")},
                        "dtheta_global_rel_l2": dt_.get("global_rel_l2"), "dtheta_max_abs_diff": dt_.get("max_abs_diff"),
                        "loss_rel": lr, "grad_norm_ratios": gn,
                        "moment_norm_ratios": {k: mr.get(k) for k in ("global", "min", "max", "zero_reference")},
                        "zero_reference_tensors": {"grad_b1": g1.get("zero_reference_tensors"),
                                                   "dtheta_final": dt_.get("zero_reference_tensors")}}
    out.update(category=cat, integrity_pass=integrity, gross_pass=gross, pass_=integrity and gross,
               overall_grade=overall if integrity and gross else "not_graded (integrity or gross failure)")
    return out


def cgroup_bytes(spec: str) -> int:
    units = {"K": 2**10, "M": 2**20, "G": 2**30}
    return int(spec[:-1]) * units[spec[-1]] if spec[-1] in units else int(spec)


def free_ram_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return 0.0


def active_units(prefix: str, timeout_s: float = 10.0) -> list[str]:
    import subprocess

    r = subprocess.run(["systemctl", "--user", "list-units", "--all", "--no-legend", "--plain", f"{prefix}*"],
                       capture_output=True, text=True, timeout=timeout_s)
    if r.returncode != 0:
        raise RuntimeError(f"systemctl list-units failed: {r.stderr.strip()}")
    names = [l.split()[0] for l in r.stdout.splitlines() if l.strip() and " active " in f" {l} "]
    return [n[:-len(".scope")] if n.endswith(".scope") else n for n in names]


def pid_in_cgroup(pid: int, cgroup: str | None) -> bool | None:
    try:
        line = next(l for l in Path(f"/proc/{pid}/cgroup").read_text().splitlines() if l.startswith("0::"))
    except (OSError, StopIteration):
        return None
    return cgroup is not None and line[3:] == cgroup


def env_for_worker(base_env: dict, *, src: Path, uuid: str, index: int, threads: int, cache: Path, run_id: str,
                   pci: str | None = None) -> dict:
    """Exactly one visible device, by integer PCI-order index (DeepSpeed 0.19.7 and similar tools parse
    CUDA_VISIBLE_DEVICES as integers - a UUID value failed DeepSpeed initialisation in smoke 20261016s case 3). The
    UUID binding is kept by the pre-launch NVML UUID check of that index and the in-process runtime CUDA UUID check
    (MGBENCH_EXPECTED_GPU_UUID); either mismatch refuses."""
    keep = {k: base_env[k] for k in ("HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TERM", "XDG_RUNTIME_DIR",
                                       "DBUS_SESSION_BUS_ADDRESS") if k in base_env}
    keep.update({
        "PATH": "/usr/local/bin:/usr/bin:/bin", "PYTHONPATH": str(src), "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        **({} if isinstance(index, (list, tuple)) else {
            "CUDA_VISIBLE_DEVICES": str(int(index)), "MGBENCH_EXPECTED_GPU_UUID": uuid,
            "MGBENCH_REQUIRE_SINGLE_DEVICE": "1", "MGBENCH_EXPECTED_GPU_INDEX": str(int(index)),
            **({"MGBENCH_EXPECTED_GPU_PCI": pci} if pci else {})}),
        "MGBENCH_RUN_ID": run_id,
        "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads), "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "NCCL_DEBUG": "WARN",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1", **private_cache_env(cache),
        "DS_BUILD_OPS": "0", "DO_NOT_TRACK": "1"})
    if isinstance(index, (list, tuple)):        # Candidate 17: declared ordered device set (rank r -> index[r])
        uu, pp = list(uuid), list(pci or [])
        if not (len(index) == len(uu) == len(pp) > 1) or len(set(index)) != len(index):
            raise ValueError(f"device set malformed: indices {index}, {len(uu)} uuids, {len(pp)} pcis")
        for k in ("MGBENCH_EXPECTED_GPU_UUID", "MGBENCH_EXPECTED_GPU_INDEX", "MGBENCH_EXPECTED_GPU_PCI",
                  "MGBENCH_REQUIRE_SINGLE_DEVICE"):
            keep.pop(k, None)
        keep.update({"CUDA_VISIBLE_DEVICES": ",".join(str(int(i)) for i in index),
                     "MGBENCH_EXPECTED_GPU_UUIDS": ",".join(uu), "MGBENCH_EXPECTED_GPU_PCIS": ",".join(pp),
                     "MGBENCH_EXPECTED_GPU_INDICES": ",".join(str(int(i)) for i in index),
                     "MGBENCH_REQUIRE_DEVICE_COUNT": str(len(index))})
    for k in list(keep):
        if k.startswith("CUDA_MPS"):
            del keep[k]
    return keep



def snapshot(root: Path) -> dict:
    """{relative path: {bytes, sha256, mtime}} of every file under root (runtime cache provenance)."""
    import hashlib

    out = {}
    if root.exists():
        for f in sorted(root.rglob("*")):
            if f.is_file():
                out[f.relative_to(root).as_posix()] = {"bytes": f.stat().st_size, "mtime": f.stat().st_mtime,
                                                       "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    return out


def snapshot_diff(before: dict, after: dict) -> dict:
    return {"created": {k: v for k, v in after.items() if k not in before},
            "modified": {k: {"before": before[k], "after": v} for k, v in after.items()
                         if k in before and before[k]["sha256"] != v["sha256"]},
            "removed": sorted(k for k in before if k not in after)}
