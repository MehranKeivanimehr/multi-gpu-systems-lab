"""Candidate 16: minimum two-device controls for the FP32 R5/C5 positive-control pair ONLY (pure functions, CPU-testable).

Scope `train-control-pair`: R5 on GPU 4 (world 1), C5 on GPUs 4 and 5 (world 2). Nothing here queries a device; the
admission functions evaluate a telemetry snapshot handed to them. Old single-GPU authorizations never admit a world-2
job (launch_controls.authorization_problems refuses len(gpus) != 1), and this scope never admits anything but R5/C5.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path

PAIR_SCOPE = "train-control-pair"
PAIR_GPUS = [4, 5]
CASE_GPUS = {"R5_fp32_world1_ga4": [4], "C5_fp32_world2_ga2": [4, 5]}
REQUIRED = ("id", "scope", "candidate_root", "candidate_digest", "gpu_indices", "gpu_uuids", "case_gpus", "max_gpus",
            "max_concurrent_jobs", "max_cases", "issued_utc", "expires_utc", "nonrenewable")
IDLE = {"max_mem_mib": 100, "max_util_pct": 0, "max_power_w": 60, "max_temp_c": 50}
GUARD = {"planned_w_per_gpu": 300, "start_budget_w": 1500, "abort_total_w": 1700, "abort_after_s": 5, "power_limit_w": 300}


ENROLL_SEQUENCE = ["enroll_gpu5", "R5_fp32_world1_ga4", "C5_fp32_world2_ga2"]


def pair_authorization_problems(auth_path, *, candidate_root: Path, uuids: dict, n_cases_done: int = 0,
                                expected_sha256: str | None = None, now: dt.datetime | None = None,
                                enrollment_policy_sha256: str | None = None) -> list[str]:
    from .launch_controls import candidate_digest, parse_utc
    from .safety import UUID_RE

    if auth_path is None:
        return ["no pair authorization given"]
    try:
        raw = Path(auth_path).read_bytes()
        auth = json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        return [f"pair authorization unreadable ({exc!r})"]
    if not isinstance(auth, dict):
        return ["pair authorization malformed (not an object)"]
    out = []
    if expected_sha256 and hashlib.sha256(raw).hexdigest() != expected_sha256:
        out.append("pair authorization file changed since it was issued")
    missing = [k for k in REQUIRED if auth.get(k) is None]
    if missing:
        return out + [f"pair authorization malformed or null: {missing}"]
    if Path(str(auth_path) + ".revoked").exists():
        out.append("pair authorization revoked")
    try:
        issued, expires = parse_utc(auth["issued_utc"]), parse_utc(auth["expires_utc"])
        now = now or dt.datetime.now(dt.timezone.utc)
        if not issued <= now < expires:
            out.append("pair authorization not valid now")
    except Exception as exc:  # noqa: BLE001
        out.append(f"pair authorization times malformed ({exc!r})")
    if auth["scope"] != PAIR_SCOPE:
        out.append(f"scope {auth['scope']!r} does not cover {PAIR_SCOPE!r}")
    if auth["nonrenewable"] is not True:
        out.append("pair authorization must be nonrenewable")
    if auth["gpu_indices"] != PAIR_GPUS or auth["max_gpus"] != 2 or auth["max_concurrent_jobs"] != 1:
        out.append(f"GPUs {auth['gpu_indices']} / max_gpus {auth['max_gpus']} / jobs {auth['max_concurrent_jobs']}: "
                   "required [4, 5] / 2 / 1")
    want = {str(k): v for k, v in uuids.items()}
    enroll = [k for k, v in want.items() if v is None]
    if set(want) != {"4", "5"} or auth["gpu_uuids"] != want or \
            not all(UUID_RE.match(str(v)) for k, v in want.items() if k not in enroll):
        out.append("per-index UUID binding mismatch")
    if enroll:     # Candidate 18: an unknown device only through the frozen enrollment policy, in this same window
        e = auth.get("enrollment")
        if enroll != ["5"] or not isinstance(e, dict) or e.get("indices") != [5] or e.get("sequence") != ENROLL_SEQUENCE \
                or enrollment_policy_sha256 is None or e.get("policy_sha256") != enrollment_policy_sha256:
            out.append("authorization does not cover enrollment of GPU 5 under the frozen policy followed by R5 and C5")
    if auth["case_gpus"] != CASE_GPUS or auth["max_cases"] != len(CASE_GPUS):
        out.append(f"case/GPU binding {auth['case_gpus']} / max_cases {auth['max_cases']} differs from {CASE_GPUS}")
    if Path(auth["candidate_root"]).resolve() != Path(candidate_root).resolve():
        out.append("pair authorization names another candidate")
    elif auth["candidate_digest"] != candidate_digest(Path(candidate_root)):
        out.append("candidate tree changed since authorization (digest mismatch)")
    if n_cases_done >= auth["max_cases"]:
        out.append(f"case ceiling reached ({n_cases_done} of {auth['max_cases']})")
    return out


def case_gpu_problems(case_id: str, gpus: list[int]) -> list[str]:
    want = CASE_GPUS.get(case_id)
    if want is None:
        return [f"case {case_id!r} is not part of the control pair"]
    return [] if list(gpus) == want else [f"case {case_id}: GPUs {gpus}, bound {want}"]


def rank_identity_problems(expected: list[dict], observed: list[dict]) -> list[str]:
    """expected/observed rows: {rank, nvml_index, uuid, pci_bus_id}. Every rank exactly once, on exactly its device."""
    out = []
    ranks = [o.get("rank") for o in observed]
    if len(set(ranks)) != len(ranks):
        out.append(f"duplicate rank rows {ranks}")
    exp = {e["rank"]: e for e in expected}
    miss, extra = sorted(set(exp) - set(ranks)), sorted(set(ranks) - set(exp), key=str)
    if miss or extra:
        out.append(f"rank rows missing {miss} / extra {extra}")
    for o in observed:
        e = exp.get(o.get("rank"))
        if e and any(o.get(k) != e[k] for k in ("nvml_index", "uuid", "pci_bus_id")):
            out.append(f"rank {o.get('rank')} on {(o.get('nvml_index'), o.get('uuid'))}, bound {(e['nvml_index'], e['uuid'])}")
    return out


def pair_admission_problems(snapshot: dict, gpus: list[int], *, idle: dict = IDLE, guard: dict = GUARD) -> list[str]:
    """Fresh per-GPU eligibility from a SUPPLIED snapshot {gpus: {index: {mem_used_mib, util_pct, power_w, temp_c,
    power_limit_w, processes}}}: each requested GPU idle on its own; start budget other GPUs' draw + planned x n <= 1500 W.
    Failures refuse; nothing is clamped or weakened."""
    out = []
    g = snapshot.get("gpus") or {}
    if len(g) != 8:
        out.append(f"{len(g)} GPUs in the snapshot, expected 8 -> unknown, do not launch")
    for i in gpus:
        s = g.get(i)
        if not isinstance(s, dict):
            out.append(f"gpu{i}: no telemetry -> unknown, do not launch")
            continue
        vals = [s.get(k) for k in ("mem_used_mib", "util_pct", "power_w", "temp_c", "power_limit_w")]
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in vals):
            out.append(f"gpu{i}: telemetry not numeric/finite -> unknown, do not launch")
            continue
        if s.get("processes") != []:
            out.append(f"gpu{i}: processes present or unreadable -> do not launch")
        if s["mem_used_mib"] > idle["max_mem_mib"] or s["util_pct"] > idle["max_util_pct"] or \
                s["power_w"] > idle["max_power_w"] or s["temp_c"] > idle["max_temp_c"]:
            out.append(f"gpu{i}: not idle")
        if abs(s["power_limit_w"] - guard["power_limit_w"]) > 0.5:
            out.append(f"gpu{i}: power limit {s['power_limit_w']} W, guard assumes {guard['power_limit_w']} W")
    others = [v.get("power_w") for k, v in g.items() if k not in gpus]
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in others):
        out.append("other GPUs' power unreadable -> unknown, do not launch")
    elif sum(others) + guard["planned_w_per_gpu"] * len(gpus) > guard["start_budget_w"]:
        out.append(f"start budget: other GPUs {sum(others):.0f} W + {guard['planned_w_per_gpu']} W x {len(gpus)} > "
                   f"{guard['start_budget_w']} W -> wait/refuse")
    return out


def collect_rank_records(run_dir: Path, world: int, max_bytes: int = 1 << 20) -> tuple[list[dict], list[str]]:
    """Bounded collection of rank{r}_record.json; missing, extra, oversized or unreadable records are problems."""
    out, recs = [], []
    present = sorted(Path(run_dir).glob("rank*_record.json"))
    want = {f"rank{r}_record.json" for r in range(world)}
    extra = sorted(p.name for p in present if p.name not in want)
    if extra:
        out.append(f"unexpected rank records {extra}")
    for r in range(world):
        p = Path(run_dir) / f"rank{r}_record.json"
        if not p.is_file():
            out.append(f"rank {r}: record missing")
            continue
        if p.stat().st_size > max_bytes:
            out.append(f"rank {r}: record larger than {max_bytes} B")
            continue
        try:
            rec = json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            out.append(f"rank {r}: record unreadable ({exc!r})")
            continue
        if rec.get("rank") != r:
            out.append(f"rank {r}: record claims rank {rec.get('rank')}")
        recs.append(rec)
    return recs, out


PAIR_COMMON = {"model": "qwen2.5-1.5b", "seq_len": 1024, "micro_batch": 1, "seed": 1234, "warmup_steps": 0,
               "measure_steps": 5, "numerical_checks": True, "canonical_init": True, "canonical_start": "bf16_representable",
               "global_inputs": True, "fp32_reference": True, "reference_role": "write",
               "result_kind": "full_control_5update", "measurement_role": "diagnostic"}
PAIR_CASES = {"R5_fp32_world1_ga4": {"framework": "ddp", "world": 1, "grad_accum": 4},
              "C5_fp32_world2_ga2": {"framework": "ddp", "world": 2, "grad_accum": 2}}


def pair_config_problems(cfg: dict, feasibility) -> list[str]:
    """The proposed R5/C5 pair: exact cases, GPUs, recipe, the corrected cushioned admission for EVERY GPU of a case,
    and one shared worker allowance for all ranks (never 46 GiB per rank)."""
    from .smoke import feasibility_problems

    out = []
    if cfg.get("scope") != PAIR_SCOPE or cfg.get("case_gpus") != CASE_GPUS or cfg.get("max_cases") != 2:
        out.append("scope/case GPUs/max_cases differ from the frozen pair")
    cases = cfg.get("cases") or []
    if [c.get("id") for c in cases] != list(PAIR_CASES):
        out.append(f"cases must be exactly {list(PAIR_CASES)}")
    for c in cases:
        want = PAIR_CASES.get(c.get("id"), {})
        if {k: c.get(k) for k in want} != want:
            out.append(f"case {c.get('id')}: {({k: c.get(k) for k in want})} != {want}")
        w, g = c.get("world"), c.get("grad_accum")
        if not (isinstance(w, int) and isinstance(g, int)) or isinstance(w, bool) or isinstance(g, bool) or w * g != 4:
            out.append(f"case {c.get('id')}: world x GA must be 4 (same global batch)")
    out += feasibility_problems(feasibility, cases)
    for k, v in PAIR_COMMON.items():
        if (cfg.get("common") or {}).get(k) != v:
            out.append(f"common.{k} differs from {v!r}")
    lim = cfg.get("limits") or {}
    if lim.get("worker_cpu_quota_pct", 0) > 1500 or lim.get("worker_memory_bytes", 0) > 46 * 2**30 or \
            lim.get("worker_cpu_quota_pct", 0) + lim.get("supervisor_cpu_quota_pct", 0) > 1600 or \
            lim.get("worker_memory_bytes", 0) + lim.get("supervisor_memory_bytes", 0) > 48 * 2**30:
        out.append("limits exceed 15 CPU / 46 GiB for ALL ranks together (+ supervisor 1 / 2 GiB)")
    t = cfg.get("timing") or {}
    from .window_contract import timing_model
    try:
        inclusive = timing_model(t, n_gpus_watched=len(PAIR_GPUS), n_cases=len(cases))["inclusive_window_s"]
    except ValueError as exc:
        inclusive = None
        out.append(f"timing model: {exc}")
    if inclusive is not None and (inclusive > t.get("window_s", 0) or t.get("window_s", 0) > 2700):
        out.append(f"inclusive budget {inclusive} s exceeds the window or the window exceeds 45 minutes")
    if not str(cfg.get("lease_dir", "")).startswith("/") or not cfg.get("enrollment_policy"):
        out.append("pair config must name an absolute host lease directory and the frozen enrollment policy")
    gu = cfg.get("gpu_uuids") or {}
    if gu.get(5) is not None or not gu.get(4):
        out.append("pair config must pin GPU 4 and leave GPU 5 to the frozen enrollment policy")
    return out


# ------------------------------------------------------------------ Candidate 17: supervisor integration helpers
def pair_experiment(case: dict, cfg: dict, attempt) -> dict:
    """One-run experiment for R5 or C5: the frozen common recipe + this case's world/GA/fraction; rank 0 writes the
    reference snapshots into the case's own directory."""
    d = dict(cfg["common"], grad_accum=case["grad_accum"], mem_fraction=case["mem_fraction"], reference_dir=str(attempt))
    return {"name": f"validation_c17_pair_{case['id']}", "kind": "train", "repeats": 1, "gpu_counts": [case["world"]],
            "defaults": d, "workloads": [{"model": cfg["common"]["model"], "frameworks": [case["framework"]],
                                          "variant": case["id"]}]}


def test_worker_hook(attempt, world: int) -> dict | None:
    """Harmless CPU stand-in for the worker, for supervisor integration tests ONLY. Active only when ALL hold: the
    loaded `pynvml` module declares itself a test stub (MGBENCH_TEST_STUB is True), CUDA is hidden in the supervisor,
    and MGBENCH_SUPERVISOR_TEST_WORKER names the script. Real NVML can never satisfy the first condition."""
    import os
    import sys

    nv = sys.modules.get("pynvml")
    script = os.environ.get("MGBENCH_SUPERVISOR_TEST_WORKER")
    if not (script and getattr(nv, "MGBENCH_TEST_STUB", False) is True and os.environ.get("CUDA_VISIBLE_DEVICES") == ""):
        return None
    passthrough = {k: v for k, v in os.environ.items() if k.startswith("MGBENCH_TEST_")}
    return {"cmd": [sys.executable, "-B", "-s", script, str(attempt), str(world)], "env": passthrough}


def evaluate_pair_case(attempt, rc, case: dict, frozen: dict, *, gpus: list[int], pins: dict, pci: dict, prestep,
                       test_hook: bool = False) -> dict:
    """Bounded, rank-identified evidence for one pair case. Missing/duplicate/extra rank output, identity rows that do
    not match the bound device of each rank, a failed pre-step, or a contract violation never pass."""
    from .control_c import run_contract_problems
    from .device_identity import normalize_pci

    checks = []

    def check(name, ok, observed=None):
        checks.append({"check": name, "pass": bool(ok), "observed": observed})

    world = len(gpus)
    check("worker exit code 0", rc == 0, rc)
    check("pre-step identity of the whole declared device set", isinstance(prestep, dict) and prestep.get("problems") == [],
          (prestep or {}).get("problems") if isinstance(prestep, dict) else prestep)
    recs, probs = collect_rank_records(attempt, world)
    check("one bounded record per rank, no missing/extra/duplicate", not probs and len(recs) == world, probs)
    exp_rows = [{"rank": r, "nvml_index": g, "uuid": pins.get(g), "pci_bus_id": normalize_pci(pci.get(g))}
                for r, g in enumerate(gpus)]
    obs_rows = []
    for r in recs:
        row = dict(r.get("identity_row") or {})
        row["pci_bus_id"] = normalize_pci(row.get("pci_bus_id"))
        try:
            row["nvml_index"] = int(row.get("nvml_index"))
        except (TypeError, ValueError):
            pass
        obs_rows.append(row)
    idp = rank_identity_problems(exp_rows, obs_rows)
    check("each rank on exactly its bound device (UUID + PCI + index)", not idp, idp)
    cr = Path(attempt) / "control_record.json"
    rec = json.loads(cr.read_text()) if cr.is_file() and cr.stat().st_size < (8 << 20) else None
    cp = run_contract_problems(rec, world=world, grad_accum=case["grad_accum"],
                               frozen={"recipe": frozen["recipe"], "global_row_digests": frozen["global_row_digests"]})
    check("R5/C5 evidence contract (5 updates per rank, sample map, rank-0 snapshots)", not cp, cp[:6])
    caps, cap_probs = [], []
    for r in range(world):                          # Candidate 18: every rank's own capacity, before any allocation
        p = Path(attempt) / f"rank{r}_capacity.json"
        try:
            c = json.loads(p.read_text()) if p.is_file() and p.stat().st_size < (1 << 20) else None
        except Exception:  # noqa: BLE001
            c = None
        if not isinstance(c, dict) or c.get("pass") is not True or c.get("rank") != r or c.get("index") != gpus[r]:
            cap_probs.append(f"rank {r}: capacity record missing or not passing ({(c or {}).get('problems')})")
        caps.append(c)
    check("every rank validated its own actual capacity before allocation; all ranks passed", not cap_probs, cap_probs)
    writers = sorted(p.name for p in Path(attempt).glob("*.f32"))
    check("only the two rank-0 snapshot files exist", writers == ["dtheta_final.f32", "grad_b1.f32"], writers)
    out = {"pass": all(c["pass"] for c in checks), "checks": checks}
    if test_hook:
        out["label"] = "TEST HOOK: harmless CPU stand-in; no GPU, no NCCL, no training"
    return out


def case_outcome(cleanup_confirmed, stop_reason, ev_pass) -> tuple[str, bool]:
    """(outcome, stop_window). Cleanup is decided FIRST: an unconfirmed cleanup can never become a successful result.
    Candidate 18: `cleanup_confirmed` may be a window_contract.classify_cleanup class; over budget never succeeds."""
    if cleanup_confirmed == "confirmed_over_budget":
        return "cleanup_over_budget", True
    if cleanup_confirmed not in (True, "confirmed"):
        return "cleanup_unconfirmed", True
    if stop_reason:
        return "stopped", True
    if ev_pass is not True:
        return "unexpected_result", True
    return "completed_valid", False
