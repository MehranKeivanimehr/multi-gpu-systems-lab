"""Candidate 18: the execution contract of ONE owner-approved R5/C5 window (pure functions + small file primitives).

1. Timing: one model generates every bound (admission, watcher, termination, scope stop, reap, final verification,
   supervisor stop allowance) and the supervisor's launch parameters. One clock: the authorization's absolute expiry.
2. Cleanup classification: confirmed | confirmed_over_budget | unconfirmed. Only the first can lead to success.
3. Enrollment: a frozen policy (in the candidate, bound by hash in the authorization) admits enrolling GPU 5 at the
   start of the SAME window; the observation is written once to a typed record OUTSIDE the candidate tree, sealed
   once, and every later identity check must match it.
4. Capacity: per device; NVML bytes (explicit units) are an UPPER bound for CUDA-usable memory, so the supervisor's
   pre-launch check is necessary-not-sufficient; each rank validates its actual Torch capacity before model
   allocation and all ranks must pass before any optimizer step.
5. Host lease: one project-wide lease directory; joint acquisition under an flock; never reclaimed automatically.
Nothing here queries a device; observations are handed in.
"""
from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import socket
from decimal import Decimal, ROUND_CEILING
from pathlib import Path

MIB, GIB = 2**20, 2**30
RECORD_TYPE = "mgbench.gpu_enrollment.v1"
SEAL_TYPE = "mgbench.gpu_enrollment_seal.v1"
LEASE_TYPE = "mgbench.host_device_lease.v1"
TIMING_KEYS = ("startup_s", "enroll_s", "gate_s", "child_s", "grace_s", "kill_wait_s", "reap_s", "metadata_call_s",
               "telemetry_timeout_s", "post_case_s", "final_s", "sup_kill_margin_s")


# ------------------------------------------------------------------ 1. timing model
def _pos(t: dict, k: str) -> float:
    v = t.get(k)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
        raise ValueError(f"timing.{k} = {v!r} is not a finite non-negative number")
    return float(v)


def timing_model(timing: dict, n_gpus_watched: int = 2, n_cases: int = 2) -> dict:
    """Every bound of the window, derived from the configured phase lengths. Units: seconds.

    cleanup_bound: scopes.cleanup_scope's worst path = before-state + SIGTERM + grace + last poll + SIGKILL +
    kill wait + last poll + final state + stop + final state = grace + kill_wait + 8 metadata calls (+1 s slack).
    watch_overshoot: one watcher iteration past the child limit = node snapshot + per-GPU process lists (each bounded
    by the telemetry timeout) + one unit listing (metadata call) (+1 s slack).
    supervisor stop allowance: what the supervisor needs after a SIGTERM from its manager = watch overshoot +
    cleanup + reap + final verification (+5 s slack). The supervisor's TimeoutStopSec is this value."""
    t = {k: _pos(timing, k) for k in TIMING_KEYS}
    m = t["metadata_call_s"]
    cleanup = t["grace_s"] + t["kill_wait_s"] + 8 * m + 1
    overshoot = t["telemetry_timeout_s"] * (1 + n_gpus_watched) + m + 1
    stop_and_reap = cleanup + t["reap_s"]
    case_total = t["gate_s"] + t["child_s"] + overshoot + stop_and_reap + t["post_case_s"]
    sup_stop = overshoot + stop_and_reap + t["final_s"] + 5
    worker_runtime_max = t["child_s"] + 2                    # from worker-unit activation, independent of the supervisor
    worker_worst = worker_runtime_max + t["grace_s"] + t["kill_wait_s"]
    inclusive = t["startup_s"] + t["enroll_s"] + n_cases * case_total + t["final_s"] + sup_stop + t["sup_kill_margin_s"]
    return {"inputs": t, "cleanup_bound_s": cleanup, "watch_overshoot_s": overshoot, "stop_and_reap_bound_s": stop_and_reap,
            "case_total_s": case_total, "supervisor_timeout_stop_s": math.ceil(sup_stop),
            "supervisor_kill_margin_s": t["sup_kill_margin_s"], "worker_runtime_max_s": int(worker_runtime_max),
            "worker_timeout_stop_s": int(t["grace_s"]), "worker_worst_case_s": worker_worst,
            "inclusive_window_s": math.ceil(inclusive), "n_cases": n_cases,
            "formula": "startup + enroll + n x (gate + child + overshoot + cleanup + reap + post_case) + final + "
                       "supervisor stop allowance + kill margin"}


def work_deadline_epoch(expires_epoch: float, model: dict) -> float:
    """Latest time the supervisor may still be working normally: expiry minus its own stop allowance and the
    manager's kill margin. Admission, the RuntimeMaxSec backstop and the final reserve all use this one deadline."""
    return expires_epoch - model["supervisor_timeout_stop_s"] - model["supervisor_kill_margin_s"]


def supervisor_launch_parameters(timing: dict, expires_epoch: float, now_epoch: float, n_cases: int = 2,
                                 n_gpus_watched: int = 2) -> dict:
    """systemd-run properties for the supervisor, computed AT LAUNCH: RuntimeMaxSec is a DURATION from now (not a
    timestamp) ending at the work deadline; TimeoutStopSec covers the supervisor's whole cleanup path; the outer limit
    ends before expiry by the kill margin. Refuses (problems) if the window cannot hold startup + enrollment + all
    cases + final verification from now."""
    m = timing_model(timing, n_gpus_watched=n_gpus_watched, n_cases=n_cases)   # Candidate 19: 1 for first-boundary
    w = work_deadline_epoch(expires_epoch, m)
    runtime = math.floor(w - now_epoch)
    need = m["inputs"]["startup_s"] + m["inputs"]["enroll_s"] + n_cases * m["case_total_s"] + m["inputs"]["final_s"]
    problems = [] if runtime >= need else [f"RuntimeMaxSec {runtime} s < required {need:.0f} s (startup + enroll + "
                                           f"{n_cases} cases + final) -> do not launch"]
    return {"RuntimeMaxSec": runtime, "TimeoutStopSec": m["supervisor_timeout_stop_s"], "work_deadline_epoch": w,
            "outer_end_epoch": now_epoch + runtime + m["supervisor_timeout_stop_s"],
            "expires_epoch": expires_epoch, "kill_margin_s": m["supervisor_kill_margin_s"], "problems": problems,
            "model": m}


def case_admission(now_epoch: float, work_deadline: float, model: dict, startup_deadline: float | None = None) -> dict:
    """One more case fits only if its whole inclusive bound and the final verification fit before the work deadline."""
    remaining = work_deadline - now_epoch
    need = model["case_total_s"] + model["inputs"]["final_s"]
    return {"remaining_s": remaining, "needed_s": need, "fits": remaining >= need,
            "gate_deadline_epoch": now_epoch + model["inputs"]["gate_s"]}


def own_scope_timing_problems(props: dict, model: dict, work_deadline: float, now_epoch: float,
                              tolerance_s: float = 5.0) -> list[str]:
    """The supervisor verifies its own scope: TimeoutStopUSec == the model's allowance; the RuntimeMaxSec backstop
    (activation + RuntimeMaxUSec) ends no later than the work deadline (+ tolerance) and not before now."""
    out = []
    try:
        stop_us, rmax_us = int(props["TimeoutStopUSec"]), int(props["RuntimeMaxUSec"])
        act_us = int(props["ActiveEnterTimestampMonotonic"])
        mono_now_us = int(props["_monotonic_now_us"])
    except (KeyError, TypeError, ValueError) as exc:
        return [f"own scope timing properties unreadable ({exc!r}) -> unknown, do not launch"]
    if stop_us != model["supervisor_timeout_stop_s"] * 10**6:
        out.append(f"supervisor TimeoutStopSec {stop_us / 1e6:g} s != model {model['supervisor_timeout_stop_s']} s")
    backstop_epoch = now_epoch + (act_us + rmax_us - mono_now_us) / 1e6
    if backstop_epoch > work_deadline + tolerance_s:
        out.append(f"supervisor RuntimeMaxSec backstop ends {backstop_epoch - work_deadline:.0f} s after the work deadline")
    if backstop_epoch < now_epoch:
        out.append("supervisor RuntimeMaxSec backstop already passed")
    return out


# ------------------------------------------------------------------ 2. cleanup classification
def classify_cleanup(receipt: dict, stop_and_reap_s: float, model: dict) -> str:
    """confirmed: emptiness observed within the modeled bound. confirmed_over_budget: emptiness observed, but later
    than the model allows (never success; the window stops). unconfirmed: emptiness not observed (hold retained)."""
    if receipt.get("confirmed") is not True:
        return "unconfirmed"
    if not (isinstance(stop_and_reap_s, (int, float)) and math.isfinite(stop_and_reap_s)):
        return "unconfirmed"
    return "confirmed" if stop_and_reap_s <= model["stop_and_reap_bound_s"] else "confirmed_over_budget"


# ------------------------------------------------------------------ 3. enrollment
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def boot_id() -> str | None:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None


def policy_problems(policy) -> list[str]:
    """The frozen enrollment policy may only enroll index 5 next to the known GPU 4 for the fixed pair."""
    if not isinstance(policy, dict):
        return ["enrollment policy malformed"]
    out = []
    want = {"policy_type": "mgbench.gpu_enrollment_policy.v1", "eligible_indices": [4, 5], "enroll_indices": [5],
            "cases_after_enrollment": ["R5_fp32_world1_ga4", "C5_fp32_world2_ga2"]}
    for k, v in want.items():
        if policy.get(k) != v:
            out.append(f"enrollment policy {k} = {policy.get(k)!r}, frozen {v!r}")
    known = policy.get("known") or {}
    if set(known) != {"4"} or known["4"].get("uuid") != "GPU-00000000-0000-0000-0000-000000000109":
        out.append("enrollment policy must pin exactly GPU 4's known UUID")
    dc = policy.get("device_class") or {}
    if not isinstance(dc.get("name"), str) or dc.get("enforced_power_limit_w") != 300:
        out.append("enrollment policy device class must name the model and require the 300 W enforced limit")
    return out


def _int_bytes(v, what: str) -> tuple[int | None, str | None]:
    """NVML/Torch capacities are BYTES (integers). MiB-sized values (< 1 GiB) are a wrong-unit error."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None, f"{what} {v!r} is not numeric"
    if not math.isfinite(v):
        return None, f"{what} {v!r} is not finite"
    if v != int(v) or v <= 0:
        return None, f"{what} {v!r} is not a positive integer byte count"
    if v < GIB:
        return None, f"{what} {v!r} < 1 GiB: wrong unit (MiB given where bytes are required?)"
    return int(v), None


def observation_problems(obs: dict, policy: dict) -> list[str]:
    """obs = {"reads": [read1, read2]}, each read {"gpus": {idx: {uuid, pci_bus_id, name, nvml_total_bytes,
    enforced_power_limit_w}}} for all 8 indices (name/bytes/limit required for 4 and 5)."""
    from .device_identity import normalize_pci
    from .safety import UUID_RE

    reads = obs.get("reads") if isinstance(obs, dict) else None
    if not isinstance(reads, list) or len(reads) != 2 or not all(isinstance(r, dict) for r in reads):
        return ["enrollment needs exactly two complete observations"]
    out = []
    if reads[0] != reads[1]:
        out.append("enrollment observations differ between the two reads (identity/capacity not stable)")
    g = reads[1].get("gpus") or {}
    if sorted(map(int, g)) != list(range(8)):
        return out + [f"enrollment observation lists GPUs {sorted(g)}, expected 0..7 -> unknown"]
    g = {int(k): v for k, v in g.items()}
    uu = [str(v.get("uuid")) for v in g.values()]
    pp = [normalize_pci(v.get("pci_bus_id")) for v in g.values()]
    if not all(UUID_RE.match(u) for u in uu) or len(set(uu)) != 8:
        out.append("node UUIDs malformed or not distinct")
    if None in pp or len(set(pp)) != 8:
        out.append("node PCI identities malformed or not distinct")
    k4 = policy["known"]["4"]
    if g[4].get("uuid") != k4["uuid"] or normalize_pci(g[4].get("pci_bus_id")) != normalize_pci(k4["pci_bus_id"]):
        out.append("index 4 does not report the known GPU-4 UUID/PCI (index reorder?) -> refuse")
    dc = policy["device_class"]
    for i in policy["eligible_indices"]:
        d = g[i]
        if d.get("name") != dc["name"]:
            out.append(f"gpu{i}: name {d.get('name')!r} is not the device class {dc['name']!r}")
        lim = d.get("enforced_power_limit_w")
        if isinstance(lim, bool) or not isinstance(lim, (int, float)) or not math.isfinite(lim) or \
                abs(lim - dc["enforced_power_limit_w"]) > 0.5:
            out.append(f"gpu{i}: enforced power limit {lim!r} W, required {dc['enforced_power_limit_w']} W")
        _, err = _int_bytes(d.get("nvml_total_bytes"), f"gpu{i} NVML total")
        if err:
            out.append(err)
    return out


def device_capacity_record(index: int, d: dict, case: dict, fcase: dict, *, torch_total_bytes=None,
                           expected_torch_total_bytes=None) -> dict:
    """Separate per-device fields. NVML reports BYTES (nvidia-smi shows MiB = bytes / 2**20); Torch reports BYTES of
    CUDA-usable memory, which is SMALLER than NVML's total (GPU 4: NVML 49,140 MiB = 51,527,024,640 B; Torch
    50,897,289,216 B = 47.40 GiB; difference ~600 MiB reserved by the driver/firmware). Fraction and headroom are
    computed against Torch's capacity, so NVML only gives an UPPER bound: passing it is necessary, never sufficient."""
    from .smoke import admission

    rec = {"index": index, "uuid": d.get("uuid"), "pci_bus_id": d.get("pci_bus_id"),
           "enforced_power_limit_w": d.get("enforced_power_limit_w"), "nvml_total_bytes": d.get("nvml_total_bytes"),
           "nvml_total_mib": None, "torch_total_bytes": torch_total_bytes, "expected_torch_total_bytes":
           expected_torch_total_bytes, "predicted_allocator_gib": fcase.get("predicted_reserved_gib"),
           "non_allocator_gib": fcase.get("non_allocator_gib"), "mem_fraction": case.get("mem_fraction"),
           "problems": []}
    nv, err = _int_bytes(d.get("nvml_total_bytes"), f"gpu{index} NVML total")
    if err:
        rec["problems"].append(err)
        return rec
    rec["nvml_total_mib"] = nv / MIB
    f = case.get("mem_fraction")
    if isinstance(f, bool) or not isinstance(f, (int, float)) or not 0 < f <= 0.92:
        rec["problems"].append(f"mem_fraction {f!r} invalid")
        return rec
    up = admission(fcase.get("predicted_reserved_gib"), nv / GIB, fcase.get("non_allocator_gib"))
    rec["upper_bound_check"] = {"basis": "NVML total (upper bound of CUDA-usable)", "total_gib": nv / GIB, **up}
    if not up["eligible"] or Decimal(str(up["fraction"])) > Decimal(str(f)):
        rec["problems"].append(f"gpu{index}: even NVML's upper bound {nv / GIB:.3f} GiB cannot hold the case at the frozen "
                               f"fraction {f} ({'; '.join(up['reasons']) or 'required ' + str(up['fraction'])})")
    if torch_total_bytes is not None:
        tb, err = _int_bytes(torch_total_bytes, f"gpu{index} Torch total")
        if err:
            rec["problems"].append(err)
            return rec
        if tb > nv:
            rec["problems"].append(f"gpu{index}: Torch total {tb} B exceeds NVML total {nv} B (inconsistent)")
        if expected_torch_total_bytes is not None and tb < int(expected_torch_total_bytes):
            rec["problems"].append(f"gpu{index}: runtime Torch total {tb} B below the admission assumption "
                                   f"{int(expected_torch_total_bytes)} B")
        a = admission(fcase.get("predicted_reserved_gib"), tb / GIB, fcase.get("non_allocator_gib"))
        allowance = Decimal(str(f)) * Decimal(tb) / GIB
        headroom = Decimal(tb) / GIB - allowance - Decimal(str(fcase.get("non_allocator_gib")))
        rec["actual_check"] = {"basis": "Torch total (actual CUDA-usable)", "total_gib": tb / GIB, **a,
                               "allowance_at_frozen_fraction_gib": str(allowance), "headroom_at_frozen_fraction_gib":
                               str(headroom)}
        need = Decimal("1.02") * Decimal(str(fcase.get("predicted_reserved_gib")))
        if not a["eligible"] or allowance < need or headroom < 2:
            rec["problems"].append(f"gpu{index}: actual capacity {tb / GIB:.3f} GiB at the frozen fraction {f}: allowance "
                                   f"{float(allowance):.3f} < {float(need):.3f} GiB or headroom {float(headroom):.3f} < 2 GiB"
                                   if (allowance < need or headroom < 2) else f"gpu{index}: {'; '.join(a['reasons'])}")
    rec["pass"] = not rec["problems"]
    return rec


def enrollment_capacity(observed: dict, cfg: dict, feasibility: dict, policy: dict) -> tuple[dict, list[str]]:
    """Pre-launch, per case and per device, from each device's OWN NVML reading (no inference from GPU 4)."""
    out, problems = {}, []
    for case in cfg["cases"]:
        fcase = (feasibility.get("cases") or {}).get(case["id"]) or {}
        for i in cfg["case_gpus"][case["id"]]:
            exp = (policy.get("known_torch_total_bytes") or {}).get(str(i))
            r = device_capacity_record(i, observed[i], case, fcase, expected_torch_total_bytes=exp)
            out[f"{case['id']}/gpu{i}"] = r
            problems += r["problems"]
    return out, problems


def build_record(*, auth: dict, auth_sha256: str, policy_sha256: str, config_sha256: str, candidate_digest: str,
                 window_dir: str, sup_unit: str, supervisor_start_utc: str, obs: dict, observed_utc: list[str],
                 capacity: dict) -> dict:
    g = {int(k): v for k, v in obs["reads"][1]["gpus"].items()}
    return {"record_type": RECORD_TYPE, "authorization_id": auth.get("id"), "authorization_sha256": auth_sha256,
            "authorization_expires_utc": auth.get("expires_utc"), "enrollment_policy_sha256": policy_sha256,
            "config_sha256": config_sha256, "candidate_digest": candidate_digest, "host": socket.gethostname(),
            "boot_id": boot_id(), "window_dir": window_dir, "supervisor_unit": sup_unit,
            "supervisor_start_utc": supervisor_start_utc, "observed_utc": observed_utc,
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "eligible_indices": [4, 5],
            "devices": {str(i): g[i] for i in (4, 5)}, "node_uuids": {str(i): g[i]["uuid"] for i in range(8)},
            "capacity_pre_launch": capacity,
            "label": "runtime enrollment record (outside the candidate tree); created exclusively, sealed once"}


def seal_record(edir: Path, record: dict) -> dict:
    """Create the record and its seal exclusively (0444). Either already existing -> FileExistsError (refuse)."""
    edir = Path(edir)
    edir.mkdir(mode=0o700, exist_ok=False)
    raw = (json.dumps(record, indent=1, sort_keys=True) + "\n").encode()
    rp, sp = edir / "ENROLLMENT_RECORD.json", edir / "ENROLLMENT_RECORD.json.seal"
    fd = os.open(rp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "wb") as f:
        f.write(raw)
    seal = {"seal_type": SEAL_TYPE, "record_sha256": sha256_bytes(raw), "authorization_sha256":
            record["authorization_sha256"], "sealed_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    sraw = (json.dumps(seal, indent=1, sort_keys=True) + "\n").encode()
    fd = os.open(sp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "wb") as f:
        f.write(sraw)
    return {"record_sha256": seal["record_sha256"], "seal_sha256": sha256_bytes(sraw), "dir": str(edir)}


def record_problems(edir: Path, binding: dict | None, *, auth: dict, auth_sha256: str, policy_sha256: str,
                    config_sha256: str, window_dir: str, sup_unit: str, supervisor_start_epoch: float,
                    expires_epoch: float) -> tuple[dict | None, list[str]]:
    """Re-validate the sealed record at every gate. Missing, altered, duplicate, inconsistent or stale -> refuse."""
    from .launch_controls import parse_utc

    edir = Path(edir)
    try:
        names = sorted(p.name for p in edir.iterdir())
    except OSError as exc:
        return None, [f"enrollment record missing ({exc!r}) -> refuse"]
    if names != ["ENROLLMENT_RECORD.json", "ENROLLMENT_RECORD.json.seal"]:
        return None, [f"enrollment directory holds {names} (missing or duplicate record) -> refuse"]
    try:
        raw, sraw = (edir / names[0]).read_bytes(), (edir / names[1]).read_bytes()
        rec, seal = json.loads(raw), json.loads(sraw)
    except Exception as exc:  # noqa: BLE001
        return None, [f"enrollment record unreadable ({exc!r}) -> refuse"]
    out = []
    if seal.get("seal_type") != SEAL_TYPE or seal.get("record_sha256") != sha256_bytes(raw):
        out.append("enrollment record altered after sealing (hash differs from its seal) -> refuse")
    if binding is not None and (binding.get("record_sha256") != sha256_bytes(raw) or
                                binding.get("seal_sha256") != sha256_bytes(sraw)):
        out.append("enrollment record or seal differs from the one this supervisor sealed -> refuse")
    want = {"record_type": RECORD_TYPE, "authorization_id": auth.get("id"), "authorization_sha256": auth_sha256,
            "enrollment_policy_sha256": policy_sha256, "config_sha256": config_sha256, "host": socket.gethostname(),
            "boot_id": boot_id(), "window_dir": window_dir, "supervisor_unit": sup_unit,
            "authorization_expires_utc": auth.get("expires_utc")}
    bad = sorted(k for k, v in want.items() if rec.get(k) != v)
    if bad:
        out.append(f"enrollment record does not belong to this window/authorization ({bad}) -> refuse (stale/foreign)")
    try:
        ts = [parse_utc(x).timestamp() for x in [*rec["observed_utc"], rec["created_utc"]]]
        if min(ts) < supervisor_start_epoch - 1 or max(ts) >= expires_epoch:
            out.append("enrollment record observed outside this window -> refuse (stale)")
    except Exception as exc:  # noqa: BLE001
        out.append(f"enrollment record timestamps malformed ({exc!r}) -> refuse")
    return (rec if not out else None), out


def enrolled_pins(rec: dict) -> tuple[dict, dict]:
    d = rec["devices"]
    return {int(k): v["uuid"] for k, v in d.items()}, {int(k): v["pci_bus_id"] for k, v in d.items()}


# ------------------------------------------------------------------ 4. worker capacity contract
def worker_capacity_contract(case: dict, fcase: dict, gpus: list[int], devices: dict, policy: dict) -> dict:
    return {"case": case["id"], "mem_fraction": case["mem_fraction"], "predicted_reserved_gib":
            fcase["predicted_reserved_gib"], "non_allocator_gib": fcase["non_allocator_gib"],
            "ranks": [{"rank": r, "index": g, "uuid": devices[g]["uuid"], "nvml_total_bytes": devices[g]["nvml_total_bytes"],
                       "expected_torch_total_bytes": (policy.get("known_torch_total_bytes") or {}).get(str(g))}
                      for r, g in enumerate(gpus)]}


def worker_capacity_check(contract, local_rank: int, torch_total_bytes) -> dict:
    """In-rank, before model allocation: the rank's ACTUAL capacity at the frozen fraction."""
    try:
        r = contract["ranks"][local_rank]
        case = {"mem_fraction": contract["mem_fraction"]}
        fcase = {k: contract[k] for k in ("predicted_reserved_gib", "non_allocator_gib")}
    except (TypeError, KeyError, IndexError) as exc:
        return {"pass": False, "problems": [f"capacity contract malformed ({exc!r})"]}
    rec = device_capacity_record(r["index"], {"uuid": r["uuid"], "nvml_total_bytes": r["nvml_total_bytes"]}, case,
                                 fcase, torch_total_bytes=torch_total_bytes,
                                 expected_torch_total_bytes=r.get("expected_torch_total_bytes"))
    if torch_total_bytes is None:
        rec["problems"].append("Torch total unavailable")
    rec["pass"] = not rec["problems"]
    rec["rank"] = local_rank
    return rec


def all_ranks_agree(ok: bool, dist=None, device=None) -> bool:
    """Collective agreement before any optimizer step: MIN over ranks of the local pass flag."""
    if dist is None or not dist.is_initialized():
        return bool(ok)
    import torch

    flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(int(flag.item()) == 1)


# ------------------------------------------------------------------ 5. host-wide lease
def _lease_dir_problems(d: Path) -> list[str]:
    try:
        st = d.stat()
    except OSError as exc:
        return [f"lease directory unreadable ({exc!r}) -> unknown, do not launch"]
    if st.st_uid != os.getuid() or (st.st_mode & 0o077):
        return [f"lease directory {d} not owner-only -> do not launch"]
    return []


def acquire_lease(lease_dir: Path, gpus: list[int], owner: dict, lock_timeout_s: float = 10.0) -> tuple[dict | None, list[str]]:
    """Joint acquisition (all or none) of gpu{i}.lease files under one directory-wide flock. An existing lease is
    NEVER reclaimed (not if its owner died, not if its window expired): refusal + manual inspection."""
    import time

    lease_dir = Path(lease_dir)
    try:
        lease_dir.mkdir(mode=0o700, exist_ok=True)
    except OSError as exc:
        return None, [f"lease directory not creatable ({exc!r}) -> unknown, do not launch"]
    p = _lease_dir_problems(lease_dir)
    if p:
        return None, p
    token = hashlib.sha256(json.dumps(owner, sort_keys=True).encode() + os.urandom(16)).hexdigest()[:32]
    body = dict(owner, lease_type=LEASE_TYPE, token=token, gpus=gpus, host=socket.gethostname(), boot_id=boot_id(),
                pid=os.getpid(), created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                policy="never reclaimed automatically; released only by its owner after confirmed cleanup")
    with open(lease_dir / ".acquire.lock", "a") as lk:
        end = time.monotonic() + lock_timeout_s
        while True:
            try:
                fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > end:
                    return None, ["lease lock busy beyond its timeout -> unknown, do not launch"]
                time.sleep(0.05)
        try:
            held = sorted(f"gpu{g}" for g in gpus if (lease_dir / f"gpu{g}.lease").exists())
            if held:
                return None, [f"host lease held for {held} (another window, or retained after unconfirmed cleanup; never "
                              "reclaimed) -> refuse"]
            made = []
            try:
                for g in gpus:
                    lp = lease_dir / f"gpu{g}.lease"
                    fd = os.open(lp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
                    with os.fdopen(fd, "w") as f:
                        json.dump(body, f, sort_keys=True)
                    made.append(lp)
            except OSError as exc:
                for lp in made:                     # ours, created a moment ago under the lock: both or neither
                    lp.unlink()
                return None, [f"joint lease not acquired ({exc!r}); nothing partial kept -> refuse"]
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)
    return body, []


def release_lease(lease_dir: Path, lease: dict, reason: str) -> list[str]:
    """Ownership-scoped release: only files carrying OUR token; each is moved to a new unique name (never overwritten)."""
    lease_dir = Path(lease_dir)
    out = []
    with open(lease_dir / ".acquire.lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            for g in lease["gpus"]:
                lp = lease_dir / f"gpu{g}.lease"
                try:
                    cur = json.loads(lp.read_text())
                except Exception as exc:  # noqa: BLE001
                    out.append(f"gpu{g} lease unreadable at release ({exc!r}); left for inspection")
                    continue
                if cur.get("token") != lease["token"]:
                    out.append(f"gpu{g} lease is not ours; untouched")
                    continue
                dst = lease_dir / "released" / f"gpu{g}.lease.{lease['token']}"
                dst.parent.mkdir(mode=0o700, exist_ok=True)
                os.link(lp, dst)                    # fails if dst exists: never overwrite
                lp.unlink()
                fd = os.open(str(dst) + ".reason", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
                with os.fdopen(fd, "w") as f:
                    f.write(reason + "\n")
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)
    return out


def lease_held_problems(lease_dir: Path, lease: dict) -> list[str]:
    out = []
    for g in lease["gpus"]:
        try:
            if json.loads((Path(lease_dir) / f"gpu{g}.lease").read_text()).get("token") != lease["token"]:
                out.append(f"gpu{g} host lease no longer ours -> do not launch")
        except Exception as exc:  # noqa: BLE001
            out.append(f"gpu{g} host lease missing or unreadable ({exc!r}) -> do not launch")
    return out


def post_enrollment_problems(snapshot: dict, rec: dict) -> list[str]:
    """Every later gate: each enrolled/known device must still report the sealed UUID, NVML total and power limit."""
    out = []
    for k, d in (rec.get("devices") or {}).items():
        g = (snapshot.get("gpus") or {}).get(int(k)) or {}
        mib, lim = g.get("mem_total_mib"), g.get("power_limit_w")
        if g.get("uuid") != d.get("uuid"):
            out.append(f"gpu{k}: UUID {g.get('uuid')!r} differs from the sealed enrollment record -> refuse")
        if not isinstance(mib, (int, float)) or not math.isfinite(mib) or round(mib * MIB) != d.get("nvml_total_bytes"):
            out.append(f"gpu{k}: NVML total {mib!r} MiB differs from the sealed enrollment record "
                       f"({d.get('nvml_total_bytes')} B) -> refuse")
        if not isinstance(lim, (int, float)) or abs(lim - d.get("enforced_power_limit_w", -1)) > 0.5:
            out.append(f"gpu{k}: enforced power limit {lim!r} W differs from the sealed enrollment record -> refuse")
    return out


def fb_feasibility_case(feasibility: dict, case_id: str) -> dict:
    """Candidate 19: the first-boundary feasibility record of one case, with the global non-allocator allowance when
    the case has none of its own."""
    rec = (feasibility.get("cases") or {}).get(case_id) or {}
    na = rec.get("non_allocator_gib")
    return {"predicted_reserved_gib": rec.get("predicted_reserved_gib"),
            "non_allocator_gib": na if na is not None else feasibility.get("non_allocator_gib")}
