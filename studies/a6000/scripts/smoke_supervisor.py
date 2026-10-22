#!/usr/bin/env python
"""Supervisor for owner-approved single-GPU windows on GPU 4: the clean-only smoke (stage 20261016s2, default
config configs/smoke_gpu4.yaml, scope train-smoke) and the S1a numerical diagnostics (configs/s1a_qwen15b.yaml,
scope train-diagnostics: FP32 control first, then up to four dependent comparisons; an invalid control blocks them).

Run ONLY inside its own bounded scope (verified), with the explicit interpreter:
  systemd-run --user --scope --unit=<sup unit> -p CPUQuota=100% -p MemoryMax=2G -p MemorySwapMax=0 \
      -p RuntimeMaxSec=<seconds to expiry> -- <python> -s scripts/smoke_supervisor.py --authorization A \
      --expectations E --expectations-sha256 H --out <window dir> --sup-unit <sup unit> [--config C] \
      [--seed-cache DIR SHA256]

Time: one absolute deadline on a monotonic clock, derived from the authorization's UTC expiry at start. Global
startup <= startup_s; per case gate+setup <= gate_s, child <= child_s, graceful stop <= grace_s, kill/reap <= reap_s;
a case is refused unless it and the final reserve fit. Identity: NVML index 4 -> approved UUID and PCI before launch;
inside the worker scope a Driver-API pre-step (no torch, no context) and the rank's pre-torch check; DeepSpeed and
torch links inside the rank before model allocation. Any problem stops the window; no retry, no substitute GPU.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import signal
import sys
import time
from pathlib import Path

A6000 = Path(__file__).resolve().parents[1]
SRC = A6000 / "src"
sys.path.insert(0, str(SRC))

import mgbench  # noqa: E402
from mgbench import launch_controls as lc  # noqa: E402
from mgbench import pair_controls as pc  # noqa: E402
from mgbench import scopes, smoke  # noqa: E402
from mgbench import window_contract as wc  # noqa: E402
from mgbench.config import cluster_config, expand_experiment, load_yaml  # noqa: E402
from mgbench.device_identity import chain_problems, normalize_pci  # noqa: E402
from mgbench.gpu_monitor import _init, bounded, gpu_processes, gpu_uuid, node_snapshot  # noqa: E402
from mgbench.safety import (PowerWatch, host_problems, idle_problems, mps_daemon_problems,  # noqa: E402
                            node_guard_problems, node_telemetry_problems)

STOP = {"signal": None}


def utc(t: float | None = None) -> str:
    return dt.datetime.fromtimestamp(time.time() if t is None else t, dt.timezone.utc).isoformat()


def _usec(v: str) -> int:
    """systemd time span text ('2min 39s', '159s', 'infinity', '500ms') -> microseconds."""
    if v.isdigit():
        return int(v)
    if v == "infinity":
        return 2**63
    units = {"us": 1, "ms": 10**3, "s": 10**6, "min": 60 * 10**6, "h": 3600 * 10**6}
    total = 0
    for part in v.split():
        num = "".join(ch for ch in part if ch.isdigit() or ch == ".")
        total += int(float(num) * units[part[len(num):]])
    return total


class Log:
    def __init__(self, path: Path):
        self.f = open(path, "x")

    def __call__(self, msg):
        line = f"{utc()} {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def write_new(path: Path, obj) -> None:
    lc.create_exclusive(path, json.loads(json.dumps(obj, default=str)))


def nvml_identity(index: int, tq: float) -> dict:
    import pynvml

    def read():
        _init()
        h = pynvml.nvmlDeviceGetHandleByIndex(index)
        u, b = pynvml.nvmlDeviceGetUUID(h), pynvml.nvmlDeviceGetPciInfo(h).busId
        return {"index": index, "uuid": u.decode() if isinstance(u, bytes) else u,
                "pci_bus_id": b.decode() if isinstance(b, bytes) else b}
    return bounded(read, tq)


def enrollment_read(tq: float) -> dict:
    """Candidate 18: one whole-node identity read plus name / NVML total (BYTES) / enforced limit of GPUs 4 and 5."""
    import pynvml

    def s(v):
        return v.decode() if isinstance(v, bytes) else v

    def read():
        _init()
        g = {}
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            g[i] = {"uuid": s(pynvml.nvmlDeviceGetUUID(h)), "pci_bus_id": s(pynvml.nvmlDeviceGetPciInfo(h).busId)}
            if i in (4, 5):
                g[i].update(name=s(pynvml.nvmlDeviceGetName(h)), nvml_total_bytes=int(pynvml.nvmlDeviceGetMemoryInfo(h).total),
                            enforced_power_limit_w=pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0)
        return {"gpus": g}
    return bounded(read, tq)


def gate(ctx) -> list[str]:
    """Full re-validation for the CURRENT case's devices (ctx["case_gpus"]; [4] outside the pair scope). Every selected
    device is checked on its own; nothing is inferred from another device or from history."""
    gpus = ctx.get("case_gpus") or [4]
    problems = []
    if ctx.get("pair"):
        problems += pc.pair_authorization_problems(ctx["auth_path"], candidate_root=ctx["candidate"],
                                                   uuids=ctx["auth_pins"], n_cases_done=ctx["cases_started"],
                                                   expected_sha256=ctx["auth_sha256"],
                                                   enrollment_policy_sha256=ctx.get("policy_sha256"))
        e = ctx.get("enrollment")                # Candidate 18: the sealed record, re-validated at EVERY gate
        if e is None:
            problems.append("enrollment record missing -> refuse")
        else:
            _, ep = wc.record_problems(e["dir"], e, **ctx["enrollment_check"])
            problems += ep
    if ctx.get("contract"):                     # Candidate 19: the host lease is held by every contract scope
        problems += wc.lease_held_problems(ctx["lease_dir"], ctx["lease"]) if ctx.get("lease") else \
            ["host lease missing -> do not launch"]
    else:
        problems += lc.authorization_problems(ctx["auth_path"], scope=ctx["scope"], candidate_root=ctx["candidate"],
                                              gpus=gpus, uuids=ctx["pins"], n_cases_done=ctx["cases_started"],
                                              expected_sha256=ctx["auth_sha256"])
    if ctx["diag"]:
        problems += smoke.bound_file_problems(ctx["bound_files"])
    problems += smoke.settings_problems(cluster_config())
    try:
        snap = node_snapshot(ctx["tq"])
        ctx["last_snapshot"] = snap
        problems += node_telemetry_problems(snap, 8, ctx["pins"])
        for g in gpus:
            problems += smoke.gpu_power_limit_problems(snap, g, smoke.FROZEN_GUARD["planned_w_per_gpu"])
        if ctx.get("pair") and ctx.get("enrollment_record"):
            problems += wc.post_enrollment_problems(snap, ctx["enrollment_record"])
    except Exception as exc:  # noqa: BLE001
        problems.append(f"node telemetry unreadable ({exc!r}) -> unknown, do not launch")
    ctx["nvml_identity"] = {}
    for g in gpus:
        try:
            nv = nvml_identity(g, ctx["tq"])
            ctx["nvml_identity"][g] = nv
            problems += [f"{p} -> do not launch" for p in chain_problems(approved_uuid=ctx["pins"].get(g),
                                                                         approved_pci=ctx["pci"].get(g), nvml=nv)]
        except Exception as exc:  # noqa: BLE001
            problems.append(f"gpu{g}: NVML identity unreadable ({exc!r}) -> unknown, do not launch")
    idle = dict(cluster_config()["idle"], query_timeout_s=ctx["tq"])
    problems += idle_problems(gpus, idle) + host_problems(idle) + node_guard_problems(gpus, idle) + mps_daemon_problems()
    try:
        other = [u for u in smoke.active_units("mgbench-", ctx.get("meta_s", 10.0)) if u != ctx["sup_unit"]]
        if other:
            problems.append(f"another mgbench scope is active ({other}) -> do not launch (one job at a time)")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"scope listing unreadable ({exc!r}) -> unknown, do not launch")
    return problems


def settle(ctx, log, deadline_mono: float) -> tuple[bool, list[str], list[dict]]:
    idle = cluster_config()["idle"]
    streak, trail = 0, []
    while time.monotonic() < deadline_mono and not STOP["signal"]:
        busy, fatal = smoke.classify(gate(ctx))
        trail.append({"t": utc(), "busy": busy, "fatal": fatal})
        if fatal:
            return False, fatal, trail
        streak = streak + 1 if not busy else 0
        if streak >= idle["consecutive_polls"]:
            return True, [], trail
        if busy and len(trail) % 4 == 1:
            log(f"  gate waiting: {'; '.join(busy)}")
        time.sleep(max(0.0, min(idle["poll_s"], deadline_mono - time.monotonic())))
    return False, ["gate not satisfied within the per-case gate budget" if not STOP["signal"]
                   else f"signal {STOP['signal']}"], trail


def watch(ctx, proc, handle, child_deadline_mono: float) -> dict:
    guard = smoke.FROZEN_GUARD
    pw = PowerWatch(guard)
    gpus = ctx.get("case_gpus") or [4]
    peaks = {"node_gpu_power_w": 0.0, "worker_memory_current_bytes": 0, "worker_memory_peak_bytes": 0,
             "worker_cpu_rate_max": 0.0, "worker_pids_max": 0, "active_worker_scopes_max": 0, "samples": 0,
             **{f"gpu{g}_{k}": 0.0 for g in gpus for k in ("power_w", "temp_c", "mem_used_mib")}}
    last_cpu, reason = None, None
    while proc.poll() is None:
        t0 = time.monotonic()
        if STOP["signal"]:
            reason = f"signal {STOP['signal']}"
            break
        if t0 >= child_deadline_mono:
            reason = "child phase time limit reached"
            break
        try:
            snap = node_snapshot(ctx["tq"])
            bad = node_telemetry_problems(snap, 8, ctx["pins"])
            if bad:
                reason = "telemetry: " + "; ".join(bad)
                break
            total = sum(g["power_w"] for g in snap["gpus"].values())
            peaks["node_gpu_power_w"] = max(peaks["node_gpu_power_w"], total)
            for g in gpus:
                for k in ("power_w", "temp_c", "mem_used_mib"):
                    peaks[f"gpu{g}_{k}"] = max(peaks[f"gpu{g}_{k}"], snap["gpus"][g][k])
            if pw.update(total):
                reason = (f"node power guard: {total:.0f} W > {guard['abort_total_w']} W for {pw.over} s"
                          if pw.invalid is None else f"node power reading invalid ({pw.invalid!r})")
                break
            import pynvml
            _init()
            for g in gpus:                    # every selected device, independently; foreign work is never touched
                procs, err = bounded(gpu_processes, ctx["tq"], pynvml.nvmlDeviceGetHandleByIndex(g))
                if procs is None:
                    reason = f"gpu{g} process list unavailable ({err})"
                    break
                foreign = [p["pid"] for p in procs if smoke.pid_in_cgroup(p["pid"], handle.cgroup) is not True]
                if foreign:
                    reason = f"process outside our scope on gpu{g} (pids {foreign}); ours stopped, theirs never signalled"
                    break
            if reason:
                break
            mps = mps_daemon_problems()
            if mps:
                reason = mps[0]
                break
            lim = scopes.cgroup_limits(handle.cgroup)
            for key, field in (("memory.current", "worker_memory_current_bytes"),
                               ("memory.peak", "worker_memory_peak_bytes"), ("pids.current", "worker_pids_max")):
                if str(lim.get(key, "")).isdigit():
                    peaks[field] = max(peaks[field], int(lim[key]))
            u = lim.get("cpu.stat.usage_usec")
            if isinstance(u, int):
                if last_cpu:
                    peaks["worker_cpu_rate_max"] = max(peaks["worker_cpu_rate_max"],
                                                       (u - last_cpu[0]) / 1e6 / max(1e-3, t0 - last_cpu[1]))
                last_cpu = (u, t0)
            peaks["active_worker_scopes_max"] = max(
                peaks["active_worker_scopes_max"],
                len([x for x in smoke.active_units("mgbench-", ctx.get("meta_s", 10.0)) if x != ctx["sup_unit"]]))
            peaks["samples"] += 1
        except Exception as exc:  # noqa: BLE001
            reason = f"guard telemetry unreadable ({exc!r})"
            break
        time.sleep(max(0.0, min(1.0 - (time.monotonic() - t0), child_deadline_mono - time.monotonic())))
    return {"stop_reason": reason, "peaks": peaks, "power_watch_peak_w": pw.peak}


def main():
    t_start_mono, t_start_wall = time.monotonic(), time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--authorization", required=True, type=Path)
    ap.add_argument("--expectations", required=True, type=Path)
    ap.add_argument("--expectations-sha256", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sup-unit", required=True)
    ap.add_argument("--config", default="configs/smoke_gpu4.yaml",
                    help="window configuration, relative to studies/a6000 (inside the authorized candidate)")
    ap.add_argument("--seed-cache", nargs=2, metavar=("DIR", "SHA256"), default=None,
                    help="verified same-owner Triton key directory to copy into the fresh cache, with its .so hash")
    args = ap.parse_args()
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, lambda n, f: STOP.__setitem__("signal", signal.Signals(n).name))

    out = args.out
    (out / "cases").mkdir(exist_ok=False)
    log = Log(out / "supervisor.log")
    summary = {"supervisor_started_utc": utc(t_start_wall), "cases": [], "stopped_because": None, "peak_concurrency": 0}
    candidate = A6000.parents[1]
    cfg_path = (A6000 / args.config).resolve()
    smoke_cfg = load_yaml(cfg_path)
    fb = smoke_cfg.get("scope") == smoke.FB_SCOPE              # Candidate 14 first-boundary scope (proposed)
    pair = smoke_cfg.get("scope") == pc.PAIR_SCOPE              # Candidate 17 R5/C5 control pair (proposed)
    contract = pair or fb      # Candidate 19: the Candidate 18 window contract also governs the first-boundary scope
    diag = smoke_cfg.get("scope") == smoke.DIAG_SCOPE or fb or pair
    scope = pc.PAIR_SCOPE if pair else smoke.FB_SCOPE if fb else smoke.DIAG_SCOPE if diag else "train-smoke"
    case_gpus_of = (lambda c: list((smoke_cfg.get("case_gpus") or {}).get(c["id"], []))) if pair else (lambda c: [4])
    window_gpus = sorted({g for c in smoke_cfg["cases"] for g in case_gpus_of(c)}) or [4]
    n_cases = len(smoke_cfg["cases"])
    timing = smoke_cfg["timing"]
    pins = {int(k): v for k, v in (smoke_cfg.get("gpu_uuids") or {}).items()}
    pci = {int(k): v for k, v in (smoke_cfg.get("gpu_pci") or {}).items()}
    cfg_scope = smoke_cfg.get("scope")                 # Candidate 16: only scopes this supervisor can execute
    unsupported = None if cfg_scope in (None, smoke.DIAG_SCOPE, smoke.FB_SCOPE, pc.PAIR_SCOPE) else \
        f"configuration scope {cfg_scope!r} is not executable by this supervisor (the two-device control-pair " \
        "execution path is not implemented) -> refuse"
    auth_raw = args.authorization.read_bytes()
    clock_problem = None                     # Candidate 14: a null/malformed authorization refuses cleanly
    try:
        auth = json.loads(auth_raw)
        auth = auth if isinstance(auth, dict) else {}
        issued = lc.parse_utc(auth["issued_utc"]).timestamp()
        expiry = lc.parse_utc(auth["expires_utc"]).timestamp()
    except Exception as exc:  # noqa: BLE001
        auth = auth if isinstance(locals().get("auth"), dict) else {}
        issued = expiry = t_start_wall
        clock_problem = f"authorization times unusable ({exc!r}) -> refuse"
    model, work_deadline = None, expiry
    if contract:  # Candidate 18/19: ONE timing model; work deadline = expiry - supervisor stop allowance - kill margin
        try:
            model = wc.timing_model(timing, n_gpus_watched=len(window_gpus), n_cases=n_cases)
            work_deadline = wc.work_deadline_epoch(expiry, model)
        except ValueError as exc:
            clock_problem = clock_problem or f"timing model unusable ({exc}) -> refuse"
    deadline_mono = t_start_mono + (work_deadline - t_start_wall)     # absolute deadline on the monotonic clock
    startup_deadline_mono = t_start_mono + (issued + timing.get("startup_s", 0) - t_start_wall)
    enroll_deadline_mono = startup_deadline_mono + timing.get("enroll_s", 0)   # same clock; never restarted
    final_reserve_start_mono = deadline_mono - timing.get("final_reserve_s", 0)
    ctx = {"auth_path": args.authorization, "auth_sha256": hashlib.sha256(auth_raw).hexdigest(), "pins": pins,
           "pci": pci, "candidate": candidate, "cases_started": 0, "tq": timing.get("telemetry_timeout_s", 5),
           "sup_unit": args.sup_unit, "scope": scope, "diag": diag, "bound_files": None, "pair": pair,
           "case_gpus": [4], "auth_pins": dict(pins), "meta_s": timing.get("metadata_call_s", 10.0),
           "contract": contract}
    summary["clock"] = {"authorization_issued_utc": auth.get("issued_utc"), "expiry_utc": auth.get("expires_utc"),
                        "supervisor_start_wall": t_start_wall, "supervisor_start_monotonic": t_start_mono,
                        "deadline_monotonic": deadline_mono, "startup_deadline_monotonic": startup_deadline_mono,
                        "budget": f"{timing.get('startup_s')} + {n_cases} x ({timing.get('gate_s')} + {timing.get('child_s')} + "
                                  f"{timing.get('grace_s')} + {timing.get('reap_s')}) + {timing.get('final_reserve_s')} = "
                                  f"{smoke.diag_budget_seconds(timing, n_cases) if all(k in timing for k in ('startup_s', 'gate_s', 'child_s', 'grace_s', 'reap_s', 'final_reserve_s')) else None} s within {timing.get('window_s')} s"}
    summary.update(mode="diagnostics" if diag else "smoke", scope=scope, config=str(cfg_path),
                   config_sha256=smoke.file_sha256(cfg_path))
    if contract:
        summary["timing_model"] = model
        summary["clock"]["work_deadline_utc"] = utc(work_deadline)
    lease, lease_dir = None, Path(str(smoke_cfg.get("lease_dir") or "/nonexistent-lease-dir"))
    ctx["lease_dir"] = lease_dir
    reservation = out / ("RESERVATION_" + "_".join(f"gpu{g}" for g in window_gpus))   # ONE joint file: all or none
    reserved_by_us = False
    lock = open(out / "supervisor.lock", "x")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    all_cleanups_confirmed = True
    cache = out / "cache"
    receipts = {}                                             # diagnostics: reference receipt per writing case

    def stop(reason):
        summary["stopped_because"] = reason
        log(f"STOP: {reason}")

    try:
        pre = [p for p in (clock_problem, unsupported) if p]
        if hashlib.sha256(args.expectations.read_bytes()).hexdigest() != args.expectations_sha256:
            pre.append("expectations file differs from its frozen hash")
        if not str(mgbench.__file__).startswith(str(SRC)):
            pre.append(f"supervisor imported mgbench from {mgbench.__file__}, not the candidate")
        if not str(cfg_path).startswith(str(A6000) + os.sep):
            pre.append(f"config {cfg_path} is outside the authorized candidate")
        expectations = json.loads(args.expectations.read_text())
        if diag:
            ctx["bound_files"] = expectations.get("bound_files")
            feas_path = next((p for p in (ctx["bound_files"] or {}) if p.endswith("memory_time_feasibility_v2.json")), None)
            tol_path = next((p for p in (ctx["bound_files"] or {}) if p.endswith("tolerances.json")), None)
            if pair:                                        # control pair: its own feasibility and input map
                feasibility, tolerances = None, {}
                fp = next((p for p in (ctx["bound_files"] or {}) if p.endswith("pair_memory_feasibility.json")), None)
                mp = next((p for p in (ctx["bound_files"] or {}) if p.endswith("global_input_map_r5_c5.json")), None)
                if fp is None or mp is None:
                    pre.append("authorization/expectations do not bind the pair feasibility and the global input map")
                else:
                    feasibility = json.loads(Path(fp).read_text())
                    if json.loads(Path(mp).read_text()).get("global_row_digests") != expectations.get("global_row_digests"):
                        pre.append("expectations' global row digests differ from the bound input map")
                missing_id = [g for g in window_gpus if not pins.get(g) or not pci.get(g)]
                pol_path = (A6000 / str(smoke_cfg.get("enrollment_policy") or "missing-policy")).resolve()
                try:
                    pol_raw = pol_path.read_bytes()
                    policy = json.loads(pol_raw)
                    ctx["policy_sha256"] = hashlib.sha256(pol_raw).hexdigest()
                except Exception as exc:  # noqa: BLE001
                    policy = None
                    pre.append(f"enrollment policy unreadable ({exc!r})")
                if not str(pol_path).startswith(str(A6000) + os.sep):
                    pre.append("enrollment policy is outside the authorized candidate")
                pre += wc.policy_problems(policy)
                if missing_id != [5]:
                    pre.append(f"device identity pending for GPUs {missing_id}: only GPU 5 may be enrolled, and only "
                               "under the frozen policy (never inferred)")
                if not args.sup_unit.startswith("mgbench-"):
                    pre.append("supervisor unit must be named mgbench-* (other windows' active-unit guard must see it)")
                summary["enrollment_policy"] = {"path": str(pol_path), "sha256": ctx.get("policy_sha256")}
            elif fb:                                        # first-boundary scope: its own feasibility, no grading
                feasibility, tolerances = None, {}
            elif feas_path is None or tol_path is None:
                pre.append("authorization/expectations do not bind the feasibility and tolerance files")
                feasibility, tolerances = {}, {}
            else:
                feasibility, tolerances = json.loads(Path(feas_path).read_text()), json.loads(Path(tol_path).read_text())
            if pair:
                pre += pc.pair_config_problems(smoke_cfg, feasibility)
            elif fb:
                feas_path = next((p for p in (ctx["bound_files"] or {}) if p.endswith("fb_memory_feasibility.json")), None)
                feasibility = json.loads(Path(feas_path).read_text()) if feas_path else None
                pre += smoke.fb_config_problems(smoke_cfg, feasibility)
            else:
                pre += smoke.diag_config_problems(smoke_cfg, feasibility)
            pre += smoke.bound_file_problems(ctx["bound_files"])
            if auth.get("bound_files") != ctx["bound_files"]:
                pre.append("authorization binds different design files than the expectations")
            if auth.get("expectations_sha256") != args.expectations_sha256:
                pre.append("authorization is bound to different expectations")
            if auth.get("config_sha256") != summary["config_sha256"]:
                pre.append("authorization is bound to a different window configuration")
            if fb:   # compare-only against the SEALED S1a r0 reference (read-only), fully re-hashed before any case
                ref_dir = Path(expectations["sealed_reference_dir"])
                receipts["sealed"] = smoke.reference_receipt(ref_dir, 13_500_000_000, **expectations["reference_counts"])
                if not receipts["sealed"]["valid"] or {k: v["sha256"] for k, v in receipts["sealed"]["files"].items()} \
                        != expectations.get("sealed_reference_sha256"):
                    pre.append("sealed S1a reference differs from its bound hashes")
            elif pair:
                ref_dir = None                               # rank 0 writes snapshots into each case's own directory
            else:
                ref_dir = out / "reference"
                ref_dir.mkdir(mode=0o700)
                os.chmod(ref_dir, 0o700)
            free = shutil.disk_usage(out).free
            summary["reference"] = {"dir": str(ref_dir), "free_bytes_at_start": free,
                                    "budget_bytes": smoke_cfg["reference_budget_bytes"]}
            if free < smoke_cfg["min_free_disk_bytes"] + smoke_cfg["reference_budget_bytes"]:
                pre.append(f"free disk {free} B below the minimum {smoke_cfg['min_free_disk_bytes']} B + reference budget")
        else:
            pre += smoke.smoke_config_problems(smoke_cfg)
        pre += smoke.settings_problems(cluster_config())
        if pair:     # an old single-GPU authorization lacks every pair binding and is refused here
            pre += pc.pair_authorization_problems(args.authorization, candidate_root=candidate, uuids=pins,
                                                  n_cases_done=0, expected_sha256=ctx["auth_sha256"],
                                                  enrollment_policy_sha256=ctx.get("policy_sha256"))
        else:
            pre += lc.authorization_problems(args.authorization, scope=scope, candidate_root=candidate, gpus=[4],
                                             uuids=pins, n_cases_done=0, expected_sha256=ctx["auth_sha256"])
        if fb and not args.sup_unit.startswith("mgbench-"):
            pre.append("supervisor unit must be named mgbench-* (other windows' active-unit guard must see it)")
        if model is not None:          # the supervisor's OWN scope must carry the generated stop allowance/backstop
            try:
                props = scopes.unit_properties(args.sup_unit, ctx["meta_s"],
                                               "TimeoutStopUSec,RuntimeMaxUSec,ActiveEnterTimestampMonotonic")
                props["_monotonic_now_us"] = int(time.monotonic() * 1e6)
                for k in ("TimeoutStopUSec", "RuntimeMaxUSec"):      # systemd prints e.g. "2min 39s" or usec
                    r = scopes.run_systemctl(["show", scopes.unit_name(args.sup_unit), "-p", k, "--value",
                                              "--timestamp=unix"], ctx["meta_s"])
                    props[k] = _usec(r.stdout.strip())
            except Exception as exc:  # noqa: BLE001
                props = {"error": repr(exc)}
            summary["supervisor_scope_timing"] = props
            pre += wc.own_scope_timing_problems(props, model, work_deadline, time.time())
        if auth.get("max_cases") != n_cases:
            pre.append(f"authorization max_cases {auth.get('max_cases')} differs from the {n_cases} configured cases")
        own = scopes.own_cgroup_limits()
        lim = smoke_cfg["limits"]
        if not str(own.get("cgroup", "")).endswith(f"/{args.sup_unit}.scope"):
            pre.append(f"supervisor is not in its own scope ({own.get('cgroup')})")
        if own.get("cpu.max") != f"{lim['supervisor_cpu_quota_pct'] * 1000} 100000":
            pre.append(f"supervisor cpu.max {own.get('cpu.max')!r} is not the reserved quota")
        if own.get("memory.max") != str(lim["supervisor_memory_bytes"]):
            pre.append(f"supervisor memory.max {own.get('memory.max')!r} is not the reserved limit")
        summary.update(supervisor_cgroup_start=own, interpreter=sys.executable,
                       interpreter_realpath=os.path.realpath(sys.executable), mgbench_file=mgbench.__file__,
                       authorization=str(args.authorization), authorization_sha256=ctx["auth_sha256"],
                       expectations_sha256=args.expectations_sha256)
        # Candidate 14: one fresh owner-only cache directory PER CASE (created at launch time, never shared), seeded
        # only with the verified binary; here the seed is verified once and its provenance recorded before any use.
        cache.mkdir(mode=0o700)
        os.chmod(cache, 0o700)
        seed = {"requested": args.seed_cache is not None, "per_case": True}
        seed_dir, seed_sha = (Path(args.seed_cache[0]), args.seed_cache[1]) if args.seed_cache else (None, None)
        if seed_dir is not None:
            entries = list(seed_dir.iterdir())
            sos = [e for e in entries if e.suffix == ".so" and e.is_file() and not e.is_symlink()]
            got = hashlib.sha256(sos[0].read_bytes()).hexdigest() if len(sos) == 1 and len(entries) == 1 else None
            seed.update(source=str(seed_dir), sha256_source=got, entries=[e.name for e in entries])
            if got != seed_sha or seed_dir.stat().st_uid != os.getuid():
                pre.append(f"seed cache not verified (hash {got}, entries {[e.name for e in entries]})")
        summary["cache"] = {"path": str(cache), "mode": oct(cache.stat().st_mode & 0o777), "seed": seed}
        if time.monotonic() > startup_deadline_mono:
            pre.append("global startup budget exceeded before the first case")
        if pre:
            stop("preconditions: " + "; ".join(pre))
            return 2
        if contract:  # Candidate 18/19: project-wide host lease, joint, never reclaimed (window-local markers are not)
            lease, lp = wc.acquire_lease(lease_dir, window_gpus, {"window_dir": str(out), "supervisor_unit": args.sup_unit,
                                                                 "authorization_id": auth.get("id"),
                                                                 "authorization_sha256": ctx["auth_sha256"]})
            summary["host_lease"] = {"dir": str(lease_dir), "acquired": lease, "problems": lp}
            if lp:
                stop("host lease: " + "; ".join(lp))
                return 2
            ctx["lease"] = lease
        try:
            lc.create_exclusive(reservation, {"gpus": window_gpus, "uuids": {str(g): pins.get(g) for g in window_gpus},
                                              "created_utc": utc(), "by": args.sup_unit, "kind": "cooperative marker, "
                                              "not an exclusive facility allocation"})
            reserved_by_us = True
        except FileExistsError:
            stop(f"joint reservation {reservation.name} already exists (not ours; never reclaimed) -> refuse; "
                 "no partial reservation was made")
            return 2
        log(f"window expiry {auth.get('expires_utc')}; deadline in {deadline_mono - time.monotonic():.0f} s (monotonic); "
            f"supervisor {own.get('cgroup')} cpu.max {own.get('cpu.max')} memory.max {own.get('memory.max')}")

        if pair:     # Candidate 18: enrollment inside the SAME window, before any case; nothing else may change
            t_e = time.monotonic()
            reads, t_obs, eprob = [], [], []
            try:
                for k in range(2):
                    reads.append(json.loads(json.dumps(enrollment_read(ctx["tq"]))))
                    t_obs.append(utc())
                    if k == 0:
                        time.sleep(1.0)
            except Exception as exc:  # noqa: BLE001
                eprob.append(f"enrollment observation unreadable ({exc!r}) -> unknown, refuse")
            obs = {"reads": reads}
            if not eprob:
                eprob += wc.observation_problems(obs, policy)
            capacity = {}
            if not eprob:
                capacity, cp = wc.enrollment_capacity({int(k): v for k, v in reads[1]["gpus"].items()}, smoke_cfg,
                                                      feasibility, policy)
                eprob += cp
            if time.monotonic() > enroll_deadline_mono:
                eprob.append("enrollment did not finish within startup + enroll budget (same clock) -> refuse")
            write_new(out / "enrollment_attempt.json", {"observation": obs, "observed_utc": t_obs, "problems": eprob,
                                                         "capacity_pre_launch": capacity,
                                                         "seconds": time.monotonic() - t_e})
            if eprob:
                stop("enrollment refused: " + "; ".join(eprob))
                return 2
            check = dict(auth=auth, auth_sha256=ctx["auth_sha256"], policy_sha256=ctx["policy_sha256"],
                         config_sha256=summary["config_sha256"], window_dir=str(out), sup_unit=args.sup_unit,
                         supervisor_start_epoch=t_start_wall, expires_epoch=expiry)
            try:
                binding = wc.seal_record(out / "enrollment", wc.build_record(
                    auth=auth, auth_sha256=ctx["auth_sha256"], policy_sha256=ctx["policy_sha256"],
                    config_sha256=summary["config_sha256"], candidate_digest=auth.get("candidate_digest"),
                    window_dir=str(out), sup_unit=args.sup_unit, supervisor_start_utc=utc(t_start_wall), obs=obs,
                    observed_utc=t_obs, capacity=capacity))
            except FileExistsError as exc:
                stop(f"enrollment record already exists ({exc.filename}); never overwritten -> refuse")
                return 2
            erec, ep = wc.record_problems(binding["dir"], binding, **check)
            if ep:
                stop("enrollment record: " + "; ".join(ep))
                return 2
            ep_pins, ep_pci = wc.enrolled_pins(erec)
            pins.update(ep_pins)
            pci.update(ep_pci)
            ctx.update(enrollment=binding, enrollment_check=check, enrollment_record=erec)
            summary["enrollment"] = binding
            log(f"enrolled GPU 5 {pins[5]} {pci[5]} under policy {ctx['policy_sha256'][:12]}; record sealed "
                f"{binding['record_sha256'][:12]}")

        eff_cluster = dict(cluster_config(), allowed_gpus=window_gpus,
                           gpu_sets={1: [4], **({2: [4, 5]} if pair else {})})
        py = sys.executable
        for n, case in enumerate(smoke_cfg["cases"], 1):
            if STOP["signal"]:
                stop(f"signal {STOP['signal']}")
                break
            rec = {"case": case["id"], "index": n, "outcome": "not_run"}
            summary["cases"].append(rec)
            if diag and (fb or case.get("depends_on")):
                blocked = smoke.reference_recheck(ref_dir, receipts.get("sealed" if fb else case["depends_on"]))
                if blocked:
                    rec.update(outcome="blocked_invalid_reference", reason="; ".join(blocked))
                    stop(f"{case['id']}: {rec['reason']}")
                    break
            if contract:
                budget = wc.case_admission(time.time(), work_deadline, model)
                budget["gate_deadline"] = time.monotonic() + timing["gate_s"]
            else:
                budget = smoke.case_budget(time.monotonic(), deadline_mono, timing)
            rec["budget_before_gate"] = budget
            if not budget["fits"]:
                rec["reason"] = "remaining time cannot hold gate + child + grace + reap and the final reserve"
                stop(rec["reason"])
                break
            gate_deadline = budget["gate_deadline"]
            ctx["case_gpus"] = case_gpus_of(case)
            if pair and pc.case_gpu_problems(case["id"], ctx["case_gpus"]):
                rec["reason"] = "; ".join(pc.case_gpu_problems(case["id"], ctx["case_gpus"]))
                stop(rec["reason"])
                break
            rec["gpus"] = ctx["case_gpus"]
            log(f"[{n}/{n_cases}] {case['id']}: gate on GPUs {ctx['case_gpus']} (<= {timing['gate_s']} s incl. scope setup)")
            ok, problems, trail = settle(ctx, log, gate_deadline - 5)          # keep 5 s of the gate for setup
            rec["gate_trail"] = trail[-6:]
            rec["gate_polls"] = len(trail)
            if not ok:
                rec["reason"] = "gate: " + "; ".join(problems)
                stop(rec["reason"])
                break
            final = gate(ctx)
            if final:
                rec["reason"] = "final pre-launch re-validation: " + "; ".join(final)
                stop(rec["reason"])
                break
            if time.monotonic() > gate_deadline:
                rec["reason"] = "gate phase exceeded its budget"
                stop(rec["reason"])
                break
            attempt = out / "cases" / f"{n:02d}_{case['id']}"
            attempt.mkdir(exist_ok=False)
            if pair:
                exp = pc.pair_experiment(case, smoke_cfg, attempt)
            elif fb:
                exp = smoke.fb_experiment(case, smoke_cfg, ref_dir)
            elif diag:
                exp = smoke.diag_experiment(case, smoke_cfg, ref_dir)
            else:
                exp = {"name": f"validation_v2_20261016s2_smoke_{case['id']}", "kind": "train", "repeats": 1,
                       "gpu_counts": [1], "defaults": dict(smoke_cfg["common"], mem_fraction=smoke_cfg["mem_fraction"]),
                       "workloads": [{"model": smoke_cfg["common"]["model"], "frameworks": [case["framework"]],
                                      "numerical_checks": case["numerical_checks"], "variant": case["id"]}]}
            specs = expand_experiment(exp, eff_cluster)
            if len(specs) != 1 or specs[0]["gpus"] != ctx["case_gpus"]:
                rec["reason"] = f"spec expansion produced {[(s['run_id'], s['gpus']) for s in specs]}"
                stop(rec["reason"])
                break
            spec = specs[0]
            write_new(attempt / "spec.json", spec)
            unit = f"mgbench-{'pair' if pair else 'diag' if diag else 'smoke'}-{out.name}-{n:02d}"
            prov = smoke.make_case_cache(cache, f"{n:02d}_{case['id']}", seed_dir, seed_sha)
            case_cache = Path(prov["dir"])
            write_new(attempt / "cache_provenance.json", prov)
            unsafe = prov["problems"] + smoke.cache_preload_problems(case_cache, prov["expected_manifest"])
            if unsafe:
                rec["reason"] = "case cache: " + "; ".join(unsafe)
                stop(rec["reason"])
                break
            cg = ctx["case_gpus"]
            thr = smoke_cfg.get("threads_per_rank", (smoke_cfg.get("limits") or {}).get("threads_per_rank"))
            thr = thr.get(case["id"]) if isinstance(thr, dict) else thr
            env = smoke.env_for_worker(os.environ, src=SRC, uuid=[pins[g] for g in cg] if len(cg) > 1 else pins[cg[0]],
                                       index=cg if len(cg) > 1 else cg[0],
                                       pci=[pci[g] for g in cg] if len(cg) > 1 else pci[cg[0]],
                                       threads=thr, cache=case_cache, run_id=f"{out.name}/{case['id']}")
            q = shlex.quote
            prestep = (f"timeout 30 {q(py)} -s -m mgbench.device_identity {q(','.join(pins[g] for g in cg))} "
                       f"{q(','.join(pci[g] for g in cg))} {','.join(map(str, cg))} "
                       f"{q(str(attempt / 'identity_prestep.json'))}")
            train = " ".join(q(x) for x in [
                "numactl", f"--cpunodebind={smoke_cfg.get('numa_node', 1)}", f"--preferred={smoke_cfg.get('numa_node', 1)}",
                py, "-s", "-m", "torch.distributed.run", "--standalone", "--nnodes=1", f"--nproc-per-node={len(cg)}",
                "-m", "mgbench.train.guarded", "--spec", str(attempt / "spec.json"), "--out", str(attempt)])
            cmd = ["sh", "-c", f"{prestep} && exec {train}"]
            hook = pc.test_worker_hook(attempt, len(cg))      # active ONLY with a declared NVML test stub + hidden CUDA
            if pair:     # Candidate 18: each rank checks its OWN actual capacity before allocation
                fcase = feasibility["cases"][case["id"]]
                env["MGBENCH_CAPACITY_CONTRACT"] = json.dumps(wc.worker_capacity_contract(
                    case, fcase, cg, {int(k): v for k, v in ctx["enrollment_record"]["devices"].items()}, policy))
                rec["enrollment_binding"] = ctx["enrollment"]
            elif fb:     # Candidate 19: GPU 4's OWN capacity - NVML upper bound now, actual Torch total in the rank
                g4 = ((ctx.get("last_snapshot") or {}).get("gpus") or {}).get(4) or {}
                mib = g4.get("mem_total_mib")
                dev = {"uuid": pins[4], "nvml_total_bytes": round(mib * wc.MIB) if isinstance(mib, (int, float)) else mib}
                fcase = wc.fb_feasibility_case(feasibility, case["id"])
                pre_cap = wc.device_capacity_record(4, dev, case, fcase, expected_torch_total_bytes=(
                    smoke_cfg.get("known_torch_total_bytes") or {}).get(4))
                rec["capacity_pre_launch"] = pre_cap
                if pre_cap["problems"]:
                    rec["reason"] = "capacity (pre-launch, NVML upper bound): " + "; ".join(pre_cap["problems"])
                    stop(rec["reason"])
                    break
                env["MGBENCH_CAPACITY_CONTRACT"] = json.dumps(wc.worker_capacity_contract(
                    case, fcase, cg, {4: dev}, {"known_torch_total_bytes": {
                        str(k): v for k, v in (smoke_cfg.get("known_torch_total_bytes") or {}).items()}}))
            if hook:
                cmd = hook["cmd"]
                env.update(hook["env"])
                summary["test_hook_used"] = True
            limits = dict(cpu_quota_pct=lim["worker_cpu_quota_pct"], memory_max=lim["worker_memory"],
                          runtime_max_s=model["worker_runtime_max_s"] if contract else timing["child_s"] + 2,
                          stop_timeout_s=model["worker_timeout_stop_s"] if contract else timing["grace_s"])
            before = smoke.snapshot(case_cache)
            write_new(attempt / "launch.json", {"unit": unit, "command": cmd, "scope_command":
                                                scopes.scope_command(unit, cmd, **limits), "env": env,
                                                "child_s": timing["child_s"], "grace_s": timing["grace_s"],
                                                "reap_s": timing["reap_s"], "launched_utc": utc(),
                                                "nvml_identity_link1": ctx.get("nvml_identity"),
                                                "pre_launch_snapshot": ctx.get("last_snapshot"),
                                                "cache_before": before})
            ctx["cases_started"] += 1
            rec["outcome"] = "launched"
            log(f"[{n}/{n_cases}] {case['id']}: launch in {unit}.scope (child <= {timing['child_s']} s)")
            t_launch = time.monotonic()
            with open(attempt / "run.log", "x") as lf:
                proc, handle = scopes.start_scope(unit, cmd, env=env, cwd=str(A6000), stdout=lf, stderr=lf, **limits)
                rec["identity"] = {"unit": f"{unit}.scope", "leader_pid": handle.leader_pid,
                                   "leader_start_time": handle.leader_start_time, "cgroup": handle.cgroup,
                                   "identity_error": handle.identity_error}
                summary["peak_concurrency"] = max(summary["peak_concurrency"], 1)
                w = watch(ctx, proc, handle, t_launch + timing["child_s"]) if handle.cgroup else \
                    {"stop_reason": f"scope identity not captured ({handle.identity_error})", "peaks": {}}
                rc = proc.poll()
                t_stop = time.monotonic()
                cleanup = scopes.cleanup_scope(handle, grace_s=timing["grace_s"],
                                               deadline=time.time() + timing["grace_s"],
                                               kill_wait_s=timing.get("kill_wait_s", 10.0), meta_s=ctx["meta_s"])
                if rc is None:
                    try:
                        rc = proc.wait(timeout=timing["reap_s"])
                    except Exception as exc:  # noqa: BLE001
                        cleanup["confirmed"] = False
                        cleanup["steps"].append(f"leader not reaped within {timing['reap_s']} s ({exc!r})")
                rec["phase_seconds"] = {"child": t_stop - t_launch, "stop_and_reap": time.monotonic() - t_stop}
                if contract:  # Candidate 18/19: three classes; observed emptiness is never reported as unconfirmed
                    rec["cleanup_class"] = wc.classify_cleanup(cleanup, rec["phase_seconds"]["stop_and_reap"], model)
                    rec["cleanup_bound_s"] = model["stop_and_reap_bound_s"]
                elif rec["phase_seconds"]["stop_and_reap"] > timing["grace_s"] + timing["reap_s"] + 2:
                    cleanup["confirmed"] = False
                    cleanup["steps"].append("stop/reap exceeded grace + reap budget")
            after = smoke.snapshot(case_cache)
            rec.update(rc=rc, guard=w, cleanup=cleanup, ended_utc=utc(), duration_s=time.monotonic() - t_launch,
                       worker_scope_limits=limits, cache_events=smoke.snapshot_diff(before, after))
            summary["peak_concurrency"] = max(summary["peak_concurrency"],
                                              w.get("peaks", {}).get("active_worker_scopes_max", 0))
            if not cleanup["confirmed"]:
                all_cleanups_confirmed = False
                rec["outcome"] = "cleanup_unconfirmed"
                write_new(attempt / "case_receipt.json", rec)
                stop(f"cleanup of {unit}.scope not confirmed; reservation, host lease and hold retained")
                break
            if rec.get("cleanup_class") == "confirmed_over_budget":
                rec["outcome"] = "cleanup_over_budget"
                write_new(attempt / "case_receipt.json", rec)
                stop(f"cleanup of {unit}.scope confirmed empty but over the modeled bound "
                     f"({rec['phase_seconds']['stop_and_reap']:.1f} > {model['stop_and_reap_bound_s']} s); not a success")
                break
            if w["stop_reason"]:
                rec["outcome"] = "stopped"
                write_new(attempt / "case_receipt.json", rec)
                stop(f"{case['id']}: {w['stop_reason']}")
                break
            result_path = attempt / "result.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else None
            prestep_rec = json.loads((attempt / "identity_prestep.json").read_text()) \
                if (attempt / "identity_prestep.json").exists() else None
            kw = dict(candidate_src=SRC, interpreter=py, uuid=pins[4],
                      worker_cpu_max=f"{lim['worker_cpu_quota_pct'] * 1000} 100000",
                      worker_memory_max=str(lim["worker_memory_bytes"]))
            prestep_check = {"check": "Driver-API pre-step (worker scope, before torchrun; pre-allocation) saw exactly "
                                      "the approved device", "pass": bool(prestep_rec) and prestep_rec.get("problems") == [],
                             "observed": prestep_rec}
            if pair:
                ev = pc.evaluate_pair_case(attempt, rc, case, expectations, gpus=ctx["case_gpus"], pins=pins, pci=pci,
                                           prestep=prestep_rec, test_hook=bool(summary.get("test_hook_used")))
                failed = [c["check"] for c in ev["checks"] if not c["pass"]]
                ev["overall_grade"] = "evidence only (the R5-vs-C5 comparison runs after the window)"
            elif fb:
                ev = smoke.evaluate_fb_case(result, rc, case, expectations, uuid=pins[4], candidate_src=SRC,
                                            interpreter=py)
                ev["checks"].append(prestep_check)
                capf = attempt / "rank0_capacity.json"            # Candidate 19: the rank's own capacity, pre-allocation
                cap = json.loads(capf.read_text()) if capf.is_file() and capf.stat().st_size < (1 << 20) else None
                ev["checks"].append({"check": "rank validated its own actual capacity before allocation", "pass":
                                     isinstance(cap, dict) and cap.get("pass") is True and cap.get("index") == 4 and
                                     cap.get("rank") == 0, "observed": (cap or {}).get("problems", cap)})
                ev["pass"] = all(c["pass"] for c in ev["checks"])
                ev["overall_grade"] = "reported, not graded (no justified equivalence margin)"
                failed = [c["check"] for c in ev["checks"] if not c["pass"]]
            elif diag:
                ev = smoke.evaluate_diag_case(result, rc, case, expectations, tolerances, **kw)
                ev["integrity_checks"].append(prestep_check)
                ev["integrity_pass"] = all(c["pass"] for c in ev["integrity_checks"])
                ev["pass"] = ev["integrity_pass"] and ev["gross_pass"]
                failed = [c["check"] for c in ev["integrity_checks"] + ev["gross_checks"] if not c["pass"]]
                if case.get("reference_role") == "write" and ev["pass"]:
                    t_r = time.monotonic()
                    receipt = smoke.reference_receipt(ref_dir, smoke_cfg["reference_budget_bytes"],
                                                      **expectations["reference_counts"])
                    receipt["hash_seconds"] = time.monotonic() - t_r
                    write_new(out / "reference_receipt.json", receipt)
                    receipts[case["id"]] = receipt
                    ev["reference_receipt_valid"] = receipt["valid"]
                    if not receipt["valid"]:
                        ev["pass"] = False
                        failed.append("reference receipt: " + "; ".join(receipt["problems"]))
            else:
                ev = smoke.evaluate_case(result, rc, case, expectations, **kw)
                ev["checks"].append(prestep_check)
                ev["pass"] = all(c["pass"] for c in ev["checks"])
                failed = [c["check"] for c in ev["checks"] if not c["pass"]]
            rec["evaluation"] = ev
            if not ev["pass"]:
                rec["outcome"] = "unexpected_result"
            elif diag:
                rec["outcome"] = "completed_valid"
                rec["grade"] = ev["overall_grade"]
            else:
                rec["outcome"] = "completed_as_expected"
            log(f"[{n}/{n_cases}] {case['id']}: {rec['outcome']}{' grade ' + rec['grade'] if rec.get('grade') else ''} "
                f"rc={rc} child {rec['phase_seconds']['child']:.0f} s; "
                f"node GPU power peak {w['peaks'].get('node_gpu_power_w', 0):.0f} W; cleanup confirmed; "
                f"cache events {len(rec['cache_events']['created'])} created / {len(rec['cache_events']['modified'])} modified")
            write_new(attempt / "case_receipt.json", rec)
            if not ev["pass"]:
                stop(f"{case['id']}: result differs from frozen expectations ({failed})")
                break
        else:
            summary["stopped_because"] = summary["stopped_because"] or "all approved cases attempted"
    finally:
        # Every configured case stays visible: cases never reached are listed with the reason.
        seen = {r["case"]: r for r in summary["cases"]}
        control = seen.get(smoke.DIAG_ORDER[0]) if diag else None
        control_failed = (control is not None and control.get("outcome") != "not_run"
                          and not receipts.get(smoke.DIAG_ORDER[0], {}).get("valid"))     # ran, but no valid reference
        for n, case in enumerate(smoke_cfg["cases"], 1):
            if case["id"] not in seen:
                summary["cases"].append({"case": case["id"], "index": n, "outcome":
                                         "blocked_invalid_reference" if case.get("depends_on") and control_failed
                                         else "not_run", "reason": f"window stopped: {summary['stopped_because']}"})
        revoked = lc.revoke(args.authorization, f"{'diagnostics' if diag else 'smoke'} supervisor ended: "
                                                f"{summary.get('stopped_because')}")
        summary["authorization_revoked_now"] = revoked
        summary["repository_hold_active_at_end"] = (cluster_config().get("launch_hold") or {}).get("active") is True
        try:
            final_idle = idle_problems(window_gpus, dict(cluster_config()["idle"], query_timeout_s=ctx["tq"]))
        except Exception as exc:  # noqa: BLE001
            final_idle = [f"final check unreadable ({exc!r})"]
        summary["final_gpu4_check"] = final_idle or "idle"
        summary["all_cleanups_confirmed"] = all_cleanups_confirmed
        summary["supervisor_cgroup_end"] = scopes.own_cgroup_limits()
        summary["ended_utc"] = utc()
        summary["seconds_before_deadline_at_end"] = deadline_mono - time.monotonic()
        summary["final_reserve_respected"] = time.monotonic() <= deadline_mono
        if all_cleanups_confirmed and reserved_by_us:
            write_new(out / f"{reservation.name}.released", {"released_utc": utc(), "all_cleanups_confirmed": True})
        if lease is not None:      # ownership-scoped; retained (manual inspection) unless every cleanup was confirmed
            if all_cleanups_confirmed:
                summary["host_lease"]["release_problems"] = wc.release_lease(
                    lease_dir, lease, f"released by {args.sup_unit}: all cleanups confirmed; {summary.get('stopped_because')}")
                summary["host_lease"]["state"] = "released" if not summary["host_lease"]["release_problems"] else "RETAINED"
            else:
                summary["host_lease"]["state"] = "RETAINED (cleanup unconfirmed; manual inspection)"
        summary["reservation"] = ("never created" if not reserved_by_us else
                                  "released" if all_cleanups_confirmed else "RETAINED (cleanup unconfirmed)")
        write_new(out / "window_summary.json", summary)
        log(f"window summary written; authorization revoked={revoked}; reservation {summary['reservation']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
