"""Exclusive systemd user scopes as the ownership and cleanup authority for launched work (A6000 smoke stage).

Every supervised command runs in its own transient scope created by `systemd-run --user --scope` with enforced
CPUQuota/MemoryMax and an absolute RuntimeMaxSec, so systemd itself ends it even if the supervisor dies. Cleanup acts
only on that scope's cgroup (`systemctl --user kill <unit>`), never on processes found by UID or an environment
marker. A scope counts as cleaned up only when its unit is gone or inactive AND its cgroup has no processes; any
error while confirming that is reported as `confirmed: False` (the caller must then keep its hold/reservation).
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

CGROUP_ROOT = Path("/sys/fs/cgroup")


def scope_command(unit: str, cmd: list[str], *, cpu_quota_pct: int, memory_max: str, runtime_max_s: int,
                  stop_timeout_s: int, extra_props: tuple[str, ...] = ()) -> list[str]:
    props = [f"CPUQuota={int(cpu_quota_pct)}%", f"MemoryMax={memory_max}", "MemorySwapMax=0",
             f"RuntimeMaxSec={int(runtime_max_s)}", f"TimeoutStopSec={int(stop_timeout_s)}",
             "KillMode=control-group", *extra_props]
    out = ["systemd-run", "--user", "--scope", "--quiet", f"--unit={unit}", "--collect"]
    for p in props:
        out += ["-p", p]
    return out + ["--", *cmd]


def proc_start_time(pid: int) -> int | None:
    """Kernel start time (clock ticks since boot) - with the PID, a process identity that survives PID reuse."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return int(stat[stat.rindex(")") + 2:].split()[19])
    except (OSError, ValueError, IndexError):
        return None


def proc_cgroup(pid: int) -> str | None:
    try:
        for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                return line[3:]
    except OSError:
        return None
    return None


def unit_name(unit: str) -> str:
    """Full unit name. systemctl assumes ".service" for a bare name, which would address a different (absent)
    unit - found by the lifecycle self-test on 2026-10-16 - so every call uses the explicit ".scope" name."""
    return unit if unit.endswith((".scope", ".service", ".slice")) else f"{unit}.scope"


def run_systemctl(args: list[str], timeout_s: float = 10.0) -> subprocess.CompletedProcess:
    return subprocess.run(["systemctl", "--user", *args], capture_output=True, text=True, timeout=timeout_s)


def unit_properties(unit: str, timeout_s: float = 10.0, props: str = "ActiveState,ControlGroup,LoadState,SubState,MainPID") -> dict:
    r = run_systemctl(["show", unit_name(unit), "-p", props], timeout_s)
    if r.returncode != 0:
        raise RuntimeError(f"systemctl show {unit} failed: {r.stderr.strip()}")
    return dict(line.split("=", 1) for line in r.stdout.strip().splitlines() if "=" in line)


def cgroup_pids(cgroup: str) -> list[int]:
    """Every PID in the cgroup and its descendants. Raises if the cgroup exists but cannot be read."""
    base = CGROUP_ROOT / cgroup.lstrip("/")
    if not base.exists():
        return []
    pids = []
    for d in [base, *[p for p in base.rglob("*") if p.is_dir()]]:
        pids += [int(x) for x in (d / "cgroup.procs").read_text().split()]
    return pids


def cgroup_limits(cgroup: str) -> dict:
    base = CGROUP_ROOT / cgroup.lstrip("/")
    out = {}
    for name in ("cpu.max", "memory.max", "memory.swap.max", "memory.current", "memory.peak", "pids.current"):
        try:
            out[name] = (base / name).read_text().strip()
        except OSError as exc:
            out[name] = f"unreadable: {exc!r}"
    try:
        out["cpu.stat.usage_usec"] = int(dict(l.split() for l in (base / "cpu.stat").read_text().splitlines())["usage_usec"])
    except (OSError, KeyError, ValueError) as exc:
        out["cpu.stat.usage_usec"] = f"unreadable: {exc!r}"
    return out


@dataclass
class ScopeHandle:
    unit: str
    leader_pid: int
    leader_start_time: int | None = None
    cgroup: str | None = None
    identity_error: str | None = None
    events: list = field(default_factory=list)

    def note(self, msg: str):
        self.events.append({"t": time.time(), "event": msg})


def start_scope(unit: str, cmd: list[str], *, env: dict, cwd: str, stdout, stderr, identity_timeout_s: float = 20.0,
                **limits) -> tuple[subprocess.Popen, ScopeHandle]:
    """Start `cmd` in a new exclusive scope (and a new session). The scope's cgroup and the leader's identity are
    captured *before* the caller starts waiting for completion."""
    p = subprocess.Popen(scope_command(unit, cmd, **limits), env=env, cwd=cwd, stdout=stdout, stderr=stderr,
                         start_new_session=True)
    h = ScopeHandle(unit=unit, leader_pid=p.pid)
    deadline = time.time() + identity_timeout_s
    while time.time() < deadline:
        cg = proc_cgroup(p.pid)
        if cg and cg.endswith(f"/{unit}.scope"):
            h.cgroup, h.leader_start_time = cg, proc_start_time(p.pid)
            h.note(f"identity captured: pid {p.pid} start {h.leader_start_time} cgroup {cg}")
            return p, h
        if p.poll() is not None:
            break
        time.sleep(0.05)
    # Fall back to systemd's record of the unit's cgroup (e.g. the leader already exited).
    try:
        props = unit_properties(unit)
        if props.get("ControlGroup", "").endswith(f"/{unit}.scope"):
            h.cgroup = props["ControlGroup"]
            h.note(f"identity from systemd: cgroup {h.cgroup}")
            return p, h
    except Exception as exc:  # noqa: BLE001
        h.identity_error = repr(exc)
    h.identity_error = h.identity_error or "scope cgroup not observed"
    h.note(f"identity NOT captured: {h.identity_error}")
    return p, h


def scope_state(h: ScopeHandle, meta_s: float = 10.0) -> dict:
    """{'pids': [...], 'active': bool}. Raises on metadata errors (including a call exceeding `meta_s`)."""
    if not h.cgroup:
        raise RuntimeError(f"no recorded cgroup for {h.unit} ({h.identity_error})")
    props = unit_properties(h.unit, meta_s)
    if props.get("LoadState") not in ("loaded", "not-found"):
        raise RuntimeError(f"unexpected LoadState for {h.unit}: {props}")
    if props.get("ControlGroup") not in ("", None, h.cgroup):
        raise RuntimeError(f"unit {h.unit} now reports cgroup {props.get('ControlGroup')}, recorded {h.cgroup}")
    return {"pids": cgroup_pids(h.cgroup), "active": props.get("ActiveState") in ("active", "activating", "deactivating")}


def cleanup_scope(h: ScopeHandle, grace_s: float, deadline: float | None = None, *, kill_wait_s: float = 10.0,
                  meta_s: float = 10.0) -> dict:
    """SIGTERM every process in the scope's cgroup, wait up to `grace_s` (never past `deadline`), then SIGKILL, then
    confirm the scope is empty. Only the unit's own cgroup is signalled.
    Candidate 18: every systemctl call is bounded by `meta_s` and the loops never sleep past their ends, so the worst
    path is grace_s + kill_wait_s + 8 x meta_s (window_contract.timing_model); `seconds` records the actual time."""
    t0 = time.monotonic()
    receipt = {"unit": h.unit, "cgroup": h.cgroup, "steps": [], "confirmed": False, "bounds": {
        "grace_s": grace_s, "kill_wait_s": kill_wait_s, "metadata_call_s": meta_s}}
    if not h.cgroup:
        receipt["steps"].append(f"no recorded scope identity ({h.identity_error}): nothing signalled")
        return receipt
    try:
        st = scope_state(h, meta_s)
        receipt["steps"].append({"before": st})
        if st["pids"] or st["active"]:
            run_systemctl(["kill", "--kill-whom=all", "--signal=SIGTERM", unit_name(h.unit)], meta_s)
            end = time.time() + grace_s if deadline is None else min(time.time() + grace_s, deadline)
            while time.time() < end:
                st = scope_state(h, meta_s)
                if not st["pids"]:
                    break
                time.sleep(max(0.0, min(0.5, end - time.time())))
            if st["pids"]:
                receipt["steps"].append({"sigkill": st["pids"]})
                run_systemctl(["kill", "--kill-whom=all", "--signal=SIGKILL", unit_name(h.unit)], meta_s)
                end = time.time() + kill_wait_s
                while time.time() < end and scope_state(h, meta_s)["pids"]:
                    time.sleep(max(0.0, min(0.2, end - time.time())))
        final = scope_state(h, meta_s)
        if final["active"] and not final["pids"]:
            run_systemctl(["stop", unit_name(h.unit)], meta_s)
            final = scope_state(h, meta_s)
        receipt["steps"].append({"after": final})
        receipt["confirmed"] = not final["pids"] and not final["active"]
    except Exception as exc:  # noqa: BLE001 - unconfirmed cleanup: caller keeps the hold/reservation
        receipt["steps"].append(f"metadata error: {exc!r}")
        receipt["confirmed"] = False
    receipt["seconds"] = time.monotonic() - t0
    return receipt


def own_cgroup_limits() -> dict:
    cg = proc_cgroup(os.getpid())
    return {"cgroup": cg, **(cgroup_limits(cg) if cg else {}), "affinity": sorted(os.sched_getaffinity(0))}
