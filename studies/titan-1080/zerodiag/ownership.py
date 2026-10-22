"""Whole-job ownership: every rank and descendant, cleanable after any exit.

Why not one process group for all ranks: joining a process group across session
boundaries is EPERM. The previous launcher called `os.setsid` in rank 0 and then
tried `os.setpgid(0, rank0_pid)` in later ranks, which raised inside
`preexec_fn` and prevented every rank after the first from launching at all.
That was reproduced with harmless fixtures before this module was written.

The model here instead:

* each rank is started with `start_new_session=True`, so it leads its own
  session and process group and its own descendants inherit that group;
* every rank receives `ZERODIAG_JOB_TOKEN` in its environment, which its
  descendants inherit too;
* ownership is the UNION of (a) the recorded per-rank process groups and (b) a
  sweep of `/proc/<pid>/environ` for the token, which still finds a descendant
  that called `setsid` and escaped its parent's group;
* the ledger is written to disk as it is built, so cleanup remains possible
  after the launcher itself exits.

`preexec_fn` is not used anywhere, so the launcher is safe to thread.

Nothing here is a resource cap. It is accounting and ownership-scoped cleanup.
"""
from __future__ import annotations

import os
import pathlib
import signal
import time

JOB_TOKEN_ENV = "ZERODIAG_JOB_TOKEN"


def _read_environ_token(pid: int) -> str | None:
    try:
        raw = pathlib.Path(f"/proc/{pid}/environ").read_bytes()
    except (OSError, PermissionError):
        return None
    for item in raw.split(b"\0"):
        if item.startswith(JOB_TOKEN_ENV.encode() + b"="):
            return item.split(b"=", 1)[1].decode("utf-8", "replace")
    return None


def _is_zombie(pid: int) -> bool:
    """A zombie has exited and is merely unreaped; it owns no resources.

    Without this, cleanup can never confirm absence: `terminate()` signals the
    ranks, but each stays in /proc as a zombie until its parent reaps it, and a
    zombie still answers getpgid() and has a readable status file.
    """
    try:
        txt = pathlib.Path(f"/proc/{pid}/stat").read_text()
        return txt.rsplit(")", 1)[1].split()[0] == "Z"
    except (OSError, IndexError):
        return False


def _proc_uid(pid: int) -> int | None:
    try:
        for line in pathlib.Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("Uid:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        return None
    return None


def _vmrss_bytes(pid: int) -> int | None:
    try:
        for line in pathlib.Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


class JobOwnership:
    """Tracks and cleans one job's ranks and their descendants."""

    def __init__(self, token: str, *, ledger_path: str | pathlib.Path | None = None,
                 excluded_pgids=()):
        if not token or len(token) < 8:
            raise ValueError("a job token of at least 8 characters is required")
        self.token = token
        self.uid = os.getuid()
        self.ledger_path = pathlib.Path(ledger_path) if ledger_path else None
        self.ranks: dict[int, dict] = {}
        self.excluded_pgids = frozenset(excluded_pgids)

    # ---------------------------------------------------------------- setup
    def env_for_child(self, base: dict | None = None) -> dict:
        env = dict(base if base is not None else os.environ)
        env[JOB_TOKEN_ENV] = self.token
        return env

    def register(self, rank: int, pid: int) -> dict:
        try:
            pgid = os.getpgid(pid)
        except (ProcessLookupError, PermissionError, OSError):
            pgid = pid            # already gone; its own pid is the best record
        if pgid in self.excluded_pgids:
            raise ValueError("cannot register an excluded process group")
        rec = {"rank": rank, "pid": pid, "pgid": pgid,
               "registered_monotonic": round(time.monotonic(), 3)}
        self.ranks[rank] = rec
        self._persist()
        return rec

    def _persist(self) -> None:
        if self.ledger_path is None:
            return
        import json
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.ledger_path.write_text(json.dumps(
            {"token": self.token, "uid": self.uid,
             "ranks": [self.ranks[r] for r in sorted(self.ranks)],
             "excluded_pgids": sorted(self.excluded_pgids),
             "model": ("per-rank session; ownership is the union of recorded "
                       "process groups and a token sweep of /proc"),
             "cleanable_after_launcher_exit": True},
            indent=2, sort_keys=True))

    # ------------------------------------------------------------ discovery
    def token_pids(self) -> list[int]:
        """Every live pid of mine whose environment carries this job token."""
        found = []
        for d in pathlib.Path("/proc").iterdir():
            if not d.name.isdigit():
                continue
            pid = int(d.name)
            if _proc_uid(pid) != self.uid or _is_zombie(pid):
                continue
            try:
                if os.getpgid(pid) in self.excluded_pgids:
                    continue
            except ProcessLookupError:
                continue
            if _read_environ_token(pid) == self.token:
                found.append(pid)
        return sorted(found)

    def group_pids(self) -> list[int]:
        """Every live pid of mine in a recorded rank process group."""
        pgids = {r["pgid"] for r in self.ranks.values()}
        found = []
        for d in pathlib.Path("/proc").iterdir():
            if not d.name.isdigit():
                continue
            pid = int(d.name)
            if _proc_uid(pid) != self.uid or _is_zombie(pid):
                continue
            try:
                if os.getpgid(pid) in pgids:
                    found.append(pid)
            except (ProcessLookupError, PermissionError, OSError):
                continue
        return sorted(found)

    def owned_pids(self) -> list[int]:
        return sorted(set(self.token_pids()) | set(self.group_pids()))

    def accounting(self) -> dict:
        """Conservative owned resident memory over the whole job."""
        pids = self.owned_pids()
        per, total, unreadable = {}, 0, []
        for pid in pids:
            b = _vmrss_bytes(pid)
            if b is None:
                unreadable.append(pid)
            else:
                per[pid] = b
                total += b
        return {"total_bytes": total, "per_pid": per, "unreadable": unreadable,
                "n_owned": len(pids),
                "discovery": "union of recorded rank pgids and a /proc token sweep",
                "method": "sum of /proc/<pid>/status VmRSS over owned pids",
                "shared_pages": "counted once per sharing process: an over-estimate",
                "enforced_cap": False}

    # -------------------------------------------------------------- cleanup
    def terminate(self, *, grace_s: float = 20.0) -> dict:
        """SIGTERM every owned group and stray, grace, then SIGKILL. Ours only."""
        steps = []
        for pgid in sorted({r["pgid"] for r in self.ranks.values()}):
            try:
                os.killpg(pgid, signal.SIGTERM)
                steps.append({"signal": "TERM", "pgid": pgid, "sent": True})
            except (ProcessLookupError, PermissionError, OSError) as exc:
                steps.append({"signal": "TERM", "pgid": pgid, "sent": False,
                              "error": type(exc).__name__})
        for pid in self.token_pids():
            if _proc_uid(pid) != self.uid:
                continue
            try:
                os.kill(pid, signal.SIGTERM)
                steps.append({"signal": "TERM", "pid": pid, "sent": True})
            except (ProcessLookupError, PermissionError, OSError):
                pass
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline and self.owned_pids():
            time.sleep(0.1)
        remaining = self.owned_pids()
        if remaining:
            for pgid in sorted({r["pgid"] for r in self.ranks.values()}):
                try:
                    os.killpg(pgid, signal.SIGKILL)
                    steps.append({"signal": "KILL", "pgid": pgid, "sent": True})
                except (ProcessLookupError, PermissionError, OSError):
                    pass
            for pid in remaining:
                if _proc_uid(pid) != self.uid:
                    continue
                try:
                    os.kill(pid, signal.SIGKILL)
                    steps.append({"signal": "KILL", "pid": pid, "sent": True})
                except (ProcessLookupError, PermissionError, OSError):
                    pass
            hard = time.monotonic() + 10.0
            while time.monotonic() < hard and self.owned_pids():
                time.sleep(0.1)
        return {"steps": steps, "scope": "owned ranks and their descendants only",
                "foreign_processes_signalled": 0, **self.confirm_absent()}

    def confirm_absent(self) -> dict:
        pids = self.owned_pids()
        return {"owned_absent_confirmed": not pids, "surviving_pids": pids,
                "basis": ("a leader's exit is not sufficient; every owned pid "
                          "and token-carrying descendant is checked. Zombies "
                          "are excluded: they have exited and hold nothing.")}


def run_ranks(*, world_size: int, argv_for, out_dir, timeout_s: float,
              budget_bytes: int, interval_s: float = 0.5,
              env_for=None, token: str | None = None,
              grace_s: float = 20.0) -> dict:
    """Spawn, supervise, account for and clean up one job's ranks.

    This is the production supervision mechanism. `argv_for(rank, rank_dir)`
    returns that rank's command line and `env_for(rank, rank_dir)` its extra
    environment, so the same code path supervises the real model worker and the
    harmless CPU fixtures used in the lifecycle tests. No `preexec_fn` is used.

    A partial launch is a failure, not a smaller job: if any rank fails to
    start, the ranks already started are cleaned up and the attempt reports it.
    """
    import json
    import subprocess
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    token = token or ("zd-" + os.urandom(8).hex())
    own = JobOwnership(token, ledger_path=out_dir / "ownership_ledger.json")

    procs: dict[int, object] = {}
    launch_errors: dict[int, str] = {}
    t0 = time.monotonic()
    for r in range(world_size):
        rd = out_dir / f"rank{r}"
        rd.mkdir(parents=True, exist_ok=True)
        env = own.env_for_child(env_for(r, rd) if env_for else None)
        try:
            p = subprocess.Popen(
                argv_for(r, rd), env=env, start_new_session=True,
                stdout=open(rd / "worker.stdout", "w"),
                stderr=open(rd / "worker.stderr", "w"))
        except Exception as exc:                 # partial launch
            launch_errors[r] = f"{type(exc).__name__}: {exc}"
            break
        procs[r] = p
        own.register(r, p.pid)

    peak = 0
    samples = 0
    breached = False
    deadline = t0 + timeout_s
    if not launch_errors:
        while time.monotonic() < deadline:
            if all(p.poll() is not None for p in procs.values()):
                break
            acc = own.accounting()
            samples += 1
            peak = max(peak, acc["total_bytes"])
            if acc["total_bytes"] > budget_bytes:
                breached = True
                break
            time.sleep(interval_s)
    acc = own.accounting()
    samples += 1
    peak = max(peak, acc["total_bytes"])

    timed_out = (not launch_errors and not breached
                 and any(p.poll() is None for p in procs.values()))
    cleanup = None
    if launch_errors or breached or timed_out or own.owned_pids():
        cleanup = own.terminate(grace_s=grace_s)
    for p in procs.values():
        try:
            p.wait(timeout=grace_s + 10)
        except subprocess.TimeoutExpired:
            pass
    # Re-confirm AFTER reaping, so a zombie awaiting wait() is not mistaken for
    # a surviving process, and so any descendant that escaped is still swept.
    cleanup = own.terminate(grace_s=grace_s)
    elapsed = time.monotonic() - t0

    report = {
        "token": token, "world_size": world_size,
        "ranks_launched": sorted(procs), "n_launched": len(procs),
        "launch_errors": launch_errors,
        "partial_launch": bool(launch_errors),
        "exit_codes": {str(r): procs[r].returncode for r in sorted(procs)},
        "elapsed_s": round(elapsed, 2),
        "timed_out": timed_out, "budget_breached": breached,
        "aggregate": {"peak_owned_rss_bytes": peak,
                      "peak_owned_rss_gib": round(peak / 1073741824, 4),
                      "budget_bytes": budget_bytes,
                      "budget_gib": round(budget_bytes / 1073741824, 3),
                      "samples": samples, "interval_s": interval_s,
                      "enforced_cap": False,
                      "limitations": (f"sampled at {interval_s}s; transients "
                                      f"shorter than the interval can be "
                                      f"missed. Monitoring, not a cap."),
                      **{k: acc[k] for k in ("discovery", "method",
                                             "shared_pages")}},
        "cleanup": cleanup,
        "owned_absent_confirmed": bool(cleanup.get("owned_absent_confirmed")),
        "ledger": str(out_dir / "ownership_ledger.json"),
    }
    (out_dir / "lifecycle_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str))
    return report
