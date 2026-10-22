"""Pre-run safety: memory feasibility estimate, the GPU idle gate, host checks, and cleanup of our own processes.

Nothing in here ever signals a process that does not carry this run's MGBENCH_RUN_ID marker.
"""
from __future__ import annotations

import math
import os
import re
import signal
import time
from pathlib import Path

from .gpu_monitor import read_gpu_bounded
from .metrics import GIB, lm_total_params

# Bytes of model state per parameter, per GPU, as a function of world size n (mixed-precision AdamW).
#   ddp  : fp32 param + fp32 grad + 2x fp32 Adam moments                      = 16
#   fsdp : the same 16 bytes, all sharded                                    = 16/n
#   zero1: bf16 param + bf16 grad (replicated) + fp32 master/grad/m/v shards = 4 + 16/n
#   zero2: bf16 param (replicated) + bf16 grad shard + fp32 shards           = 2 + 18/n
#   zero3: everything sharded                                                 = 20/n
STATE_BYTES = {
    "ddp": lambda n: 16.0,
    "fsdp": lambda n: 16.0 / n,
    "zero1": lambda n: 4.0 + 16.0 / n,
    "zero2": lambda n: 2.0 + 18.0 / n,
    "zero3": lambda n: 20.0 / n,
}


def estimate_state_gib(n_params: int, framework: str, n_gpus: int) -> float:
    return n_params * STATE_BYTES[framework](n_gpus) / GIB


def check_feasible(spec: dict, hf_cfg: dict | None, cluster: dict) -> tuple[bool, dict]:
    """Lower-bound check: if the model states alone exceed the cap, the run cannot fit, so it is not launched.
    Activations are not part of the bound; runs near the limit are launched and protected by the per-process cap."""
    if spec["kind"] != "train" or hf_cfg is None:
        return True, {}
    n_params = lm_total_params(hf_cfg)
    state = estimate_state_gib(n_params, spec["framework"], spec["n_gpus"])
    info = {"n_params_est": n_params, "state_gib_per_gpu_est": round(state, 2), "cap_gib": cluster["mem_cap_gib"]}
    return state <= cluster["mem_cap_gib"], info


# ----------------------------------------------------------------------------------------------- idle gate

def _uid_of(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("Uid:"):
                return int(line.split()[1])
    except OSError:
        return None
    return None


# ------------------------------------------------------------------ telemetry plausibility (smoke stage)
# Readings that are missing, NaN/inf, negative or physically implausible are refusals, never "idle".
UUID_RE = re.compile(r"^GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
PLAUSIBLE = {"power_w": (0.0, 600.0), "temp_c": (1.0, 110.0), "util_pct": (0.0, 100.0),
             "mem_util_pct": (0.0, 100.0), "mem_used_mib": (0.0, 65536.0), "mem_total_mib": (1024.0, 65536.0),
             "power_limit_w": (50.0, 600.0)}


def value_problem(name: str, value) -> str | None:
    if value is None:
        return f"{name} missing"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"{name} not numeric ({value!r})"
    if not math.isfinite(value):
        return f"{name} not finite ({value!r})"
    lo, hi = PLAUSIBLE.get(name, (0.0, math.inf))
    if value < lo or value > hi:
        return f"{name} implausible ({value!r} outside [{lo}, {hi}])"
    return None


def reading_problems(r: dict, label: str, keys=("power_w", "temp_c", "util_pct", "mem_used_mib", "mem_total_mib")) -> list[str]:
    out = [f"{label}: {p} -> unknown, do not launch" for k in keys if (p := value_problem(k, r.get(k)))]
    if not out and r["mem_used_mib"] > r["mem_total_mib"]:
        out.append(f"{label}: memory used {r['mem_used_mib']} > total {r['mem_total_mib']} -> do not launch")
    return out


def mps_daemon_problems(proc: Path = Path("/proc")) -> list[str]:
    """An MPS control/server daemon of any user changes GPU sharing semantics: refuse while one is running."""
    found = []
    for d in proc.iterdir():
        if d.name.isdigit():
            try:
                comm = (d / "comm").read_text().strip()
            except OSError:
                continue
            if comm.startswith("nvidia-cuda-mps"):
                found.append(f"{d.name}:{comm}")
    if os.environ.get("CUDA_MPS_PIPE_DIRECTORY") or os.environ.get("CUDA_MPS_LOG_DIRECTORY"):
        found.append("CUDA_MPS_* set in this environment")
    return [f"MPS active ({', '.join(found)}) -> do not launch"] if found else []


def idle_problems(gpus: list[int], idle: dict) -> list[str]:
    """Return a list of reasons why the GPUs are not idle (empty list = idle in every respect).
    Fail closed: an unreadable GPU, an implausible reading or an unknown process list is a reason not to launch,
    never "idle". Each NVML read is time-bounded."""
    problems = []
    for g in gpus:
        try:
            r = read_gpu_bounded(g, idle.get("query_timeout_s", 5.0))
        except Exception as exc:  # noqa: BLE001 - any NVML failure or timeout means the state is unknown
            problems.append(f"gpu{g}: NVML state unreadable ({exc!r}) -> unknown, do not launch")
            continue
        bad = reading_problems(r, f"gpu{g}")
        if bad:
            problems += bad
            continue
        if r.get("processes") is None:
            problems.append(f"gpu{g}: process list unavailable ({r.get('process_query_error')}) -> unknown, "
                            "do not launch")
        elif r["processes"]:
            owners = {p["pid"]: _uid_of(p["pid"]) for p in r["processes"]}
            problems.append(f"gpu{g}: processes {owners} (pid: uid)")
        if r["mem_used_mib"] > idle["max_mem_mib"]:
            problems.append(f"gpu{g}: mem {r['mem_used_mib']:.0f} MiB")
        if r["util_pct"] > idle["max_util_pct"]:
            problems.append(f"gpu{g}: util {r['util_pct']}%")
        if r["power_w"] > idle["max_power_w"]:
            problems.append(f"gpu{g}: power {r['power_w']:.0f} W")
        if r["temp_c"] > idle["max_temp_c"]:
            problems.append(f"gpu{g}: temp {r['temp_c']} C")
    return problems


def host_problems(idle: dict) -> list[str]:
    problems = []
    avail_kib = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            avail_kib = int(line.split()[1])
    if avail_kib / 2**20 < idle["min_free_ram_gib"]:
        problems.append(f"host: only {avail_kib / 2**20:.0f} GiB RAM available")
    return problems


# ------------------------------------------------------------------------------------------- node guard
# Added 2026-10-15 after the host stopped at 2026-10-14 22:14 during a 4-GPU run while another user loaded GPUs 0-3
# (cause unknown; see validation evidence INCIDENT_20261014_2214_host_failure.md). Conservative, empirical limits:
# the node ran stably for > 1 h at up to ~1.7 kW total GPU draw and stopped during a run at roughly 1.8-2.3 kW.

def node_gpu_power(timeout_s: float = 5.0) -> dict:
    """Current board power (W) of every GPU on the node, any user (read-only, time-bounded NVML queries).
    Raises if any reading is missing, non-finite, negative or implausible."""
    import pynvml

    from .gpu_monitor import _init, bounded

    def read():
        _init()
        return {i: pynvml.nvmlDeviceGetPowerUsage(pynvml.nvmlDeviceGetHandleByIndex(i)) / 1000.0
                for i in range(pynvml.nvmlDeviceGetCount())}
    power = bounded(read, timeout_s)
    bad = [f"gpu{i}: {p}" for i, w in power.items() if (p := value_problem("power_w", w))]
    if bad:
        raise ValueError(f"invalid node power readings: {'; '.join(bad)}")
    return power


def node_telemetry_problems(snapshot: dict, expected_count: int, pins: dict | None = None) -> list[str]:
    """Whole-node read-only telemetry: exactly `expected_count` GPUs, every one readable and plausible; pinned
    indices must report their approved UUID."""
    problems = []
    if snapshot.get("count") != expected_count or len(snapshot.get("gpus", {})) != expected_count:
        problems.append(f"node: NVML lists {snapshot.get('count')} GPUs, expected {expected_count} -> do not launch")
    for i, g in sorted(snapshot.get("gpus", {}).items()):
        if "error" in g:
            problems.append(f"gpu{i}: unreadable ({g['error']}) -> unknown, do not launch")
            continue
        problems += reading_problems(g, f"gpu{i}", keys=("power_w", "power_limit_w", "temp_c", "util_pct",
                                                         "mem_used_mib", "mem_total_mib"))
        if not UUID_RE.match(str(g.get("uuid", ""))):
            problems.append(f"gpu{i}: malformed UUID {g.get('uuid')!r} -> do not launch")
    for idx, want in (pins or {}).items():
        got = snapshot.get("gpus", {}).get(int(idx), {}).get("uuid")
        if got != want:
            problems.append(f"gpu{idx}: UUID {got!r} is not the approved {want!r} -> do not launch")
    return problems


def node_guard_problems(gpus: list[int], idle: dict) -> list[str]:
    """Refuse to start unless (other GPUs' measured draw + planned_w_per_gpu x our GPUs) <= start_budget_w and the
    host 1-minute load is below max_load1."""
    guard = idle.get("node_guard")
    if not guard:  # fail closed (recovery review 20261015r): a missing guard is not permission to launch
        return ["node: node_guard not configured -> do not launch"]
    problems = []
    try:
        power = node_gpu_power()
    except Exception as exc:  # noqa: BLE001
        return [f"node: GPU power unreadable ({exc!r}) -> unknown, do not launch"]
    expected = guard.get("node_gpu_count")
    if expected is not None and len(power) != expected:
        # e.g. a GPU that fell off the bus after a fault would make the node total look lower than it is
        problems.append(f"node: NVML lists {len(power)} GPUs, expected {expected} -> do not launch")
    others = sum(w for i, w in power.items() if i not in gpus)
    planned = others + guard["planned_w_per_gpu"] * len(gpus)
    if planned > guard["start_budget_w"]:
        problems.append(f"node: other GPUs draw {others:.0f} W + {len(gpus)} x {guard['planned_w_per_gpu']} W planned"
                        f" = {planned:.0f} W > start budget {guard['start_budget_w']} W")
    load1 = os.getloadavg()[0]
    if load1 > guard["max_load1"]:
        problems.append(f"node: 1-min load {load1:.0f} > {guard['max_load1']}")
    return problems


class PowerWatch:
    """In-run watchdog: trips after `abort_after_s` consecutive 1-second samples of total node GPU power above
    `abort_total_w`."""

    def __init__(self, guard: dict):
        self.limit = guard["abort_total_w"]
        self.needed = int(guard["abort_after_s"])
        self.over = 0
        self.peak = 0.0
        self.invalid = None

    def update(self, total_w: float) -> bool:
        """True = stop. A missing, non-finite or negative total trips immediately (fail closed)."""
        if value_problem("power_w", total_w / 8.0 if isinstance(total_w, (int, float)) else total_w):
            self.invalid = total_w
            return True
        self.peak = max(self.peak, total_w)
        self.over = self.over + 1 if total_w > self.limit else 0
        return self.over >= self.needed


# ---------------------------------------------------------------- device identity and launch hold
# Added in recovery review 20261015r (after the 2026-10-14 host failure and reboots). Both fail closed.

def device_identity_problems(gpus: list[int], cluster: dict) -> list[str]:
    """Every physical index must still be the GPU the owner approved (UUID pin `gpu_uuids` in
    configs/cluster.yaml). A missing pin, an unreadable UUID or a mismatch blocks the launch."""
    from .gpu_monitor import gpu_uuid

    pins = {int(k): v for k, v in (cluster.get("gpu_uuids") or {}).items()}
    problems = []
    for g in gpus:
        if g not in pins:
            problems.append(f"gpu{g}: no approved UUID in configs/cluster.yaml gpu_uuids -> do not launch")
            continue
        try:
            actual = gpu_uuid(g)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"gpu{g}: UUID unreadable ({exc!r}) -> unknown, do not launch")
            continue
        if actual != pins[g]:
            problems.append(f"gpu{g}: UUID {actual} is not the approved {pins[g]} -> do not launch")
    return problems


def launch_hold_problems(cluster: dict) -> list[str]:
    """Owner-controlled hold on all GPU launches (`launch_hold` in configs/cluster.yaml). A missing hold entry or a
    missing `active` flag counts as an active hold; only an explicit `active: false` lets launches proceed."""
    hold = cluster.get("launch_hold")
    if not isinstance(hold, dict):
        return ["launch_hold missing from configs/cluster.yaml -> treated as active, do not launch"]
    if hold.get("active", True) is not False:
        return [f"launch hold active (since {hold.get('since')}): {hold.get('reason')}"]
    return []


def wait_for_idle(gpus: list[int], idle: dict, log=print) -> tuple[bool, float, list[str]]:
    """Block until every GPU in `gpus` is idle (no processes of any user, ~0 MiB, 0 % util, idle power,
    cooled down) for `consecutive_polls` polls in a row. Returns (ok, waited_s, last_problems)."""
    t0 = time.time()
    streak = 0
    last: list[str] = []
    last_logged = 0.0
    while True:
        problems = idle_problems(gpus, idle) + host_problems(idle) + node_guard_problems(gpus, idle)
        if problems:
            streak = 0
            last = problems
            if time.time() - last_logged > 60:
                log(f"[idle-gate] waiting for GPUs {gpus}: {'; '.join(problems)}")
                last_logged = time.time()
        else:
            streak += 1
            if streak >= idle["consecutive_polls"]:
                return True, time.time() - t0, []
        if time.time() - t0 > idle["max_wait_s"]:
            return False, time.time() - t0, last
        time.sleep(idle["poll_s"])


# ------------------------------------------------------------------------------------- process cleanup

def our_marked_pids(run_id: str) -> list[int]:
    """PIDs owned by us whose environment carries MGBENCH_RUN_ID=<run_id> (i.e. spawned by this run)."""
    marker = f"MGBENCH_RUN_ID={run_id}".encode()
    me = os.getuid()
    pids = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        pid = int(d.name)
        if pid == os.getpid() or _uid_of(pid) != me:
            continue
        try:
            env = (d / "environ").read_bytes()
        except OSError:
            continue
        if marker in env.split(b"\0"):
            pids.append(pid)
    return pids


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def terminate_process_group(pgid: int, grace_s: float = 30, log=print) -> bool:
    """SIGTERM, then SIGKILL, whatever is left in the run's own process group (the runner starts every run in a
    new session, so this group contains only that run's processes - including children that renamed themselves,
    e.g. vLLM's `VLLM::Worker_TP*`, whose environment marker is no longer readable after setproctitle)."""
    if not _group_alive(pgid):
        return False
    log(f"[cleanup] processes left in run process group {pgid} -> SIGTERM")
    os.killpg(pgid, signal.SIGTERM)
    deadline = time.time() + grace_s
    while time.time() < deadline and _group_alive(pgid):
        time.sleep(1)
    if _group_alive(pgid):
        log(f"[cleanup] process group {pgid} still alive -> SIGKILL")
        os.killpg(pgid, signal.SIGKILL)
        time.sleep(2)
    return True


def terminate_run_processes(run_id: str, grace_s: float = 30, log=print) -> int:
    """SIGTERM, then SIGKILL, every leftover process of this run. Returns how many were found."""
    pids = our_marked_pids(run_id)
    if not pids:
        return 0
    log(f"[cleanup] {len(pids)} leftover process(es) of {run_id}: {pids} -> SIGTERM")
    for p in pids:
        try:
            os.kill(p, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + grace_s
    while time.time() < deadline and our_marked_pids(run_id):
        time.sleep(1)
    rest = our_marked_pids(run_id)
    for p in rest:
        log(f"[cleanup] SIGKILL {p}")
        try:
            os.kill(p, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return len(pids)
