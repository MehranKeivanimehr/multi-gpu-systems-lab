"""
Shared telemetry, safety guards and result plumbing for the Stage-1 benchmark suite.

Design notes for this cluster (TITAN_NODE, 10 mixed GPUs, shared machine):

* Every benchmark process is launched with an explicit CUDA_VISIBLE_DEVICES and
  CUDA_DEVICE_ORDER=PCI_BUS_ID. Inside the process, torch device i maps to the
  i-th entry of that list. nvidia-smi, however, only speaks *physical* indices,
  so the launcher also exports BENCH_PHYS_DEVICES with the same physical list and
  all telemetry below keys off that.
* Nothing here ever signals a PID it does not own. The "is this GPU busy" guard
  refuses to run; it never tries to make room.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from scientific_metrics import PROTOCOL_VERSION, integrate_power

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_RAW = Path(os.environ.get("BENCH_RESULTS_DIR", str(PROJECT_ROOT / "results" / "validation-v2")))

# --- safety knobs -----------------------------------------------------------
# A foreign process holding more than this much VRAM, or pushing utilisation
# above this, means we yield the GPU rather than compete for it.
FOREIGN_MEM_LIMIT_MIB = 500
FOREIGN_UTIL_LIMIT_PCT = 5
# Cap our own allocator so an OOM lands inside this process instead of
# squeezing the node. Applies per visible device.
MEMORY_FRACTION = 0.85

#: Frozen telemetry sampling interval, in seconds, for EVERY sampled path
#: (training, CNN, HF, vLLM). Previously bench_cnn/infer_hf/infer_vllm
#: hardcoded 0.1 while the protocol specified 0.2, so the energy gap rule
#: (3 x nominal) was being declared against the wrong nominal. One
#: constant now defines it; every record persists the value actually used.
SAMPLE_INTERVAL_S = 0.2


# ---------------------------------------------------------------------------
# nvidia-smi helpers
# ---------------------------------------------------------------------------

def _smi(args: list[str]) -> str:
    out = subprocess.run(
        ["nvidia-smi", *args], capture_output=True, text=True, timeout=30
    )
    if out.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {out.stderr.strip()}")
    return out.stdout


def _as_float(tok: str) -> float | None:
    tok = tok.strip()
    if not tok or tok in ("N/A", "[N/A]", "[Not Supported]", "Not Supported"):
        return None
    try:
        return float(tok)
    except ValueError:
        return None


QUERY_FIELDS = [
    "index",
    "name",
    "compute_cap",
    "memory.total",
    "memory.used",
    "utilization.gpu",
    "power.draw",
    "power.limit",
    "clocks.sm",
    "temperature.gpu",
    "pstate",
]


def query_gpus(indices: list[int] | None = None) -> list[dict[str, Any]]:
    """Snapshot the requested physical GPUs (all of them when indices is None)."""
    args = ["--query-gpu=" + ",".join(QUERY_FIELDS), "--format=csv,noheader,nounits"]
    if indices:
        # Must be two argv entries: nvidia-smi rejects a fused "-i0".
        args += ["-i", ",".join(str(i) for i in indices)]
    rows = []
    for line in _smi(args).strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(QUERY_FIELDS):
            continue
        rec: dict[str, Any] = {}
        for key, tok in zip(QUERY_FIELDS, parts):
            if key in ("name", "compute_cap", "pstate"):
                rec[key] = tok
            else:
                rec[key] = _as_float(tok)
        rec["index"] = int(rec["index"])
        rows.append(rec)
    return rows


def compute_processes() -> list[dict[str, Any]]:
    """All *compute* processes on the box, with owning GPU and VRAM footprint.

    Graphics processes (Xorg) are deliberately excluded: Xorg holds ~4 MiB on
    every GPU here and is not a sign of contention.
    """
    args = [
        "--query-compute-apps=gpu_uuid,pid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    uuid_to_idx = {}
    for line in _smi(
        ["--query-gpu=index,uuid", "--format=csv,noheader,nounits"]
    ).strip().splitlines():
        idx, uuid = [p.strip() for p in line.split(",")]
        uuid_to_idx[uuid] = int(idx)

    procs = []
    text = _smi(args).strip()
    if not text:
        return procs
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        uuid, pid, mem = parts
        procs.append(
            {
                "gpu": uuid_to_idx.get(uuid, -1),
                "pid": int(pid) if pid.isdigit() else -1,
                "used_mib": _as_float(mem) or 0.0,
            }
        )
    return procs


# ---------------------------------------------------------------------------
# Pre-flight guard
# ---------------------------------------------------------------------------

class GpuBusy(RuntimeError):
    """Raised when a target GPU is in use by somebody else."""


def _own_pgid() -> int:
    try:
        return os.getpgid(0)
    except OSError:  # pragma: no cover
        return -1


def _proc_ppid(pid: int) -> int | None:
    """Parent pid from /proc, tolerating a comm field containing ') '."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as fh:
            data = fh.read()
        tail = data[data.rindex(b")") + 1:].split()
        return int(tail[1])
    except (OSError, ValueError, IndexError):
        return None


def _proc_uid(pid: int) -> int | None:
    try:
        return os.stat(f"/proc/{pid}").st_uid
    except OSError:
        return None


def _shares_launcher_ancestry(pid: int, max_depth: int = 16) -> bool:
    """True if pid descends from us, or from the very agent that launched us.

    torchrun puts each rank in its own process group, so a sibling rank is not
    detectable by pgid. Walking up to our own parent is the narrowest test that
    still recognises siblings: an unrelated process of ours -- including one of
    our own ancestors -- does not pass it.
    """
    own_pid, own_ppid = os.getpid(), os.getppid()
    cur, depth = pid, 0
    while cur and cur > 1 and depth < max_depth:
        if cur == own_pid or (own_ppid > 1 and cur == own_ppid):
            return True
        nxt = _proc_ppid(cur)
        if nxt is None or nxt == cur:
            return False
        cur, depth = nxt, depth + 1
    return False


def _is_ours(pid: int) -> bool:
    """True if pid is ours: same process group, our descendant, or a sibling
    rank of the same launcher. Anything else counts as foreign contention."""
    if pid == os.getpid():
        return True
    try:
        same_pgid = os.getpgid(pid) == _own_pgid()
    except (OSError, ProcessLookupError):
        return False  # not ours and already gone, or not inspectable
    if same_pgid:
        return True
    # Only ever widen ownership on hard evidence: identical UID *and* descent
    # from our own launcher. Never on UID alone -- that would silently accept a
    # different job of ours competing for the same device.
    if _proc_uid(pid) != os.getuid():
        return False
    return _shares_launcher_ancestry(pid)


def guard_devices(
    phys: list[int], *, strict: bool = True, settle_retries: int = 6,
    settle_sleep: float = 2.0,
) -> dict[str, Any]:
    """Refuse to start if any target GPU is occupied by *another user*.

    Signal priority matters here. A foreign compute PID and the VRAM it holds
    are authoritative evidence of contention. Instantaneous utilisation is NOT:
    nvidia-smi keeps reporting a decaying utilisation figure for seconds after
    a process exits, so a back-to-back sweep would flag its own previous step
    and abort. Utilisation is therefore only treated as blocking when a foreign
    process is actually resident; otherwise a high reading is just our own
    tail and we wait briefly for it to settle.

    Never kills, never reconfigures -- only declines.
    """
    problems: list[str] = []
    snap: list[dict[str, Any]] = []
    foreign: list[dict[str, Any]] = []

    for attempt in range(settle_retries + 1):
        snap = query_gpus(phys)
        foreign = [
            p
            for p in compute_processes()
            if p["gpu"] in phys and not _is_ours(p["pid"]) and p["used_mib"] > 0
        ]
        foreign_mem: dict[int, float] = {}
        for p in foreign:
            foreign_mem[p["gpu"]] = foreign_mem.get(p["gpu"], 0.0) + p["used_mib"]

        problems = []
        for p in foreign:
            problems.append(
                f"GPU{p['gpu']} holds foreign compute PID {p['pid']} "
                f"({p['used_mib']:.0f} MiB)"
            )
        for row in snap:
            idx = row["index"]
            used = row["memory.used"] or 0.0
            util = row["utilization.gpu"] or 0.0
            # Total VRAM above the limit with no foreign PID attributable to it
            # still blocks: something is resident that we cannot account for.
            if used > FOREIGN_MEM_LIMIT_MIB:
                problems.append(
                    f"GPU{idx} has {used:.0f} MiB in use "
                    f"(limit {FOREIGN_MEM_LIMIT_MIB} MiB)"
                )
            # Utilisation only counts against a GPU that a foreign process holds.
            if foreign_mem.get(idx, 0.0) > 0 and util > FOREIGN_UTIL_LIMIT_PCT:
                problems.append(
                    f"GPU{idx} at {util:.0f}% utilisation under foreign load "
                    f"(limit {FOREIGN_UTIL_LIMIT_PCT}%)"
                )

        if not problems:
            break
        # Only worth waiting if nothing foreign is resident -- i.e. the reading
        # is plausibly our own decaying tail.
        if foreign or attempt == settle_retries:
            break
        time.sleep(settle_sleep)

    if problems and strict:
        raise GpuBusy(
            "Refusing to launch -- target GPUs are not free:\n  "
            + "\n  ".join(problems)
        )
    return {"snapshot": snap, "foreign_procs": foreign, "warnings": problems}


def wait_for_idle(phys: list[int], max_wait: float = 60.0) -> float:
    """Block until our own prior run's utilisation has decayed. Returns waited s.

    Used between sweep steps so each measurement starts from a quiet GPU and
    the power baseline is not contaminated by the previous configuration.
    """
    t0 = time.time()
    while time.time() - t0 < max_wait:
        rows = query_gpus(phys)
        if all((r["utilization.gpu"] or 0) <= FOREIGN_UTIL_LIMIT_PCT for r in rows):
            break
        time.sleep(1.0)
    return time.time() - t0


# ---------------------------------------------------------------------------
# Background power / utilisation sampler
# ---------------------------------------------------------------------------

@dataclass
class SampleStats:
    n_samples: int = 0
    duration_s: float = 0.0
    util_mean_pct: dict[int, float] = field(default_factory=dict)
    util_max_pct: dict[int, float] = field(default_factory=dict)
    power_mean_w: dict[int, float] = field(default_factory=dict)
    power_max_w: dict[int, float] = field(default_factory=dict)
    mem_max_mib: dict[int, float] = field(default_factory=dict)
    sm_clock_mean_mhz: dict[int, float] = field(default_factory=dict)
    temp_max_c: dict[int, float] = field(default_factory=dict)
    energy_total_j: float = 0.0
    energy_per_gpu_j: dict[int, float] = field(default_factory=dict)
    sample_interval_s: float | None = None
    sample_interval_frozen_default_s: float = SAMPLE_INTERVAL_S
    sample_interval_matches_frozen: bool | None = None
    max_permitted_gap_s: float = round(3 * SAMPLE_INTERVAL_S, 9)
    protocol_version: str = PROTOCOL_VERSION
    window_start_s: float | None = None
    window_stop_s: float | None = None
    timebase: str = "time.perf_counter, monotonic, host-local"
    integration: dict = field(default_factory=dict)
    raw_samples: list[dict] = field(default_factory=list)
    idle_raw_samples: list[dict] = field(default_factory=list)
    sampling_errors: list[dict] = field(default_factory=list)
    energy_scope: str = "sampled GPU-board energy; not whole-node energy; sensor accuracy uncalibrated"
    net_energy_status: str = "diagnostic_only; baseline not a calibrated incremental-energy measurement"


class GpuSampler:
    """Polls nvidia-smi on a background thread and integrates power over time.

    Explicit workload markers use the same monotonic clock as sample queries.
    Raw readings bracket the window. Energy is unavailable if coverage fails.
    Boundary interpolation does not correct sensor lag or calibrate accuracy.
    """

    def __init__(self, phys: list[int], interval: float | None = None):
        interval = SAMPLE_INTERVAL_S if interval is None else interval
        if not phys or interval <= 0:
            raise ValueError("Explicit nonempty physical GPU list and positive sample interval required")
        self.phys = phys
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._rows: list[tuple[float, list[dict[str, Any]]]] = []
        self.idle_power_w: dict[int, float] = {}
        self.idle_raw_samples = []
        self.raw_samples = []
        self.errors = []
        self.t_start = self.t_stop = None

    def _sample(self):
        before = time.perf_counter()
        try:
            rows = query_gpus(self.phys)
        except Exception as exc:
            self.errors.append({"time_s": before, "error": repr(exc)})
            rows = []
        after = time.perf_counter()
        timestamp = (before + after) / 2
        self._rows.append((timestamp, rows))
        self.raw_samples.append({"time_s": timestamp, "query_start_s": before,
                                 "query_stop_s": after, "gpus": rows})

    def mark_start(self):
        self.t_start = time.perf_counter()
        return self.t_start

    def mark_stop(self):
        self.t_stop = time.perf_counter()
        return self.t_stop

    def measure_idle(self, seconds: float = 2.0) -> dict[int, float]:
        """Loaded-but-not-executing diagnostic; not a calibrated counterfactual."""
        acc: dict[int, list[float]] = {i: [] for i in self.phys}
        t_end = time.perf_counter() + seconds
        while time.perf_counter() < t_end:
            before = time.perf_counter()
            rows = query_gpus(self.phys)
            self.idle_raw_samples.append({"query_start_s": before,
                                         "query_stop_s": time.perf_counter(), "gpus": rows})
            for row in rows:
                p = row["power.draw"]
                if p is not None:
                    acc[row["index"]].append(p)
            time.sleep(0.2)
        self.idle_power_w = {
            i: (sum(v) / len(v) if v else 0.0) for i, v in acc.items()
        }
        if any(not values for values in acc.values()):
            raise RuntimeError("Missing idle power readings; no zero-filled baseline allowed")
        return self.idle_power_w

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            self._stop.wait(self.interval)

    def __enter__(self) -> "GpuSampler":
        self._rows.clear()
        self.raw_samples.clear()
        self.errors.clear()
        self.t_start = self.t_stop = None
        self._stop.clear()
        self._sample()  # pre-workload bracket, outside measured time
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=35)
            if self._thread.is_alive():
                raise RuntimeError("Telemetry query did not stop; measurement rejected")
        self._sample()  # post-workload bracket, outside measured time

    def stats(self) -> SampleStats:
        st = SampleStats(n_samples=len(self._rows))
        st.sample_interval_s = self.interval
        st.sample_interval_matches_frozen = self.interval == SAMPLE_INTERVAL_S
        if self.t_start is None or self.t_stop is None or self.t_stop <= self.t_start:
            raise RuntimeError("Explicit valid workload boundaries are required")
        st.window_start_s, st.window_stop_s = self.t_start, self.t_stop
        st.duration_s = self.t_stop - self.t_start
        st.raw_samples = list(self.raw_samples)
        st.idle_raw_samples = list(self.idle_raw_samples)
        st.sampling_errors = list(self.errors)

        series: dict[int, dict[str, list[float]]] = {
            i: {"util": [], "power": [], "mem": [], "clk": [], "temp": [], "t": []}
            for i in self.phys
        }
        for ts, rows in self._rows:
            for row in rows:
                i = row["index"]
                if i not in series:
                    continue
                s = series[i]
                if not self.t_start <= ts <= self.t_stop:
                    continue
                s["t"].append(ts)
                for key, col in (
                    ("util", "utilization.gpu"),
                    ("power", "power.draw"),
                    ("mem", "memory.used"),
                    ("clk", "clocks.sm"),
                    ("temp", "temperature.gpu"),
                ):
                    v = row[col]
                    s[key].append(v if v is not None else float("nan"))

        def _clean(xs: list[float]) -> list[float]:
            return [x for x in xs if x == x]  # drops NaN

        for i, s in series.items():
            u, p, m = _clean(s["util"]), _clean(s["power"]), _clean(s["mem"])
            c, tp = _clean(s["clk"]), _clean(s["temp"])
            if u:
                st.util_mean_pct[i] = round(sum(u) / len(u), 2)
                st.util_max_pct[i] = round(max(u), 2)
            if p:
                st.power_mean_w[i] = round(sum(p) / len(p), 2)
                st.power_max_w[i] = round(max(p), 2)
            if m:
                st.mem_max_mib[i] = round(max(m), 1)
            if c:
                st.sm_clock_mean_mhz[i] = round(sum(c) / len(c), 1)
            if tp:
                st.temp_max_c[i] = round(max(tp), 1)

            power = [(ts, next((row["power.draw"] for row in rows if row["index"] == i), None))
                     for ts, rows in self._rows]
            integration = integrate_power(power, self.t_start, self.t_stop)
            st.integration[i] = integration
            if integration["status"] != "ok":
                raise RuntimeError(f"GPU {i}: invalid energy coverage: {integration['status']}")
            st.energy_per_gpu_j[i] = integration["energy_j"]
            st.power_mean_w[i] = integration["energy_j"] / st.duration_s

        st.energy_total_j = round(sum(st.energy_per_gpu_j.values()), 2)
        return st


# ---------------------------------------------------------------------------
# torch-side helpers
# ---------------------------------------------------------------------------

def apply_memory_cap(fraction: float = MEMORY_FRACTION, *, torch_module=None) -> None:
    """Cap the allocator on every visible device WITHOUT losing the current device.

    torch.cuda.set_per_process_memory_fraction(fraction, d) sets the current CUDA
    device to d as a side effect and does not restore it (see the native caching
    allocator's setMemoryFraction). Looping over every visible device therefore
    left the process bound to the last index, silently overwriting the
    torch.cuda.set_device(LOCAL_RANK) performed immediately before this call.
    Every rank then issued collectives from one shared GPU and NCCL aborted with
    "Duplicate GPU detected".

    The cap is unchanged: the same `fraction` still reaches the same full range of
    visible devices. Only the caller's current device is preserved, restored in a
    finally block so a raising cap call cannot leak a changed binding either.

    `torch_module` exists so the restoration contract can be tested with a mock on
    a CPU-only host; production callers leave it None.
    """
    if torch_module is None:
        import torch as torch_module

    if not torch_module.cuda.is_available():
        return
    previous = torch_module.cuda.current_device()
    try:
        for d in range(torch_module.cuda.device_count()):
            torch_module.cuda.set_per_process_memory_fraction(fraction, d)
    finally:
        torch_module.cuda.set_device(previous)


def runtime_device_identity(local_rank: int, torch_module=None) -> dict:
    """Record what the rank is ACTUALLY bound to, from the CUDA runtime itself.

    Echoing CUDA_VISIBLE_DEVICES or querying a physical index proves nothing about
    the device this process will compute on. This reads the live current device and
    its hardware UUID through the runtime. If the UUID cannot be established, the
    caller MUST NOT certify identity: `uuid_established` is False and `uuid` is None.
    """
    if torch_module is None:
        import torch as torch_module
    out = {"local_rank": int(local_rank), "current_device": None,
           "device_name": None, "uuid": None, "uuid_source": None,
           "uuid_established": False, "visible_devices_env": os.environ.get(
               "CUDA_VISIBLE_DEVICES"), "phys_devices_env": os.environ.get(
               "BENCH_PHYS_DEVICES")}
    if not torch_module.cuda.is_available():
        out["uuid_source"] = "cuda_unavailable"
        return out
    out["current_device"] = int(torch_module.cuda.current_device())
    props = torch_module.cuda.get_device_properties(out["current_device"])
    out["device_name"] = props.name
    raw = getattr(props, "uuid", None)
    if raw is not None:
        text = str(raw)
        out["uuid"] = text if text.startswith("GPU-") else f"GPU-{text}"
        out["uuid_source"] = "torch.cuda.get_device_properties().uuid"
        out["uuid_established"] = True
    else:
        out["uuid_source"] = "unavailable_on_this_torch_build"
    return out


def phys_devices() -> list[int]:
    """Physical GPU indices this process was given, for telemetry."""
    raw = os.environ.get("BENCH_PHYS_DEVICES") or os.environ.get(
        "CUDA_VISIBLE_DEVICES", ""
    )
    return [int(x) for x in raw.split(",") if x.strip() != ""]


def torch_env() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nccl": None,
        "arch_list": [],
        "device_count": 0,
        "devices": [],
    }
    try:
        info["arch_list"] = torch.cuda.get_arch_list()
    except Exception:
        pass
    try:
        info["nccl"] = ".".join(str(x) for x in torch.cuda.nccl.version())
    except Exception:
        pass
    if torch.cuda.is_available():
        info["device_count"] = torch.cuda.device_count()
        for d in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(d)
            info["devices"].append(
                {
                    "local_index": d,
                    "name": p.name,
                    "capability": f"{p.major}.{p.minor}",
                    "total_mem_mib": round(p.total_memory / 1024**2, 1),
                    "multi_processor_count": p.multi_processor_count,
                }
            )
    return info


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------

def write_result(record: dict[str, Any], name: str) -> Path:
    RESULTS_RAW.mkdir(parents=True, exist_ok=True)
    if Path(name).name != name:
        raise ValueError("Result name must not contain a directory")
    path = RESULTS_RAW / f"{name}.json"
    if path.exists():
        raise FileExistsError(f"Refusing to replace an existing run: {path}; use a new session directory")
    record.setdefault("meta", {})
    record["meta"].update(
        {
            "host": socket.gethostname(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_device_order": os.environ.get("CUDA_DEVICE_ORDER"),
            "phys_devices": phys_devices(),
            "sample_interval_frozen_default_s": SAMPLE_INTERVAL_S,
            "max_permitted_gap_s": round(3 * SAMPLE_INTERVAL_S, 9),
            "protocol_version": PROTOCOL_VERSION,
            "historical_dataset_overwritten": False,
        }
    )
    with open(path, "x") as fh:
        json.dump(record, fh, indent=2, sort_keys=True, default=str)
    return path


def stats_to_dict(st: SampleStats) -> dict[str, Any]:
    d = asdict(st)
    # JSON object keys must be strings; keep the physical index readable.
    for key in list(d):
        if isinstance(d[key], dict):
            d[key] = {str(k): v for k, v in d[key].items()}
    return d
