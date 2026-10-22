"""NVML helpers: one-shot GPU readings and a background sampler for utilization, memory, power and clocks.

GPU indices here are physical NVML indices (same order as nvidia-smi). Runs set CUDA_DEVICE_ORDER=PCI_BUS_ID,
so CUDA_VISIBLE_DEVICES uses the same numbering.
"""
from __future__ import annotations

import statistics
import threading
import time

import pynvml

_initialized = False


def _init():
    global _initialized
    if not _initialized:
        pynvml.nvmlInit()
        _initialized = True


def _s(x):
    return x.decode() if isinstance(x, bytes) else x


def mem_info(h):
    """v2 memory info: `used` excludes the ~600 MiB the driver reserves (matches nvidia-smi); v1 includes it."""
    try:
        return pynvml.nvmlDeviceGetMemoryInfo(h, version=pynvml.nvmlMemory_v2)
    except (pynvml.NVMLError, TypeError, AttributeError):
        return pynvml.nvmlDeviceGetMemoryInfo(h)


def gpu_processes(h) -> tuple[list[dict] | None, str | None]:
    """Compute and graphics processes on one GPU (any user).

    Fail closed: if either NVML query fails or is unavailable, the process list is UNKNOWN (None) and the reason is
    returned, so callers cannot mistake a failed query for an empty, idle GPU. (Recovery review 20261015r: the
    previous version turned an NVMLError into [] - i.e. "no processes".)"""
    procs = []
    for kind, query in (("compute", "nvmlDeviceGetComputeRunningProcesses"),
                        ("graphics", "nvmlDeviceGetGraphicsRunningProcesses"),
                        ("mps", "nvmlDeviceGetMPSComputeRunningProcesses")):
        try:
            listed = getattr(pynvml, query)(h)
            procs += [{"pid": p.pid, "used_mib": (p.usedGpuMemory or 0) / 2**20, "type": kind} for p in listed]
        except Exception as exc:  # NVMLError, missing binding, driver/library mismatch: all mean "unknown"
            return None, f"{query} failed: {exc!r}"
    return procs, None


def read_gpu(index: int) -> dict:
    """Current state of one GPU, including every compute and graphics process on it (any user).
    `processes` is None (unknown) when the process query failed; see gpu_processes()."""
    _init()
    h = pynvml.nvmlDeviceGetHandleByIndex(index)
    mem = mem_info(h)
    util = pynvml.nvmlDeviceGetUtilizationRates(h)
    procs, proc_error = gpu_processes(h)
    return {
        "index": index,
        "name": _s(pynvml.nvmlDeviceGetName(h)),
        "bus_id": _s(pynvml.nvmlDeviceGetPciInfo(h).busId),
        "mem_used_mib": mem.used / 2**20,
        "mem_total_mib": mem.total / 2**20,
        "util_pct": util.gpu,
        "mem_util_pct": util.memory,
        "power_w": pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0,
        "temp_c": pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
        "pstate": pynvml.nvmlDeviceGetPerformanceState(h),
        "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
        "processes": procs,
        "process_query_error": proc_error,
    }


# ---------------------------------------------------------------- bounded, validated node telemetry
# Added for the A6000 GPU-4 smoke stage (session 20261016s). Every query is time-bounded; a slow or failing query is
# reported as an error string, never as a plausible-looking value.

class TelemetryTimeout(RuntimeError):
    pass


def bounded(fn, timeout_s: float, *args):
    """Run fn(*args) on a daemon thread and wait at most timeout_s. A hung driver call is abandoned (the daemon
    thread may stay blocked) and reported as TelemetryTimeout - the caller must then refuse."""
    box = {}

    def run():
        try:
            box["value"] = fn(*args)
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc
    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        raise TelemetryTimeout(f"{getattr(fn, '__name__', fn)} exceeded {timeout_s} s")
    if "error" in box:
        raise box["error"]
    return box["value"]


def _node_snapshot() -> dict:
    _init()
    count = pynvml.nvmlDeviceGetCount()
    gpus = {}
    for i in range(count):
        entry = {"index": i}
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = mem_info(h)
            entry.update(uuid=_s(pynvml.nvmlDeviceGetUUID(h)),
                         power_w=pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0,
                         power_limit_w=pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0,
                         temp_c=pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
                         util_pct=pynvml.nvmlDeviceGetUtilizationRates(h).gpu,
                         mem_used_mib=mem.used / 2**20, mem_total_mib=mem.total / 2**20)
        except Exception as exc:  # noqa: BLE001
            entry["error"] = repr(exc)
        gpus[i] = entry
    return {"count": count, "gpus": gpus}


def node_snapshot(timeout_s: float = 5.0) -> dict:
    """Read-only state of every GPU on the node (any user): uuid, power, enforced limit, temperature, utilization,
    memory. Raises TelemetryTimeout or the NVML error; per-GPU failures are recorded in that GPU's `error`."""
    return bounded(_node_snapshot, timeout_s)


def read_gpu_bounded(index: int, timeout_s: float = 5.0) -> dict:
    return bounded(read_gpu, timeout_s, index)


def gpu_uuid(index: int) -> str:
    """NVML UUID of the GPU at physical index `index` (read-only)."""
    _init()
    return _s(pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(index)))


def driver_info() -> dict:
    _init()
    return {"driver_version": _s(pynvml.nvmlSystemGetDriverVersion()),
            "nvml_cuda_driver_version": pynvml.nvmlSystemGetCudaDriverVersion()}


class GpuSampler:
    """Samples the given GPUs every `interval_s` on a daemon thread. Call mark_start/mark_stop around
    the measured window; summary() reports statistics over that window only."""

    def __init__(self, gpus: list[int], interval_s: float = 0.2):
        _init()
        self.gpus = list(gpus)
        self.interval_s = interval_s
        self.handles = {g: pynvml.nvmlDeviceGetHandleByIndex(g) for g in self.gpus}
        self.samples: list[tuple[float, int, float, int, float, int, float]] = []
        self.raw_samples = []
        self.errors = []
        self.t_start = None
        self.t_stop = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            for g, h in self.handles.items():
                before = time.perf_counter()
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
                    mem = mem_info(h).used / 2**30
                    power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                    temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                    clk = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
                    pstate = pynvml.nvmlDeviceGetPerformanceState(h)
                    after = time.perf_counter()
                    t = (before + after) / 2
                    self.samples.append((t, g, util, mem, power, temp, clk))
                    self.raw_samples.append({"time_s": t, "query_start_s": before,
                        "query_stop_s": after, "gpu": g, "util_pct": util,
                        "mem_used_gib": mem, "power_w": power, "temp_c": temp,
                        "sm_clock_mhz": clk, "pstate": pstate})
                except pynvml.NVMLError as exc:
                    self.errors.append({"time_s": before, "gpu": g, "error": repr(exc)})
            self._stop.wait(self.interval_s)

    def start(self):
        self._thread.start()
        return self

    def mark_start(self):
        self.t_start = time.perf_counter()

    def mark_stop(self):
        self.t_stop = time.perf_counter()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def summary(self) -> dict:
        if self.t_start is None or self.t_stop is None or self.t_stop <= self.t_start:
            raise ValueError("Explicit monotonic measurement boundaries required")
        t0, t1 = self.t_start, self.t_stop
        out = {}
        for g in self.gpus:
            win = [s for s in self.samples if s[1] == g and t0 <= s[0] <= t1]
            allg = [s for s in self.samples if s[1] == g]
            if not win:
                continue
            out[str(g)] = {
                "n_samples": len(win),
                "util_mean_pct": statistics.fmean(s[2] for s in win),
                "util_median_pct": statistics.median(s[2] for s in win),
                "mem_used_peak_gib": max(s[3] for s in allg),  # whole run incl. setup
                "mem_used_peak_timed_gib": max(s[3] for s in win),
                "window_start_s": t0,
                "window_stop_s": t1,
                "timebase": "time.perf_counter, monotonic, host-local",
                "memory_peak_scope": "mem_used_peak_gib includes setup; timed peak reported separately",
                "power_mean_w": statistics.fmean(s[4] for s in win),
                "power_max_w": max(s[4] for s in win),
                "temp_max_c": max(s[5] for s in win),
                "sm_clock_mean_mhz": statistics.fmean(s[6] for s in win),
            }
        return out

    def raw_record(self) -> dict:
        return {"protocol_version": "scientific-v2", "timebase": "time.perf_counter",
                "window_start_s": self.t_start, "window_stop_s": self.t_stop,
                "requested_interval_s": self.interval_s,
                "samples": self.raw_samples, "errors": self.errors,
                "energy_claim": "not calculated here; sensor behavior and integration require separate validation"}


def aggregate_gpu_summary(per_gpu: dict) -> dict:
    """Collapse per-GPU summaries into run-level numbers."""
    if not per_gpu:
        return {}
    vals = list(per_gpu.values())
    return {
        "gpu_util_mean_pct": statistics.fmean(v["util_mean_pct"] for v in vals),
        "gpu_util_min_pct": min(v["util_mean_pct"] for v in vals),
        "nvml_mem_peak_gib": max(v["mem_used_peak_gib"] for v in vals),
        "power_mean_w_per_gpu": statistics.fmean(v["power_mean_w"] for v in vals),
        "temp_max_c": max(v["temp_max_c"] for v in vals),
        "sm_clock_mean_mhz": statistics.fmean(v["sm_clock_mean_mhz"] for v in vals),
    }
