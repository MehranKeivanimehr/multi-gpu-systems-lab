"""DECLARED TEST STUB of pynvml for CPU-only supervisor integration tests (Candidate 17). Never touches hardware.
State comes from the JSON file named by MGBENCH_TEST_NVML_STATE, re-read on every call:
  {"uuids": {"0": ..}, "pci": {"0": ..}, "power_w": {"5": 20.0}, "unsafe_after_calls": {"5": 40},
   "foreign_flag": "<path>", "foreign_pid": 123, "foreign_gpu": 5}
The supervisor's test-worker hook activates only if this module is loaded (MGBENCH_TEST_STUB)."""
import json
import os
import types

MGBENCH_TEST_STUB = True
NVML_TEMPERATURE_GPU, NVML_CLOCK_SM, nvmlMemory_v2 = 0, 1, 2
_calls = {"power": 0}


class NVMLError(Exception):
    pass


def _state():
    st = json.loads(open(os.environ["MGBENCH_TEST_NVML_STATE"]).read())
    late = st.get("late") or {}                 # Candidate 18: overrides that apply once a flag file appears
    if late.get("flag") and os.path.exists(late["flag"]):
        for k, v in late.items():
            if isinstance(v, dict):
                st[k] = dict(st.get(k) or {}, **v)
    return st


def nvmlInit():
    return None


def nvmlDeviceGetCount():
    return 8


def nvmlDeviceGetHandleByIndex(i):
    return int(i)


def nvmlDeviceGetUUID(h):
    return _state()["uuids"][str(h)]


def nvmlDeviceGetPciInfo(h):
    return types.SimpleNamespace(busId=_state()["pci"][str(h)].encode())


def nvmlDeviceGetName(h):
    return str((_state().get("names") or {}).get(str(h), "NVIDIA RTX A6000")).encode()


def nvmlDeviceGetMemoryInfo(h, version=None):
    return types.SimpleNamespace(used=2**20, total=(_state().get("mem_total_bytes") or {}).get(str(h), 48 * 2**30))


def nvmlDeviceGetUtilizationRates(h):
    return types.SimpleNamespace(gpu=0, memory=0)


def _procs(h):
    st = _state()
    if st.get("foreign_flag") and os.path.exists(st["foreign_flag"]) and int(h) == int(st.get("foreign_gpu", -1)):
        return [types.SimpleNamespace(pid=int(st["foreign_pid"]), usedGpuMemory=2**30)]
    return []


nvmlDeviceGetComputeRunningProcesses = _procs


def nvmlDeviceGetGraphicsRunningProcesses(h):
    return []


def nvmlDeviceGetMPSComputeRunningProcesses(h):
    return []


def nvmlDeviceGetPowerUsage(h):
    st = _state()
    _calls["power"] += 1
    lim = (st.get("unsafe_after_calls") or {}).get(str(h))
    if lim is not None and _calls["power"] > lim:
        return 120000
    return int(float((st.get("power_w") or {}).get(str(h), 20.0)) * 1000)


def nvmlDeviceGetEnforcedPowerLimit(h):
    return int(float((_state().get("power_limit_w") or {}).get(str(h), 300.0)) * 1000)


def nvmlDeviceGetTemperature(h, sensor):
    return 25


def nvmlDeviceGetPerformanceState(h):
    return 8


def nvmlDeviceGetClockInfo(h, c):
    return 210
