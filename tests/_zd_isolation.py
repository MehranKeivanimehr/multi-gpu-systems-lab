"""Device-denying isolation for tests that execute the GPU-capable worker.

INCIDENT_001: an acceptance test executed the real model worker on this
unisolated host and initialised CUDA on a shared GPU. Synthetic UUIDs, prefix
assertions, environment labels and monkeypatches are NOT device isolation —
they are input hygiene. The only thing that actually prevents device access
here is the absence of the device nodes.

This module provides that, using `bubblewrap`, which is already installed on
the host (no install, no sudo, no host configuration change). Inside the
sandbox `--dev /dev` supplies a minimal /dev without any `/dev/nvidia*` node,
so the driver cannot be opened at all.

The guard has an implemented, tested effect: `probe()` runs a capability query
INSIDE the sandbox and reports what it found. It never attempts an allocation,
and it never probes by allocating on the host.

If bubblewrap is unavailable or the probe fails, `available()` is False and the
dependent tests are skipped with an honest reason rather than silently running
unprotected.
"""
from __future__ import annotations

import functools
import os
import tempfile
import json
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent / "studies" / "titan-1080"
SANDBOX_TMP = pathlib.Path(os.environ.get("ZD_SANDBOX_TMP") or
                          (pathlib.Path(tempfile.gettempdir()) / "zd_sandbox"))


def bwrap_path() -> str | None:
    return shutil.which("bwrap")


def wrap(argv: list[str], *, extra_env: dict | None = None,
         writable: list | None = None) -> list[str]:
    """Wrap a command so it runs with the nvidia device nodes absent.

    `/` is bound read-only, so any directory the command must write to has to
    be listed in `writable`.
    """
    bw = bwrap_path()
    if not bw:
        raise RuntimeError("bubblewrap is not available")
    SANDBOX_TMP.mkdir(parents=True, exist_ok=True)
    cmd = [bw, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
           "--tmpfs", "/tmp",
           "--ro-bind", str(ROOT), str(ROOT),
           "--bind", str(SANDBOX_TMP), str(SANDBOX_TMP),
           "--setenv", "XDG_CACHE_HOME", str(SANDBOX_TMP),
           "--setenv", "HOME", str(SANDBOX_TMP),
           "--setenv", "TMPDIR", str(SANDBOX_TMP),
           "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
           "--setenv", "PYTHONUNBUFFERED", "1",
           "--setenv", "PYTHONUNBUFFERED", "1",
           "--die-with-parent"]
    for d in (writable or []):
        p = pathlib.Path(d)
        p.mkdir(parents=True, exist_ok=True)
        cmd += ["--bind", str(p), str(p)]
    for d in (writable or []):
        q = pathlib.Path(d)
        q.mkdir(parents=True, exist_ok=True)
        cmd += ["--bind", str(q), str(q)]
    for k, v in (extra_env or {}).items():
        cmd += ["--setenv", str(k), str(v)]
    return cmd + list(argv)


@functools.lru_cache(maxsize=1)
def probe() -> dict:
    """Capability query inside the sandbox. No allocation, ever."""
    if not bwrap_path():
        return {"available": False, "reason": "bubblewrap is not installed"}
    script = (
        "import glob, json, sys\n"
        "d = glob.glob('/dev/nvidia*')\n"
        "out = {'dev_nodes': d}\n"
        "try:\n"
        "    import torch\n"
        "    out['torch'] = torch.__version__\n"
        "    out['cuda_is_available'] = bool(torch.cuda.is_available())\n"
        "    out['device_count'] = int(torch.cuda.device_count())\n"
        "except Exception as exc:\n"
        "    out['torch_error'] = f'{type(exc).__name__}: {exc}'\n"
        "print(json.dumps(out))\n")
    try:
        r = subprocess.run(wrap([sys.executable, "-I", "-c", script]),
                           capture_output=True, text=True, timeout=300)
    except Exception as exc:
        return {"available": False, "reason": f"sandbox failed: {exc}"}
    if r.returncode != 0:
        return {"available": False,
                "reason": f"sandbox exited {r.returncode}: {r.stderr[-200:]}"}
    try:
        info = json.loads(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"available": False, "reason": "sandbox probe returned no JSON"}
    denied = (info.get("dev_nodes") == []
              and info.get("cuda_is_available") is False
              and info.get("device_count") == 0)
    return {"available": bool(denied), "probe": info,
            "mechanism": "bubblewrap --dev /dev (no nvidia nodes present)",
            "allocation_attempted": False,
            "reason": (None if denied else
                       "the sandbox did not deny device access")}


def available() -> bool:
    return bool(probe().get("available"))


def skip_reason() -> str:
    p = probe()
    return ("DEFERRED: executing the GPU-capable worker needs device-denying "
            "isolation, which is unavailable here (" + str(p.get("reason")) +
            "). Per INCIDENT_001 this test does not run unprotected; it is "
            "deferred to an approved GPU allocation.")
