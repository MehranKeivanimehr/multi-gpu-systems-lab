"""Result files and environment capture (everything needed to reproduce or audit a run)."""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import time
from importlib import metadata
from pathlib import Path

ENV_KEYS = ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "OMP_NUM_THREADS", "NCCL_DEBUG", "NCCL_ALGO",
            "NCCL_PROTO", "NCCL_P2P_DISABLE", "TORCH_NCCL_ASYNC_ERROR_HANDLING", "HF_HUB_OFFLINE")


def write_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def executed_source_record(package_dir: str | Path | None = None) -> dict:
    """Which mgbench sources this process actually imported (path + SHA-256 of every package .py file).

    Added on the A6000 node (scientific-v2 validation, session 20261014a): the shared environment also has an
    older editable install of mgbench, so each result must prove it executed the new checkout's bytes."""
    import hashlib

    pkg = Path(package_dir) if package_dir else Path(__file__).resolve().parent
    files = sorted(pkg.rglob("*.py"))
    return {"mgbench_package_dir": str(pkg),
            "sha256": {f.relative_to(pkg.parent).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}


def read_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def _ver(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def software_versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("torch", "torchvision", "transformers", "deepspeed", "vllm", "nvidia-ml-py"):
        out[pkg] = _ver(pkg)
    try:
        import torch
        out["torch_cuda"] = torch.version.cuda
        out["nccl"] = ".".join(map(str, torch.cuda.nccl.version()))
        out["cudnn"] = torch.backends.cudnn.version()
    except Exception:  # noqa: BLE001 - best effort
        pass
    return out


def topology_matrix() -> str:
    try:
        return subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=30).stdout
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e}"


def link_types(gpus: list[int], topo: str) -> dict:
    """Pairwise link type between the GPUs of a run, parsed from `nvidia-smi topo -m` (NV4, NODE, SYS, ...)."""
    rows = {}
    for line in topo.splitlines():
        parts = line.replace("\x1b[4m", "").replace("\x1b[0m", "").split()
        if parts and parts[0].startswith("GPU") and parts[0][3:].isdigit():
            rows[int(parts[0][3:])] = parts[1:]
    out = {}
    for i in gpus:
        for j in gpus:
            if i < j and i in rows and j < len(rows[i]):
                out[f"{i}-{j}"] = rows[i][j]
    return out


def environment(gpus: list[int]) -> dict:
    from .gpu_monitor import driver_info, read_gpu

    topo = topology_matrix()
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hostname": socket.gethostname(),
        "kernel": platform.release(),
        "software": software_versions(),
        "driver": driver_info(),
        "gpus": [{k: v for k, v in read_gpu(g).items() if k in ("index", "name", "bus_id", "mem_total_mib")}
                 for g in gpus],
        "links": link_types(gpus, topo),
        "env": {k: os.environ.get(k) for k in ENV_KEYS},
    }
