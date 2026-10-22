"""
Measure achieved GEMM throughput per dtype on one GPU of each type.

PyTorch 2.4.1 includes emulation in is_bf16_supported() by default. Record both
the emulation-inclusive and native-capability flags for the selected device.
GEMM observations do not establish a universal precision policy for other tasks.

Run with one GPU visible; it benchmarks whatever cuda:0 is.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from common import apply_memory_cap, guard_devices, phys_devices, write_result  # noqa: E402

N = 4096  # 4096^3 GEMM: 2*N^3 FLOPs; a bottleneck claim would require profiling


def bench_dtype(dtype: torch.dtype, iters: int = 30) -> dict[str, object]:
    try:
        a = torch.randn(N, N, device="cuda", dtype=dtype)
        b = torch.randn(N, N, device="cuda", dtype=dtype)
    except Exception as exc:
        return {"supported": False, "error": f"{type(exc).__name__}: {exc}"[:200]}

    try:
        for _ in range(5):  # warmup / autotune
            c = a @ b
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            c = a @ b
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    except Exception as exc:
        del a, b
        torch.cuda.empty_cache()
        return {"supported": False, "error": f"{type(exc).__name__}: {exc}"[:200]}

    flops = 2.0 * N**3 * iters
    tflops = flops / dt / 1e12
    finite = bool(torch.isfinite(c.float()).all().item())
    del a, b, c
    torch.cuda.empty_cache()
    return {
        "supported": True,
        "tflops": round(tflops, 3),
        "ms_per_gemm": round(dt / iters * 1000, 3),
        "finite": finite,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--label", required=True)
    args = p.parse_args()

    guard_devices(phys_devices())
    apply_memory_cap()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    props = torch.cuda.get_device_properties(0)
    out: dict[str, object] = {
        "device": {
            "name": props.name,
            "capability": f"{props.major}.{props.minor}",
            "sm_count": props.multi_processor_count,
            "total_mem_mib": round(props.total_memory / 1024**2, 1),
        },
        "flags": {
            "bf16_including_emulation": bool(torch.cuda.is_bf16_supported(including_emulation=True)),
            "bf16_native_api_flag": bool(torch.cuda.is_bf16_supported(including_emulation=False)),
            "scope": "only this selected cuda:0; not all devices on the node",
        },
        "gemm_n": N,
        "results": {},
    }

    for name, dt in (
        ("fp32", torch.float32),
        ("fp16", torch.float16),
        ("bf16", torch.bfloat16),
    ):
        r = bench_dtype(dt)
        out["results"][name] = r
        status = (
            f"{r['tflops']:>8.2f} TFLOP/s  ({r['ms_per_gemm']:.2f} ms/GEMM)"
            if r.get("supported")
            else f"UNSUPPORTED: {r.get('error', '')[:80]}"
        )
        print(f"  {name:<6} {status}", flush=True)

    res = out["results"]
    if res["fp32"].get("supported") and res["fp16"].get("supported"):
        ratio = res["fp16"]["tflops"] / res["fp32"]["tflops"]
        out["fp16_vs_fp32_speedup"] = round(ratio, 3)
        print(f"  fp16/fp32 speedup: {ratio:.2f}x", flush=True)
    if res["fp32"].get("supported") and res["bf16"].get("supported"):
        ratio = res["bf16"]["tflops"] / res["fp32"]["tflops"]
        out["bf16_vs_fp32_speedup"] = round(ratio, 3)
        print(f"  bf16/fp32 speedup: {ratio:.2f}x", flush=True)

    path = write_result(out, f"01_dtype_{args.label}")
    print(f"  -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
