"""
ResNet-50 forward-only batch sweep, scientific-v2.

The supplied A6000 corpus measures training, not this inference task. Missing
external baselines are not accepted comparisons. The approximate 4.1 GMAC/image
is 8.2 GFLOP/image with FMA=2. Mixed precision has no FP32-peak percentage.
Power samples bracket the exact measured interval, but sensor accuracy still
needs target-host validation. Idle-subtracted values are diagnostic only.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

from common import (
    SAMPLE_INTERVAL_S,  # noqa: E402
    GpuSampler,
    apply_memory_cap,
    guard_devices,
    phys_devices,
    stats_to_dict,
    torch_env,
    wait_for_idle,
    write_result,
)

# Approximate convolution/linear work: 4.1 GMAC, FMA = two operations.
# Excludes activation, normalization and pooling operations.
GFLOP_PER_IMAGE = 8.2

# Vendor theoretical FP32 peak (cores x 2 x boost clock), for compute-utilisation
# percentages on the same basis as the A6000 study's 38.7 TFLOP/s.
THEORETICAL_FP32_TFLOPS = {
    "NVIDIA TITAN RTX": 16.31,       # 4608 sp x 2 x 1770 MHz
    "NVIDIA GeForce GTX 1080 Ti": 11.34,  # 3584 sp x 2 x 1582 MHz
    "NVIDIA TITAN Xp": 12.15,        # 3840 sp x 2 x 1582 MHz
    "NVIDIA RTX A6000": 38.71,       # reference only, not present on this host
}


def run_batch(model, bs: int, target_secs: float, device, amp: bool) -> dict:
    x = torch.randn(bs, 3, 224, 224, device=device)
    ctx = torch.amp.autocast("cuda", dtype=torch.float16) if amp else None

    def fwd():
        with torch.no_grad():
            if ctx is not None:
                with ctx:
                    return model(x)
            return model(x)

    # warmup: cudnn.benchmark algorithm selection happens here
    for _ in range(10):
        fwd()
    torch.cuda.synchronize(device)

    # calibrate iteration count to hit ~target_secs
    t0 = time.perf_counter()
    for _ in range(5):
        fwd()
    torch.cuda.synchronize(device)
    per_iter = (time.perf_counter() - t0) / 5
    iters = max(20, min(1_000_000, math.ceil(target_secs / max(per_iter, 1e-6))))

    torch.cuda.reset_peak_memory_stats(device)
    sampler = GpuSampler(phys_devices(), interval=SAMPLE_INTERVAL_S)
    idle = sampler.measure_idle(1.5)

    with sampler:
        torch.cuda.synchronize(device)
        t_start = sampler.mark_start()
        for _ in range(iters):
            fwd()
        torch.cuda.synchronize(device)
        elapsed = sampler.mark_stop() - t_start
    st = sampler.stats()

    n_images = iters * bs
    throughput = n_images / elapsed
    idle_w = sum(idle.values())
    energy_gross = st.energy_total_j
    # Scale the idle baseline to the measured window, then subtract.
    energy_idle = idle_w * st.duration_s if st.duration_s else 0.0
    energy_net = energy_gross - energy_idle  # diagnostic; retain negative values

    achieved_tflops = n_images * GFLOP_PER_IMAGE / elapsed / 1000.0
    memory = {
        "memory_allocated_gb": torch.cuda.memory_allocated(device) / 1024**3,
        "memory_reserved_gb": torch.cuda.memory_reserved(device) / 1024**3,
        "max_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
        "max_memory_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
        "memory_scope": "current at workload stop before input cleanup; allocator peaks reset before timing",
    }
    del x
    torch.cuda.empty_cache()

    return {
        "batch_size": bs,
        "iters": iters,
        "n_images": n_images,
        "elapsed_s": round(elapsed, 4),
        "runtime": round(elapsed / iters, 6),          # seconds per batch
        "throughput": round(throughput, 2),            # images/s
        "energy_per_image": round(energy_gross / n_images, 6),      # J, gross
        "energy_per_image_net": round(energy_net / n_images, 6),    # J, idle-subtracted
        "energy_total_j": round(energy_gross, 2),
        "energy_idle_j": round(energy_idle, 2),
        "idle_power_w": round(idle_w, 2),
        "power_mean_w": st.power_mean_w,
        "power_max_w": st.power_max_w,
        "util_mean_pct": st.util_mean_pct,
        "achieved_tflops": round(achieved_tflops, 3),
        **memory,
        "telemetry": stats_to_dict(st),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--group", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--batches", default="1,2,4,8,16,32,64")
    p.add_argument("--target-secs", type=float, default=30.0)
    p.add_argument("--precision", default="fp32", choices=["fp32", "amp_fp16"])
    p.add_argument("--memory-fraction", type=float, default=0.85)
    args = p.parse_args()
    if not 0 < args.target_secs <= 300:
        p.error("target-secs must be in (0, 300]; use an external run timeout")

    guard_devices(phys_devices())
    wait_for_idle(phys_devices())
    apply_memory_cap(args.memory_fraction)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(1234)

    device = torch.device("cuda", 0)
    props = torch.cuda.get_device_properties(0)
    peak = THEORETICAL_FP32_TFLOPS.get(props.name)

    from torchvision.models import resnet50

    # weights=None: this is a throughput/energy benchmark, and random vs trained
    # weights execute the identical graph. Keeps the suite offline-reproducible.
    model = resnet50(weights=None).to(device).eval()

    rows = []
    for bs in [int(b) for b in args.batches.split(",")]:
        try:
            r = run_batch(model, bs, args.target_secs, device, args.precision == "amp_fp16")
        except torch.cuda.OutOfMemoryError:
            print(f"  bs={bs:<4} OOM (recorded, stopping sweep)", flush=True)
            rows.append({"batch_size": bs, "status": "oom"})
            torch.cuda.empty_cache()
            break
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(f"  bs={bs:<4} OOM (recorded, stopping sweep)", flush=True)
                rows.append({"batch_size": bs, "status": "oom"})
                torch.cuda.empty_cache()
                break
            raise
        r["status"] = "ok"
        if peak and args.precision == "fp32":
            r["compute_util_pct"] = round(100.0 * r["achieved_tflops"] / peak, 2)
        else:
            r["compute_util_pct"] = None
        rows.append(r)
        print(
            f"  bs={bs:<4} {r['throughput']:>9.1f} img/s  "
            f"{r['energy_per_image']*1000:>8.1f} mJ/img gross  "
            f"{r['energy_per_image_net']*1000:>8.1f} mJ/img net  "
            f"{r['achieved_tflops']:>6.2f} TFLOP/s"
            + (f"  ({r['compute_util_pct']:.1f}% of FP32 peak)" if r["compute_util_pct"] is not None else ""),
            flush=True,
        )

    record = {
        "config": {
            "model": "resnet50",
            "task": "inference_forward_only",
            "precision": args.precision,
            "input": "3x224x224 synthetic",
            "gflop_per_image": GFLOP_PER_IMAGE,
            "operation_definition": "approximate conv+linear work; FMA=2; not hardware occupancy",
            "protocol_version": "scientific-v2",
            "target_secs": args.target_secs,
            "group": args.group,
            "label": args.label,
            "memory_fraction": args.memory_fraction,
        },
        "device": {
            "name": props.name,
            "capability": f"{props.major}.{props.minor}",
            "theoretical_fp32_tflops": peak,
        },
        "rows": rows,
        "torch_env": torch_env(),
    }
    path = write_result(record, f"B_cnn_{args.label}_{args.precision}")
    print(f"  -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
