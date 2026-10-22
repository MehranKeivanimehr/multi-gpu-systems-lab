"""
NCCL all-reduce bandwidth and P2P reachability per device group.

This exists to explain the DDP scaling numbers rather than leave them inferred.
GPT-2's recorded 124,439,808 parameters correspond to 497,759,232 bytes of
FP32 gradients (about 474.70 MiB). The historical largest probe setting is
497 MiB, a nearby synthetic payload about 4.7% larger, not an exact match.

Reported bandwidths:
  algbw = bytes / time                (what the algorithm sees)
  busbw = algbw * 2(n-1)/n            (what the wire actually carries for a
                                       ring all-reduce; comparable across n)

Launch under torchrun with one rank per GPU in the group.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from common import apply_memory_cap, phys_devices, torch_env, write_result  # noqa: E402

# Historical field name: these allocations are MiB; retain it for compatibility.
#: Payloads in MEBIBYTES (MiB = 1024*1024 B). Exact byte counts are
#: recorded per row. 497 MiB = 521,142,272 B is NOT the GPT-2 124M FP32
#: gradient volume (124,439,808 params x 4 B = 497,759,232 B); it is a
#: nearby synthetic payload about 4.7% larger.
SIZES_MIB = [4, 16, 64, 256, 497]
GPT2_124M_FP32_GRADIENT_BYTES = 124_439_808 * 4

# The 497 MiB point is 521,142,272 B. The actual FP32 gradient volume of GPT-2
# 124M is 497,759,232 B -- 497.76 MB decimal, but only 474.7 MiB. Labelling the
# sweep's 497 MiB point "the 497 MB gradient payload" conflated the two units
# and left the real DDP payload unmeasured, which is why every row reported
# is_exact_gpt2_124m_fp32_gradient_volume = false. The exact volume is measured
# explicitly here so one row carries the true payload.
EXTRA_PAYLOAD_BYTES = [GPT2_124M_FP32_GRADIENT_BYTES]


def measurement_payloads() -> list[tuple[int | None, int]]:
    """(size_mib or None, payload_bytes) for every point in the sweep."""
    pts = [(mb, mb * 1024 * 1024 // 4 * 4) for mb in SIZES_MIB]
    have = {b for _, b in pts}
    pts += [(None, b) for b in EXTRA_PAYLOAD_BYTES if b not in have]
    return sorted(pts, key=lambda t: t[1])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--group", required=True)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])

    dist.init_process_group("nccl", init_method="env://")
    torch.cuda.set_device(local)
    apply_memory_cap()
    device = torch.device("cuda", local)

    # P2P reachability matrix (local indices; physical list is in meta)
    p2p = {}
    for i in range(world):
        for j in range(world):
            if i == j:
                continue
            try:
                p2p[f"{i}-{j}"] = bool(torch.cuda.can_device_access_peer(i, j))
            except Exception:
                p2p[f"{i}-{j}"] = None

    rows = []
    for mb, target_bytes in measurement_payloads():
        n_elem = target_bytes // 4
        buf = torch.ones(n_elem, device=device, dtype=torch.float32)

        for _ in range(5):  # warmup / ring construction
            dist.all_reduce(buf)
        torch.cuda.synchronize(device)
        dist.barrier()

        t0 = time.perf_counter()
        for _ in range(args.iters):
            dist.all_reduce(buf)
        torch.cuda.synchronize(device)
        dt = (time.perf_counter() - t0) / args.iters
        dist.barrier()

        nbytes = n_elem * 4
        algbw = nbytes / dt / 1e9
        busbw = algbw * 2 * (world - 1) / world
        if rank == 0:
            rows.append(
                {
                    "size_mib": mb,
                    "size_mib_exact": round(nbytes / 1024 / 1024, 3),
                    "payload_bytes": nbytes,
                    "is_exact_gpt2_124m_fp32_gradient_volume":
                        nbytes == GPT2_124M_FP32_GRADIENT_BYTES,
                    "time_ms": round(dt * 1000, 3),
                    "algbw_gb_s": round(algbw, 2),
                    "ring_equivalent_bw_gb_s": round(busbw, 2),
                    "ring_equivalent_bw_note":
                        "algbw x 2(n-1)/n ring-equivalent NORMALISATION; an analytical rescaling, not measured wire traffic",
                }
            )
            tag = (f"{mb:>7} MiB" if mb is not None
                   else f"{nbytes/1024/1024:>7.1f} MiB")
            exact = "  <- exact GPT-2 124M FP32 gradient volume" if \
                nbytes == GPT2_124M_FP32_GRADIENT_BYTES else ""
            print(
                f"  {tag} {dt*1000:>8.2f} ms  "
                f"algbw {algbw:>6.2f} GB/s  ring-eq {busbw:>6.2f} GB/s{exact}",
                flush=True,
            )
        del buf
        torch.cuda.empty_cache()

    if rank == 0:
        record = {
            "config": {"group": args.group, "world_size": world, "iters": args.iters},
            "p2p_access": p2p,
            # An unknown entry must NOT be skipped into a vacuous "all enabled".
            "p2p_unknown_entries": sum(1 for v in p2p.values() if v is None),
            "p2p_all_enabled": (None if any(v is None for v in p2p.values())
                                else all(p2p.values())),
            "rows": rows,
            "torch_env": torch_env(),
        }
        path = write_result(record, f"C_comm_{args.group}")
        print(f"  -> {path}", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
