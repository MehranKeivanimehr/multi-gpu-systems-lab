"""NCCL all-reduce bandwidth sweep (nccl-tests style bus bandwidth). Launched with torchrun."""
from __future__ import annotations

import argparse
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist

from ..metrics import bus_bandwidth_allreduce
from ..results import environment, read_json, write_json

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    spec = read_json(args.spec)

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(minutes=5))
    world = dist.get_world_size()
    dtype = DTYPES[spec["dtype"]]
    elem = torch.tensor([], dtype=dtype).element_size()

    rows = []
    for size_mib in spec["sizes_mib"]:
        n = size_mib * 2**20 // elem
        x = torch.ones(n, dtype=dtype, device=device)
        for _ in range(spec["warmup_iters"]):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dist.barrier()
        t0 = time.perf_counter()
        for _ in range(spec["iters"]):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / spec["iters"]
        t = torch.tensor([dt], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        dt = t.item()
        algbw, busbw = bus_bandwidth_allreduce(n * elem, dt, world)
        rows.append({"size_mib": size_mib, "time_us": dt * 1e6, "algbw_gbs": algbw, "busbw_gbs": busbw})
        del x

    if dist.get_rank() == 0:
        env = environment(spec["gpus"])
        write_json(os.path.join(args.out, "result.json"),
                   {"status": "ok", "spec": spec, "rows": rows, "env": env})
        for r in rows:
            print(f"[mgbench] allreduce gpus={spec['gpus']} links={env['links']} {r['size_mib']:>5} MiB: "
                  f"{r['time_us']:9.1f} us  busbw {r['busbw_gbs']:6.2f} GB/s", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
