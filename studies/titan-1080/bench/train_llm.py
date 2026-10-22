"""
GPT-2 124M training benchmark: single-GPU, DDP, FSDP and DeepSpeed ZeRO.

The modes share a workload shape, but their precision and optimizer
implementations are distinct recipes. New evidence uses scientific-v2.

Deliberate choices for this cluster:

* Model is built from GPT2Config() defaults -- that is exactly GPT-2 small
  (12 layers, d_model 768, 12 heads, vocab 50257), 124M parameters -- with
  random init rather than downloaded weights. Throughput is identical either
  way, and this keeps the benchmark hermetic and byte-for-byte reproducible.
* Data is synthetic random token IDs. There is no dataloader, no disk I/O and
  no host-side preprocessing, so the number measured is GPU + interconnect
  throughput rather than a storage benchmark.
* FP32 everywhere by default. This box lacks native BF16 acceleration;
  tensor allocation/emulation support is a separate capability. FP16 via AMP is offered on TITAN RTX (sm_75, Tensor
  Cores); on GP102 Pascal, fp16 arithmetic runs at 1/64 of fp32 rate, so AMP
  there is a pessimisation and the suite does not pretend otherwise.
* No CUDA JIT anywhere. The host nvcc is 10.1 against a 12.2 driver, so
  DeepSpeed runs with DS_BUILD_OPS=0, no offload and a plain torch AdamW
  instead of FusedAdam. ZeRO 1/2 are pure Python + NCCL and need no compiler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
from torch.nn.parallel import DistributedDataParallel as DDP  # noqa: E402

from launch_gate import terminate_owned  # noqa: F401  (re-exported for operators)
from run_report import apply_certification, build_update_record
from update_observer import UpdateObserver, not_applicable_native_summary
from common import (
    SAMPLE_INTERVAL_S,
    runtime_device_identity,  # noqa: E402
    GpuSampler,
    apply_memory_cap,
    guard_devices,
    phys_devices,
    stats_to_dict,
    torch_env,
    wait_for_idle,
    write_result,
)

SEQ_LEN_DEFAULT = 512


# ---------------------------------------------------------------------------
# distributed plumbing
# ---------------------------------------------------------------------------

def dist_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world, local


def is_primary() -> bool:
    return int(os.environ.get("RANK", "0")) == 0


def log(msg: str) -> None:
    if is_primary():
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# model + data
# ---------------------------------------------------------------------------

def build_model(seq_len: int, grad_ckpt: bool = False):
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(
        n_positions=max(1024, seq_len),
        n_ctx=max(1024, seq_len),
        use_cache=False,
    )
    model = GPT2LMHeadModel(cfg)
    if grad_ckpt:
        model.gradient_checkpointing_enable()
    return model


def synthetic_batch(bs: int, seq_len: int, device: torch.device, gen: torch.Generator):
    ids = torch.randint(
        0, 50257, (bs, seq_len), device=device, generator=gen, dtype=torch.long
    )
    return ids


# ---------------------------------------------------------------------------
# the benchmark
# ---------------------------------------------------------------------------

def run(args) -> int:
    rank, world, local = dist_info()
    multi = args.mode in ("ddp", "fsdp", "deepspeed")

    if multi:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        torch.cuda.set_device(0)
        device = torch.device("cuda", 0)

    identity_before_cap = runtime_device_identity(local)
    apply_memory_cap(args.memory_fraction)
    identity_after_cap = runtime_device_identity(local)

    torch.manual_seed(args.seed)  # identical initial full model on every rank
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed + rank)  # deliberate rank-distinct data streams
    torch.backends.cudnn.benchmark = True
    # TF32 is Ampere+; on sm_61/sm_75 these are no-ops, set explicitly so the
    # FP32 path means the same thing if this suite is ever run elsewhere.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    amp = args.precision == "amp_fp16"
    model = build_model(args.seq_len, args.grad_ckpt)
    n_params = sum(p.numel() for p in model.parameters())
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(str((tuple(value.shape), value.dtype)).encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    initial_model_sha256 = digest.hexdigest()
    if multi:
        hashes = [None] * world
        dist.all_gather_object(hashes, initial_model_sha256)
        if len(set(hashes)) != 1:
            raise RuntimeError("Ranks did not start from identical full model states")

    engine = None
    ds_cfg = None
    scaler = None
    optimizer = None

    if args.mode == "deepspeed":
        import deepspeed

        with open(args.ds_config) as fh:
            ds_cfg = json.load(fh)
        ds_cfg["train_micro_batch_size_per_gpu"] = args.batch_size
        ds_cfg["gradient_accumulation_steps"] = 1
        ds_cfg.setdefault("zero_optimization", {})["stage"] = args.zero_stage
        if amp:
            ds_cfg["fp16"] = {"enabled": True, "loss_scale": 0, "initial_scale_power": 12}
        else:
            ds_cfg["fp16"] = {"enabled": False}
            ds_cfg["bf16"] = {"enabled": False}
        # Plain torch AdamW: DeepSpeed's FusedAdam would JIT-compile against the
        # host nvcc 10.1, which cannot target this driver. ZeRO 1/2 themselves
        # need no custom kernels.
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        engine, _, _, _ = deepspeed.initialize(
            model=model, optimizer=optimizer, config=ds_cfg,
            model_parameters=model.parameters(),
        )
        step_model = engine
        device = torch.device("cuda", engine.local_rank)
    elif args.mode == "fsdp":
        import functools

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from transformers.models.gpt2.modeling_gpt2 import GPT2Block

        policy = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls={GPT2Block}
        )
        # Pure FP32: native BF16 acceleration is absent on sm_61/sm_75; fp16 FSDP would
        # need ShardedGradScaler, changing the numerics versus DDP/ZeRO. Keeping
        # FP32 preserves a common compute dtype, not full recipe equivalence.
        model = FSDP(
            model.to(device),
            auto_wrap_policy=policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=local,
            use_orig_params=True,
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        step_model = model
    else:
        model = model.to(device)
        if args.mode == "ddp":
            model = DDP(model, device_ids=[local], output_device=local)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        step_model = model

    if amp and args.mode != "deepspeed":
        scaler = torch.amp.GradScaler("cuda")

    # Attempts are counted by the loop, gradients observed outside the
    # conditional optimizer call. A step-pre-hook cannot see a GradScaler skip:
    # GradScaler omits optimizer.step() entirely, so neither hook fires on
    # exactly the non-finite steps. See bench/update_observer.py.
    observer = UpdateObserver(numerical_checks=args.numerical_checks)
    observer.attach(optimizer)

    autocast = (
        torch.amp.autocast("cuda", dtype=torch.float16) if amp else nullcontext()
    )

    def one_step() -> float:
        ids = synthetic_batch(args.batch_size, args.seq_len, device, gen)
        if args.mode == "deepspeed":
            with autocast if amp else nullcontext():
                out = step_model(input_ids=ids, labels=ids)
            loss = out.loss
            step_model.backward(loss)
            step_model.step()
            return float(loss.detach())
        optimizer.zero_grad(set_to_none=True)
        observer.begin_step()
        with autocast:
            out = step_model(input_ids=ids, labels=ids)
            loss = out.loss
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.numerical_checks:
                # Diagnostic path only. unscale_ brings gradients back to true
                # scale BEFORE inspection; scaler.step() detects the prior
                # unscale and will not unscale twice. get_scale() synchronises
                # CPU and GPU, so it is confined to this branch and never runs
                # in a clean timing path.
                scaler.unscale_(optimizer)
                observer.observe_gradients(optimizer, scaling="unscaled")
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                observer.record_scale(scale_before, scaler.get_scale())
            else:
                scaler.step(optimizer)
                scaler.update()
        else:
            loss.backward()
            if args.numerical_checks:
                observer.observe_gradients(optimizer, scaling="none")
            optimizer.step()
        observer.end_step()
        return float(loss.detach())

    # --- warmup (allocator settling, cudnn autotune, NCCL ring setup) -------
    status = "ok"
    err = None
    try:
        for _ in range(args.warmup):
            one_step()
        torch.cuda.synchronize(device)
        if multi:
            dist.barrier()
    except torch.cuda.OutOfMemoryError as exc:
        status, err = "oom", str(exc)[:400]
    except RuntimeError as exc:
        status = "oom" if "out of memory" in str(exc).lower() else "error"
        err = str(exc)[:400]

    # Agree on warmup outcome across ALL ranks before branching. Without this,
    # a rank that OOMs skips the measurement loop while its healthy peers block
    # forever in a collective it will never join -- the job then burns the full
    # wall-clock timeout instead of failing fast. Every rank calls this exactly
    # once, including a rank that just failed.
    if multi:
        if status != "ok":
            torch.cuda.empty_cache()
        flag = torch.tensor(
            [0 if status == "ok" else 1], device=device, dtype=torch.int32
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item() > 0 and status == "ok":
            status, err = "peer_failed", "another rank failed during warmup"

    record: dict[str, object] = {}
    step_times: list[float] = []
    losses: list[float] = []
    stats = None

    # Collective participation downstream keys off this, NOT off the live
    # status: after the sync above every rank agrees on it, so every rank makes
    # the same set of collective calls even if one of them fails mid-loop.
    entered_measurement = status == "ok"
    engine_start = (int(engine.global_steps), int(engine.skipped_steps)) if engine is not None else None

    if entered_measurement:
        # Discard warm-up accounting by rebuilding the observer, so measured
        # attempts and base calls start from zero.
        observer = UpdateObserver(numerical_checks=args.numerical_checks)
        observer.attach(optimizer)
        torch.cuda.reset_peak_memory_stats(device)
        sampler = None
        if is_primary():
            sampler = GpuSampler(phys_devices(), interval=args.sample_interval)

        try:
            ctx = sampler if sampler is not None else nullcontext()
            with ctx:
                t_start = sampler.mark_start() if sampler else time.perf_counter()
                for _ in range(args.steps):
                    t0 = time.perf_counter()
                    losses.append(one_step())
                    torch.cuda.synchronize(device)
                    step_times.append(time.perf_counter() - t0)
                if multi:
                    dist.barrier()
                torch.cuda.synchronize(device)
                t_total = (sampler.mark_stop() if sampler else time.perf_counter()) - t_start
            if sampler is not None:
                stats = sampler.stats()
        except torch.cuda.OutOfMemoryError as exc:
            status, err = "oom", str(exc)[:400]
            t_total = float("nan")
        except RuntimeError as exc:
            status = "oom" if "out of memory" in str(exc).lower() else "error"
            err = str(exc)[:400]
            t_total = float("nan")
    else:
        t_total = float("nan")

    peak_alloc = torch.cuda.max_memory_allocated(device) / 1024**2
    peak_resv = torch.cuda.max_memory_reserved(device) / 1024**2
    expected_updates = args.steps if entered_measurement else 0
    engine_counters = None
    if args.mode == "deepspeed":
        native_summary = not_applicable_native_summary(
            "scaling, overflow detection, the skip decision and the base "
            "optimizer call all occur inside engine.step(); the native observer "
            "does not bracket it")
    else:
        native_summary = observer.summary()
    if engine_start is not None:
        boundaries = int(engine.global_steps) - engine_start[0]
        skipped = int(engine.skipped_steps) - engine_start[1]
        engine_counters = {"engine_update_boundaries": boundaries,
                           "engine_skipped_updates": skipped,
                           "engine_base_optimizer_calls": observer.base_optimizer_calls}
    update_audit = build_update_record(
        mode=args.mode, numerical_checks=args.numerical_checks,
        expected_logical_updates=expected_updates,
        native_summary=native_summary, engine_counters=engine_counters,
        losses=losses)
    apply_certification(update_audit)
    if entered_measurement and status == "ok" and not update_audit["update_claim_verified"]:
        status, err = ("invalid_numerics_or_updates",
                       "; ".join(update_audit["certification_reasons"])[:400]
                       or "update claim not verified")
    elif entered_measurement and status == "ok" and \
            update_audit["numerical_diagnostic_certified"] is False:
        status, err = (update_audit["status_verdict"],
                       "; ".join(update_audit["certification_reasons"])[:400])
    if multi and entered_measurement:
        per_rank_audit = [None] * world
        dist.all_gather_object(per_rank_audit, update_audit)
    else:
        per_rank_audit = [update_audit]

    # --- aggregate across ranks --------------------------------------------
    # Both collectives below are gated on entered_measurement, which is
    # identical on every rank, so they are always matched.
    if multi and entered_measurement:
        buf = torch.tensor([peak_alloc, peak_resv], device=device)
        gathered = [torch.zeros_like(buf) for _ in range(world)]
        dist.all_gather(gathered, buf)
        per_rank_mem = [[round(x.item(), 1) for x in g] for g in gathered]

        # Catch a failure that appeared during the measured loop rather than
        # warmup, so one rank's mid-loop OOM is reported instead of silently
        # producing a throughput number from the survivors.
        flag = torch.tensor(
            [0 if status == "ok" else 1], device=device, dtype=torch.int32
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item() > 0 and status == "ok":
            status, err = "peer_failed", "another rank failed during measurement"
    else:
        per_rank_mem = [[round(peak_alloc, 1), round(peak_resv, 1)]]

    if not is_primary():
        if multi:
            dist.barrier()
            dist.destroy_process_group()
        return 0 if status == "ok" else 1

    # --- rank-0 reporting ---------------------------------------------------
    tokens_per_step = args.batch_size * args.seq_len * world
    if status == "ok" and step_times:
        sorted_t = sorted(step_times)
        median = sorted_t[len(sorted_t) // 2]
        mean = sum(step_times) / len(step_times)
        throughput = tokens_per_step * len(step_times) / t_total
        per_gpu_tp = throughput / world
    else:
        median = mean = throughput = per_gpu_tp = float("nan")

    record = {
        "config": {
            "mode": args.mode,
            "zero_stage": args.zero_stage if args.mode == "deepspeed" else None,
            "precision": args.precision,
            "model": "gpt2-124m",
            "n_params": n_params,
            "seq_len": args.seq_len,
            "micro_batch_per_gpu": args.batch_size,
            "global_batch": args.batch_size * world,
            "world_size": world,
            "steps": args.steps,
            "warmup": args.warmup,
            "grad_ckpt": args.grad_ckpt,
            "memory_fraction": args.memory_fraction,
            "group": args.group,
            "protocol_version": "scientific-v2",
            "seed": args.seed,
            "initial_model_sha256": initial_model_sha256,
            "numerical_checks": args.numerical_checks,
            "runtime_device_identity_before_memory_cap": identity_before_cap,
            "runtime_device_identity_after_memory_cap": identity_after_cap,
            "device_binding_preserved_across_memory_cap": (
                identity_before_cap.get("current_device")
                == identity_after_cap.get("current_device")
                and identity_before_cap.get("uuid") == identity_after_cap.get("uuid")),
            "device_identity_established": bool(
                identity_after_cap.get("uuid_established")),
            "deepspeed_resolved_config": ds_cfg,
            "recipe_comparison_note": "DeepSpeed clipping/precision defaults may differ; not an isolated framework effect",
        },
        "status": status,
        "error": err,
        "perf": {
            "wall_time_s": round(t_total, 4) if t_total == t_total else None,
            "tokens_per_step": tokens_per_step,
            "step_time_mean_s": round(mean, 5) if mean == mean else None,
            "step_time_median_s": round(median, 5) if median == median else None,
            "step_time_min_s": round(min(step_times), 5) if step_times else None,
            "step_time_max_s": round(max(step_times), 5) if step_times else None,
            "tokens_per_s": round(throughput, 1) if throughput == throughput else None,
            "tokens_per_s_per_gpu": (
                round(per_gpu_tp, 1) if per_gpu_tp == per_gpu_tp else None
            ),
        },
        "memory": {
            "peak_alloc_mib_rank0": round(peak_alloc, 1),
            "peak_reserved_mib_rank0": round(peak_resv, 1),
            "per_rank_alloc_reserved_mib": per_rank_mem,
        },
        "telemetry": stats_to_dict(stats) if stats else None,
        "loss_first_last": [losses[0], losses[-1]] if losses else None,
        "losses": losses,
        "optimizer_update_audit_per_rank": per_rank_audit,
        "measurement_kind": "numerical_diagnostic" if args.numerical_checks else "throughput_with_update_counter",
        "torch_env": torch_env(),
    }

    name = args.out_name or (
        f"{args.mode}_{args.group}_bs{args.batch_size}_{args.precision}"
    )
    path = write_result(record, name)
    log(
        f"[{status}] {args.mode} {args.group} ws={world} bs={args.batch_size} "
        f"{args.precision}: "
        + (
            f"{throughput:.0f} tok/s total, {per_gpu_tp:.0f} tok/s/gpu, "
            f"peak {peak_alloc:.0f} MiB"
            if status == "ok"
            else f"{err}"
        )
    )
    log(f"  -> {path}")

    if multi:
        dist.barrier()
        dist.destroy_process_group()
    return 0 if status == "ok" else (42 if status == "oom" else 1)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode", required=True, choices=["single", "ddp", "fsdp", "deepspeed"]
    )
    p.add_argument("--precision", default="fp32", choices=["fp32", "amp_fp16"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=SEQ_LEN_DEFAULT)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--numerical-checks", action="store_true", help="diagnostic finite-gradient checks; timings are not performance results")
    p.add_argument("--zero-stage", type=int, default=1, choices=[0, 1, 2, 3])
    p.add_argument("--ds-config", default=None)
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--memory-fraction", type=float, default=0.85)
    p.add_argument("--sample-interval", type=float, default=SAMPLE_INTERVAL_S)
    p.add_argument("--group", default="unknown", help="label for the device group")
    p.add_argument("--out-name", default=None)
    p.add_argument(
        "--skip-guard",
        action="store_true",
        help="runner already guarded; avoids re-querying smi on every rank",
    )
    args = p.parse_args()
    if args.mode == "fsdp" and args.precision != "fp32":
        p.error("FSDP validation currently supports FP32 only; do not silently relabel its recipe")
    if args.steps < 1 or args.warmup < 1:
        p.error("positive measurement and warmup counts are required")

    if args.mode == "deepspeed" and not args.ds_config:
        args.ds_config = str(
            Path(__file__).resolve().parent.parent
            / "configs"
            / f"ds_zero{args.zero_stage}.json"
        )

    if is_primary() and not args.skip_guard:
        guard_devices(phys_devices())
        # Let the previous sweep step's utilisation decay so this measurement
        # starts from a quiet GPU and the power baseline is clean.
        wait_for_idle(phys_devices())

    return run(args)


if __name__ == "__main__":
    sys.exit(main())
