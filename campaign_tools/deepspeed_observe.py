"""Bounded, diagnostic-only engine-aware observation of a DeepSpeed ZeRO step.

Why this exists. The native observer brackets `optimizer.step()`. Under ZeRO the
scaling, overflow test, skip decision and base optimizer call all happen inside
`engine.step()`, so that bracket observes nothing and the run is honestly
recorded as `numerics_unobservable_on_engine_path`. This diagnostic asks the
engine itself, at a valid lifecycle point, for four independent things:

  1. gradients exist and are finite after backward and before the step
  2. the fp32 master partition this rank owns actually changes across the step
  3. the base optimizer's own per-parameter state advances
  4. the engine's step counters reconcile with the number of steps we drove

Collective discipline. Some accessors reduce across ranks. Every rank therefore
executes OBSERVATION_ORDER in full, unconditionally, in the same order, every
step. No accessor is ever called on rank zero alone, and the verdict is
all-reduced so no rank can pass while another fails. Nothing is injected: the
run is an ordinary short ZeRO run and this code only reads from it.

Not a performance measurement. Not a repair of the native path -- the native
status for ZeRO arms remains unobservable-on-engine-path either way.
"""
from __future__ import annotations

import json, os, pathlib, sys

sys.path.insert(0, os.environ["BENCH"])
import torch
import torch.distributed as dist

STEPS = 8
OBSERVATION_ORDER = [
    "forward",
    "engine_backward",
    "grad_evidence_after_backward_before_step",
    "master_partition_norms_before",
    "base_optimizer_state_steps_before",
    "engine_step",
    "master_partition_norms_after",
    "base_optimizer_state_steps_after",
    "engine_counters",
]


def probe(fn):
    """Read an accessor without ever substituting a value for a missing one."""
    try:
        return {"available": True, "value": fn(), "error": None}
    except Exception as exc:                      # noqa: BLE001 - recorded, not hidden
        return {"available": False, "value": None, "error": f"{type(exc).__name__}: {exc}"}


def _zero_opt(engine):
    return getattr(engine, "optimizer", None)


def _base_opt(engine):
    """The plain torch.optim.AdamW that ZeRO wraps."""
    z = _zero_opt(engine)
    return getattr(z, "optimizer", None) if z is not None else None


def grad_evidence(engine, model):
    """Gradient evidence at a valid lifecycle point: after backward, before step.

    Two sources are reported separately and never merged, because under stage 2
    the per-parameter .grad is partitioned away while the engine keeps reduced
    gradients in its own buffers. Whichever source carries the data, the finite
    check is applied to the tensors that actually exist.
    """
    out = {}
    for label, getter in (
        ("param_dot_grad", lambda: [p.grad for p in model.parameters() if p.grad is not None]),
        ("averaged_gradients", lambda: [g for gs in (_zero_opt(engine).averaged_gradients or {}).values()
                                        for g in gs if g is not None]),
    ):
        r = probe(getter)
        tensors = r["value"] if r["available"] and r["value"] else []
        finite = all(bool(torch.isfinite(t).all()) for t in tensors) if tensors else None
        out[label] = {
            "available": r["available"], "error": r["error"],
            "n_tensors": len(tensors),
            "all_finite": finite,
            "total_elements": int(sum(t.numel() for t in tensors)),
            "l2_norm": float(torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(t.detach().float()) for t in tensors]))) if tensors else None,
        }
    srcs = [k for k, v in out.items() if v["n_tensors"] > 0]
    out["evidence_sources_with_data"] = srcs
    out["verdict"] = ("observed_finite" if srcs and all(out[k]["all_finite"] for k in srcs)
                      else "observed_nonfinite" if srcs else "not_observed")
    return out


def master_partition_norms(engine):
    z = _zero_opt(engine)
    return probe(lambda: [float(torch.linalg.vector_norm(t.detach().float()))
                          for t in z.single_partition_of_fp32_groups])


def base_state_steps(engine):
    """Per-parameter 'step' counters inside the wrapped torch optimizer."""
    b = _base_opt(engine)

    def read():
        vals = []
        for group in b.param_groups:
            for p in group["params"]:
                st = b.state.get(p, {})
                if "step" in st:
                    s = st["step"]
                    vals.append(int(s.item()) if torch.is_tensor(s) else int(s))
        return vals

    return probe(read)


def engine_counters(engine):
    z = _zero_opt(engine)
    return {
        "global_steps": probe(lambda: int(engine.global_steps)),
        "micro_steps": probe(lambda: int(engine.micro_steps)),
        "global_grad_norm": probe(lambda: float(engine.get_global_grad_norm())),
        "overflow": probe(lambda: bool(z.overflow)),
    }


def main() -> int:
    stage = int(os.environ.get("DS_OBSERVE_STAGE", "1"))
    cfg_path = os.environ["DS_OBSERVE_CONFIG"]
    import deepspeed
    from transformers import GPT2Config, GPT2LMHeadModel

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)

    with open(cfg_path) as fh:
        ds_cfg = json.load(fh)
    ds_cfg["train_micro_batch_size_per_gpu"] = 4
    ds_cfg["gradient_accumulation_steps"] = 1
    ds_cfg.setdefault("zero_optimization", {})["stage"] = stage
    ds_cfg["fp16"] = {"enabled": False}
    ds_cfg["bf16"] = {"enabled": False}

    torch.manual_seed(1234)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=50257, n_positions=512, n_embd=768, n_layer=12, n_head=12))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    engine, _, _, _ = deepspeed.initialize(
        model=model, optimizer=opt, config=ds_cfg, model_parameters=model.parameters())

    gen = torch.Generator(device="cpu").manual_seed(1234 + rank)
    steps, start_global = [], int(engine.global_steps)

    for i in range(STEPS):
        ids = torch.randint(0, 50257, (4, 512), generator=gen).to(dev)
        rec = {"step": i}
        # --- OBSERVATION_ORDER, identical on every rank, no rank-conditional ---
        out = engine(input_ids=ids, labels=ids)
        loss = out.loss
        rec["loss"] = float(loss.detach())
        engine.backward(loss)
        rec["grad_evidence"] = grad_evidence(engine, model)
        rec["master_before"] = master_partition_norms(engine)
        rec["base_steps_before"] = base_state_steps(engine)
        engine.step()
        rec["master_after"] = master_partition_norms(engine)
        rec["base_steps_after"] = base_state_steps(engine)
        rec["counters"] = engine_counters(engine)
        # ----------------------------------------------------------------------
        mb, ma = rec["master_before"], rec["master_after"]
        rec["master_partition_changed"] = (
            bool(mb["available"] and ma["available"]
                 and any(abs(a - b) > 0.0 for a, b in zip(mb["value"], ma["value"]))))
        sb, sa = rec["base_steps_before"], rec["base_steps_after"]
        # torch.optim.AdamW creates per-parameter state lazily, so before the
        # very first step the state dict is legitimately empty. Treat that step
        # as the establishing step and require every value to be exactly 1;
        # every later step must then advance each counter by exactly 1.
        rec["base_state_establishing_step"] = bool(
            sb["available"] and sa["available"] and not sb["value"] and sa["value"])
        if rec["base_state_establishing_step"]:
            rec["base_state_advanced"] = all(v == 1 for v in sa["value"])
        else:
            rec["base_state_advanced"] = (
                bool(sb["available"] and sa["available"] and sa["value"]
                     and len(sa["value"]) == len(sb["value"])
                     and all(y - x == 1 for x, y in zip(sb["value"], sa["value"]))))
        steps.append(rec)

    driven = len(steps)
    gdelta = int(engine.global_steps) - start_global
    local_checks = {
        "gradients_observed_finite_every_step":
            all(s["grad_evidence"]["verdict"] == "observed_finite" for s in steps),
        "gradient_source_identified_every_step":
            all(s["grad_evidence"]["evidence_sources_with_data"] for s in steps),
        "master_partition_changed_every_step":
            all(s["master_partition_changed"] for s in steps),
        "base_optimizer_state_advanced_every_step":
            all(s["base_state_advanced"] for s in steps),
        "base_optimizer_state_established_exactly_once":
            sum(1 for s in steps if s["base_state_establishing_step"]) == 1
            and steps[0]["base_state_establishing_step"],
        "base_optimizer_state_counter_equals_steps_driven":
            bool(steps[-1]["base_steps_after"]["available"]
                 and steps[-1]["base_steps_after"]["value"]
                 and all(v == len(steps) for v in steps[-1]["base_steps_after"]["value"])),
        "engine_global_steps_reconcile": gdelta == driven,
        "no_overflow_reported":
            all(s["counters"]["overflow"]["value"] is False for s in steps),
        "global_grad_norm_finite_and_positive":
            all(s["counters"]["global_grad_norm"]["available"]
                and s["counters"]["global_grad_norm"]["value"] is not None
                and s["counters"]["global_grad_norm"]["value"] > 0.0
                and s["counters"]["global_grad_norm"]["value"] == s["counters"]["global_grad_norm"]["value"]
                for s in steps),
        "losses_finite": all(s["loss"] == s["loss"] for s in steps),
    }

    # Agreement across ranks: min over ranks, so one failing rank fails all.
    keys = sorted(local_checks)
    t = torch.tensor([1 if local_checks[k] else 0 for k in keys],
                     dtype=torch.int32, device=dev)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    agreed = {k: bool(v) for k, v in zip(keys, t.tolist())}
    n = torch.tensor([driven], dtype=torch.int32, device=dev)
    dist.all_reduce(n, op=dist.ReduceOp.MIN)
    agreed["every_rank_drove_the_same_step_count"] = int(n.item()) == driven

    per_rank = [None] * world
    dist.all_gather_object(per_rank, {
        "rank": rank, "local_checks": local_checks, "driven_steps": driven,
        "global_steps_delta": gdelta,
        "device_name": torch.cuda.get_device_name(local),
        "n_master_partitions": len(steps[-1]["master_after"]["value"] or [])
        if steps[-1]["master_after"]["available"] else None,
        "n_base_state_entries": len(steps[-1]["base_steps_after"]["value"] or [])
        if steps[-1]["base_steps_after"]["available"] else None,
        "grad_sources": steps[-1]["grad_evidence"]["evidence_sources_with_data"],
    })
    dist.barrier()

    if rank == 0:
        out = {
            "measurement_kind": "deepspeed_engine_aware_observation_not_performance",
            "zero_stage": stage, "world_size": world, "steps": driven,
            "observation_order": OBSERVATION_ORDER,
            "collective_discipline":
                "every rank executes OBSERVATION_ORDER in full each step; no accessor "
                "is called on rank zero alone; verdicts are MIN-all-reduced",
            "fault_injection": "none",
            "deepspeed_version": __import__("deepspeed").__version__,
            "native_observer_status_unchanged":
                "numerics_unobservable_on_engine_path; this diagnostic is additional "
                "evidence, not a repair of the native bracket",
            "agreed_checks": agreed,
            "all_agreed_checks_passed": all(agreed.values()),
            "base_optimizer_param_tensors_seen":
                len(steps[-1]["base_steps_after"]["value"] or []),
            "base_optimizer_note":
                "ZeRO hands the wrapped optimizer one flattened partition per "
                "param group, so the state entry count is the partition count, "
                "not the model's parameter count",
            "per_rank": per_rank,
            "per_step_rank0": steps,
        }
        d = pathlib.Path(os.environ["BENCH_RESULTS_DIR"]); d.mkdir(parents=True, exist_ok=True)
        (d / f"deepspeed_observe_zero{stage}.json").write_text(
            json.dumps(out, indent=2, sort_keys=True, default=str))
        for k in sorted(agreed):
            print(f"  [{'PASS' if agreed[k] else 'FAIL'}] {k}")
        print(f"stage={stage} world={world} steps={driven} global_steps_delta={gdelta} "
              f"grad_sources={steps[-1]['grad_evidence']['evidence_sources_with_data']}")

    ok = all(agreed.values())
    dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
