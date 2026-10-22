"""One setup function per framework. Each returns a Trainer whose step() runs one optimizer step
(grad_accum micro-batches) and returns the last micro-batch loss."""
from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..models import (build_causal_lm, build_vision, has_tied_params, materialize_meta_model,
                      shard_units)
from ..metrics import GIB
from .diagnostics import (NumericalDiagnostics, canonical_model, ddp_recording_comm_hook, fsdp_reduce_dtypes,
                          gathered_state_fingerprints)
from .boundary_ledger import BoundaryLedger, local_numel, scan_tensors, scan_units
from .reference import FirstBoundarySession, ReferenceSession
from .update_evidence import AdamWCallSampler, build_plan, plan_record

LR = 1e-4


@dataclass
class Trainer:
    step: Callable[[list[dict]], torch.Tensor]
    n_params: int
    data_dtype: torch.dtype = torch.float32
    info: dict = field(default_factory=dict)
    update_audit: dict = field(default_factory=dict)
    # Node-side additions (A6000 validation, session 20261014a); all None for performance runs.
    diag: object = None                 # NumericalDiagnostics
    canonical: dict = None              # canonical CPU-fp32 initial-state fingerprints (rank 0)
    state_fingerprint: Callable = None  # collective: fingerprints of the full logical state after wrapping
    extra_diag: Callable = None         # framework-specific live facts (e.g. FSDP2 reduce dtypes)
    # Smoke stage 20261016s: per-boundary ledger with expected-coverage accounting and sampled update evidence.
    ledger: object = None               # BoundaryLedger
    # Stage S1a: reference write/compare session, canonical start on the CPU (theta0), final-state items.
    reference: object = None            # ReferenceSession
    theta0: dict = None                 # {canonical name: CPU float32 tensor}
    final_items: Callable = None        # -> [(name, param-or-master, exp_avg, exp_avg_sq)]
    final_checks: Callable = None       # -> dict (e.g. DeepSpeed bf16 module vs master, full)


def snapshot_rank_allowed(spec) -> bool:
    """Candidate 16: in the full-control kind only rank 0 writes snapshots and keeps theta0 (gradients and parameters are
    identical on every DDP rank after the all-reduce); other ranks never open a snapshot file."""
    return spec.get("result_kind") != "full_control_5update" or not dist.is_initialized() or dist.get_rank() == 0


def reference_session(spec):
    """S1a write/compare session, or Candidate 14's first-boundary session (own snapshot + compare, no update)."""
    if not snapshot_rank_allowed(spec):
        return None
    if spec.get("result_kind") == "first_boundary_gradient":
        return FirstBoundarySession.from_spec(spec, spec["_out_dir"])
    return ReferenceSession.from_spec(spec)


def first_boundary(spec) -> bool:
    return spec.get("result_kind") == "first_boundary_gradient"


def native_bf16_params(spec) -> bool:
    """G3: native bf16 parameters/buffers (module cast), no autocast; .grad and its accumulation are bf16."""
    return spec.get("param_dtype") == "bf16" and spec["framework"] == "ddp"


def diagnostics_enabled(spec) -> bool:
    return any(spec.get(k) for k in ("numerical_checks", "reference_capture", "diagnostic_fault", "canonical_init"))


def adam_branch(optimizer) -> dict:
    """The AdamW implementation branch torch selects for this call: the same rule as torch/optim/adam.py `adam()`
    when foreach/fused are unset (_default_to_fused_or_foreach on the parameters that have gradients, then the
    tensor-lr exception). A resolution record taken at the first base call, not an execution trace."""
    from torch.optim.optimizer import _default_to_fused_or_foreach

    g0 = optimizer.param_groups[0]
    params = [p for g in optimizer.param_groups for p in g["params"] if p.grad is not None]
    foreach, fused = g0.get("foreach"), g0.get("fused")
    rec = {"configured": {k: g0.get(k) for k in ("foreach", "fused", "capturable", "differentiable")},
           "optimizer_class": f"{type(optimizer).__module__}.{type(optimizer).__name__}", "params_with_grad": len(params),
           "basis": "torch.optim.optimizer._default_to_fused_or_foreach(params_with_grad, differentiable, use_fused=False) "
                    "at the first base call, as adam() applies it when foreach and fused are both None"}
    if foreach is None and fused is None:
        _, foreach = _default_to_fused_or_foreach(params, bool(g0.get("differentiable")), use_fused=False)
        if foreach and isinstance(g0.get("lr"), torch.Tensor) and not g0.get("capturable"):
            foreach = False
    rec.update(foreach=bool(foreach), fused=bool(fused))
    rec["branch"] = "fused" if rec["fused"] else "foreach (_multi_tensor_adam)" if rec["foreach"] else \
        "single tensor (_single_tensor_adam)"
    return rec


def observe_optimizer(opt, model, spec, ledger=None, observe_in_hook=False, hook_coverage="unknown"):
    """Count actual torch optimizer calls; diagnostic checks are opt-in.

    DeepSpeed's wrapping behavior must be verified by a target-host pilot. A
    zero/mismatched counter invalidates the result rather than assuming success.

    These counters are per *base optimizer call*. A step whose optimizer call is skipped is never seen here, so the
    per-attempt evidence lives in the BoundaryLedger: native frameworks inspect gradients in the step itself
    (observe_in_hook=False); DeepSpeed can only be observed here (observe_in_hook=True), and a skipped DeepSpeed
    boundary therefore stays explicitly unobserved.
    """
    state = {"optimizer_calls": 0, "successful_updates": 0, "skipped_updates": 0,
             "engine_update_boundaries": 0, "logical_updates_from_engine": False,
             "gradient_checks": 0, "nonfinite_gradient_steps": 0,
             "gradient_checks_enabled": bool(spec.get("numerical_checks", False))}
    dtypes = {}
    for parameter in model.parameters():
        key = str(parameter.dtype)
        dtypes[key] = dtypes.get(key, 0) + parameter.numel()

    def before(optimizer, args, kwargs):
        if "optimizer_branch" not in state:
            try:
                state["optimizer_branch"] = adam_branch(optimizer)
            except Exception as exc:  # noqa: BLE001
                state["optimizer_branch"] = {"branch": None, "error": repr(exc)[:300]}
        if state["gradient_checks_enabled"]:
            finite = True
            count = 0
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        grad = parameter.grad
                        if hasattr(grad, "to_local"):
                            grad = grad.to_local()
                        finite = finite and bool(torch.isfinite(grad).all().item())
                        count += 1
            state["gradient_checks"] += 1
            state["nonfinite_gradient_steps"] += int(not finite or count == 0)
            if ledger is not None and observe_in_hook:
                ledger.observe(finite if count else None, count, "base optimizer pre-hook", hook_coverage)

    def after(optimizer, args, kwargs):
        state["optimizer_calls"] += 1
        if ledger is not None:
            ledger.base_call()
        if not state["logical_updates_from_engine"]:
            state["successful_updates"] += 1

    opt.register_step_pre_hook(before)
    opt.register_step_post_hook(after)
    return state, dtypes


def native_param_units(opt) -> dict:
    """Expected gradient units of a native torch optimizer: every trainable tensor of every param group, keyed by
    (group, position), with its *local* element count (FSDP2 DTensor: this rank's shard)."""
    return {(gi, pi): p for gi, g in enumerate(opt.param_groups) for pi, p in enumerate(g["params"]) if p.requires_grad}


def native_unit_map(model, opt) -> dict:
    """Independent map from MODEL parameter identities (named_parameters, including tied aliases) to optimizer units.
    Every trainable model parameter must occupy exactly one optimizer slot, and the optimizer must hold nothing that
    is not a model parameter; otherwise there is no expected-coverage accounting ("problems")."""
    slots = {}
    for gi, g in enumerate(opt.param_groups):
        for pi, p in enumerate(g["params"]):
            slots.setdefault(id(p), []).append((gi, pi))
    by_id, tensors = {}, {}
    for name, p in model.named_parameters(remove_duplicate=False):
        by_id.setdefault(id(p), []).append(name)
        tensors[id(p)] = p
    problems, units, names, aliases, frozen = [], {}, {}, {}, []
    for pid, pnames in by_id.items():
        p = tensors[pid]
        if not p.requires_grad:
            frozen.append(pnames[0])
            continue
        where = slots.get(pid, [])
        if len(where) != 1:
            problems.append(f"{pnames[0]} occupies {len(where)} optimizer slots")
            continue
        units[where[0]] = p
        names[where[0]] = pnames[0]
        if len(pnames) > 1:
            aliases[pnames[0]] = pnames[1:]
    foreign = [k for pid, ks in slots.items() if pid not in by_id for k in ks]
    if foreign:
        problems.append(f"optimizer holds {len(foreign)} tensor(s) that are not model parameters")
    return {"units": units, "names": names, "tied_aliases": aliases, "frozen": frozen,
            "declared_unused": [], "problems": problems,
            "summary": {"model_parameter_names": sum(len(v) for v in by_id.values()), "unique_trainable": len(units),
                        "tied_alias_names": sum(len(v) for v in aliases.values()), "frozen": len(frozen),
                        "optimizer_slots": sum(len(v) for v in slots.values())}}


def canonical_theta0(model, spec) -> dict | None:
    """CPU float32 copy of the canonical start, keyed by canonical (first) parameter name; only for reference runs."""
    if not spec.get("reference_role") or not snapshot_rank_allowed(spec):
        return None
    return {n: p.detach().to("cpu", torch.float32).clone() for n, p in model.named_parameters()}


def native_reference_items(umap: dict, attr: str = "grad") -> list:
    return [(umap["names"][k], getattr(umap["units"][k], attr)) for k in sorted(umap["units"])]


def native_final_items(umap: dict, opt) -> list:
    out = []
    for k in sorted(umap["units"]):
        p = umap["units"][k]
        st = opt.state[p]
        out.append((umap["names"][k], p, st["exp_avg"], st["exp_avg_sq"]))
    return out


def native_sample_plan(unit_map: dict) -> dict:
    """First, middle and last trainable parameter in optimizer order (fixed before any step)."""
    keys = sorted(unit_map["units"])
    picks = sorted({0, len(keys) // 2, len(keys) - 1}) if keys else []
    return build_plan([(unit_map["names"][keys[i]], unit_map["units"][keys[i]]) for i in picks])


def attach_boundary_hooks(opt, ledger, *, units_in_hook=None, sampler=None, ref_in_hook=None):
    """Base-optimizer hooks feeding the ledger: optional gradient-unit inspection inside the call (DeepSpeed, where
    gradients exist only there), sampled update evidence, and one base_call() per actual call."""
    def pre(optimizer, args, kwargs):
        if units_in_hook is not None and ledger.gradient_checks:
            ledger.observe_units(units_in_hook(optimizer), "base optimizer pre-hook")
        if sampler is not None:
            sampler.pre(optimizer)
        if ref_in_hook is not None:
            ref_in_hook(optimizer)

    def post(optimizer, args, kwargs):
        evidence = None
        if sampler is not None:
            sampler.post(optimizer)
            evidence = sampler.calls[-1]
        ledger.base_call(evidence)

    opt.register_step_pre_hook(pre)
    opt.register_step_post_hook(post)


def loss_fn(model, batch: dict, kind: str) -> torch.Tensor:
    if kind == "causal_lm":
        return model(input_ids=batch["input_ids"], labels=batch["labels"]).loss
    return F.cross_entropy(model(batch["images"]), batch["labels"])


def _count_params(model) -> int:
    from torch.distributed.tensor import DTensor

    seen, n = set(), 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        n += p.numel() if not isinstance(p, DTensor) else p.size().numel()
    return n


def _enable_ckpt(model, spec):
    if spec.get("activation_checkpointing") and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})


def _build(entry, spec, device, dtype):
    if entry["kind"] == "causal_lm":
        model = build_causal_lm(entry, device, dtype)
    else:
        model = build_vision(entry, device, dtype)
    _enable_ckpt(model, spec)
    return model


# ------------------------------------------------------------------------------------------------- DDP

def setup_ddp(spec, entry, fw, device) -> Trainer:
    from torch.nn.parallel import DistributedDataParallel as DDP

    canonical = None
    theta0 = None
    if spec.get("canonical_init"):
        model, canonical = canonical_model(entry, spec, _build)
        theta0 = canonical_theta0(model, spec)
        model = model.to(device)
    else:
        model = _build(entry, spec, device, torch.float32)
    if native_bf16_params(spec):
        model = model.to(torch.bfloat16)          # parameters AND buffers, like DeepSpeed's native bf16 module cast
    n_params = _count_params(model)
    ddp = DDP(model, device_ids=[device.index], bucket_cap_mb=fw["bucket_cap_mb"],
              gradient_as_bucket_view=fw["gradient_as_bucket_view"])
    opt = torch.optim.AdamW(ddp.parameters(), lr=LR)
    kind = entry["kind"]
    # fp32_reference: the declared numerical reference recipe (fp32 params AND fp32 compute, no autocast).
    use_autocast = not spec.get("fp32_reference", False) and not native_bf16_params(spec)
    diag = NumericalDiagnostics(spec, "ddp", ddp.module, opt, kind) if diagnostics_enabled(spec) else None
    if diag is not None:
        ddp.register_comm_hook(None, ddp_recording_comm_hook(diag))
    umap = native_unit_map(ddp.module, opt)
    units = umap["units"]
    ledger = BoundaryLedger("ddp", spec.get("numerical_checks", False), "in_step",
                            {k: local_numel(p) for k, p in units.items()} if not umap["problems"] else None,
                            calls_per_boundary=1,
                            multiplicity_basis="native torch optimizer: step() called once per logical update")
    plan = native_sample_plan(umap) if spec.get("numerical_checks") else None
    ledger.layout = {"unit_map": umap["summary"], "tied_aliases": umap["tied_aliases"], "problems": umap["problems"],
                     "sample_plan": plan_record(plan) if plan else None}
    ref = reference_session(spec)

    def step(micro_batches):
        ledger.begin_step()
        if diag is not None:
            diag.before_logical_step()
        n, raw = len(micro_batches), []
        for i, b in enumerate(micro_batches):
            sync_ctx = ddp.no_sync() if i < n - 1 else contextlib.nullcontext()
            with sync_ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
                    loss = loss_fn(ddp, b, kind)
                    if diag is not None and i == 0 and diag.fault_now("nonfinite_loss"):
                        loss = diag.poison(loss)
                (loss / n).backward()
            raw.append(loss.detach())
        ledger.record_microbatch_losses(raw)
        if ledger.gradient_checks:  # all-reduced gradients, inspected whether or not the optimizer is called
            ledger.observe_units(scan_units(units), "in-step after final backward, before the optimizer call")
        if ref is not None and ledger.phase == "measured":
            ref.at_boundary(ledger.current_attempt(), native_reference_items(umap))
        if first_boundary(spec):              # Candidate 14: gradient observed at the boundary; no optimizer call
            ledger.end_step(False, "first-boundary result kind: optimizer intentionally not called",
                            extra={"intended_updates": 0, "optimizer_calls": audit["optimizer_calls"]})
            return torch.stack(raw).float().mean()
        calls_before = audit["optimizer_calls"]
        if not (diag is not None and diag.fault_now("skip_optimizer_step")):
            opt.step()
        opt.zero_grad(set_to_none=True)
        ledger.end_step(audit["optimizer_calls"] > calls_before, "native optimizer post-hook (step() returned)")
        if diag is not None:
            diag.after_logical_step()
        # Reported loss: mean of the unscaled micro-batch losses (same definition for every framework).
        return torch.stack(raw).float().mean()

    audit, dtypes = observe_optimizer(opt, ddp, spec)
    attach_boundary_hooks(opt, ledger, sampler=AdamWCallSampler(plan) if plan is not None else None)
    return Trainer(step=step, n_params=n_params, update_audit=audit, diag=diag, canonical=canonical, ledger=ledger,
                   reference=ref, theta0=theta0, final_items=lambda: native_final_items(umap, opt),
                   state_fingerprint=lambda: gathered_state_fingerprints("ddp", ddp.module, deepspeed=False),
                   info={"precision": "fp32 params + autocast bf16" if use_autocast else "fp32 params + fp32 compute (reference)",
                         "live_parameter_dtypes": dtypes, "initialization_equivalence_verified": False})


# ------------------------------------------------------------------------------------------------ FSDP2

def setup_fsdp(spec, entry, fw, device) -> Trainer:
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    dtypes = {"bf16": torch.bfloat16, "fp32": torch.float32}
    mp = MixedPrecisionPolicy(param_dtype=dtypes[fw["param_dtype"]], reduce_dtype=dtypes[fw["grad_reduce_dtype"]])

    # Count parameters on the meta device to decide where to build.
    probe = _build(entry, {}, "meta", torch.float32)
    n_params = _count_params(probe)
    use_meta = n_params * 4 / GIB > fw["meta_init_threshold_gib"]
    if use_meta and has_tied_params(probe):
        raise RuntimeError("meta init path does not support tied embeddings")
    canonical = None
    theta0 = None
    if spec.get("canonical_init"):
        if use_meta:
            raise RuntimeError("canonical_init is not implemented for the meta-device (8B) path; "
                               "those runs remain complete-recipe comparisons")
        model, canonical = canonical_model(entry, spec, _build)
        theta0 = canonical_theta0(model, spec)
        model = model.to(device)
    else:
        model = probe if use_meta else _build(entry, spec, device, torch.float32)
    if use_meta:
        _enable_ckpt(model, spec)
    del probe

    for unit in shard_units(model):
        fully_shard(unit, mp_policy=mp, reshard_after_forward=fw["reshard_after_forward"])
    fully_shard(model, mp_policy=mp, reshard_after_forward=fw["reshard_after_forward"])
    if use_meta:
        materialize_meta_model(model, device, seed=spec["seed"])

    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    kind = entry["kind"]
    diag = NumericalDiagnostics(spec, "fsdp", model, opt, kind) if diagnostics_enabled(spec) else None
    umap = native_unit_map(model, opt)
    units = umap["units"]
    ledger = BoundaryLedger("fsdp", spec.get("numerical_checks", False), "in_step",
                            {k: local_numel(p) for k, p in units.items()} if not umap["problems"] else None,
                            calls_per_boundary=1,
                            multiplicity_basis="native torch optimizer: step() called once per logical update")
    plan = native_sample_plan(umap) if spec.get("numerical_checks") else None
    ledger.layout = {"unit_map": umap["summary"], "tied_aliases": umap["tied_aliases"], "problems": umap["problems"],
                     "sample_plan": plan_record(plan) if plan else None}
    ref = reference_session(spec)

    def step(micro_batches):
        ledger.begin_step()
        if diag is not None:
            diag.before_logical_step()
        n, raw = len(micro_batches), []
        for i, b in enumerate(micro_batches):
            model.set_requires_gradient_sync(i == n - 1)
            loss = loss_fn(model, b, kind)
            if diag is not None and i == 0 and diag.fault_now("nonfinite_loss"):
                loss = diag.poison(loss)
            (loss / n).backward()
            raw.append(loss.detach())
        ledger.record_microbatch_losses(raw)
        if ref is not None and ledger.phase == "measured":
            ref.at_boundary(ledger.current_attempt(), native_reference_items(umap))
        if ledger.gradient_checks:  # this rank's reduce-scattered shards; every rank reports its own
            ledger.observe_units(scan_units(units), "in-step after final backward, before the optimizer call "
                                                    "(local DTensor shards)")
        if first_boundary(spec):              # Candidate 14: no optimizer call
            ledger.end_step(False, "first-boundary result kind: optimizer intentionally not called",
                            extra={"intended_updates": 0, "optimizer_calls": audit["optimizer_calls"]})
            return torch.stack(raw).float().mean()
        calls_before = audit["optimizer_calls"]
        if not (diag is not None and diag.fault_now("skip_optimizer_step")):
            opt.step()
        opt.zero_grad(set_to_none=True)
        ledger.end_step(audit["optimizer_calls"] > calls_before, "native optimizer post-hook (step() returned)")
        if diag is not None:
            diag.after_logical_step()
        return torch.stack(raw).float().mean()

    audit, dtypes = observe_optimizer(opt, model, spec)
    attach_boundary_hooks(opt, ledger, sampler=AdamWCallSampler(plan) if plan is not None else None)
    return Trainer(step=step, n_params=n_params, update_audit=audit, diag=diag, canonical=canonical, ledger=ledger,
                   reference=ref, theta0=theta0, final_items=lambda: native_final_items(umap, opt),
                   state_fingerprint=lambda: gathered_state_fingerprints("fsdp", model, deepspeed=False),
                   extra_diag=lambda: {"fsdp_reduce_dtypes_live": fsdp_reduce_dtypes(model),
                                       "fsdp_mp_policy": {"param_dtype": str(mp.param_dtype),
                                                          "reduce_dtype": str(mp.reduce_dtype)}},
                   info={"precision": f"{fw['param_dtype']} compute, fp32 sharded params, "
                                      f"{fw['grad_reduce_dtype']} reduce-scatter",
                         "meta_init": use_meta, "fsdp_units": len(shard_units(model)) + 1,
                         "live_parameter_dtypes": dtypes, "initialization_equivalence_verified": False})


# ----------------------------------------------------------------------------------------- DeepSpeed ZeRO

def setup_deepspeed(spec, entry, fw, device) -> Trainer:
    import deepspeed

    world = dist.get_world_size()
    cfg = copy.deepcopy(fw["deepspeed"])
    cfg["train_micro_batch_size_per_gpu"] = spec["micro_batch"]
    cfg["gradient_accumulation_steps"] = spec.get("grad_accum", 1)
    cfg["train_batch_size"] = spec["micro_batch"] * spec.get("grad_accum", 1) * world

    # ds_precision "bf16" (default): DeepSpeed bf16 mode - bf16 weights, fp32 master copy in the ZeRO optimizer.
    # ds_precision "autocast": fp32 weights + torch.autocast(bf16) inside the engine (DeepSpeed torch_autocast),
    # i.e. the same recipe as DDP; used for ResNet, where bf16 BatchNorm weights fall off the cuDNN fast path.
    if spec.get("ds_grad_accum_dtype"):        # G5: data_types.grad_accum_dtype (DeepSpeed 0.19.7 constants.py)
        cfg["data_types"] = {"grad_accum_dtype": spec["ds_grad_accum_dtype"]}
    precision = spec.get("ds_precision", "bf16")
    if precision == "autocast":
        cfg.pop("bf16", None)
        cfg["torch_autocast"] = {"enabled": True, "dtype": "bfloat16"}
        build_dtype, data_dtype = torch.float32, torch.float32
    else:
        # LMs are built directly in bf16 (bf16 mode would cast an fp32 model anyway), halving the transient
        # memory of the 8B model. ResNet is small, so it is built in fp32 and cast by DeepSpeed.
        build_dtype = torch.bfloat16 if entry["kind"] == "causal_lm" else torch.float32
        data_dtype = torch.bfloat16
    canonical = None
    theta0 = None
    if spec.get("canonical_init"):
        model, canonical = canonical_model(entry, spec, _build)
        theta0 = canonical_theta0(model, spec)
        model = model.to(build_dtype).to(device)     # the bf16 recipe stores bf16 weights (DeepSpeed would cast)
    else:
        model = _build(entry, spec, device, build_dtype)
    n_params = _count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    engine, _, _, _ = deepspeed.initialize(model=model, optimizer=opt, config=cfg)
    kind = entry["kind"]
    diag = NumericalDiagnostics(spec, spec["framework"], engine.module, opt, kind) if diagnostics_enabled(spec) else None
    stage = cfg["zero_optimization"]["stage"]
    # mgbench can inspect DeepSpeed gradients only inside the base optimizer call. Expected-coverage accounting is
    # built only for the installed ZeRO-1/2 class whose lifecycle was read from its source: step() calls
    # _optimizer_step(i) once per bit16 group, restricting param_groups to [group i] whose single parameter is the
    # fp32 partition with partition_size[i] elements (overflow returns before any call). The identity of every
    # partition is verified here; any other class (BF16_Optimizer, ZeRO-3) gets no accounting -> unverified.
    ds_optimizer_class = type(engine.optimizer).__name__
    ds_impl = deepspeed_zero12_layout(engine, opt) if stage in (1, 2) and ds_optimizer_class == "DeepSpeedZeroOptimizer" \
        else {"accounting": None, "reason": f"no accounting for {ds_optimizer_class} (stage {stage})"}
    ledger = BoundaryLedger(spec["framework"], spec.get("numerical_checks", False), "base_optimizer_pre_hook",
                            ds_impl["accounting"], calls_per_boundary=ds_impl.get("calls_per_boundary"),
                            multiplicity_basis=ds_impl.get("basis", ds_impl.get("reason")))
    partition_key = {id(t): k for k, t in ds_impl.get("partitions", {}).items()}

    def ds_units(optimizer):
        units = {}
        for g in optimizer.param_groups:
            for t in g["params"]:
                units[partition_key.get(id(t), ("unexpected", id(t)))] = t
        return scan_units(units)

    def step(micro_batches):
        ledger.begin_step()
        if diag is not None:
            diag.before_logical_step()
        before_steps, before_skips = int(engine.global_steps), int(engine.skipped_steps)
        n, raw = len(micro_batches), []
        flags = []
        if first_boundary(spec) and (not engine.managed_gradient_accumulation()
                                     or engine.gradient_accumulation_steps() != n):
            raise RuntimeError("first-boundary needs managed gradient accumulation with GA == microbatches per boundary")
        for i, b in enumerate(micro_batches):
            if first_boundary(spec):      # Candidate 15: the engine's own predicate must mark ONLY the last microbatch
                flags.append(bool(engine.is_gradient_accumulation_boundary()))
                if flags[-1] != (i == n - 1):
                    raise RuntimeError(f"engine boundary predicate {flags} disagrees with microbatch {i} of {n}")
            loss = loss_fn(engine, b, kind)
            if diag is not None and i == 0 and diag.fault_now("nonfinite_loss"):
                loss = diag.poison(loss)
            engine.backward(loss)
            if (diag is not None and i == n - 1 and diag.measure_index == 0 and spec.get("reference_capture")
                    and not diag.reference["grad_first_update"]):
                diag.capture_reference_grads_deepspeed()   # reduced gradients, before the optimizer step
            raw.append(loss.detach())
            if first_boundary(spec) and i == n - 1:
                break                         # the boundary backward reduced the gradients; engine.step() NOT called
            engine.step()
        ledger.record_microbatch_losses(raw)
        if first_boundary(spec):
            g32 = deepspeed_boundary_gradient(engine.optimizer)
            ledger.observe_units(scan_tensors({"group0": g32}),         # Candidate 20: g32 IS the gradient
                                 "first-boundary extraction: step()'s flattening, no step")
            if ref is not None:
                ref.at_boundary(ledger.current_attempt(), flat_slices(g32, slices))
            del g32
            ledger.end_step(False, "first-boundary result kind: engine.step() intentionally not called at the boundary",
                            framework_reported_skip=None,
                            extra={"intended_updates": 0, "engine_global_steps": int(engine.global_steps),
                                   "engine_micro_steps": int(getattr(engine, "micro_steps", -1)),
                                   "boundary_flags_before_backward": flags})
            return torch.stack(raw).float().mean()
        # ZeRO-3 may call the base optimizer once per parameter subgroup.
        # Those calls are retained, but are NOT the logical update count.
        boundaries = int(engine.global_steps) - before_steps
        skipped = int(engine.skipped_steps) - before_skips
        audit["engine_update_boundaries"] += boundaries
        audit["skipped_updates"] += skipped
        audit["successful_updates"] += boundaries - skipped
        # One attempted step = one accumulation boundary. More than one cannot be attributed: no evidence either way.
        logical = (boundaries == 1 and skipped == 0) if boundaries <= 1 else None
        extra = None
        if sampler is not None and ds_impl["accounting"] is not None and logical:
            extra = {"module_bf16_vs_master": deepspeed_bf16_consistency(engine, ds_impl, sampler)}
        ledger.end_step(logical, "DeepSpeed engine global_steps / skipped_steps", framework_reported_skip=skipped > 0,
                        extra=extra)
        if diag is not None:
            diag.after_logical_step()
        return torch.stack(raw).float().mean()

    desc = (f"DeepSpeed bf16 (bf16 params, fp32 master in ZeRO-{stage} optimizer)" if precision == "bf16"
            else "DeepSpeed torch_autocast (fp32 params + autocast bf16)")
    audit, dtypes = observe_optimizer(opt, model, spec)
    plan = build_plan([(k, t) for k, t in ds_impl.get("partitions", {}).items()]) \
        if spec.get("numerical_checks") and ds_impl["accounting"] is not None else None
    sampler = AdamWCallSampler(plan) if plan is not None else None
    ledger.layout = {**{k: v for k, v in ds_impl.items() if k != "partitions"},
                     "sample_plan": plan_record(plan) if plan else None}
    ref = reference_session(spec)
    if ref is not None and (ds_impl["accounting"] is None or ds_impl.get("groups") != 1
                            or not ds_impl.get("param_slices_storage_verified")):
        raise RuntimeError("reference mode needs identity-verified accounting with one bit16 group and storage-verified "
                           f"parameter slices: {ds_impl.get('reason')} {ds_impl.get('param_slices_unverified')}")
    slices = ds_impl.get("param_slices", [])
    zo = engine.optimizer

    def ds_ref_in_hook(optimizer):
        if ref is None or ledger.phase != "measured":
            return
        params = [t for g in optimizer.param_groups for t in g["params"]]
        part = ds_impl["partitions"]["group0"]
        if len(params) != 1 or params[0] is not part or part.grad is None:
            raise RuntimeError("DeepSpeed base call does not hold exactly the group-0 fp32 partition with a gradient")
        ref.at_boundary(ledger.current_attempt(), flat_slices(part.grad, slices))

    def ds_final_items():
        part = zo.single_partition_of_fp32_groups[0]
        st = opt.state[part]
        return [(n, t, m, v) for (n, t), (_, m), (_, v) in zip(flat_slices(part.detach(), slices),
                                                               flat_slices(st["exp_avg"], slices),
                                                               flat_slices(st["exp_avg_sq"], slices))]

    def ds_final_checks():
        return {"module_bf16_vs_master_full": module_vs_master_full(zo.single_partition_of_fp32_groups[0],
                                                                    engine.module, slices)}

    attach_boundary_hooks(opt, ledger, units_in_hook=ds_units, sampler=sampler, ref_in_hook=ds_ref_in_hook)
    audit["logical_updates_from_engine"] = True
    audit["deepspeed_optimizer_class"] = ds_optimizer_class
    audit["deepspeed_layout"] = {k: v for k, v in ds_impl.items() if k != "partitions"}
    return Trainer(step=step, n_params=n_params, data_dtype=data_dtype, update_audit=audit, diag=diag,
                   canonical=canonical, ledger=ledger, reference=ref, theta0=theta0,
                   final_items=ds_final_items if ds_impl["accounting"] is not None else None,
                   final_checks=ds_final_checks if ds_impl["accounting"] is not None else None,
                   state_fingerprint=lambda: gathered_state_fingerprints(spec["framework"], engine.module, deepspeed=True),
                   extra_diag=lambda: {"deepspeed_communication_data_type": str(engine.communication_data_type),
                                       "deepspeed_model_and_grad_accum_dtypes": [str(x) for x in engine.get_data_types()],
                                       "deepspeed_bf16_enabled": bool(engine.bfloat16_enabled()),
                                       "deepspeed_check_grad_overflow": bool(getattr(engine.optimizer, "check_grad_overflow", False))},
                   info={"precision": desc,
                         "zero_stage": stage, "communication_data_type": cfg.get("communication_data_type"),
                         "live_parameter_dtypes": dtypes, "initialization_equivalence_verified": False})


def deepspeed_boundary_gradient(zo):
    """The fp32 gradient partition exactly as DeepSpeedZeroOptimizer.step() (0.19.7, stage_1_and_2.py:2560-2578) builds
    it from averaged_gradients after the boundary reduction: the pre-flattened partition if present, else
    flatten_dense_tensors_aligned(averaged_gradients[0], partition_size), cast to the fp32 master dtype. With
    clip_grad == 0 and loss_scale == 1, unscale_and_clip_grads() returns without changing it, so no transform is
    applied here (never scaled twice, never silently omitted)."""
    ag = zo.averaged_gradients
    grads0 = ag.get(0) if isinstance(ag, dict) else (ag[0] if len(ag) else None)
    if len(zo.bit16_groups) != 1 or grads0 is None:
        raise RuntimeError("first-boundary extraction needs one bit16 group with reduced (averaged) gradients")
    if float(getattr(zo, "clip_grad", 0.0)) != 0.0 or float(getattr(zo, "loss_scale", 1.0)) != 1.0:
        raise RuntimeError("first-boundary extraction assumes clip_grad 0 and loss_scale 1 (identity unscale/clip)")
    flat = zo._get_preflattened_grad_partition(0)
    if flat is None:
        flat = zo.flatten_dense_tensors_aligned(zo.averaged_gradients[0], int(zo.partition_size[0]))
    if flat.numel() != int(zo.partition_size[0]):
        raise RuntimeError(f"boundary gradient has {flat.numel()} elements, partition_size {zo.partition_size[0]}")
    return flat.to(zo.single_partition_of_fp32_groups[0].dtype)


def flat_slices(flat, slices) -> list:
    """[(canonical name, view)] of a flat DeepSpeed group buffer (gradient, master or moment), one view per model
    parameter in DeepSpeed's flattening order (`param_slices`); trailing alignment padding is not part of any view."""
    flat = flat.reshape(-1)
    return [(n, flat[a:a + k].view(shape)) for n, _, a, k, shape in slices]


def module_vs_master_full(master, module, slices) -> dict:
    """Full (every element) check that each low-precision module parameter equals its fp32 master slice cast to the
    module dtype (DeepSpeed bf16: the module must hold exactly bf16(master) after the copy-back)."""
    named = dict(module.named_parameters())
    mism, total, worst = 0, 0, None
    for n, view in flat_slices(master.detach(), slices):
        mod = named[n].detach()
        bad = int((view.to(mod.dtype) != mod).sum())
        if bad and worst is None:
            worst = n
        mism += bad
        total += view.numel()
    return {"compared": total, "mismatched": mism, "first_mismatched_tensor": worst, "pass": total > 0 and mism == 0,
            "coverage": "full (every element of every parameter)"}


def deepspeed_zero12_layout(engine, opt) -> dict:
    """Expected per-boundary gradient units for DeepSpeedZeroOptimizer (ZeRO-1/2), verified by identity."""
    import hashlib
    import inspect

    zo = engine.optimizer
    src = inspect.getsourcefile(type(zo))
    out = {"optimizer_class": f"{type(zo).__module__}.{type(zo).__name__}", "source_file": src,
           "source_sha256": hashlib.sha256(open(src, "rb").read()).hexdigest(),
           "groups": len(zo.bit16_groups), "partition_size": [int(x) for x in zo.partition_size],
           "check_grad_overflow": bool(getattr(zo, "check_grad_overflow", False)),
           "clip_grad": float(getattr(zo, "clip_grad", 0.0)), "loss_scale": float(getattr(zo, "loss_scale", 1.0)),
           "partition_id": [int(dist.get_rank(group=g)) for g in zo.real_dp_process_group]}
    partitions, problems = {}, []
    # Independent mapping: every trainable MODEL parameter (incl. tied aliases) -> exactly one bit16 group;
    # every bit16 parameter is a model parameter; the flat group = sum of member elements + alignment padding.
    group_of, model_names = {}, {}
    for i, grp in enumerate(zo.bit16_groups):
        for p in grp:
            group_of.setdefault(id(p), []).append(i)
    for name, p in engine.module.named_parameters(remove_duplicate=False):
        model_names.setdefault(id(p), []).append((name, p))
    groups_detail = [{"params": 0, "elements": 0, "names": 0} for _ in zo.bit16_groups]
    for pid, entries in model_names.items():
        p = entries[0][1]
        if not p.requires_grad:
            continue
        gs = group_of.get(pid, [])
        if len(gs) != 1:
            problems.append(f"model parameter {entries[0][0]} is in {len(gs)} bit16 groups")
            continue
        groups_detail[gs[0]]["params"] += 1
        groups_detail[gs[0]]["elements"] += p.numel()
        groups_detail[gs[0]]["names"] += len(entries)
    stray = [pid for pid in group_of if pid not in model_names]
    if stray:
        problems.append(f"{len(stray)} bit16 parameter(s) are not model parameters")
    align = int(getattr(zo, "nccl_start_alignment_factor", 1))
    for i, grp in enumerate(zo.bit16_groups):
        world_i = int(dist.get_world_size(group=zo.real_dp_process_group[i]))
        flat = int(zo.partition_size[i]) * world_i
        pad = flat - groups_detail[i]["elements"]
        groups_detail[i].update(dp_world=world_i, flat_elements=flat, padding=pad,
                                flat_buffer_elements=int(zo.bit16_groups_flat[i].numel()),
                                alignment_elements=align * world_i)
        if not 0 <= pad < max(1, align * world_i) or zo.bit16_groups_flat[i].numel() != flat:
            problems.append(f"group {i}: flat {flat} vs members {groups_detail[i]['elements']} + padding {pad} "
                            f"(buffer {zo.bit16_groups_flat[i].numel()}, alignment {align * world_i})")
    out["model_to_partition_map"] = groups_detail
    # Per-parameter slices of each flat group in DeepSpeed's own flattening order and offsets (round_robin_bit16_groups
    # / round_robin_bit16_offsets when present; per-parameter alignment padding is possible in this DeepSpeed version).
    # Each slice is verified by storage identity: the live bit16 parameter must be a view of the flat buffer at exactly
    # that offset. At world size 1 the fp32 partition, its gradient and its moments share this layout.
    first_name = {pid: entries[0][0] for pid, entries in model_names.items()}
    order = getattr(zo, "round_robin_bit16_groups", None) or zo.bit16_groups
    offsets = getattr(zo, "round_robin_bit16_offsets", None)
    param_slices, unverified = [], []
    for i, grp in enumerate(order):
        start, flat = 0, zo.bit16_groups_flat[i]
        for j, p in enumerate(grp):
            off = int(offsets[i][j]) if offsets is not None else start
            name = first_name.get(id(p), f"<unnamed {id(p)}>")
            param_slices.append((name, i, off, p.numel(), tuple(p.shape)))
            if p.data_ptr() != flat.data_ptr() + off * flat.element_size() or off + p.numel() > flat.numel():
                unverified.append(name)
            start = off + p.numel()
    out["param_slices"] = param_slices
    out["param_slices_basis"] = ("round_robin_bit16_groups/offsets" if offsets is not None else "bit16_groups, cumulative") + \
        "; each slice checked by storage identity (param.data_ptr == flat.data_ptr + offset x element size)"
    out["param_slices_unverified"] = unverified[:10]
    out["param_slices_storage_verified"] = bool(param_slices) and not unverified
    for i, part in enumerate(zo.single_partition_of_fp32_groups):
        if [id(t) for t in opt.param_groups[i]["params"]] != [id(part)]:
            problems.append(f"base param group {i} is not exactly the fp32 partition {i}")
        if part.numel() != int(zo.partition_size[i]):
            problems.append(f"partition {i}: {part.numel()} elements != partition_size {zo.partition_size[i]}")
        partitions[f"group{i}"] = part
    out["partitions"] = partitions
    if problems or len(opt.param_groups) != len(zo.bit16_groups):
        out.update(accounting=None, reason="; ".join(problems) or "param group count mismatch")
        return out
    out.update(accounting={k: int(t.numel()) for k, t in partitions.items()}, calls_per_boundary=len(zo.bit16_groups),
               basis=f"{src} (sha256 {out['source_sha256'][:16]}): step() -> _optimizer_step(i) once per bit16 group "
                     f"({len(zo.bit16_groups)}), each call restricted to the group's fp32 partition")
    return out


def deepspeed_bf16_consistency(engine, layout: dict, sampler) -> dict:
    """After engine.step(): the bf16 partition must equal the bf16-rounded fp32 master at the sampled indices of the
    last base calls (DeepSpeed copies master -> bf16 after each _optimizer_step)."""
    zo = engine.optimizer
    out = {"compared": 0, "mismatched": 0, "coverage": "sampled"}
    recent = sampler.calls[-layout["groups"]:]
    for gi, call in enumerate(recent):
        part = zo.single_partition_of_fp32_groups[gi]
        bit16 = zo.parallel_partitioned_bit16_groups[gi][layout["partition_id"][gi]]
        for chk in call.get("checks", []):
            if not chk.get("idx"):
                continue
            idx = torch.tensor(chk["idx"], dtype=torch.long)
            master = part.detach().reshape(-1).index_select(0, idx.to(part.device))
            low = bit16.detach().reshape(-1).index_select(0, idx.to(bit16.device))
            out["compared"] += int(idx.numel())
            out["mismatched"] += int((master.to(low.dtype) != low).sum().item())
    out["pass"] = out["compared"] > 0 and out["mismatched"] == 0
    return out


SETUP = {"ddp": setup_ddp, "fsdp": setup_fsdp, "zero1": setup_deepspeed, "zero2": setup_deepspeed,
         "zero3": setup_deepspeed}
