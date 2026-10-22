"""Numerical diagnostics for scientific-v2 validation (added on the A6000 node, session 20261014a).

Everything here is active only for runs with ``numerical_checks: true`` (or a canonical-state / reference / fault
option). Performance runs never construct these objects, so their timings are unaffected.

What is recorded (findings A08, A10, A16):
  * live dtypes after optimizer-state initialisation: module parameters (local shards), gradients, optimizer
    master parameters and their gradients, optimizer state tensors, compute (forward-hook) and reduction dtypes;
  * per-update evidence that an update happened: representative optimizer-parameter samples change and the AdamW
    ``step`` state advances on every base-optimizer call; module parameters change on every logical step;
  * canonical initial state: one CPU fp32 model built from the model seed, its fingerprint, and the fingerprint of
    the framework's full logical state gathered *after* wrapping (not inferred from the seed);
  * reference tensors: fixed slices of full gradients at the first measured update and of parameters at the end;
  * isolated fault injection to prove that a skipped / non-finite update is detected.
"""
from __future__ import annotations

import hashlib
import math

import torch
import torch.distributed as dist

SAMPLE_ELEMS = 65536          # elements sampled per representative tensor for change detection
REFERENCE_ELEMS = 262144      # leading elements saved per reference tensor (bounded file size)


# --------------------------------------------------------------------------------------------- helpers

def local_tensor(t):
    """The locally stored data of a parameter: DTensor local shard, ZeRO-3 partition, or the tensor itself."""
    if t is None:
        return None
    if getattr(t, "ds_tensor", None) is not None:      # DeepSpeed ZeRO-3 partitioned parameter
        return t.ds_tensor
    return t.to_local() if hasattr(t, "to_local") else t


def dtype_counts(tensors) -> dict:
    out: dict[str, int] = {}
    for t in tensors:
        loc = local_tensor(t)
        if loc is None:
            continue
        out[str(loc.dtype)] = out.get(str(loc.dtype), 0) + loc.numel()
    return out


class Fingerprint:
    """Order-sensitive SHA-256 over (name, dtype, shape, bytes) of a sequence of tensors."""

    def __init__(self):
        self.h = hashlib.sha256()
        self.n = 0

    def add(self, name: str, tensor: torch.Tensor):
        t = tensor.detach().contiguous().cpu()
        self.h.update(f"{name}|{t.dtype}|{tuple(t.shape)}|".encode())
        # reshape(-1): 0-dim tensors (e.g. BatchNorm num_batches_tracked) cannot be byte-viewed directly.
        self.h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes() if t.numel() else b"")
        self.n += 1

    def hexdigest(self) -> str:
        return self.h.hexdigest()


def fingerprint_state(items, cast=None) -> dict:
    """items: iterable of (name, tensor) in a fixed order. cast: optional dtype applied before hashing."""
    fp = Fingerprint()
    for name, t in items:
        fp.add(name, t.to(cast) if cast is not None and t.is_floating_point() else t)
    return {"sha256": fp.hexdigest(), "tensors": fp.n}


def representative_names(model, kind: str) -> list[str]:
    names = [n for n, _ in model.named_parameters()]
    if kind == "causal_lm":
        layers = sorted({int(n.split(".")[2]) for n in names if n.startswith("model.layers.")})
        last = layers[-1]
        want = ["model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.down_proj.weight",
                f"model.layers.{last}.mlp.up_proj.weight", "model.norm.weight"]
    else:
        want = ["conv1.weight", "layer1.0.conv1.weight", "layer4.2.conv3.weight", "fc.weight"]
    return [w for w in want if w in names]


def slice_flat(t: torch.Tensor) -> torch.Tensor:
    return t.detach().float().reshape(-1)[:REFERENCE_ELEMS].cpu().clone()


def _change(before: torch.Tensor, after: torch.Tensor) -> dict:
    """Element changes between two samples. Non-finite values are counted separately: NaN - NaN is not > 0, so a
    NaN-poisoned parameter must not be reported as 'unchanged' (fix after the 20261014a fault diagnostics)."""
    b, a = before.float(), after.float()
    finite = torch.isfinite(a) & torch.isfinite(b)
    diff = (a - b).abs()
    return {"changed_elems": int(((diff > 0) & finite).sum().item()),
            "nonfinite_after": int((~torch.isfinite(a)).sum().item()),
            "sampled": int(a.numel()),
            "max_abs_delta": float(diff[finite].max().item()) if bool(finite.any()) else None}


# ------------------------------------------------------------------------------------------- recorder

class NumericalDiagnostics:
    """Attached to one rank's model/optimizer. ``phase``/``measure_index`` are set by the training loop."""

    def __init__(self, spec: dict, framework: str, module, opt, kind: str):
        self.spec = spec
        self.framework = framework
        self.module = module            # the underlying nn.Module (DDP: ddp.module, DS: engine.module)
        self.opt = opt                  # the *base* torch optimizer (DeepSpeed calls it internally)
        self.kind = kind
        self.measure_index = -1         # -1 during warmup
        self.fault = spec.get("diagnostic_fault") or {}
        self.record = {"compute": {}, "reduction": {}, "live_dtypes_after_first_update": None,
                       "optimizer_param_change_per_call": [], "optimizer_state_step_delta_per_call": [],
                       "module_param_change_per_step": [], "reduction_dtypes_seen": [],
                       "fault": self.fault or None, "fault_applied_at": None}
        self._pre_samples = None
        self._pre_steps = None
        self._module_samples = None
        self.rep_names = representative_names(module, kind)
        self.reference = {"names": self.rep_names, "grad_first_update": {}, "param_final": {},
                          "param_final_master_fp32": {}}
        self._hook_compute()
        opt.register_step_pre_hook(self._pre_step)
        opt.register_step_post_hook(self._post_step)

    # ---------------- compute dtype (forward hooks on the first and the last matmul/conv layer)
    def _hook_compute(self):
        mods = [(n, m) for n, m in self.module.named_modules()
                if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d))]
        if not mods:
            return
        for label, (name, mod) in (("first_layer", mods[0]), ("last_layer", mods[-1])):
            def hook(m, inputs, output, label=label, name=name):
                if self.measure_index == 0 and label not in self.record["compute"]:
                    x = inputs[0] if inputs else None
                    w = getattr(m, "weight", None)
                    self.record["compute"][label] = {
                        "module": name, "input_dtype": str(getattr(x, "dtype", None)),
                        "weight_dtype_at_forward": str(getattr(local_tensor(w), "dtype", None)),
                        "output_dtype": str(getattr(output, "dtype", None)),
                        "autocast_enabled": torch.is_autocast_enabled()}
            mod.register_forward_hook(hook)

    # ---------------- optimizer-call evidence
    def _opt_tensors(self):
        return [p for g in self.opt.param_groups for p in g["params"]]

    def _sample_targets(self):
        ps = [p for p in self._opt_tensors() if local_tensor(p) is not None and local_tensor(p).numel() > 0]
        if not ps:
            return []
        idx = sorted({0, len(ps) // 2, len(ps) - 1})
        return [ps[i] for i in idx]

    def _state_steps(self, params):
        out = []
        for p in params:
            st = self.opt.state.get(p, {})
            s = st.get("step")
            out.append(None if s is None else float(s.item() if torch.is_tensor(s) else s))
        return out

    def _pre_step(self, optimizer, args, kwargs):
        if self.measure_index < 0:
            return
        targets = self._sample_targets()
        self._pre_samples = [(p, local_tensor(p).detach().reshape(-1)[:SAMPLE_ELEMS].clone()) for p in targets]
        self._pre_steps = self._state_steps(targets)
        if (self.measure_index == 0 and self.spec.get("reference_capture") and self.framework in ("ddp", "fsdp")
                and not self.reference["grad_first_update"]):
            self.capture_reference_grads_from_module()

    def _post_step(self, optimizer, args, kwargs):
        if self.measure_index < 0 or self._pre_samples is None:
            return
        changes = []
        for p, before in self._pre_samples:
            after = local_tensor(p).detach().reshape(-1)[:SAMPLE_ELEMS]
            changes.append(_change(before, after))
        steps_after = self._state_steps([p for p, _ in self._pre_samples])
        self.record["optimizer_param_change_per_call"].append(
            {"measure_index": self.measure_index, "targets": changes})
        self.record["optimizer_state_step_delta_per_call"].append(
            {"measure_index": self.measure_index,
             "delta": [None if a is None or b is None else a - b for a, b in zip(steps_after, self._pre_steps)]})
        if self.record["live_dtypes_after_first_update"] is None:
            self.record["live_dtypes_after_first_update"] = self.live_dtypes()
        self._pre_samples = None

    def live_dtypes(self) -> dict:
        opt_params = self._opt_tensors()
        state_tensors: dict[str, dict] = {}
        for p in opt_params:
            for key, val in self.opt.state.get(p, {}).items():
                if torch.is_tensor(val):
                    d = state_tensors.setdefault(key, {})
                    loc = local_tensor(val)
                    d[str(loc.dtype)] = d.get(str(loc.dtype), 0) + loc.numel()
        return {
            "module_parameters": dtype_counts(self.module.parameters()),
            "module_gradients": dtype_counts(p.grad for p in self.module.parameters()),
            "optimizer_parameters": dtype_counts(opt_params),
            "optimizer_gradients": dtype_counts(p.grad for p in opt_params),
            "optimizer_state": state_tensors,
        }

    # ---------------- logical-step evidence on the module parameters
    def before_logical_step(self):
        if self.measure_index < 0:
            return
        names = dict(self.module.named_parameters())
        self._module_samples = []
        for n in self.rep_names:
            loc = local_tensor(names[n])
            if loc is not None and loc.numel():
                self._module_samples.append((n, names[n], loc.detach().reshape(-1)[:SAMPLE_ELEMS].clone()))

    def after_logical_step(self):
        if self.measure_index < 0 or self._module_samples is None:
            return
        out = []
        for n, p, before in self._module_samples:
            out.append({"name": n, **_change(before, local_tensor(p).detach().reshape(-1)[:SAMPLE_ELEMS])})
        self.record["module_param_change_per_step"].append({"measure_index": self.measure_index, "params": out})
        self._module_samples = None

    # ---------------- faults (isolated diagnostics only)
    def fault_now(self, kind: str) -> bool:
        hit = (self.fault.get("kind") == kind and self.measure_index == int(self.fault.get("at_measured_step", -1)))
        if hit and self.record["fault_applied_at"] is None:
            self.record["fault_applied_at"] = self.measure_index
        return hit

    def poison(self, loss: torch.Tensor) -> torch.Tensor:
        """Non-finite loss on rank 0 only, to test cross-rank propagation and detection."""
        if dist.get_rank() == 0:
            return loss * float("nan")
        return loss

    # ---------------- reference tensors (full logical tensors, gathered collectively)
    def _full(self, t):
        return t.full_tensor() if hasattr(t, "full_tensor") else t

    def capture_reference_grads_from_module(self):
        names = dict(self.module.named_parameters())
        for n in self.rep_names:
            g = names[n].grad
            if g is not None:
                self.reference["grad_first_update"][n] = slice_flat(self._full(g))

    def capture_reference_grads_deepspeed(self):
        from deepspeed.utils import safe_get_full_grad

        names = dict(self.module.named_parameters())
        for n in self.rep_names:
            g = safe_get_full_grad(names[n])       # collective for ZeRO partitions
            if g is not None:
                self.reference["grad_first_update"][n] = slice_flat(g)

    def capture_reference_params_final(self, deepspeed: bool):
        names = dict(self.module.named_parameters())
        for n in self.rep_names:
            p = names[n]
            if deepspeed:
                from deepspeed.utils import safe_get_full_fp32_param

                master = safe_get_full_fp32_param(p)
                if master is not None:
                    self.reference["param_final_master_fp32"][n] = slice_flat(master)
                if getattr(p, "ds_tensor", None) is None:
                    self.reference["param_final"][n] = slice_flat(p)
            else:
                self.reference["param_final"][n] = slice_flat(self._full(p))

    # ---------------- reduction dtype
    def note_reduction_dtype(self, dtype):
        s = str(dtype)
        if s not in self.record["reduction_dtypes_seen"]:
            self.record["reduction_dtypes_seen"].append(s)


def ddp_recording_comm_hook(diag: NumericalDiagnostics):
    """DDP comm hook that records the bucket dtype actually all-reduced, then runs PyTorch's default all-reduce."""
    from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook

    def hook(state, bucket):
        diag.note_reduction_dtype(bucket.buffer().dtype)
        return allreduce_hook(state, bucket)
    return hook


def fsdp_reduce_dtypes(model) -> dict:
    """Effective FSDP2 gradient-reduction dtypes, read from the live FSDPParam objects."""
    out = {}
    for m in model.modules():
        get_state = getattr(m, "_get_fsdp_state", None)
        if get_state is None:
            continue
        group = getattr(get_state(), "_fsdp_param_group", None)
        for fp in getattr(group, "fsdp_params", []) if group else []:
            red = getattr(fp, "reduce_dtype", None) or getattr(fp, "param_dtype", None) or getattr(fp, "orig_dtype", None)
            out[str(red)] = out.get(str(red), 0) + 1
    return out


# ------------------------------------------------------------------------------------- canonical state

def canonical_model(entry, spec, build_fn):
    """Build the canonical initial model on the CPU in fp32 from the model seed (CPU RNG is deterministic across
    processes). Every rank builds the same model; rank 0 records its fingerprint (fp32 and bf16-cast)."""
    torch.manual_seed(int(spec.get("model_seed", spec["seed"])))
    model = build_fn(entry, spec, "cpu", torch.float32)
    if spec.get("canonical_start") == "bf16_representable":
        # Declared common representable start (stage S1a): every trainable value is rounded to the nearest bf16 and
        # stored as fp32, so fp32-weight recipes, bf16-weight recipes and their fp32 masters can all start from the
        # SAME values. Tied parameters are one tensor (parameters() de-duplicates) and stay tied. Buffers unchanged.
        with torch.no_grad():
            for p in model.parameters():
                p.copy_(p.to(torch.bfloat16).to(torch.float32))
    record = None
    if dist.get_rank() == 0:
        items = sorted(model.state_dict().items())
        params = dict(model.named_parameters())
        record = {"model_seed": int(spec.get("model_seed", spec["seed"])),
                  "canonical_start": spec.get("canonical_start", "fp32 random init"),
                  "canonical_fp32": fingerprint_state(items),
                  "canonical_bf16": fingerprint_state(items, cast=torch.bfloat16),
                  **canonical_param_fingerprints(model),
                  "param_initial_slices": {n: slice_flat(params[n])
                                           for n in representative_names(model, entry["kind"])}}
    return model, record


def gathered_state_fingerprints(framework: str, module, deepspeed: bool) -> dict | None:
    """Fingerprint the framework's *full logical* state after wrapping. Collective: call on every rank."""
    sd = module.state_dict()
    items = []
    for name in sorted(sd):
        t = sd[name]
        t = t.full_tensor() if hasattr(t, "full_tensor") else t
        items.append((name, t))
    out = None
    if dist.get_rank() == 0:
        out = {"module_state": fingerprint_state(items),
               "module_state_dtypes": sorted({str(t.dtype) for _, t in items})}
    if deepspeed:
        from deepspeed.utils import safe_get_full_fp32_param

        masters = []
        for name, p in sorted(module.named_parameters()):
            m = safe_get_full_fp32_param(p)
            masters.append((name, m if m is not None else torch.empty(0)))
        if dist.get_rank() == 0:
            # Compared against the canonical state's named *parameters* (buffers excluded).
            out["optimizer_master_fp32_params"] = fingerprint_state(masters)
    return out


def canonical_param_fingerprints(canonical_cpu_model) -> dict:
    items = sorted(canonical_cpu_model.named_parameters())
    return {"params_fp32": fingerprint_state(items),
            "params_bf16_as_fp32": fingerprint_state((n, t.to(torch.bfloat16).float()) for n, t in items)}


def finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))
