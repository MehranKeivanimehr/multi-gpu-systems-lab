"""Instrumentation around the REAL DeepSpeed ZeRO optimizer path.

Scope and honesty
-----------------
This reads state out of a live `DeepSpeedZeroOptimizer` (stage 1/2) around the
base optimizer call. It is NOT a substitute model and NOT a hand-written mock
optimizer: every value is pulled from the engine's own attributes, named below
against DeepSpeed 0.15.1 source.

What the resulting check establishes, stated exactly:
  * **Optimizer-update agreement conditional on the observed gradient.** Given
    the post-unscale/post-clip gradient the base optimizer actually consumed,
    the master parameters and optimizer state moved by the amount the resolved
    AdamW semantics require.
  * It is **NOT** independent proof of gradient correctness: the gradient is
    taken as observed, not recomputed. Dropout is active (p=0.1) in the retained
    recipe, so an independent gradient recomputation would require capturing and
    replaying RNG state and is not attempted here.
  * It is **NOT** evidence of training quality, convergence, or of any
    historical run's validity. No historical row becomes certified by it.

Engine attributes used (DeepSpeed 0.15.1, stage_1_and_2.py
sha256 26ca67af1ea9b78ad1ce0ac787f2945bed6a9d62e1d0bbff6b5a7ba9daa25207):
  single_partition_of_fp32_groups  master FP32 partition this rank owns, per group
  averaged_gradients               reduced gradients per group (stage 2)
  bit16_groups / fp16_groups       the module parameters flattened per group
  partition_size                   per-group partition element count
  groups_padding                   per-group padding added to the last partition
  optimizer                        the wrapped base optimizer (torch AdamW)
  overflow                         the engine's own overflow flag

Refusals, never fallbacks: an unsupported layout, an unavailable attribute, an
empty or partial observation, or a storage-correspondence failure produces a
REFUSAL. There is no mechanics-only path that can be labelled certification.
"""
from __future__ import annotations

import hashlib
from typing import Any

SUPPORTED_ZERO_STAGES = (1, 2)


class ProbeRefusal(RuntimeError):
    """Evidence is unavailable or the layout is unsupported. Never a pass."""


def _req(obj, name: str):
    if not hasattr(obj, name):
        raise ProbeRefusal(
            f"the optimizer does not expose {name!r}; this layout is not "
            f"supported and no inference is attempted")
    return getattr(obj, name)


def _storage_key(t) -> int:
    return (t.untyped_storage().data_ptr() if hasattr(t, "untyped_storage")
            else t.storage().data_ptr())


def describe_optimizer(zopt) -> dict:
    """Resolve the engine's layout WITHOUT assuming any of it."""
    base = _req(zopt, "optimizer")
    masters = _req(zopt, "single_partition_of_fp32_groups")
    if not isinstance(masters, (list, tuple)) or not masters:
        raise ProbeRefusal("single_partition_of_fp32_groups is empty: nothing to verify")
    part_sizes = getattr(zopt, "partition_size", None)
    padding = getattr(zopt, "groups_padding", None)
    bit16 = (getattr(zopt, "bit16_groups", None)
             or getattr(zopt, "fp16_groups", None))
    out = {
        "zero_optimizer_class": type(zopt).__name__,
        "zero_optimizer_module": type(zopt).__module__,
        "base_optimizer_class": type(base).__name__,
        "base_optimizer_module": type(base).__module__,
        "n_groups": len(masters),
        "master_numels": [int(m.numel()) for m in masters],
        "master_dtypes": [str(m.dtype) for m in masters],
        "partition_size": [int(x) for x in part_sizes] if part_sizes is not None else None,
        "groups_padding": [int(x) for x in padding] if padding is not None else None,
        "bit16_group_sizes": ([[int(p.numel()) for p in g] for g in bit16]
                              if bit16 is not None else None),
        "overflow_flag_present": hasattr(zopt, "overflow"),
    }
    if out["partition_size"] is None:
        raise ProbeRefusal("partition_size is unavailable; partition identities "
                           "cannot be reconciled from lengths alone")
    if out["groups_padding"] is None:
        raise ProbeRefusal("groups_padding is unavailable; padding/offset "
                           "correctness cannot be established")
    return out


def expected_partition_map(zopt, *, rank: int, world_size: int) -> dict:
    """Enumerate the EXPECTED partition identities explicitly.

    Identity is (group_index, rank, offset, numel_without_padding). It is derived
    from the engine's own partition_size and groups_padding, never from equal
    lengths or matching norms.
    """
    d = describe_optimizer(zopt)
    ids = []
    for gi, (psize, pad) in enumerate(zip(d["partition_size"], d["groups_padding"])):
        offset = psize * rank
        real = psize - (pad if rank == world_size - 1 else 0)
        if real < 0:
            raise ProbeRefusal(
                f"group {gi}: padding {pad} exceeds partition size {psize}")
        ids.append({"group": gi, "rank": rank, "offset": offset,
                    "partition_numel": psize, "padding": pad,
                    "unpadded_numel": real})
    return {"expected_partitions": ids, "layout": d}


def verify_storage_correspondence(zopt, *, rank: int) -> dict:
    """Confirm master<->module correspondence by STORAGE and offset, not names.

    Equal lengths, equal norms and a matching optimizer class name are all
    insufficient; this checks that the base optimizer's parameter objects ARE
    the master partition tensors.
    """
    base = _req(zopt, "optimizer")
    masters = _req(zopt, "single_partition_of_fp32_groups")
    in_groups = [p for g in base.param_groups for p in g["params"]]
    mk = [_storage_key(m) for m in masters]
    gk = [_storage_key(p) for p in in_groups]
    missing = [i for i, k in enumerate(mk) if k not in gk]
    extra = [i for i, k in enumerate(gk) if k not in mk]
    ok = not missing and not extra and len(mk) == len(gk)
    return {
        "base_optimizer_sees_the_master_partitions": ok,
        "n_master_tensors": len(mk), "n_base_optimizer_params": len(gk),
        "masters_absent_from_base_optimizer": missing,
        "base_optimizer_params_not_masters": extra,
        "basis": "storage data_ptr identity, not lengths, norms or class names",
    }


DEFAULT_CHUNK_ELEMS = 4_194_304          # route-1 bound, shared by snapshot + oracle


def cpu_snapshot(t, *, name: str, chunk_elems: int = DEFAULT_CHUNK_ELEMS,
                 check_finite: bool = True):
    """An immutable, independently owned CPU copy of `t`, filled chunk by chunk.

    No full-size clone is ever made on the source device: the destination is
    allocated on the host first and each bounded slice is copied into it, so the
    transient on the source device is one chunk.

    `detach()` alone shares storage, and `to()` can return `self` when the
    device and dtype already match, so neither is sufficient to own the data.
    The destination here comes from `torch.empty(..., device="cpu")` and is
    filled with `copy_`, which always writes into that storage.

    Finite checking is done per chunk on the host copy, so no whole-partition
    boolean mask is ever allocated.
    """
    import torch
    if t is None:
        return None
    if int(chunk_elems) < 1:
        raise ProbeRefusal(f"chunk_elems must be >= 1, got {chunk_elems!r}")
    src = t.detach()
    if not src.is_contiguous():
        raise ProbeRefusal(
            f"{name}: source is not contiguous, so it cannot be sliced without "
            f"a whole-tensor copy on the source device; refusing rather than "
            f"allocating one")
    flat = src.view(-1)                      # view, not a copy
    n = int(flat.numel())
    out = torch.empty(n, dtype=src.dtype, device="cpu")
    for off in range(0, n, int(chunk_elems)):
        k = min(int(chunk_elems), n - off)
        dst = out.narrow(0, off, k)
        dst.copy_(flat.narrow(0, off, k))    # bounded D2H (or H2H) transfer
        if src.is_cuda:
            # copy_ into unpinned host memory is synchronous, but make the
            # completion explicit rather than relying on that.
            torch.cuda.synchronize(src.device)
        if check_finite and not bool(torch.isfinite(dst).all()):
            raise ProbeRefusal(
                f"{name}: non-finite value in elements "
                f"[{off}, {off + k}) of the observed state")
    snap = out.view(src.shape)
    snap.requires_grad_(False)
    return snap


def snapshot_meta(t, snap, *, name: str) -> dict:
    """Identity and placement metadata retained alongside a CPU snapshot."""
    if t is None:
        return {"name": name, "present": False}
    return {
        "name": name, "present": True,
        "source_device": str(t.device), "source_dtype": str(t.dtype),
        "source_data_ptr": int(t.data_ptr()),
        "snapshot_device": str(snap.device), "snapshot_dtype": str(snap.dtype),
        "snapshot_data_ptr": int(snap.data_ptr()),
        "independent_storage": int(t.data_ptr()) != int(snap.data_ptr()),
        "shape": list(t.shape), "numel": int(t.numel()),
        "requires_grad": bool(snap.requires_grad),
        "transfer": ("bounded per-chunk copy_ into host memory allocated "
                     "before the transfer; no full clone on the source device"),
        "cuda_transfer_unverified_on_cpu": True,
    }



def capture_boundary(zopt, *, rank: int, world_size: int, phase: str,
                     chunk_elems: int = DEFAULT_CHUNK_ELEMS) -> dict:
    """Snapshot one logical boundary. `phase` is 'before' or 'after' the step.

    The gradient captured is `master.grad`, which in stage 1/2 is the
    post-unscale/post-clip gradient the base optimizer consumes. Hook timing and
    observed-gradient semantics are unchanged; only where the copy lands and how
    it is transferred have changed.

    Snapshots are independent CPU tensors. Live parameters, gradients and
    optimizer state are read only and never mutated.
    """
    if phase not in ("before", "after"):
        raise ProbeRefusal(f"phase must be 'before' or 'after', got {phase!r}")
    base = _req(zopt, "optimizer")
    masters = _req(zopt, "single_partition_of_fp32_groups")
    snap = {"phase": phase, "rank": rank, "groups": [],
            "snapshot_placement": "cpu",
            "snapshot_chunk_elems": int(chunk_elems)}
    for gi, m in enumerate(masters):
        st = base.state.get(m, {})
        g = m.grad
        if phase == "before" and g is None:
            raise ProbeRefusal(
                f"group {gi}: master.grad is None before the step, so the "
                f"gradient the optimizer consumes was not observed")
        # Each snapshot is taken chunk-wise onto the host, and the finite check
        # happens inside that loop on the host copy. No whole-partition clone
        # and no whole-partition boolean mask is allocated anywhere.
        live = {"master": m, "grad": g, "exp_avg": st.get("exp_avg"),
                "exp_avg_sq": st.get("exp_avg_sq"),
                "max_exp_avg_sq": st.get("max_exp_avg_sq")}
        snaps, meta = {}, {}
        for nm, t in live.items():
            snaps[nm] = cpu_snapshot(t, name=f"group {gi} {nm}",
                                     chunk_elems=chunk_elems,
                                     check_finite=nm in ("master", "grad"))
            meta[nm] = snapshot_meta(t, snaps[nm], name=nm)
        step = st.get("step")
        snap["groups"].append({
            "group": gi,
            "master": snaps["master"],
            "grad": snaps["grad"],
            "exp_avg": snaps["exp_avg"],
            "exp_avg_sq": snaps["exp_avg_sq"],
            "max_exp_avg_sq": snaps["max_exp_avg_sq"],
            "snapshot_meta": meta,
            "state_step": (int(step.item()) if hasattr(step, "item")
                           else int(step) if step is not None else None),
            "numel": int(m.numel()),
        })
    return snap



def boundary_capture(zopt, *, rank: int, world_size: int,
                     chunk_elems: int = DEFAULT_CHUNK_ELEMS):
    """Capture the boundary where the base optimizer CONSUMES the gradient.

    In DeepSpeed 0.15.1 ZeRO-1 the master partition's `.grad` exists only inside
    `DeepSpeedZeroOptimizer.step()`. Reading the installed source
    (`stage_1_and_2.py`, digest recorded in the evidence):

        :1899  single_partition_of_fp32_groups[i].grad = single_grad_partition
        :1903  averaged_gradients[i] = None
        :1905  unscale_and_clip_grads([single_grad_partition], ...)
        :1911  _optimizer_step(i)        -> self.optimizer.step()
        :1913  single_partition_of_fp32_groups[i].grad = None

    So a capture taken around `engine.step()` can never observe it: before the
    call it has not been assigned, and after the call it has been freed. That is
    exactly the refusal the first hardware attempt produced, and the refusal was
    correct - the gradient genuinely had not been observed.

    This wrapper snapshots immediately before and immediately after the base
    optimizer's own `step()`, which is the only instant at which the consumed,
    post-unscale/post-clip gradient is live. The observed object is therefore
    the gradient the base optimizer actually used, not a reconstruction of it
    from `averaged_gradients`.

    Nothing about the recipe, the optimizer, the tolerances or the coverage
    rules changes; only WHERE the observation is taken.
    """
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        base = _req(zopt, "optimizer")
        original = base.step
        box = {"before": None, "after": None, "base_step_calls": 0,
               "capture_point": ("entry/exit of the base optimizer's step(), "
                                 "called from DeepSpeedZeroOptimizer."
                                 "_optimizer_step"),
               "observed_gradient": ("master partition .grad as assigned at "
                                     "stage_1_and_2.py:1899 and clipped at "
                                     ":1905; post-unscale, post-clip"),
               "reconstructed_from_averaged_gradients": False}

        def wrapped(*args, **kwargs):
            box["base_step_calls"] += 1
            first = box["before"] is None
            if first:
                box["before"] = capture_boundary(
                    zopt, rank=rank, world_size=world_size, phase="before",
                    chunk_elems=chunk_elems)
            result = original(*args, **kwargs)
            if first:
                box["after"] = capture_boundary(
                    zopt, rank=rank, world_size=world_size, phase="after",
                    chunk_elems=chunk_elems)
            return result

        base.step = wrapped
        try:
            yield box
        finally:
            base.step = original          # always restored

    return _ctx()


def _int_view_dtype(dtype):
    """The same-width integer dtype for a bit-pattern comparison, or None."""
    import torch
    return {torch.float32: torch.int32, torch.float16: torch.int16,
            torch.bfloat16: torch.int16, torch.float64: torch.int64}.get(dtype)


def bitwise_identical(a, b, *, chunk_elems: int = DEFAULT_CHUNK_ELEMS) -> dict:
    """Bit-pattern equality, compared chunk by chunk through an integer view.

    This is NOT the same predicate as numerical equality, and the two are
    deliberately given different names. Two cases separate them:
      * -0.0 == +0.0 numerically, but their bit patterns differ.
      * NaN != NaN numerically, but two NaNs with the same payload are
        bitwise identical.
    A caller that means one must not report the other.
    """
    import torch
    iv = _int_view_dtype(a.dtype)
    if iv is None or a.dtype != b.dtype:
        return {"comparable": False, "bitwise_identical": None,
                "reason": (f"no integer view defined for {a.dtype} / {b.dtype}, "
                           f"so a bit-pattern comparison is NOT established")}
    if tuple(a.shape) != tuple(b.shape):
        return {"comparable": False, "bitwise_identical": None,
                "reason": f"shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}"}
    af, bf = a.reshape(-1), b.reshape(-1)
    if not (af.is_contiguous() and bf.is_contiguous()):
        return {"comparable": False, "bitwise_identical": None,
                "reason": "non-contiguous input; refusing a strided bit compare"}
    n = int(af.numel())
    differing, first = 0, None
    for off in range(0, n, int(chunk_elems)):
        k = min(int(chunk_elems), n - off)
        ne = af.narrow(0, off, k).view(iv) != bf.narrow(0, off, k).view(iv)
        c = int(ne.sum().item())
        if c and first is None:
            first = off + int(torch.argmax(ne.to(torch.uint8)).item())
        differing += c
    return {"comparable": True, "bitwise_identical": differing == 0,
            "n_differing_bit_patterns": differing,
            "first_differing_index": first,
            "integer_view": str(iv), "chunk_elems": int(chunk_elems)}


def _compare_exact(master, working, *, chunk_elems: int) -> dict:
    """Exact elementwise numerical equality, chunked. No tolerance anywhere."""
    import torch
    mf, wf = master.reshape(-1), working.reshape(-1)
    n = int(mf.numel())
    differing, first, worst, nonfinite = 0, None, 0.0, 0
    for off in range(0, n, int(chunk_elems)):
        k = min(int(chunk_elems), n - off)
        a = mf.narrow(0, off, k)
        b = wf.narrow(0, off, k)
        nonfinite += int((~torch.isfinite(a)).sum().item())
        nonfinite += int((~torch.isfinite(b)).sum().item())
        ne = a != b
        c = int(ne.sum().item())
        if c:
            if first is None:
                first = {"global_index":
                         off + int(torch.argmax(ne.to(torch.uint8)).item())}
            worst = max(worst, float(
                (a.to(torch.float64) - b.to(torch.float64)).abs().max()))
        differing += c
    return {"n_differing_elements": differing, "first_difference": first,
            "max_abs_difference": worst, "n_non_finite": nonfinite,
            "elements_compared": n}


def module_reflects_masters(zopt, *, chunk_elems: int = DEFAULT_CHUNK_ELEMS) -> dict:
    """Did the engine's working parameters take up the updated master values?

    Observation phase is unchanged: the caller invokes this AFTER
    `engine.step()` has returned, by which point DeepSpeed has copied the
    masters into the partitioned working view and propagated them
    (`stage_1_and_2.py:1915`, `all_gather_dp_groups`,
    `_update_model_bit16_weights`).

    The generic 1e-3 tolerance that previously applied to every layout is
    WITHDRAWN. The criterion now depends on the resolved contract:

    * **fp32 master -> fp32 working copy** (the frozen Gate-0 contract):
      correct identities, complete expected comparison coverage, matching
      shapes and dtypes with no broadcasting, finite values on both sides, and
      **exact elementwise numerical equality**. A bit-pattern comparison also
      runs and is reported under its own separate name.

    * **any other identical floating dtype pair**: also exact equality, under
      the separate name `same_dtype_exact_equality`.

    * **differing floating dtypes** (e.g. fp32 master -> fp16/bf16 working):
      compared against the resolved expected cast `master.to(working_dtype)`,
      bitwise. This criterion is not inherited from the fp32 one and carries no
      tolerance.

    * **anything else**, including a working dtype with no defined bit view:
      reported `unchecked`. Unresolved conversion semantics stay unverified
      rather than passing.

    Absence of the working view is a refusal, not a pass. Partial coverage is a
    refusal, not a prefix comparison.
    """
    import torch
    masters = _req(zopt, "single_partition_of_fp32_groups")
    working = (getattr(zopt, "bit16_groups_flat", None)
               or getattr(zopt, "fp16_groups_flat", None))
    base = {"observation_phase": ("after engine.step() returned, i.e. after "
                                 "DeepSpeed propagated the updated masters"),
            "supersedes": ("the generic 1e-3 tolerance criterion used up to and "
                           "including Gate-0 attempt GATE0-20261020T041835Z"),
            "chunk_elems": int(chunk_elems), "tolerance_used": None,
            "tolerance_note": ("no criterion here uses a tolerance; the previous "
                               "generic 1e-3 rule is withdrawn")}
    if working is None:
        return dict(base, checked=False, criterion="unchecked",
                    verdict_successful=False, per_group=[],
                    reason=("the engine exposes no flattened working-parameter "
                            "view in this layout; module/master reflection is "
                            "NOT established and is reported as unchecked"))
    if len(working) != len(masters):
        return dict(base, checked=False, criterion="unchecked",
                    verdict_successful=False, per_group=[],
                    reason=(f"expected {len(masters)} working views for "
                            f"{len(masters)} master partitions, found "
                            f"{len(working)}; incomplete comparison coverage is "
                            f"refused rather than compared as a prefix"))

    groups, ok = [], True
    for gi, (m, w) in enumerate(zip(masters, working)):
        g = {"group": gi, "master_dtype": str(m.dtype),
             "working_dtype": str(w.dtype),
             "master_numel": int(m.numel()), "working_numel": int(w.numel()),
             "master_shape": list(m.shape), "working_shape": list(w.shape),
             "distinct_storage": int(m.data_ptr()) != int(w.data_ptr()),
             "broadcasting_used": False, "shapes_match": False,
             "dtypes_match": str(m.dtype) == str(w.dtype)}
        if int(m.numel()) != int(w.numel()) or tuple(m.shape) != tuple(w.shape):
            g.update(criterion="unchecked", verdict_successful=False,
                     coverage_complete=False,
                     reason=(f"master shape {tuple(m.shape)} vs working "
                             f"{tuple(w.shape)}; shapes must match exactly and "
                             f"no broadcasting is permitted"))
            groups.append(g); ok = False
            continue
        g["shapes_match"] = True
        if not (m.is_contiguous() and w.is_contiguous()):
            g.update(criterion="unchecked", verdict_successful=False,
                     coverage_complete=False,
                     reason="non-contiguous view; refusing a strided comparison")
            groups.append(g); ok = False
            continue
        g["coverage_complete"] = True
        g["elements_expected"] = int(m.numel())
        md, wd = m.detach(), w.detach()
        same = md.dtype == wd.dtype
        floaty = md.dtype.is_floating_point and wd.dtype.is_floating_point
        if same and floaty:
            # Identical dtypes: the copy is a plain copy, so exact equality is
            # the contract. fp32/fp32 is the frozen Gate-0 case and keeps its
            # own name so it stays identifiable in the evidence.
            g["criterion"] = ("fp32_exact_equality"
                              if md.dtype == torch.float32
                              else "same_dtype_exact_equality")
            cmp = _compare_exact(md, wd, chunk_elems=chunk_elems)
            g.update(cmp)
            g["all_finite"] = cmp["n_non_finite"] == 0
            g["numerically_equal"] = cmp["n_differing_elements"] == 0
            bw = bitwise_identical(md, wd, chunk_elems=chunk_elems)
            g["bit_pattern"] = bw
            g["bitwise_identical"] = bw.get("bitwise_identical")
            g["verdict_successful"] = bool(
                g["all_finite"] and g["numerically_equal"]
                and cmp["elements_compared"] == int(m.numel()))
            if not g["verdict_successful"]:
                g["reason"] = (
                    f"{cmp['n_differing_elements']} of {m.numel()} elements "
                    f"differ (max |diff| {cmp['max_abs_difference']:.6g}) or "
                    f"{cmp['n_non_finite']} non-finite values are present; the "
                    f"fp32 contract requires exact equality and finiteness")
        elif floaty and _int_view_dtype(wd.dtype) is not None:
            g["criterion"] = "mixed_precision_expected_cast"
            g["expected_cast"] = f"master.to({wd.dtype})"
            expected = md.to(wd.dtype)
            bw = bitwise_identical(expected, wd, chunk_elems=chunk_elems)
            g["bit_pattern"] = bw
            g["bitwise_identical"] = bw.get("bitwise_identical")
            cmp = _compare_exact(expected, wd, chunk_elems=chunk_elems)
            g["n_differing_elements"] = cmp["n_differing_elements"]
            g["max_abs_difference"] = cmp["max_abs_difference"]
            g["elements_compared"] = cmp["elements_compared"]
            g["all_finite"] = cmp["n_non_finite"] == 0
            g["numerically_equal"] = cmp["n_differing_elements"] == 0
            g["verdict_successful"] = bool(
                bw.get("bitwise_identical") is True and g["all_finite"])
            g["note"] = ("compared against the resolved expected cast, not "
                         "against a tolerance; the fp32 rule is not inherited")
            if not g["verdict_successful"]:
                g["reason"] = ("the working copy does not match the expected "
                               "cast of the master into its dtype")
        else:
            g.update(criterion="unchecked", verdict_successful=False,
                     reason=(f"conversion semantics for master {md.dtype} -> "
                             f"working {wd.dtype} are not resolved in this "
                             f"implementation; reported unverified rather than "
                             f"passed"))
        ok = ok and bool(g.get("verdict_successful"))
        groups.append(g)

    crits = sorted({g["criterion"] for g in groups})
    bws = [g.get("bitwise_identical") for g in groups]
    return dict(base,
                checked=all(g["criterion"] != "unchecked" for g in groups),
                criterion=crits[0] if len(crits) == 1 else crits,
                per_group=groups,
                coverage={"expected_groups": len(masters),
                          "compared_groups": sum(1 for g in groups
                                                 if g.get("coverage_complete")),
                          "expected_elements": sum(int(x.numel()) for x in masters),
                          "compared_elements": sum(g.get("elements_compared", 0)
                                                   for g in groups),
                          "complete": all(g.get("coverage_complete")
                                          for g in groups)},
                numerically_equal=all(g.get("numerically_equal") is True
                                      for g in groups),
                bitwise_identical=(all(b is True for b in bws)
                                   if all(b is not None for b in bws) else None),
                max_abs_difference=max([g.get("max_abs_difference") or 0.0
                                        for g in groups] or [0.0]),
                verdict_successful=bool(ok))


def partition_working_map(zopt, *, rank: int, world_size: int) -> dict:
    """Map each rank's master partition to its OWN working-model elements.

    Derived from the installed implementation, not from equal lengths. In
    DeepSpeed 0.15.1 `stage_1_and_2.py` the copy-back is

        bit16_partitions = self.parallel_partitioned_bit16_groups[i]
        fp32_partition   = self.single_partition_of_fp32_groups[i]
        bit16_partitions[partition_id].data.copy_(fp32_partition.data)   # :1915

    so the element range that corresponds to rank r's master partition is
    `parallel_partitioned_bit16_groups[i][r]`, a contiguous `narrow()` view of
    the flattened group produced by `get_data_parallel_partitions` (:1585),
    which splits `total_num_elements` into `dp` parts of `base_size`, giving the
    first `remaining` partitions one extra element.

    The earlier implementation compared the master partition against the WHOLE
    flattened group and therefore refused at world size > 1. Comparing a prefix
    would have been wrong; comparing the rank's own partition view is correct.

    Padding: `groups_padding[i]` pads the LAST rank's partition, so that rank's
    real element count is `partition_size - padding`. The padded tail is
    reported separately and never counted as model coverage.
    """
    masters = _req(zopt, "single_partition_of_fp32_groups")
    parts = getattr(zopt, "parallel_partitioned_bit16_groups", None)
    flat = (getattr(zopt, "bit16_groups_flat", None)
            or getattr(zopt, "fp16_groups_flat", None))
    d = describe_optimizer(zopt)
    out = {"derived_from": ("parallel_partitioned_bit16_groups[i][rank], the "
                           "view stage_1_and_2.py:1915 copies into"),
           "inferred_from_equal_lengths": False,
           "prefix_comparison_used": False,
           "rank": rank, "world_size": world_size, "groups": []}
    if parts is None:
        out["mapped"] = False
        out["reason"] = ("the engine exposes no parallel_partitioned_bit16_groups, "
                         "so the per-rank working view cannot be located; the "
                         "mapping is NOT established")
        return out
    for gi, m in enumerate(masters):
        g = {"group": gi, "master_numel": int(m.numel())}
        try:
            views = parts[gi]
        except (IndexError, TypeError):
            g.update(mapped=False, reason="no partition views for this group")
            out["groups"].append(g); continue
        if not (0 <= rank < len(views)):
            g.update(mapped=False,
                     reason=f"rank {rank} outside {len(views)} partition views")
            out["groups"].append(g); continue
        w = views[rank]
        psize = int(d["partition_size"][gi])
        pad = int(d["groups_padding"][gi])
        offset = sum(int(x.numel()) for x in views[:rank])
        real = psize - (pad if rank == world_size - 1 else 0)
        g.update(
            mapped=True,
            working_numel=int(w.numel()),
            partition_size=psize, padding=pad,
            is_last_rank=(rank == world_size - 1),
            unpadded_numel=real,
            flat_offset=offset,
            flat_group_numel=(int(flat[gi].numel()) if flat is not None else None),
            numel_matches=int(w.numel()) == int(m.numel()),
            working_dtype=str(w.dtype), master_dtype=str(m.dtype),
            working_is_view_of_flat=(
                flat is not None
                and int(w.data_ptr()) >= int(flat[gi].data_ptr())),
        )
        out["groups"].append(g)
    out["mapped"] = all(x.get("mapped") and x.get("numel_matches")
                        for x in out["groups"]) and bool(out["groups"])
    if not out["mapped"]:
        out["reason"] = ("one or more groups could not be mapped to a same-sized "
                         "per-rank working view; reported unverified")
    return out


def module_reflects_masters_partitioned(zopt, *, rank: int, world_size: int,
                                        chunk_elems: int = DEFAULT_CHUNK_ELEMS) -> dict:
    """Working/master consistency against the rank's OWN partition view.

    Same dtype contracts as `module_reflects_masters`: fp32 requires finite
    values and exact elementwise equality; a differing float dtype is compared
    against the resolved expected cast; numerical equality and bitwise identity
    are reported under separate names; an unsupported or incomplete mapping
    stays unverified and is never reported successful.

    The caller must invoke this AFTER `engine.step()` has fully returned, so
    DeepSpeed has finished the copy-back and propagation.
    """
    import torch
    mp = partition_working_map(zopt, rank=rank, world_size=world_size)
    base = {"observation_phase": ("after engine.step() returned, i.e. after "
                                  "DeepSpeed propagated the updated masters"),
            "mapping": mp, "chunk_elems": int(chunk_elems),
            "tolerance_used": None,
            "tolerance_note": ("no criterion here uses a tolerance; the generic "
                               "1e-3 rule remains withdrawn"),
            "supersedes": ("the whole-group comparison, which refused at world "
                           "size > 1 because a partition is not the group")}
    if not mp.get("mapped"):
        return dict(base, checked=False, criterion="unchecked",
                    verdict_successful=False, per_group=[],
                    reason=mp.get("reason", "mapping not established"))
    masters = _req(zopt, "single_partition_of_fp32_groups")
    parts = zopt.parallel_partitioned_bit16_groups
    groups, ok = [], True
    for gi, (m, gmap) in enumerate(zip(masters, mp["groups"])):
        w = parts[gi][rank]
        g = {"group": gi, "flat_offset": gmap["flat_offset"],
             "elements_expected": gmap["master_numel"],
             "unpadded_numel": gmap["unpadded_numel"],
             "padding": gmap["padding"], "is_last_rank": gmap["is_last_rank"],
             "master_dtype": str(m.dtype), "working_dtype": str(w.dtype),
             "broadcasting_used": False, "prefix_comparison_used": False,
             "distinct_storage": int(m.data_ptr()) != int(w.data_ptr())}
        md, wd = m.detach(), w.detach()
        if not (md.is_contiguous() and wd.is_contiguous()):
            g.update(criterion="unchecked", verdict_successful=False,
                     reason="non-contiguous view; refusing a strided comparison")
            groups.append(g); ok = False; continue
        same = md.dtype == wd.dtype
        floaty = md.dtype.is_floating_point and wd.dtype.is_floating_point
        if same and floaty:
            g["criterion"] = ("fp32_exact_equality" if md.dtype == torch.float32
                              else "same_dtype_exact_equality")
            cmp = _compare_exact(md, wd, chunk_elems=chunk_elems)
            g.update(cmp)
            g["all_finite"] = cmp["n_non_finite"] == 0
            g["numerically_equal"] = cmp["n_differing_elements"] == 0
            bw = bitwise_identical(md, wd, chunk_elems=chunk_elems)
            g["bit_pattern"] = bw
            g["bitwise_identical"] = bw.get("bitwise_identical")
            g["verdict_successful"] = bool(g["all_finite"] and g["numerically_equal"])
        elif floaty and _int_view_dtype(wd.dtype) is not None:
            g["criterion"] = "mixed_precision_expected_cast"
            g["expected_cast"] = f"master.to({wd.dtype})"
            expected = md.to(wd.dtype)
            bw = bitwise_identical(expected, wd, chunk_elems=chunk_elems)
            cmp = _compare_exact(expected, wd, chunk_elems=chunk_elems)
            g["bit_pattern"] = bw
            g["bitwise_identical"] = bw.get("bitwise_identical")
            g["n_differing_elements"] = cmp["n_differing_elements"]
            g["max_abs_difference"] = cmp["max_abs_difference"]
            g["all_finite"] = cmp["n_non_finite"] == 0
            g["numerically_equal"] = cmp["n_differing_elements"] == 0
            g["verdict_successful"] = bool(bw.get("bitwise_identical") is True
                                           and g["all_finite"])
            g["note"] = ("compared against the resolved expected cast, not a "
                         "tolerance")
        else:
            g.update(criterion="unchecked", verdict_successful=False,
                     reason=(f"conversion semantics for master {md.dtype} -> "
                             f"working {wd.dtype} are not resolved; reported "
                             f"unverified rather than passed"))
        if not g.get("verdict_successful") and "reason" not in g:
            g["reason"] = (f"{g.get('n_differing_elements')} element(s) differ "
                           f"(max |diff| {g.get('max_abs_difference')})")
        ok = ok and bool(g.get("verdict_successful"))
        groups.append(g)
    crits = sorted({g["criterion"] for g in groups})
    bws = [g.get("bitwise_identical") for g in groups]
    return dict(base,
                checked=all(g["criterion"] != "unchecked" for g in groups),
                criterion=crits[0] if len(crits) == 1 else crits,
                per_group=groups,
                coverage={"expected_groups": len(masters),
                          "compared_groups": sum(1 for g in groups
                                                 if g.get("elements_compared")),
                          "expected_elements_this_rank":
                              sum(g["elements_expected"] for g in groups),
                          "unpadded_elements_this_rank":
                              sum(g["unpadded_numel"] for g in groups),
                          "padding_elements":
                              sum(g["padding"] for g in groups
                                  if g.get("is_last_rank")),
                          "padding_note": ("padded tail elements are reported "
                                           "and never counted as model "
                                           "coverage")},
                numerically_equal=all(g.get("numerically_equal") is True
                                      for g in groups),
                bitwise_identical=(all(b is True for b in bws)
                                   if all(b is not None for b in bws) else None),
                verdict_successful=bool(ok))


def reconcile_partition_coverage(rank_maps: list[dict], *,
                                 unique_model_elements: int,
                                 world_size: int) -> dict:
    """Every expected unique element accounted for exactly once across ranks.

    Padding is subtracted explicitly; tied aliases were already deduplicated by
    storage when the unique-element count was resolved.
    """
    spans, per_rank, bad = [], {}, []
    for mp in rank_maps:
        r = mp.get("rank")
        if not mp.get("mapped"):
            bad.append({"rank": r, "reason": mp.get("reason")})
            continue
        tot = 0
        for g in mp["groups"]:
            spans.append((g["group"], g["flat_offset"], g["unpadded_numel"], r))
            tot += g["unpadded_numel"]
        per_rank[str(r)] = tot
    ranks = sorted(per_rank)
    expected = set(str(r) for r in range(world_size))
    missing = sorted(expected - set(ranks))
    unexpected = sorted(set(ranks) - expected)
    dup = [r for r in ranks if ranks.count(r) > 1]
    covered = sum(per_rank.values())
    # within each group the spans must tile without overlap
    overlaps = []
    by_group = {}
    for gi, off, n, r in spans:
        by_group.setdefault(gi, []).append((off, n, r))
    for gi, lst in by_group.items():
        lst.sort()
        for a, b in zip(lst, lst[1:]):
            if a[0] + a[1] > b[0]:
                overlaps.append({"group": gi, "a": a, "b": b})
    return {
        "unique_model_elements": unique_model_elements,
        "elements_covered_once": covered,
        "elements_unaccounted": unique_model_elements - covered,
        "per_rank_elements": per_rank,
        "missing_ranks": missing, "duplicate_ranks": sorted(set(dup)),
        "unexpected_ranks": unexpected, "unmapped_ranks": bad,
        "overlapping_spans": overlaps,
        "complete": bool(covered == unique_model_elements and not missing
                         and not unexpected and not dup and not overlaps
                         and not bad),
        "basis": ("per-rank unpadded partition spans, tiled per group; padding "
                  "excluded; tied aliases already deduplicated by storage"),
    }
