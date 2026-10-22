"""Independent intended-update oracle with a strict, documented contract.

CPU-TESTED ONLY. **NOT integrated into live ZeRO validation** and **never used
to certify historical runs.** It verifies one supplied update against closed-form
AdamW/Adam semantics; it never calls the optimizer.

Contract (every clause is enforced and tested)
----------------------------------------------
* Non-finite anything is refused: parameters, gradients, moments, observed
  values, hyperparameters, tolerances.
* Types, shapes, step indices, domains and coverage cardinalities are validated.
  Booleans are refused where counts are required (`isinstance(True, int)` is
  True in Python, so this must be explicit).
* Resolved semantics are REQUIRED. `weight_decay` and `decoupled_weight_decay`
  must be supplied; nothing that changes the update is ever assumed.
* Partition aggregation reconciles exact IDENTITIES, not counts. Duplicate,
  missing and unexpected identities are all refused.
* Coverage is separate from success. A failed check can never report
  `verification_successful`, and verifying every supplied element is NOT
  whole-model verification: that needs an explicit reconciled model map.
* Every result records whether the comparison could actually DISTINGUISH the
  intended update from a no-op at the chosen resolution
  (`discriminating`). Non-discriminating evidence is labelled, not counted as
  proof. A genuinely zero intended update is not forced to fail.

Tolerance policy, justified from arithmetic rather than from desired outcomes
----------------------------------------------------------------------------
The comparison tolerance is `atol + rtol * scale`, where `scale` is the
magnitude of the expected post-update parameter. Defaults are derived from the
dtype's unit roundoff u (float64 u≈1.11e-16, float32 u≈5.96e-8): `rtol = 64*u`
and `atol = 64*u*max(1,|p|)`, allowing roughly 64 rounding events across the
~10 elementwise operations of an Adam step plus accumulation slack.

That tolerance is for *agreement*. Whether the evidence DISCRIMINATES is a
separate question, answered by comparing the intended movement against the
tolerance: if `|expected - before| <= tolerance`, agreement is uninformative,
because an unmoved parameter would agree equally well. This is reported as
`discriminating: False` with `verification_successful: False`, rather than
either a pass or a numerical failure. Historical experimental acceptance
thresholds are untouched by this module.
"""
from __future__ import annotations

import math
from typing import Any

FULL, SAMPLED = "full", "sampled"
SUPPORTED_MODES = ("adamw_decoupled", "adam_coupled")
REQUIRED_HPARAMS = ("lr", "beta1", "beta2", "eps", "weight_decay",
                    "decoupled_weight_decay")


class OracleRefusal(ValueError):
    """The supplied evidence cannot be judged; never a pass, never a failure."""


def _is_real_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _finite_scalar(v: Any) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(float(v)))


def _all_finite(x: Any, name: str) -> None:
    """Refuse non-finite tensors or scalars."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            if not bool(torch.isfinite(x).all()):
                raise OracleRefusal(f"{name} contains non-finite values")
            return
    except ImportError:
        pass
    if hasattr(x, "__iter__"):
        for i, v in enumerate(x):
            if not _finite_scalar(v):
                raise OracleRefusal(f"{name}[{i}] is non-finite or non-numeric")
        return
    if not _finite_scalar(x):
        raise OracleRefusal(f"{name} is non-finite or non-numeric")


def _shape(x: Any):
    return tuple(getattr(x, "shape", ()))


def _max_abs(x) -> float:
    try:
        return float(x.abs().max())
    except AttributeError:
        try:
            return max(abs(float(v)) for v in x)
        except TypeError:
            return abs(float(x))


def _dtype_unit_roundoff(x) -> float:
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return float(torch.finfo(x.dtype).eps) / 2.0
    except ImportError:
        pass
    return 1.1102230246251565e-16


def _dtype_unit_roundoff_from_dtype(dtype) -> float:
    """Same policy as `_dtype_unit_roundoff`, keyed by dtype alone.

    The chunked path knows the expected dtype without holding a whole-partition
    expected tensor, so the unit roundoff is resolved from the dtype directly.
    """
    try:
        import torch
        return float(torch.finfo(dtype).eps) / 2.0
    except (ImportError, TypeError):
        return 1.1102230246251565e-16


def default_tolerances(param_before, expected_param) -> tuple[float, float]:
    """(atol, rtol) from the supported numerical precision. See module docstring."""
    u = _dtype_unit_roundoff(expected_param)
    rtol = 64.0 * u
    atol = 64.0 * u * max(1.0, _max_abs(param_before))
    return atol, rtol


def resolve_mode(hparams: dict) -> str:
    dw = hparams.get("decoupled_weight_decay")
    if dw is None:
        raise OracleRefusal(
            "decoupled_weight_decay is not supplied; AdamW and Adam apply weight "
            "decay differently and the intended update is undefined without it")
    if not isinstance(dw, bool):
        raise OracleRefusal("decoupled_weight_decay must be a real boolean")
    return "adamw_decoupled" if dw else "adam_coupled"


def expected_adam_update(*, param_before, grad, exp_avg, exp_avg_sq, step: int,
                         lr: float, beta1: float, beta2: float, eps: float,
                         weight_decay: float, decoupled_weight_decay: bool,
                         amsgrad: bool = False, max_exp_avg_sq=None):
    """Closed-form expected (param_after, exp_avg, exp_avg_sq)."""
    if not _is_real_int(step) or step < 1:
        raise OracleRefusal(f"step must be a positive int, got {step!r}")
    for nm, v in (("lr", lr), ("beta1", beta1), ("beta2", beta2), ("eps", eps),
                  ("weight_decay", weight_decay)):
        if not _finite_scalar(v):
            raise OracleRefusal(f"{nm} must be a finite number, got {v!r}")
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise OracleRefusal("beta1 and beta2 must lie in [0,1)")
    if eps <= 0.0:
        raise OracleRefusal("eps must be positive")
    if not isinstance(amsgrad, bool):
        raise OracleRefusal("amsgrad must be a real boolean")
    for nm, t in (("param_before", param_before), ("grad", grad),
                  ("exp_avg", exp_avg), ("exp_avg_sq", exp_avg_sq)):
        _all_finite(t, nm)
    shapes = {_shape(param_before), _shape(grad), _shape(exp_avg), _shape(exp_avg_sq)}
    if len(shapes) != 1:
        raise OracleRefusal(f"shape mismatch among inputs: {shapes}")
    p, g = param_before, grad
    if weight_decay:
        if decoupled_weight_decay:
            p = p - lr * weight_decay * p
        else:
            g = g + weight_decay * param_before
    m = beta1 * exp_avg + (1.0 - beta1) * g
    v = beta2 * exp_avg_sq + (1.0 - beta2) * (g * g)
    mh = m / (1.0 - beta1 ** step)
    if amsgrad:
        if max_exp_avg_sq is None:
            raise OracleRefusal("amsgrad=True requires max_exp_avg_sq")
        _all_finite(max_exp_avg_sq, "max_exp_avg_sq")
        try:
            import torch
            vmax = torch.maximum(max_exp_avg_sq, v)
        except ImportError:
            vmax = v
        vh = vmax / (1.0 - beta2 ** step)
    else:
        vh = v / (1.0 - beta2 ** step)
    return p - lr * mh / (vh ** 0.5 + eps), m, v


def verify_update(*, param_before, param_after, grad, exp_avg, exp_avg_sq,
                  step: int, hparams: dict, observed_exp_avg=None,
                  observed_exp_avg_sq=None, atol: float | None = None,
                  rtol: float | None = None, coverage: str = FULL,
                  n_elements_checked: int | None = None,
                  n_elements_total: int | None = None,
                  model_map: dict | None = None) -> dict:
    """Judge one intended update. Raises OracleRefusal when it cannot judge."""
    if coverage not in (FULL, SAMPLED):
        raise OracleRefusal(f"coverage must be {FULL!r} or {SAMPLED!r}, got {coverage!r}")
    missing = [k for k in REQUIRED_HPARAMS if k not in hparams]
    if missing:
        raise OracleRefusal(
            f"resolved hyperparameters missing: {missing}. The intended update is "
            f"undefined without them and nothing is assumed.")
    mode = resolve_mode(hparams)
    for nm, c in (("n_elements_checked", n_elements_checked),
                  ("n_elements_total", n_elements_total)):
        if c is not None and not _is_real_int(c):
            raise OracleRefusal(f"{nm} must be a real int (bool refused), got {c!r}")
        if c is not None and c < 0:
            raise OracleRefusal(f"{nm} must be non-negative, got {c!r}")
    if (n_elements_checked is not None and n_elements_total is not None
            and n_elements_checked > n_elements_total):
        raise OracleRefusal(
            f"n_elements_checked {n_elements_checked} exceeds n_elements_total "
            f"{n_elements_total}")
    for nm, t in (("param_before", param_before), ("param_after", param_after),
                  ("grad", grad), ("exp_avg", exp_avg), ("exp_avg_sq", exp_avg_sq)):
        _all_finite(t, nm)
    if observed_exp_avg is not None:
        _all_finite(observed_exp_avg, "observed_exp_avg")
    if observed_exp_avg_sq is not None:
        _all_finite(observed_exp_avg_sq, "observed_exp_avg_sq")
    if _shape(param_after) != _shape(param_before):
        raise OracleRefusal("param_after shape differs from param_before")

    exp_p, exp_m, exp_v = expected_adam_update(
        param_before=param_before, grad=grad, exp_avg=exp_avg,
        exp_avg_sq=exp_avg_sq, step=step,
        lr=hparams["lr"], beta1=hparams["beta1"], beta2=hparams["beta2"],
        eps=hparams["eps"], weight_decay=hparams["weight_decay"],
        decoupled_weight_decay=hparams["decoupled_weight_decay"],
        amsgrad=hparams.get("amsgrad", False),
        max_exp_avg_sq=hparams.get("max_exp_avg_sq"))

    if atol is None or rtol is None:
        d_atol, d_rtol = default_tolerances(param_before, exp_p)
        atol = d_atol if atol is None else atol
        rtol = d_rtol if rtol is None else rtol
    for nm, v in (("atol", atol), ("rtol", rtol)):
        if not _finite_scalar(v) or v < 0:
            raise OracleRefusal(f"{nm} must be a finite non-negative number, got {v!r}")

    scale = max(_max_abs(exp_p), 1.0 if _max_abs(exp_p) == 0 else 0.0, 1e-300)
    tol = atol + rtol * scale
    dev_p = _max_abs(param_after - exp_p)
    intended_move = _max_abs(exp_p - param_before)
    observed_move = _max_abs(param_after - param_before)
    agrees = dev_p <= tol

    # Does the comparison distinguish the intended update from a no-op?
    zero_intended = intended_move == 0.0
    discriminating = zero_intended or intended_move > tol

    checks: dict[str, bool] = {"parameter_agrees": agrees}
    reasons: list[str] = []
    if not agrees:
        reasons.append(f"parameter deviates {dev_p:.6g} > tolerance {tol:.6g}")
    if observed_exp_avg is not None:
        d = _max_abs(observed_exp_avg - exp_m)
        checks["exp_avg_agrees"] = d <= atol + rtol * max(_max_abs(exp_m), 1e-300)
        if not checks["exp_avg_agrees"]:
            reasons.append(f"exp_avg deviates {d:.6g}")
    if observed_exp_avg_sq is not None:
        d = _max_abs(observed_exp_avg_sq - exp_v)
        checks["exp_avg_sq_agrees"] = d <= atol + rtol * max(_max_abs(exp_v), 1e-300)
        if not checks["exp_avg_sq_agrees"]:
            reasons.append(f"exp_avg_sq deviates {d:.6g}")
    if not discriminating:
        reasons.append(
            f"NON-DISCRIMINATING: intended movement {intended_move:.6g} is within "
            f"the tolerance {tol:.6g}, so agreement does not show the update "
            f"occurred; an unmoved parameter would agree equally well")

    all_agree = all(checks.values())
    # coverage is NOT success, and all-supplied-elements is NOT whole-model
    elements_complete = (n_elements_checked is not None
                         and n_elements_total is not None
                         and n_elements_checked == n_elements_total)
    map_ok = False
    if model_map is not None:
        want = model_map.get("expected_elements")
        map_ok = (_is_real_int(want) and n_elements_total == want
                  and elements_complete and coverage == FULL)
    return {
        "verification_successful": bool(all_agree and discriminating),
        "agrees_within_tolerance": bool(all_agree),
        "discriminating": bool(discriminating),
        "zero_intended_update": bool(zero_intended),
        "checks": checks,
        "reasons": reasons,
        "coverage": coverage,
        "elements_checked": n_elements_checked,
        "elements_total": n_elements_total,
        "all_supplied_elements_checked": bool(elements_complete),
        "whole_model_verified": bool(map_ok and all_agree and discriminating),
        "whole_model_basis": ("reconciled model_map" if map_ok else
                              "NOT established: a reconciled model/partition map "
                              "was not supplied, so element completeness alone "
                              "does not imply whole-model coverage"),
        "optimizer_mode": mode,
        "resolved_hparams": {k: hparams.get(k) for k in
                             REQUIRED_HPARAMS + ("amsgrad",)},
        "step": step,
        "max_abs_param_deviation": dev_p,
        "tolerance": tol, "atol": atol, "rtol": rtol,
        "intended_movement": intended_move,
        "observed_movement": observed_move,
        "integration_status": "CPU-tested only; NOT integrated into live ZeRO "
                              "validation and not used to certify historical runs",
    }


def verify_partition(results: list[dict], *, expected_partitions) -> dict:
    """Aggregate per-partition verdicts, reconciling exact IDENTITIES.

    `expected_partitions` is the set/sequence of partition identities expected.
    Duplicates, missing identities and unexpected identities are all refused;
    counting rows is not reconciliation.
    """
    if not isinstance(results, list) or not results:
        raise OracleRefusal("no partition results supplied")
    try:
        expected = list(expected_partitions)
    except TypeError:
        raise OracleRefusal("expected_partitions must be an iterable of identities")
    if not expected:
        raise OracleRefusal("expected_partitions is empty")
    if len(set(expected)) != len(expected):
        raise OracleRefusal(f"expected_partitions contains duplicates: {expected}")

    ids = []
    for i, r in enumerate(results):
        if "partition" not in r:
            raise OracleRefusal(f"result {i} carries no 'partition' identity")
        ids.append(r["partition"])
    dup = sorted({x for x in ids if ids.count(x) > 1})
    missing = sorted(set(expected) - set(ids), key=str)
    unexpected = sorted(set(ids) - set(expected), key=str)
    failing = [r["partition"] for r in results if not r.get("verification_successful")]
    nondisc = [r["partition"] for r in results if not r.get("discriminating", True)]
    sampled = [r["partition"] for r in results if r.get("coverage") == SAMPLED]
    identities_ok = not dup and not missing and not unexpected
    all_ok = identities_ok and not failing
    return {
        "verification_successful": bool(all_ok),
        "identities_reconciled": bool(identities_ok),
        "duplicate_identities": dup,
        "missing_identities": missing,
        "unexpected_identities": unexpected,
        "failing_partitions": failing,
        "non_discriminating_partitions": nondisc,
        "sampled_partitions": sampled,
        "partitions_supplied": len(results),
        "partitions_expected": len(expected),
        "whole_model_verified": bool(
            all_ok and not sampled and not nondisc
            and all(r.get("whole_model_verified") for r in results)),
        "coverage_declaration": (
            "FULL and identity-reconciled across every expected partition"
            if all_ok and not sampled else
            "REFUSED: " + "; ".join(filter(None, [
                f"duplicates {dup}" if dup else "",
                f"missing {missing}" if missing else "",
                f"unexpected {unexpected}" if unexpected else "",
                f"failing {failing}" if failing else "",
                f"sampled {sampled}" if sampled else "",
                f"non-discriminating {nondisc}" if nondisc else ""]))),
        "integration_status": "CPU-tested only; NOT integrated into live ZeRO validation",
    }


DEFAULT_CHUNK_ELEMS = 4_194_304


def _chunk_offsets(n: int, chunk_elems: int):
    """Tile [0, n) exactly once, final partial chunk included."""
    if not _is_real_int(n) or n < 0:
        raise OracleRefusal(f"element count must be a non-negative int, got {n!r}")
    if not _is_real_int(chunk_elems) or chunk_elems < 1:
        raise OracleRefusal(f"chunk_elems must be a positive int, got {chunk_elems!r}")
    off, spans = 0, []
    while off < n:
        k = min(chunk_elems, n - off)
        spans.append((off, k))
        off += k
    covered = sum(k for _, k in spans)
    if covered != n:
        raise OracleRefusal(
            f"chunk tiling covers {covered} of {n} elements; coverage must be "
            f"exact and each element must be checked exactly once")
    for i in range(1, len(spans)):
        if spans[i][0] != spans[i - 1][0] + spans[i - 1][1]:
            raise OracleRefusal(
                f"chunk spans are not contiguous at index {i}: {spans[i - 1]} "
                f"then {spans[i]}; duplicated or skipped elements are refused")
    return spans


def verify_update_chunked(*, param_before, param_after, grad, exp_avg, exp_avg_sq,
                          step: int, hparams: dict, observed_exp_avg=None,
                          observed_exp_avg_sq=None, atol: float | None = None,
                          rtol: float | None = None, coverage: str = FULL,
                          n_elements_checked: int | None = None,
                          n_elements_total: int | None = None,
                          model_map: dict | None = None,
                          chunk_elems: int = DEFAULT_CHUNK_ELEMS) -> dict:
    """Chunked equivalent of `verify_update`, bounded by `chunk_elems`.

    Every reduction used by the verdict is a maximum, and the maximum of
    per-chunk maxima is exactly the global maximum, so chunking changes no
    reported figure. This is stated explicitly because it would NOT hold for a
    sum or a mean, where chunk ordering would alter the floating-point result.

    Memory: the only buffers allocated per chunk are the oracle temporaries for
    that chunk and the comparison buffers for that chunk. No whole-partition
    tensor, conversion, subtraction or finite mask is allocated at any point,
    and chunk outputs are never concatenated into a whole-partition expected
    tensor. Each chunk's temporaries are released before the next is read.

    Tolerance policy is unchanged: `default_tolerances` still consumes the
    global max |param_before| and the dtype of the expected parameter, both of
    which are accumulated across chunks before the tolerance is applied.
    """
    import torch
    missing = [k for k in REQUIRED_HPARAMS if k not in hparams]
    if missing:
        raise OracleRefusal(f"missing required hyperparameters: {missing}")
    mode = resolve_mode(hparams)
    if _shape(param_after) != _shape(param_before):
        raise OracleRefusal("param_after shape differs from param_before")

    tensors = {"param_before": param_before, "param_after": param_after,
               "grad": grad, "exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq}
    for nm, t in tensors.items():
        if not isinstance(t, torch.Tensor):
            raise OracleRefusal(f"{nm} must be a tensor for the chunked path")
        if t.device.type != "cpu":
            raise OracleRefusal(
                f"{nm} is on {t.device}; the chunked oracle runs on independent "
                f"CPU snapshots so that no device allocation is made here")
        if not t.is_contiguous():
            raise OracleRefusal(f"{nm} is not contiguous and cannot be sliced")
    n = int(param_before.numel())
    flat = {nm: t.view(-1) for nm, t in tensors.items()}
    obs = {}
    for nm, t in (("observed_exp_avg", observed_exp_avg),
                  ("observed_exp_avg_sq", observed_exp_avg_sq)):
        if t is not None:
            if t.device.type != "cpu":
                raise OracleRefusal(f"{nm} must be a CPU snapshot")
            if int(t.numel()) != n:
                raise OracleRefusal(f"{nm} has {t.numel()} elements, expected {n}")
            obs[nm] = t.reshape(-1) if not t.is_contiguous() else t.view(-1)

    mx = hparams.get("max_exp_avg_sq")
    mx_flat = None
    if hparams.get("amsgrad"):
        if mx is None:
            raise OracleRefusal("amsgrad=True requires max_exp_avg_sq")
        if not isinstance(mx, torch.Tensor) or mx.device.type != "cpu":
            raise OracleRefusal("max_exp_avg_sq must be a CPU snapshot")
        mx_flat = mx.view(-1)

    spans = _chunk_offsets(n, chunk_elems)
    acc = {"max_abs_pb": 0.0, "max_abs_exp_p": 0.0, "max_abs_exp_m": 0.0,
           "max_abs_exp_v": 0.0, "dev_p": 0.0, "dev_m": 0.0, "dev_v": 0.0,
           "intended": 0.0, "observed": 0.0}
    witness = {"param": None, "exp_avg": None, "exp_avg_sq": None}
    checked = 0
    hp_chunk = {k: hparams[k] for k in REQUIRED_HPARAMS}
    peak_chunk_elems = 0

    for off, k in spans:
        peak_chunk_elems = max(peak_chunk_elems, k)
        pb = flat["param_before"].narrow(0, off, k)
        pa = flat["param_after"].narrow(0, off, k)
        hp = dict(hp_chunk)
        hp["amsgrad"] = bool(hparams.get("amsgrad", False))
        # expected_adam_update validates finiteness of ITS inputs, which here
        # are chunk views, so every finite check is chunk-bounded.
        exp_p, exp_m, exp_v = expected_adam_update(
            param_before=pb, grad=flat["grad"].narrow(0, off, k),
            exp_avg=flat["exp_avg"].narrow(0, off, k),
            exp_avg_sq=flat["exp_avg_sq"].narrow(0, off, k),
            step=step, lr=hp["lr"], beta1=hp["beta1"], beta2=hp["beta2"],
            eps=hp["eps"], weight_decay=hp["weight_decay"],
            decoupled_weight_decay=hp["decoupled_weight_decay"],
            amsgrad=hp["amsgrad"],
            max_exp_avg_sq=(None if mx_flat is None
                            else mx_flat.narrow(0, off, k)))
        acc["max_abs_pb"] = max(acc["max_abs_pb"], _max_abs(pb))
        acc["max_abs_exp_p"] = max(acc["max_abs_exp_p"], _max_abs(exp_p))
        d = pa - exp_p
        local = int(torch.argmax(d.abs()).item()) if k else 0
        dmax = _max_abs(d)
        if dmax > acc["dev_p"]:
            acc["dev_p"] = dmax
            witness["param"] = {"global_index": off + local,
                                "chunk_offset": off, "local_index": local,
                                "deviation": dmax}
        acc["intended"] = max(acc["intended"], _max_abs(exp_p - pb))
        acc["observed"] = max(acc["observed"], _max_abs(pa - pb))
        if "observed_exp_avg" in obs:
            om = obs["observed_exp_avg"].narrow(0, off, k)
            dm = om - exp_m
            dmv = _max_abs(dm)
            acc["max_abs_exp_m"] = max(acc["max_abs_exp_m"], _max_abs(exp_m))
            if dmv > acc["dev_m"]:
                acc["dev_m"] = dmv
                witness["exp_avg"] = {
                    "global_index": off + int(torch.argmax(dm.abs()).item()),
                    "deviation": dmv}
            del om, dm
        if "observed_exp_avg_sq" in obs:
            ov = obs["observed_exp_avg_sq"].narrow(0, off, k)
            dv = ov - exp_v
            dvv = _max_abs(dv)
            acc["max_abs_exp_v"] = max(acc["max_abs_exp_v"], _max_abs(exp_v))
            if dvv > acc["dev_v"]:
                acc["dev_v"] = dvv
                witness["exp_avg_sq"] = {
                    "global_index": off + int(torch.argmax(dv.abs()).item()),
                    "deviation": dvv}
            del ov, dv
        checked += k
        # release this chunk's temporaries before advancing
        exp_dtype = exp_p.dtype
        del exp_p, exp_m, exp_v, d, pb, pa

    if checked != n:
        raise OracleRefusal(
            f"chunked coverage checked {checked} of {n} elements; incomplete "
            f"coverage cannot be reported as verified")

    if atol is None or rtol is None:
        u = _dtype_unit_roundoff_from_dtype(exp_dtype)
        d_rtol = 64.0 * u
        d_atol = 64.0 * u * max(1.0, acc["max_abs_pb"])
        atol = d_atol if atol is None else atol
        rtol = d_rtol if rtol is None else rtol
    for nm, v in (("atol", atol), ("rtol", rtol)):
        if not _finite_scalar(v) or v < 0:
            raise OracleRefusal(f"{nm} must be a finite non-negative number, got {v!r}")

    scale = max(acc["max_abs_exp_p"],
                1.0 if acc["max_abs_exp_p"] == 0 else 0.0, 1e-300)
    tol = atol + rtol * scale
    dev_p, intended_move, observed_move = acc["dev_p"], acc["intended"], acc["observed"]
    agrees = dev_p <= tol
    zero_intended = intended_move == 0.0
    discriminating = zero_intended or intended_move > tol

    checks: dict[str, bool] = {"parameter_agrees": agrees}
    reasons: list[str] = []
    if not agrees:
        reasons.append(f"parameter deviates {dev_p:.6g} > tolerance {tol:.6g}")
    if "observed_exp_avg" in obs:
        checks["exp_avg_agrees"] = acc["dev_m"] <= atol + rtol * max(
            acc["max_abs_exp_m"], 1e-300)
        if not checks["exp_avg_agrees"]:
            reasons.append(f"exp_avg deviates {acc['dev_m']:.6g}")
    if "observed_exp_avg_sq" in obs:
        checks["exp_avg_sq_agrees"] = acc["dev_v"] <= atol + rtol * max(
            acc["max_abs_exp_v"], 1e-300)
        if not checks["exp_avg_sq_agrees"]:
            reasons.append(f"exp_avg_sq deviates {acc['dev_v']:.6g}")
    if not discriminating:
        reasons.append(
            f"NON-DISCRIMINATING: intended movement {intended_move:.6g} is within "
            f"the tolerance {tol:.6g}, so agreement does not show the update "
            f"occurred; an unmoved parameter would agree equally well")

    all_agree = all(checks.values())
    if n_elements_checked is None:
        n_elements_checked = checked
    if n_elements_checked != checked:
        raise OracleRefusal(
            f"declared elements_checked {n_elements_checked} does not match the "
            f"{checked} elements the chunk loop actually visited")
    elements_complete = (n_elements_total is not None
                         and n_elements_checked == n_elements_total)
    map_ok = False
    if model_map is not None:
        want = model_map.get("expected_elements")
        map_ok = (_is_real_int(want) and n_elements_total == want
                  and elements_complete and coverage == FULL)
    return {
        "verification_successful": bool(all_agree and discriminating),
        "agrees_within_tolerance": bool(all_agree),
        "discriminating": bool(discriminating),
        "zero_intended_update": bool(zero_intended),
        "checks": checks, "reasons": reasons, "coverage": coverage,
        "elements_checked": n_elements_checked,
        "elements_total": n_elements_total,
        "all_supplied_elements_checked": bool(elements_complete),
        "whole_model_verified": bool(map_ok and all_agree and discriminating),
        "whole_model_basis": ("reconciled model_map" if map_ok else
                              "NOT established: a reconciled model/partition map "
                              "was not supplied, so element completeness alone "
                              "does not imply whole-model coverage"),
        "optimizer_mode": mode,
        "resolved_hparams": {k: hparams.get(k) for k in
                             REQUIRED_HPARAMS + ("amsgrad",)},
        "step": step,
        "max_abs_param_deviation": dev_p,
        "tolerance": tol, "atol": atol, "rtol": rtol,
        "intended_movement": intended_move,
        "observed_movement": observed_move,
        "integration_status": "CPU-tested only; NOT integrated into live ZeRO "
                              "validation and not used to certify historical runs",
        # chunked-path specific evidence
        "chunked": True,
        "chunk_elems": int(chunk_elems),
        "n_chunks": len(spans),
        "largest_chunk_elems": peak_chunk_elems,
        "final_chunk_elems": spans[-1][1] if spans else 0,
        "elements_visited": checked,
        "witnesses": witness,
        "arithmetic_device": "cpu",
        "arithmetic_dtype": str(exp_dtype),
        "reduction": ("max of per-chunk maxima, which equals the global maximum "
                      "exactly; no sum or mean reduction is used"),
        "cuda_bitwise_agreement": ("NOT established: the arithmetic ran on CPU, "
                                   "which does not show bitwise agreement with "
                                   "the same computation on CUDA"),
    }
