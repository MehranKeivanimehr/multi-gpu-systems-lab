"""Bounded, sampled evidence that a base AdamW call applied the intended update (A6000 smoke stages).

A hook firing, equal call counts or a tensor version bump do not show that the intended update happened. For the
tensors and element indices fixed in a SAMPLE PLAN before the first step, this records parameter, gradient and
optimizer state before each base call. It recomputes torch.optim.AdamW's update in float64 from those values, the
group's hyper-parameters and the post-increment step. It then compares the result with the parameter and state
actually present after the call. The installed torch/optim/adam.py was read for the semantics: decoupled decay
param *= (1 - lr*wd); exp_avg.lerp_(g, 1-b1); exp_avg_sq = b2*v + (1-b2)*g*g; step += 1;
param -= (lr/(1-b1^step)) * m / (sqrt(v)/sqrt(1-b2^step) + eps), on both the single-tensor and foreach paths.

A zero change passes when the recomputed change is zero. A recomputed change larger than the tolerance that did not
happen fails; such samples are counted as "discriminating" (they distinguish an applied update from a no-op).
Coverage is reported exactly and labelled "sampled" - it corroborates sampled updates, not the whole model and not
a matched reference.

FROZEN_TOLERANCES (unchanged since stage 20261016s):
  param:  |theta_obs - theta_exp| <= 8 * ulp32(max(|theta_before|, |theta_exp|)) + 1e-4 * |theta_exp - theta_before|
  m:      |m_obs - m_exp|         <= 8 * ulp32(max(|m_exp|, |m_before|, 0.1*|g|))
  v:      |v_obs - v_exp|         <= 8 * ulp32(max(|v_exp|, |v_before|, 1e-3*g*g))
  step:   exact (+1 per call)
  where ulp32(x) is the float32 spacing at |x| (2^(e-24) for x = f*2^e, 0.5 <= f < 1; 2^-149 below the normal range).
"""
from __future__ import annotations

import math

FROZEN_TOLERANCES = {
    # |observed - recomputed| <= ULPS * ulp32(scale) + REL * |recomputed change|
    "param_ulps": 8, "param_rel_of_change": 1e-4,
    "moment_ulps": 8,
    "step_exact": True,
    "basis": "float32 storage: each AdamW elementwise stage rounds once (<= 0.5 ulp); 8 ulp leaves a 4x margin for "
             "the ~4 rounded stages; 1e-4 of the change covers float32 rounding of the update term (rel ~1e-7)",
    "formula": {"param": "8*ulp32(max(|theta_before|,|theta_exp|)) + 1e-4*|theta_exp-theta_before|",
                "m": "8*ulp32(max(|m_exp|,|m_before|,0.1*|g|))", "v": "8*ulp32(max(|v_exp|,|v_before|,1e-3*g^2))",
                "ulp32": "float32 spacing at the given magnitude"},
}
SAMPLE_TENSORS_PER_CALL = 3
SAMPLE_ELEMS_PER_TENSOR = 64


def ulp32(x: float) -> float:
    x = abs(x)
    if x < 2.0 ** -126:
        return 2.0 ** -149
    m, e = math.frexp(x)
    return 2.0 ** (e - 24)


def _local(t):
    return t.to_local() if hasattr(t, "to_local") else t


def sample_indices(n: int, k: int = SAMPLE_ELEMS_PER_TENSOR):
    """Deterministic rule fixed before outcomes: the first k/2 elements plus k/2 evenly spaced over [0, n-1]."""
    import torch

    if n <= 0:
        return torch.empty(0, dtype=torch.long)
    head = list(range(min(n, k // 2)))
    stride = sorted({int(i * (n - 1) / max(1, k // 2 - 1)) for i in range(k // 2)}) if n > 1 else [0]
    return torch.tensor(sorted(set(head) | set(stride)), dtype=torch.long)


def build_plan(named_tensors: list) -> dict:
    """{id(tensor): {"name", "local_elements", "idx"}} for [(name, tensor), ...], chosen before any step."""
    plan = {}
    for name, t in named_tensors:
        idx = sample_indices(_local(t).numel())
        plan[id(t)] = {"name": name, "local_elements": _local(t).numel(), "idx": idx}
    return plan


def plan_record(plan: dict) -> list:
    return [{"name": v["name"], "local_elements": v["local_elements"], "indices": v["idx"].tolist()}
            for v in plan.values()]


def _take(t, idx):
    flat = _local(t).detach().reshape(-1)
    return flat.index_select(0, idx.to(flat.device)).double().cpu().tolist()


def _flags(optimizer, group) -> dict:
    keys = ("lr", "betas", "eps", "weight_decay", "amsgrad", "maximize", "foreach", "fused", "capturable",
            "differentiable")
    return {"optimizer": f"{type(optimizer).__module__}.{type(optimizer).__name__}",
            **{k: (list(group[k]) if k == "betas" else group.get(k)) for k in keys if k in group}}


class AdamWCallSampler:
    """Attach `pre(optimizer)` / `post(optimizer)` to a torch.optim.AdamW pre/post step hook.

    With a plan, only planned tensors are sampled (selection fixed before outcomes). Without one (legacy/CPU use),
    the first/middle/last tensors with gradients are chosen at call time and labelled as such."""

    def __init__(self, plan: dict | None = None):
        self.plan = plan
        self._pending = None
        self.calls: list[dict] = []

    def _choices(self, params):
        if self.plan is not None:
            return [(j, p, self.plan[id(p)]["name"], self.plan[id(p)]["idx"]) for j, p in enumerate(params)
                    if id(p) in self.plan]
        picks = sorted({0, len(params) // 2, len(params) - 1}) if params else []
        return [(j, params[j], f"position {j} (chosen at call time)", sample_indices(_local(params[j]).numel()))
                for j in picks[:SAMPLE_TENSORS_PER_CALL]]

    def pre(self, optimizer):
        import torch

        records, total_tensors, total_elems = [], 0, 0
        for group in optimizer.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            total_tensors += len(params)
            total_elems += sum(_local(p).numel() for p in params)
            if not isinstance(optimizer, torch.optim.AdamW) or group.get("amsgrad") or group.get("maximize"):
                records.append({"unsupported": f"{type(optimizer).__name__} amsgrad={group.get('amsgrad')} "
                                               f"maximize={group.get('maximize')}"})
                continue
            for j, p, name, idx in self._choices(params):
                st = optimizer.state.get(p, {})
                step = st.get("step")
                records.append({
                    "tensor": j, "name": name, "elements": _local(p).numel(), "idx": idx, "param_ref": p,
                    "theta": _take(p, idx), "grad": _take(p.grad, idx),
                    "m": _take(st["exp_avg"], idx) if "exp_avg" in st else [0.0] * len(idx),
                    "v": _take(st["exp_avg_sq"], idx) if "exp_avg_sq" in st else [0.0] * len(idx),
                    "step": float(step.item() if hasattr(step, "item") else step) if step is not None else 0.0,
                    "lr": float(group["lr"]), "betas": tuple(map(float, group["betas"])), "eps": float(group["eps"]),
                    "wd": float(group["weight_decay"]), "flags": _flags(optimizer, group),
                    "dtypes": {"param": str(_local(p).dtype), "grad": str(_local(p.grad).dtype),
                               "exp_avg": str(_local(st["exp_avg"]).dtype) if "exp_avg" in st else "absent (first step)",
                               "exp_avg_sq": str(_local(st["exp_avg_sq"]).dtype) if "exp_avg_sq" in st else
                               "absent (first step)"}})
        self._pending = {"records": records, "stepped_tensors": total_tensors, "stepped_elements": total_elems}

    def post(self, optimizer):
        if self._pending is None:
            self.calls.append({"verdict": "unverified", "reason": "post-hook without pre-hook"})
            return
        pend, self._pending = self._pending, None
        out = {"stepped_tensors": pend["stepped_tensors"], "stepped_elements": pend["stepped_elements"],
               "sampled_tensors": 0, "sampled_elements": 0, "discriminating_samples": 0, "checks": [],
               "selection": "plan fixed before step 1" if self.plan is not None else "chosen at call time",
               "verdict": "verified_sampled"}
        for r in pend["records"]:
            if "unsupported" in r:
                out["verdict"] = "unverified"
                out["checks"].append({"unsupported": r["unsupported"]})
                continue
            p, idx = r["param_ref"], r["idx"]
            st = optimizer.state.get(p, {})
            theta1 = _take(p, idx)
            m1 = _take(st["exp_avg"], idx) if "exp_avg" in st else None
            v1 = _take(st["exp_avg_sq"], idx) if "exp_avg_sq" in st else None
            s1 = st.get("step")
            s1 = float(s1.item() if hasattr(s1, "item") else s1) if s1 is not None else None
            chk = check_adamw(r, theta1, m1, v1, s1)
            chk.update(tensor=r["tensor"], name=r["name"], elements=r["elements"], sampled=len(idx),
                       idx=idx.tolist(), flags=r["flags"], dtypes=r["dtypes"])
            out["checks"].append(chk)
            out["sampled_tensors"] += 1
            out["sampled_elements"] += len(idx)
            out["discriminating_samples"] += chk.get("discriminating", 0)
            if not chk["pass"]:
                out["verdict"] = "failed"
        if out["sampled_elements"] == 0 and out["verdict"] == "verified_sampled":
            out["verdict"] = "unverified"
        self.calls.append(out)


def check_adamw(r: dict, theta1, m1, v1, s1) -> dict:
    """Recompute one AdamW step (decoupled weight decay, no amsgrad) in float64 and compare. Expected values (and
    the count of discriminating samples) are computed from the pre-call record first, so a call that did nothing -
    even one that left no optimizer state - is still measured against a known expected change."""
    b1, b2 = r["betas"]
    lr, eps, wd = r["lr"], r["eps"], r["wd"]
    step = r["step"] + 1.0
    bc1, bc2 = 1.0 - b1 ** step, 1.0 - b2 ** step
    res = {"pass": True, "step_before": r["step"], "step_after": s1, "bias_correction1": bc1, "bias_correction2": bc2,
           "max_param_err_over_tol": 0.0, "max_moment_err_over_tol": 0.0, "max_abs_err_param": 0.0,
           "max_rel_err_param": 0.0, "max_rel_err_param_discriminating": None, "max_abs_err_m": 0.0,
           "max_abs_err_v": 0.0, "zero_change_elements": 0,
           "nonzero_expected_change": 0, "discriminating": 0, "worst": None}
    reasons = []
    if s1 is None or s1 != step:
        reasons.append(f"step {r['step']} -> {s1}, expected {step}")
    if m1 is None or v1 is None:
        reasons.append("optimizer state missing after the call")
    m1 = m1 if m1 is not None else [float("nan")] * len(r["theta"])
    v1 = v1 if v1 is not None else [float("nan")] * len(r["theta"])
    for k, (t0, g, m0, v0, ta, ma, va) in enumerate(zip(r["theta"], r["grad"], r["m"], r["v"], theta1, m1, v1)):
        m_exp = m0 + (g - m0) * (1.0 - b1)
        v_exp = v0 * b2 + (1.0 - b2) * g * g
        t_exp = t0 * (1.0 - lr * wd) - (lr / bc1) * m_exp / (math.sqrt(v_exp) / math.sqrt(bc2) + eps)
        change = t_exp - t0
        ulp_basis = max(abs(t0), abs(t_exp))
        tol_t = FROZEN_TOLERANCES["param_ulps"] * ulp32(ulp_basis) + FROZEN_TOLERANCES["param_rel_of_change"] * abs(change)
        tol_m = FROZEN_TOLERANCES["moment_ulps"] * ulp32(max(abs(m_exp), abs(m0), 0.1 * abs(g)))
        tol_v = FROZEN_TOLERANCES["moment_ulps"] * ulp32(max(abs(v_exp), abs(v0), 1e-3 * g * g))
        res["discriminating"] += int(abs(change) > tol_t)
        res["nonzero_expected_change"] += int(abs(change) > tol_t)
        if not math.isfinite(ta) or not math.isfinite(t_exp):
            reasons.append("non-finite value in sampled parameter")
            continue
        err_t = abs(ta - t_exp)
        ratio = err_t / tol_t
        res["max_abs_err_param"] = max(res["max_abs_err_param"], err_t)
        if abs(change) > 0:
            res["max_rel_err_param"] = max(res["max_rel_err_param"], err_t / abs(change))
        if abs(change) > tol_t:          # reporting only: relative error restricted to discriminating samples
            res["max_rel_err_param_discriminating"] = max(res["max_rel_err_param_discriminating"] or 0.0,
                                                          err_t / abs(change))
        if math.isfinite(ma) and math.isfinite(va):
            err_m, err_v = abs(ma - m_exp), abs(va - v_exp)
            res["max_abs_err_m"] = max(res["max_abs_err_m"], err_m)
            res["max_abs_err_v"] = max(res["max_abs_err_v"], err_v)
            res["max_moment_err_over_tol"] = max(res["max_moment_err_over_tol"], err_m / tol_m, err_v / tol_v)
        if res["worst"] is None or ratio > res["max_param_err_over_tol"]:
            res["worst"] = {"sample": k, "flat_index": None, "theta_before": t0, "grad": g, "m_before": m0,
                            "v_before": v0, "step_after": step, "theta_expected": t_exp, "theta_observed": ta,
                            "abs_err": err_t, "tolerance": tol_t, "ulp_basis": ulp_basis, "expected_change": change}
        res["max_param_err_over_tol"] = max(res["max_param_err_over_tol"], ratio)
        res["zero_change_elements"] += int(ta == t0)
    if res["worst"] is not None and "idx" in r:
        res["worst"]["flat_index"] = int(r["idx"][res["worst"]["sample"]])
    if res["max_param_err_over_tol"] > 1.0 or res["max_moment_err_over_tol"] > 1.0:
        reasons.append("observed parameter/state differs from the recomputed AdamW update beyond frozen tolerance")
    res["pass"] = not reasons
    if reasons:
        res["reason"] = "; ".join(dict.fromkeys(reasons))
    return res
