"""Per-boundary accounting with expected-coverage bookkeeping (A6000 smoke stage, supersedes update_ledger use).

Kept separate for every logical attempt, by phase (warmup / measured):
  microbatches          - each microbatch loss (all observed, not only the returned mean)
  inspected boundary    - one gradient inspection per accumulation boundary, accounted against EXPECTED units
                          (tensor/partition keys and element counts), not merely "some non-empty tensors"
  base optimizer calls  - every call of the underlying torch optimizer, compared with the multiplicity the
                          framework's implementation implies (native: 1; DeepSpeed ZeRO-1/2: one per param group)
  successful update     - counter evidence (hook / engine) AND, when collected, sampled parameter/state evidence
                          (update_evidence.AdamWCallSampler) - reported as separate fields
Coverage of a boundary is not inspection of every microbatch gradient: accumulated gradients are inspected once,
at the boundary. Finite zero gradients are present gradients.
"""
from __future__ import annotations

import math

COMPLETE, PARTIAL, EMPTY, REPEATED, NONFINITE = "complete", "partial", "empty", "repeated", "nonfinite"
UNOBSERVED, NOT_CHECKED, UNACCOUNTED = "unobserved", "not_checked", "no_expected_accounting"


def scan_tensors(tensors_by_key: dict) -> dict:
    """Candidate 20: {key: (elements, finite)} for GRADIENT TENSORS themselves (e.g. DeepSpeed's flattened boundary
    partition). scan_units() reads `.grad` of parameters; a plain gradient tensor has no `.grad`, so passing one there
    counted 0 elements (found on hardware in window w19fb_20261021T224030Z, case G4)."""
    import torch

    out, flags, keys = {}, [], []
    for key, t in tensors_by_key.items():
        if t is None:
            out[key] = (0, None)
            continue
        loc = t.to_local() if hasattr(t, "to_local") else t
        out[key] = (loc.numel(), None if loc.numel() else True)
        if loc.numel():
            flags.append(torch.isfinite(loc).all())
            keys.append(key)
    if flags:
        vals = torch.stack(flags).cpu().tolist()
        for k, v in zip(keys, vals):
            out[k] = (out[k][0], bool(v))
    return out


def scan_units(params_by_key: dict) -> dict:
    """{key: (gradient elements present, finite or None)} for the given parameters' *local* gradients, with a single
    device synchronisation. A missing gradient is (0, None)."""
    import torch

    out, flags, keys = {}, [], []
    for key, p in params_by_key.items():
        g = p.grad
        if g is None:
            out[key] = (0, None)
            continue
        loc = g.to_local() if hasattr(g, "to_local") else g
        out[key] = (loc.numel(), None)
        if loc.numel():
            flags.append(torch.isfinite(loc).all())
            keys.append(key)
        else:
            out[key] = (0, True)
    if flags:
        vals = torch.stack(flags).cpu().tolist()
        for k, v in zip(keys, vals):
            out[k] = (out[k][0], bool(v))
    return out


def local_numel(t) -> int:
    return (t.to_local() if hasattr(t, "to_local") else t).numel()


class BoundaryLedger:
    def __init__(self, framework: str, gradient_checks: bool, observer: str, expected_units: dict | None,
                 calls_per_boundary: int | None, multiplicity_basis: str):
        """expected_units: {unit key: expected local elements} or None when no accounting is available.
        calls_per_boundary: base-optimizer calls the implementation makes per non-skipped boundary."""
        self.framework, self.gradient_checks, self.observer = framework, bool(gradient_checks), observer
        self.expected_units = dict(expected_units) if expected_units is not None else None
        self.calls_per_boundary, self.multiplicity_basis = calls_per_boundary, multiplicity_basis
        self.phases = {"warmup": [], "measured": []}
        self.phase = "warmup"
        self._cur = None
        self.layout = None                  # independent unit map / partition map and the pre-step sample plan

    # ---- lifecycle
    def start_measured(self):
        self.phase = "measured"

    def current_attempt(self) -> int:
        if self._cur is None:
            raise RuntimeError("no attempt in progress")
        return self._cur["attempt"]

    def begin_step(self):
        self._cur = {"attempt": len(self.phases[self.phase]), "phase": self.phase, "microbatch_losses": [],
                     "observations": [], "base_optimizer_calls": 0, "update_evidence": []}

    def record_microbatch_losses(self, losses):
        if self._cur is not None:
            self._cur["microbatch_losses"] = list(losses)        # tensors; converted after timing

    def observe_units(self, units: dict, source: str):
        if self._cur is None or not self.gradient_checks:
            return
        self._cur["observations"].append({"source": source, "units": units})

    def base_call(self, evidence: dict | None = None):
        if self._cur is not None:
            self._cur["base_optimizer_calls"] += 1
            if evidence is not None:
                self._cur["update_evidence"].append(evidence)

    def end_step(self, counter_update: bool | None, counter_source: str, framework_reported_skip: bool | None = None,
                 extra: dict | None = None):
        cur, self._cur = self._cur, None
        if cur is None:
            raise RuntimeError("end_step() without begin_step()")
        cur.update({"counter_update": counter_update, "counter_source": counter_source,
                    "framework_reported_skip": framework_reported_skip, **(extra or {})})
        cur["coverage"] = self._coverage(cur["observations"])
        self.phases[cur["phase"]].append(cur)

    # ---- assessment
    def _coverage(self, observations: list) -> dict:
        if not self.gradient_checks:
            return {"state": NOT_CHECKED}
        if not observations:
            return {"state": UNOBSERVED}
        seen, repeated, nonfinite, absent = {}, [], [], []
        for obs in observations:
            for key, (elems, finite) in obs["units"].items():
                if key in seen:
                    repeated.append(key)
                seen[key] = elems
                if finite is False:
                    nonfinite.append(key)
                if finite is None and elems == 0:
                    absent.append(key)
        out = {"observed_units": len(seen), "observed_elements": sum(seen.values()),
               "repeated_units": sorted(set(map(str, repeated))), "nonfinite_units": sorted(map(str, nonfinite))}
        if self.expected_units is None:
            return {**out, "state": UNACCOUNTED}
        missing = [k for k in self.expected_units if seen.get(k, 0) != self.expected_units[k]]
        extra = [k for k in seen if k not in self.expected_units]
        out.update(expected_units=len(self.expected_units), expected_elements=sum(self.expected_units.values()),
                   missing_or_short_units=[str(k) for k in missing][:20], n_missing_or_short=len(missing),
                   unexpected_units=[str(k) for k in extra][:20])
        if nonfinite:
            state = NONFINITE
        elif not seen or out["observed_elements"] == 0:
            state = EMPTY
        elif repeated:
            state = REPEATED
        elif missing or extra or absent:
            state = PARTIAL
        else:
            state = COMPLETE
        return {**out, "state": state}

    def summary(self, expected_measured: int, expected_warmup: int, microbatches_per_attempt: int) -> dict:
        def phase_summary(steps, expected):
            losses = [float(t.float().item()) if hasattr(t, "item") else float(t)
                      for s in steps for t in s["microbatch_losses"]]
            cov = [s["coverage"]["state"] for s in steps]
            calls = [s["base_optimizer_calls"] for s in steps]
            ev = [e for s in steps for e in s["update_evidence"]]
            mult_ok = (self.calls_per_boundary is not None
                       and all(c == self.calls_per_boundary for s, c in zip(steps, calls) if s["counter_update"]))
            plan = sorted(x["name"] for x in ((self.layout or {}).get("sample_plan") or []))
            per_boundary = [sorted(c.get("name") for e in s["update_evidence"] for c in e.get("checks", [])
                                   if "name" in c) for s in steps]
            return {
                "logical_attempts": len(steps), "expected_logical_attempts": expected,
                "microbatches": len(losses), "expected_microbatches": expected * microbatches_per_attempt,
                "microbatch_losses": losses,
                "microbatch_losses_all_finite": (all(math.isfinite(x) for x in losses) if losses
                                                 else expected * microbatches_per_attempt == 0),
                "inspected_boundaries": sum(c not in (UNOBSERVED, NOT_CHECKED) for c in cov),
                "boundary_coverage": cov,
                "base_optimizer_calls": sum(calls), "base_calls_per_attempt": calls,
                "base_call_multiplicity_expected": self.calls_per_boundary,
                "base_call_multiplicity_matches": mult_ok,
                "counter_updates": sum(s["counter_update"] is True for s in steps),
                "counter_update_unknown": sum(s["counter_update"] is None for s in steps),
                "framework_reported_skips": sum(bool(s["framework_reported_skip"]) for s in steps),
                "update_evidence_calls": len(ev),
                "update_evidence_verdicts": [e.get("verdict") for e in ev],
                "sampled_elements_per_call": [e.get("sampled_elements") for e in ev],
                "discriminating_samples_per_call": [e.get("discriminating_samples") for e in ev],
                "sample_plan_complete_each_boundary": (all(b == plan for b in per_boundary) if plan else None),
                "sampled_names_per_boundary": per_boundary,
                "stepped_elements_per_call": [e.get("stepped_elements") for e in ev],
                "steps": [{k: v for k, v in s.items() if k not in ("microbatch_losses", "observations")} |
                          {"microbatch_losses": [float(t.float().item()) if hasattr(t, "item") else float(t)
                                                 for t in s["microbatch_losses"]]} for s in steps],
            }
        return {"framework": self.framework, "gradient_observer": self.observer,
                "gradient_checks_requested": self.gradient_checks,
                "expected_units": len(self.expected_units) if self.expected_units is not None else None,
                "expected_elements": sum(self.expected_units.values()) if self.expected_units is not None else None,
                "base_call_multiplicity_basis": self.multiplicity_basis,
                "layout": self.layout,
                "warmup": phase_summary(self.phases["warmup"], expected_warmup),
                "measured": phase_summary(self.phases["measured"], expected_measured)}


def assess(per_rank: list[dict], *, checks_requested: bool, update_evidence_requested: bool,
           measurement_role: str, diagnostics_in_timed_path: bool) -> dict:
    """Four separate verdicts for the measured phase of a run (all ranks)."""
    m = [r["measured"] for r in per_rank]
    w = [r["warmup"] for r in per_rank]
    reasons = []
    counts_ok = all(x["logical_attempts"] == x["expected_logical_attempts"]
                    and x["microbatches"] == x["expected_microbatches"] for x in m + w)
    if not counts_ok:
        reasons.append("logical attempt or microbatch count differs from the prescribed count")
    losses_ok = all(x["microbatch_losses_all_finite"] for x in m + w)
    if not losses_ok:
        reasons.append("a microbatch loss is missing or non-finite")
    updates_ok = all(x["counter_updates"] == x["expected_logical_attempts"] and not x["counter_update_unknown"]
                     and not x["framework_reported_skips"] for x in m)
    if not updates_ok:
        reasons.append("counter evidence does not show every measured attempt as an update")
    mult_ok = all(x["base_call_multiplicity_matches"] for x in m)
    if not mult_ok:
        reasons.append("base optimizer calls differ from the implementation's per-boundary multiplicity")
    execution = "ok" if counts_ok and losses_ok and updates_ok and mult_ok else "problem_detected"

    states = [s for x in m for s in x["boundary_coverage"]]
    if not checks_requested:
        gradient = {"status": "not_checked", "gradient_validity": None,
                    "condition": "gradient checks not requested: finiteness and coverage UNVERIFIED"}
    elif NONFINITE in states:
        gradient = {"status": "nonfinite_detected", "gradient_validity": False}
    elif states and all(s == COMPLETE for s in states):
        gradient = {"status": "complete_at_every_measured_boundary", "gradient_validity": True}
    else:
        gradient = {"status": "incomplete_or_unaccounted", "gradient_validity": None,
                    "states": sorted(set(states))}

    verdicts = [v for x in m for v in x["update_evidence_verdicts"]]
    if not update_evidence_requested:
        update = {"status": "not_collected", "condition": "parameter/state evidence not collected: UNVERIFIED"}
    elif verdicts and "failed" in verdicts:
        update = {"status": "failed"}
    elif verdicts and all(v == "verified_sampled" for v in verdicts) and \
            all(x["update_evidence_calls"] == x["base_optimizer_calls"] for x in m) and \
            all(x.get("sample_plan_complete_each_boundary") is not False for x in m):
        update = {"status": "verified_sampled_every_base_call", "coverage": "sampled"}
    else:
        update = {"status": "unverified", "verdicts": sorted(set(map(str, verdicts)))}

    if measurement_role == "smoke":
        eligibility = "ineligible: smoke test, not a performance measurement"
    elif diagnostics_in_timed_path:
        eligibility = "ineligible: diagnostic instrumentation inside the timed path"
    else:
        eligibility = "unvalidated: ledger bookkeeping is in the timed path; overhead not measured"
    return {"execution": execution, "execution_reasons": reasons, "gradient": gradient, "update_state": update,
            "measurement_eligibility": eligibility}
