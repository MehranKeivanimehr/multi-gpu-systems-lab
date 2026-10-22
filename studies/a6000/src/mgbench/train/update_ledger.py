"""Per-attempt accounting of gradient observation and parameter updates (recovery review 20261015r).

Optimizer hooks alone cannot show that every attempted step's gradients were inspected: when a framework skips the
optimizer call (a skipped step, a DeepSpeed overflow skip), a pre-step hook never runs. The ledger therefore keeps
four things distinct for every attempted logical step:

  attempted step          - the training loop called Trainer.step() once
  observed gradients      - mgbench itself inspected the step's gradients (where, and with what coverage)
  successful logical update - the evidence named in `logical_update_source` says the update was applied
  base optimizer calls    - how many times the underlying torch optimizer ran (ZeRO-3 may call it per subgroup)

Anything not observed stays explicitly unverified; it is never reported as finite or as a pass.
"""
from __future__ import annotations

# Per-step gradient states.
FINITE, NONFINITE = "finite", "nonfinite"
UNOBSERVED = "unobserved"                    # no observation ran for this attempted step
NO_GRADIENTS = "no_gradients"                # an observation ran but found no gradient tensors
INCOMPLETE = "incomplete_coverage"           # observed tensors were finite, but coverage is not established
NOT_CHECKED = "not_checked"                  # gradient checks were not requested (throughput runs)


def scan_gradients(tensors) -> tuple[bool | None, int]:
    """(all finite?, number of non-empty gradient tensors). None when there was nothing to inspect.
    DTensor gradients (FSDP2) are inspected as this rank's local shard."""
    import torch

    flags = []
    for t in tensors:
        if t is None:
            continue
        if hasattr(t, "to_local"):
            t = t.to_local()
        if t.numel() == 0:
            continue
        flags.append(torch.isfinite(t).all())
    if not flags:
        return None, 0
    return bool(torch.stack(flags).all().item()), len(flags)


class UpdateLedger:
    def __init__(self, framework: str, gradient_checks: bool, observer: str):
        """observer: where gradients are inspected - "in_step" (after the final backward, before and independent of
        the optimizer call) or "base_optimizer_pre_hook" (only when the framework calls the base optimizer)."""
        self.framework = framework
        self.gradient_checks = bool(gradient_checks)
        self.observer = observer
        self.steps: list[dict] = []
        self._cur: dict | None = None

    def reset(self):
        """Forget warmup steps; called once before the measured window."""
        self.steps = []
        self._cur = None

    def begin_step(self):
        self._cur = {"attempt": len(self.steps), "observations": [], "base_optimizer_calls": 0}

    def observe(self, finite: bool | None, tensors: int, source: str, coverage: str):
        """Record one gradient inspection. coverage: "full_local" (every gradient this rank holds for the step) or
        "unknown". Ignored outside an attempted step or when checks were not requested."""
        if self._cur is None or not self.gradient_checks:
            return
        self._cur["observations"].append({"finite": finite, "tensors": tensors, "source": source,
                                          "coverage": coverage})

    def base_call(self):
        if self._cur is not None:
            self._cur["base_optimizer_calls"] += 1

    def end_step(self, logical_update: bool | None, source: str, framework_reported_skip: bool | None = None,
                 expected_observations: int | None = 1):
        """Close the attempted step. logical_update None = no evidence either way (unverified).
        expected_observations: inspections that together cover every local gradient (None = not established)."""
        cur, self._cur = self._cur, None
        if cur is None:
            raise RuntimeError("end_step() without begin_step()")
        obs = cur.pop("observations")
        if not self.gradient_checks:
            grad = NOT_CHECKED
        elif not obs:
            grad = UNOBSERVED
        elif any(o["finite"] is False for o in obs):
            grad = NONFINITE
        elif any(o["finite"] is None for o in obs):
            grad = NO_GRADIENTS
        elif (expected_observations is None or len(obs) != expected_observations
              or any(o["coverage"] != "full_local" for o in obs)):
            grad = INCOMPLETE
        else:
            grad = FINITE
        cur.update({"gradient": grad,
                    "gradient_sources": sorted({o["source"] for o in obs}),
                    "gradient_tensors_inspected": sum(o["tensors"] for o in obs),
                    "logical_update": logical_update, "logical_update_source": source,
                    "framework_reported_skip": framework_reported_skip})
        self.steps.append(cur)

    def summary(self, expected_logical_updates: int) -> dict:
        steps = self.steps
        attempted = len(steps)
        grads = [s["gradient"] for s in steps]
        unverified_steps = [s["attempt"] for s in steps if s["gradient"] not in (FINITE, NONFINITE)]
        if not self.gradient_checks:
            finiteness = "unverified"
        elif NONFINITE in grads:
            finiteness = "nonfinite_detected"
        elif attempted and all(g == FINITE for g in grads):
            finiteness = "verified_finite"
        else:
            finiteness = "unverified"
        logical = sum(s["logical_update"] is True for s in steps)
        unknown = sum(s["logical_update"] is None for s in steps)
        conditions = []
        if not self.gradient_checks:
            conditions.append("gradient finiteness: checks not requested (throughput run) - not verified")
        elif unverified_steps:
            conditions.append(f"gradient finiteness: attempted steps {unverified_steps} not fully observed "
                              f"({', '.join(sorted({grads[i] for i in unverified_steps}))})")
        if unknown:
            conditions.append(f"logical update: {unknown} attempted step(s) without evidence either way")
        if attempted != expected_logical_updates:
            conditions.append(f"attempted steps {attempted} != prescribed {expected_logical_updates}")
        return {
            "framework": self.framework,
            "gradient_observer": self.observer,
            "gradient_checks_requested": self.gradient_checks,
            "attempted_steps": attempted,
            "gradient_observed_steps": sum(g in (FINITE, NONFINITE) for g in grads),
            "gradient_unverified_steps": unverified_steps,
            "gradient_nonfinite_steps": [s["attempt"] for s in steps if s["gradient"] == NONFINITE],
            "gradient_finiteness": finiteness,
            "base_optimizer_calls": sum(s["base_optimizer_calls"] for s in steps),
            "logical_updates": logical,
            "logical_updates_failed_or_skipped": sum(s["logical_update"] is False for s in steps),
            "logical_updates_unknown": unknown,
            "framework_reported_skips": sum(bool(s["framework_reported_skip"]) for s in steps),
            "expected_logical_updates": expected_logical_updates,
            "logical_updates_complete": (attempted == expected_logical_updates == logical and unknown == 0),
            "unverified_conditions": conditions,
            "steps": steps,
        }


def run_status(per_rank_audits: list[dict], gradient_checks_requested: bool) -> tuple[str, list[str]]:
    """Run-level status from every rank's audit (each with a `verification` ledger summary).

    invalid_numerics_or_updates - a problem was detected (missing/skipped update, non-finite loss or gradient)
    unverified_gradients        - nothing wrong detected, but gradient checks were requested and some attempted
                                  step's gradients were not fully observed on some rank
    ok                          - otherwise; for throughput runs the unverified gradient condition stays listed
    """
    reasons, unverified = [], []
    for rank, a in enumerate(per_rank_audits):
        v = a.get("verification") or {}
        if not v:
            reasons.append(f"rank {rank}: no update ledger")
            continue
        if not v.get("logical_updates_complete"):
            reasons.append(f"rank {rank}: logical updates {v.get('logical_updates')} of "
                           f"{v.get('expected_logical_updates')} (attempted {v.get('attempted_steps')})")
        if v.get("gradient_finiteness") == "nonfinite_detected":
            reasons.append(f"rank {rank}: non-finite gradients at attempted steps {v.get('gradient_nonfinite_steps')}")
        if not a.get("losses_all_finite", False):
            reasons.append(f"rank {rank}: non-finite loss")
        if a.get("nonfinite_gradient_steps"):
            reasons.append(f"rank {rank}: base-optimizer gradient check flagged {a['nonfinite_gradient_steps']} call(s)")
        if v.get("gradient_finiteness") != "verified_finite":
            unverified += [f"rank {rank}: {c}" for c in v.get("unverified_conditions", [])]
    if reasons:
        return "invalid_numerics_or_updates", reasons + unverified
    if gradient_checks_requested and any((a.get("verification") or {}).get("gradient_finiteness") != "verified_finite"
                                         for a in per_rank_audits):
        return "unverified_gradients", unverified
    return "ok", unverified
