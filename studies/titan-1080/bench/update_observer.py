"""Per-attempt update/gradient accounting that cannot certify missing evidence.

Why this exists (defect history)
--------------------------------
v1 attached the gradient check to `optimizer.register_step_pre_hook`. GradScaler
omits `optimizer.step()` on non-finite gradients, so neither hook fired on
exactly the steps that went non-finite: an injected-inf diagnostic still
reported `nonfinite_gradient_steps == 0`.

v2 moved observation outside the conditional call but defined coverage as an
aggregate count, which produced three false certifications (review R01):
  * five attempts with *no gradient tensors at all* reported coverage complete;
  * zero observer calls reported `coverage == False` yet the run still certified;
  * the DeepSpeed path never enters this lifecycle, so `0 == 0` was a vacuous
    "complete coverage" pass.

v3 (this module) therefore records a per-attempt verdict and defines coverage as
"every attempt carries a VALIDATED observation", where validated means observed,
non-empty, and finite. Absence of evidence is never coverage. When no attempt was
made at all, coverage is `None` (unknown) rather than True.
"""

from __future__ import annotations

import math
from typing import Any, Callable

#: per-attempt gradient verdicts
OBSERVED_FINITE = "observed_finite"
OBSERVED_NONFINITE = "observed_nonfinite"
OBSERVED_EMPTY = "observed_but_no_gradient_tensors"
NOT_OBSERVED = "not_observed"

#: sentinel for fields the native observer structurally cannot measure
NOT_APPLICABLE = "not_applicable_native_observer_not_in_this_path"
UNOBSERVED = "unobserved"

VALID_SCALING = ("unscaled", "scaled", "none")


def _default_grad_reader(optimizer: Any) -> list:
    return [p.grad for group in optimizer.param_groups
            for p in group["params"] if p.grad is not None]


def _default_finite_fn(gradients: list) -> bool:
    import torch
    return all(bool(torch.isfinite(g).all().item()) for g in gradients)


class UpdateObserver:
    """Native-path observer. Counts attempts in the loop, never in a hook."""

    def __init__(self, numerical_checks: bool = False,
                 finite_fn: Callable[[list], bool] | None = None,
                 grad_reader: Callable[[Any], list] | None = None) -> None:
        self.numerical_checks = bool(numerical_checks)
        self._finite_fn = finite_fn or _default_finite_fn
        self._grad_reader = grad_reader or _default_grad_reader
        self.attempted_steps = 0
        self.base_optimizer_calls = 0
        self.successful_logical_updates = 0
        self.per_step: list[dict] = []
        self._open: dict | None = None
        self._calls_at_step_start = 0

    # -- attachment -----------------------------------------------------------
    def attach(self, optimizer: Any) -> None:
        optimizer.register_step_post_hook(self._on_base_call)

    def _on_base_call(self, *args, **kwargs) -> None:
        self.base_optimizer_calls += 1

    # -- lifecycle ------------------------------------------------------------
    def begin_step(self) -> None:
        if self._open is not None:
            raise RuntimeError("begin_step() called twice without end_step()")
        self._calls_at_step_start = self.base_optimizer_calls
        self.attempted_steps += 1
        self._open = {"step": self.attempted_steps - 1,
                      "gradient_verdict": NOT_OBSERVED,
                      "n_grad_tensors": None, "gradient_scaling": None,
                      "base_calls_in_step": 0, "successful_update": False,
                      "scale_before": None, "scale_after": None}

    def observe_gradients(self, optimizer: Any, *, scaling: str = "unscaled") -> str:
        """Inspect gradients BEFORE the conditional optimizer call.

        Returns the per-attempt verdict. An empty gradient list yields
        OBSERVED_EMPTY, which is explicitly NOT a finite result.
        """
        if self._open is None:
            raise RuntimeError("observe_gradients() outside a step")
        if scaling not in VALID_SCALING:
            raise ValueError(f"unknown scaling semantics: {scaling!r}")
        if not self.numerical_checks:
            return NOT_OBSERVED
        gradients = self._grad_reader(optimizer)
        self._open["n_grad_tensors"] = len(gradients)
        self._open["gradient_scaling"] = scaling
        if not gradients:
            self._open["gradient_verdict"] = OBSERVED_EMPTY
            return OBSERVED_EMPTY
        verdict = (OBSERVED_FINITE if bool(self._finite_fn(gradients))
                   else OBSERVED_NONFINITE)
        self._open["gradient_verdict"] = verdict
        return verdict

    def record_scale(self, before: float | None, after: float | None) -> None:
        """Corroboration only. A scale change is never evidence of an update."""
        if self._open is None:
            raise RuntimeError("record_scale() outside a step")
        self._open["scale_before"] = before
        self._open["scale_after"] = after

    def end_step(self) -> dict:
        if self._open is None:
            raise RuntimeError("end_step() without begin_step()")
        made = self.base_optimizer_calls - self._calls_at_step_start
        self._open["base_calls_in_step"] = made
        self._open["successful_update"] = made > 0
        if made > 0:
            self.successful_logical_updates += 1
        record, self._open = self._open, None
        self.per_step.append(record)
        return record

    # -- derived counts -------------------------------------------------------
    @property
    def skipped_steps(self) -> int:
        return self.attempted_steps - self.successful_logical_updates

    def _count(self, verdict: str) -> int:
        return sum(1 for r in self.per_step if r["gradient_verdict"] == verdict)

    def summary(self) -> dict:
        """Native-observer facts only. Certification lives in run_report.py."""
        finite = self._count(OBSERVED_FINITE)
        nonfinite = self._count(OBSERVED_NONFINITE)
        empty = self._count(OBSERVED_EMPTY)
        unobserved = self._count(NOT_OBSERVED)
        if not self.numerical_checks:
            coverage = None            # not requested -> unknown, never True
        elif self.attempted_steps == 0:
            coverage = None            # nothing attempted -> unknown, never True
        else:
            coverage = (finite + nonfinite) == self.attempted_steps and empty == 0
        return {
            "observer_path": "native",
            "attempted_steps": self.attempted_steps,
            "native_successful_logical_updates": self.successful_logical_updates,
            "base_optimizer_calls": self.base_optimizer_calls,
            "native_skipped_steps": self.skipped_steps,
            "gradient_checks_enabled": self.numerical_checks,
            "gradient_observations_finite": finite,
            "nonfinite_gradient_steps": nonfinite,
            "missing_gradient_steps": empty,
            "unobserved_attempt_steps": unobserved,
            # Coverage means: EVERY attempt carries a validated, non-empty
            # observation. None means unknown. It is never True by vacuity.
            "gradient_coverage_complete": coverage,
            "gradient_scaling_semantics": sorted(
                {r["gradient_scaling"] for r in self.per_step
                 if r["gradient_scaling"]}),
            "scale_change_is_not_success_evidence": True,
            "update_counter_source":
                "attempts counted by the training loop; base calls by an "
                "optimizer post-step hook; a logical update requires a base call "
                "inside that attempt; gradients observed before the conditional "
                "optimizer call",
            "per_step": list(self.per_step),
        }


def not_applicable_native_summary(reason: str) -> dict:
    """Native-observer fields for a path this observer structurally cannot see.

    Used for DeepSpeed, where scaling, overflow detection, the skip decision and
    the base call all happen inside `engine.step()`. Every native field is an
    explicit sentinel so no reader can mistake a zero for a measurement, and
    coverage is None (unknown) rather than a vacuous True.
    """
    return {
        "observer_path": "engine_internal",
        "native_observer_status": NOT_APPLICABLE,
        "native_observer_reason": reason,
        "attempted_steps": UNOBSERVED,
        "native_successful_logical_updates": NOT_APPLICABLE,
        "base_optimizer_calls": NOT_APPLICABLE,
        "native_skipped_steps": NOT_APPLICABLE,
        "gradient_observations_finite": UNOBSERVED,
        "nonfinite_gradient_steps": UNOBSERVED,
        "missing_gradient_steps": UNOBSERVED,
        "unobserved_attempt_steps": UNOBSERVED,
        "gradient_coverage_complete": None,
        "gradient_scaling_semantics": UNOBSERVED,
        "scale_change_is_not_success_evidence": True,
        "update_counter_source":
            "engine counters only; the native observer does not bracket "
            "engine.step(), so native gradient coverage is unobserved",
        "per_step": [],
    }
