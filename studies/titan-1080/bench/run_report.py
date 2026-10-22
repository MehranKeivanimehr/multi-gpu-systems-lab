"""Builds the update record and decides certification. Pure, no torch, testable.

Separated from train_llm.py so the FINAL status decision can be exercised
directly by CPU tests, which is what review R01 required: the previous candidate
tested helper counters and source ordering but never the status actually emitted.

Certification contract
----------------------
`status == "ok"` asserts two different things and only when both hold:

  * the UPDATE claim: the prescribed number of logical updates demonstrably
    happened, evidenced by counters appropriate to the execution path;
  * the NUMERICAL DIAGNOSTIC claim, *if and only if* diagnostics were requested:
    every attempt carries a validated, non-empty, finite gradient observation.

Absence of evidence never satisfies either. Unknown (`None`) is treated as
failure, not as a pass. A disclosure string never substitutes for a field.
"""

from __future__ import annotations

import math
from typing import Any

NATIVE_MODES = ("single", "ddp", "fsdp")
ENGINE_MODES = ("deepspeed",)

STATUS_OK = "ok"
STATUS_UPDATES_UNVERIFIED = "updates_unverified"
STATUS_NUMERICS_UNVERIFIED = "numerics_unverified"
STATUS_NUMERICS_UNOBSERVABLE = "numerics_unobservable_on_engine_path"


DS_EVIDENCE_ENV = "BENCH_DS_EVIDENCE"
# Checks a conforming engine-aware observation artifact must carry AND pass.
DS_REQUIRED_CHECKS = (
    "gradients_observed_finite_every_step",
    "gradient_source_identified_every_step",
    "master_partition_changed_every_step",
    "base_optimizer_state_advanced_every_step",
    "base_optimizer_state_counter_equals_steps_driven",
    "engine_global_steps_reconcile",
    "every_rank_drove_the_same_step_count",
    "global_grad_norm_finite_and_positive",
)


def deepspeed_engine_aware_evidence(path: str | None = None) -> dict[str, Any]:
    """Resolve the engine-aware DeepSpeed validation status from an artifact.

    Never defaults to validated. A missing, unreadable, incomplete or failing
    artifact yields a non-validating status, and the reason is recorded. The
    artifact must itself claim every check in DS_REQUIRED_CHECKS and pass it,
    so adding a check to this tuple invalidates older artifacts rather than
    silently accepting them.
    """
    import json
    import os

    src = path if path is not None else os.environ.get(DS_EVIDENCE_ENV)
    out: dict[str, Any] = {"validated": False, "evidence_path": src,
                           "reason": None, "checks_passed": None,
                           "zero_stage": None, "world_size": None}
    if not src:
        out["reason"] = "no engine-aware observation artifact supplied"
        return out
    try:
        with open(src) as fh:
            ev = json.load(fh)
    except (OSError, ValueError) as exc:
        out["reason"] = f"artifact unreadable: {type(exc).__name__}: {exc}"
        return out
    agreed = ev.get("agreed_checks")
    if not isinstance(agreed, dict):
        out["reason"] = "artifact has no agreed_checks map"
        return out
    missing = [k for k in DS_REQUIRED_CHECKS if k not in agreed]
    if missing:
        out["reason"] = f"artifact omits required checks: {missing}"
        return out
    failed = [k for k in DS_REQUIRED_CHECKS if not agreed[k]]
    if failed:
        out["reason"] = f"artifact reports failing checks: {failed}"
        return out
    if ev.get("fault_injection") not in (None, "none"):
        out["reason"] = "artifact reports fault injection; not an unperturbed observation"
        return out
    out.update({"validated": True, "reason": None,
                "checks_passed": sorted(DS_REQUIRED_CHECKS),
                "zero_stage": ev.get("zero_stage"),
                "world_size": ev.get("world_size"),
                "deepspeed_version": ev.get("deepspeed_version")})
    return out


def build_update_record(*, mode: str, numerical_checks: bool,
                        expected_logical_updates: int,
                        native_summary: dict | None = None,
                        engine_counters: dict | None = None,
                        losses: list[float] | None = None) -> dict:
    """Assemble one rank's record. Native and engine counters stay separate.

    No `successful_updates` alias is emitted. The previous candidate's alias
    reported the ENGINE boundary count under a name that readers compared
    against NATIVE attempt counts, producing a record that simultaneously said
    `attempted_steps=0` and `successful_updates=5`.
    """
    record: dict[str, Any] = dict(native_summary or {})
    record["mode"] = mode
    record["expected_logical_updates"] = int(expected_logical_updates)
    record["gradient_checks_requested"] = bool(numerical_checks)

    if mode in ENGINE_MODES:
        counters = dict(engine_counters or {})
        boundaries = counters.get("engine_update_boundaries")
        skipped = counters.get("engine_skipped_updates")
        record["engine_update_boundaries"] = boundaries
        record["engine_skipped_updates"] = skipped
        record["engine_successful_updates"] = (
            boundaries - skipped
            if isinstance(boundaries, int) and isinstance(skipped, int) else None)
        record["engine_base_optimizer_calls"] = counters.get(
            "engine_base_optimizer_calls")
        # The NATIVE claim never becomes true: nothing brackets engine.step().
        record["deepspeed_hardware_validated"] = False
        record["deepspeed_observer_status"] = (
            "engine counters only; no native gradient observation brackets "
            "engine.step(); the DeepSpeed skip path has never been exercised on "
            "any host; this is NOT a native hardware validation of DeepSpeed")
        # A separate engine-aware observation can nonetheless supply positive
        # evidence. It is reported alongside, never folded into, the native
        # status, so a reader cannot mistake one for the other.
        ev = deepspeed_engine_aware_evidence()
        record["deepspeed_engine_aware_validation"] = ev
        record["deepspeed_engine_aware_validated"] = bool(ev["validated"])
        record["deepspeed_skip_path_exercised"] = False
    record["losses_all_finite"] = bool(losses) and all(
        isinstance(x, (int, float)) and math.isfinite(x) for x in losses)
    record["parameter_state_delta_evidence"] = (
        "not collected in benchmark runs; requires a separate approved diagnostic")
    return record


def certify(record: dict) -> dict:
    """Decide the final status. Returns {status, certified, reasons, flags}."""
    mode = record.get("mode")
    expected = record.get("expected_logical_updates")
    requested = bool(record.get("gradient_checks_requested"))
    reasons: list[str] = []

    if not isinstance(expected, int) or expected < 0:
        reasons.append(f"expected_logical_updates is not a valid count: {expected!r}")
        expected = None

    # ---- update claim -----------------------------------------------------
    updates_ok = False
    if mode in NATIVE_MODES:
        attempted = record.get("attempted_steps")
        success = record.get("native_successful_logical_updates")
        calls = record.get("base_optimizer_calls")
        if not all(isinstance(v, int) for v in (attempted, success, calls)):
            reasons.append(
                "native update counters are not integers "
                f"(attempted={attempted!r}, successful={success!r}, calls={calls!r})")
        elif expected is None:
            pass
        elif attempted != expected:
            reasons.append(f"attempted_steps {attempted} != expected {expected}")
        elif success != expected:
            reasons.append(
                f"native_successful_logical_updates {success} != expected {expected}")
        elif calls < expected:
            reasons.append(f"base_optimizer_calls {calls} < expected {expected}")
        else:
            updates_ok = True
    elif mode in ENGINE_MODES:
        success = record.get("engine_successful_updates")
        skipped = record.get("engine_skipped_updates")
        if not isinstance(success, int) or not isinstance(skipped, int):
            reasons.append(
                "engine counters are not integers "
                f"(successful={success!r}, skipped={skipped!r})")
        elif expected is None:
            pass
        elif success != expected:
            reasons.append(f"engine_successful_updates {success} != expected {expected}")
        elif skipped != 0:
            reasons.append(f"engine reported {skipped} skipped update(s)")
        else:
            updates_ok = True
    else:
        reasons.append(f"unknown execution mode {mode!r}")

    if not record.get("losses_all_finite"):
        reasons.append("losses are absent or not all finite")
        updates_ok = False

    # ---- numerical diagnostic claim ---------------------------------------
    diagnostic_certified = None
    if requested:
        if mode in ENGINE_MODES:
            diagnostic_certified = False
            reasons.append(
                "numerical diagnostic requested but gradient observation is "
                "structurally unobservable on the engine path; "
                "no native observation brackets engine.step()")
        else:
            coverage = record.get("gradient_coverage_complete")
            missing = record.get("missing_gradient_steps")
            unobserved = record.get("unobserved_attempt_steps")
            nonfinite = record.get("nonfinite_gradient_steps")
            problems: list[str] = []
            if coverage is not True:
                problems.append(
                    f"gradient_coverage_complete is {coverage!r}, not True")
            if not isinstance(missing, int) or missing != 0:
                problems.append(f"missing_gradient_steps={missing!r}")
            if not isinstance(unobserved, int) or unobserved != 0:
                problems.append(f"unobserved_attempt_steps={unobserved!r}")
            if not isinstance(nonfinite, int) or nonfinite != 0:
                problems.append(f"nonfinite_gradient_steps={nonfinite!r}")
            diagnostic_certified = not problems
            reasons.extend(problems)

    # ---- final status -----------------------------------------------------
    if not updates_ok:
        status = STATUS_UPDATES_UNVERIFIED
    elif requested and diagnostic_certified is False:
        status = (STATUS_NUMERICS_UNOBSERVABLE if mode in ENGINE_MODES
                  else STATUS_NUMERICS_UNVERIFIED)
    else:
        status = STATUS_OK

    return {
        "status": status,
        "certified": status == STATUS_OK,
        "update_claim_verified": updates_ok,
        "numerical_diagnostic_certified": diagnostic_certified,
        "reasons": reasons,
    }


def apply_certification(record: dict) -> dict:
    """Merge the verdict into the record. Never writes a positive field it
    cannot support."""
    verdict = certify(record)
    record["status_verdict"] = verdict["status"]
    record["numerical_diagnostic_certified"] = verdict["numerical_diagnostic_certified"]
    record["update_claim_verified"] = verdict["update_claim_verified"]
    record["certification_reasons"] = verdict["reasons"]
    record["all_updates_observed"] = verdict["update_claim_verified"]
    return record
