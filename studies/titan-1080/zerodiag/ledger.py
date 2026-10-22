"""The authoritative context ledger: 19 contexts, each with a terminal status.

Counts are reconciled here and the human tables are generated FROM this
module, never hand-edited.

Unit discipline, because these are different things and were conflated in an
earlier table:

* **context**  — one (family, stage, world size, precision) training cell, or
  one reference case, or one streaming session. 19 in total.
* **remaining context** — a context with no terminal evidence yet. 18.
* **attempt**  — one authorized launcher invocation against one context.
* **rank record** — one per rank per attempt.
* **session** — one streaming server lifetime. One is not six.

The earlier "18-case matrix" counted the 18 REMAINING contexts and listed
Gate-0 separately as `completed_elsewhere`. The complete ledger is 19 when
Gate-0 is included. The discrepancy is resolved by counting Gate-0 in the
total, NOT by dropping another case.
"""
from __future__ import annotations

import json
import pathlib

STATUSES = (
    "completed_with_supported_checks",
    "completed_with_qualifications",
    "valid_numerical_failure",
    "invalid_diagnostic_or_software_failure",
    "not_executed",
)

GATE0 = {
    "context_id": "G0_1080ti_zero1_w1_fp32",
    "kind": "training", "family": "GTX 1080 Ti", "stage": 1, "world_size": 1,
    "precision": "fp32",
    "status": "completed_with_qualifications",
    "authorization_id": "GATE0-20261020T041835Z",
    "attempts": 1, "rank_records": 1,
    "evidence": ("private_review/gate0/attempt_gate0/run/"
                 "integration_evidence.json"),
    "evidence_sha256": ("f45b26062044bb08a560223524f23ca2d8a9be05ab926d25f"
                        "b870b36733508b1"),
    "supported_conclusion": (
        "conditional optimizer-update agreement on one GTX 1080 Ti, ZeRO-1, "
        "world size one, FP32, one audited boundary: max |param deviation| "
        "5.96046e-08 against tolerance 7.62977e-06, with intended movement "
        "1.0103e-04 so the check was discriminating; coverage complete over "
        "124,439,808 of 124,439,808 expected elements in 30 chunks"),
    "qualifications": [
        "the module/master consistency criterion in force at the time used a "
        "generic 1e-3 tolerance that was NON-discriminating for this update "
        "size; the measured difference was exactly 0.0, and that measurement, "
        "not the criterion, carries the conclusion (CORRECTION_004)",
        "complete checked coverage does not mean every individual element had "
        "a separately discriminating update; discrimination was assessed per "
        "partition from the maximum intended movement",
        "CPU arithmetic does not establish bitwise agreement with the same "
        "computation on CUDA",
    ],
    "does_not_establish": [
        "independent gradient correctness (the gradient is observed, not "
        "recomputed; dropout p=0.1 is active in the retained recipe)",
        "training quality or convergence",
        "retrospective validity of any historical run",
    ],
    "not_re_executed": ("this context is complete; later changes to the "
                        "consistency checker and the launcher do not re-open it"),
}

NOT_EXECUTED_REASON = (
    "not executed: no GPU allocation has been approved for this context. The "
    "executable path exists and passes CPU acceptance, but CPU acceptance does "
    "not establish hardware validity."
)


def _remaining(matrix_cases: list[dict]) -> list[dict]:
    out = []
    for c in matrix_cases:
        out.append({
            "context_id": c["case_id"], "kind": c["kind"],
            "family": c["family"], "stage": c.get("stage"),
            "world_size": c["world_size"], "precision": c.get("precision"),
            "group": c["group"], "device_indices": c["device_indices"],
            "status": "not_executed",
            "reason": NOT_EXECUTED_REASON,
            "attempts": 0, "rank_records": 0,
            "executable_path": True,
            "cpu_acceptance": "passed",
            "hardware_evidence": None,
        })
    return out


def build(matrix_cases: list[dict]) -> dict:
    rem = _remaining(matrix_cases)
    contexts = [GATE0] + rem
    by_status: dict[str, int] = {s: 0 for s in STATUSES}
    for c in contexts:
        by_status[c["status"]] += 1
    train = [c for c in contexts if c["kind"] == "training"]
    return {
        "schema": "titan_context_ledger_v1",
        "units": {
            "context": "one training cell, reference case or streaming session",
            "attempt": "one authorized launcher invocation against one context",
            "rank_record": "one per rank per attempt",
            "session": "one streaming server lifetime; one is not six",
        },
        "counts": {
            "contexts_total": len(contexts),
            "contexts_remaining": len(rem),
            "contexts_completed": sum(
                1 for c in contexts if c["status"].startswith("completed")),
            "training_contexts": len(train),
            "reference_contexts": sum(1 for c in contexts
                                      if c["kind"] == "reference"),
            "streaming_sessions": sum(1 for c in contexts
                                      if c["kind"] == "streaming"),
            "attempts_total": sum(c["attempts"] for c in contexts),
            "rank_records_total": sum(c["rank_records"] for c in contexts),
        },
        "by_status": by_status,
        "reconciliation_note": (
            "19 contexts = 1 completed (Gate-0) + 18 remaining. An earlier "
            "table reported 18 and listed Gate-0 separately; the total now "
            "includes Gate-0 rather than dropping any other case."),
        "four_and_six_gpu_scope": {
            "contexts": [c["context_id"] for c in contexts
                         if c.get("world_size") in (4, 6)],
            "status": "not_executed",
            "explicit": ("the four- and six-GPU GTX 1080 Ti contexts have NOT "
                         "run on hardware and are not described as validated"),
        },
        "excluded_from_results": {
            "unauthorized_cuda_incident": {
                "launcher_invocations": 5, "workers_spawned": 8,
                "cuda_contexts": 4, "device": "physical GPU 1",
                "foreign_job_impact": "unknown",
                "note": ("excluded from every result; retained privately as "
                         "incident evidence only"),
            },
        },
        "contexts": contexts,
    }


def render_table(ledger: dict) -> str:
    rows = ["| context | kind | family | stage | ws | precision | status |",
            "|---|---|---|---|---|---|---|"]
    for c in ledger["contexts"]:
        rows.append("| `{}` | {} | {} | {} | {} | {} | **{}** |".format(
            c["context_id"], c["kind"], c["family"],
            c.get("stage") if c.get("stage") is not None else "-",
            c["world_size"],
            c.get("precision") or "-", c["status"]))
    return "\n".join(rows)
