"""The frozen diagnostic case matrix: 18 remaining cases.

Gate-0 (`G0_1080ti_zero1_w1_fp32`, the GTX 1080 Ti / ZeRO-1 / world size 1 /
FP32 context) is COMPLETE and counts once. It is listed here as `completed`
for reconciliation and is never re-executed by this matrix.

Device identities are carried from the frozen campaign plan
(`protocols/device_groups_frozen.json`), not retyped. Groups are homogeneous by
construction: the TITAN Xp singleton is present in the source plan but is NOT a
member of any group used here, is never substituted for a GTX 1080 Ti, and is
never pooled with the TITAN RTX pair.

These are DIAGNOSTIC cases, not performance measurements. Deviations from the
performance recipe are declared in `protocols/diagnostic_protocol.md`.
"""
from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
def _groups_path() -> pathlib.Path:
    """Where the frozen device inventory is read from.

    `ZERODIAG_GROUPS_PATH` substitutes the INPUT file. It does not disable,
    weaken or bypass any check: every identity, homogeneity, rank-binding and
    receipt comparison still runs, against whatever inventory is supplied. CPU
    acceptance tests point it at a synthetic inventory whose UUIDs exist on no
    host, so a test can never open a real device.
    """
    import os
    env = os.environ.get("ZERODIAG_GROUPS_PATH")
    return (pathlib.Path(env) if env
            else ROOT / "protocols" / "device_groups_labelled.json")

# Frozen per-case diagnostic defaults. These differ from the performance
# recipe on purpose; see the protocol document.
DIAG_DEFAULTS = {
    "model": "retained GPT2 (GPT2Config defaults, 124,439,808 unique params)",
    "batch_size": 4,
    "seq_len": 512,
    "accumulation_steps": 1,
    "warmup_steps": 0,
    "audited_boundaries": 1,
    "worker_timeout_s": 180,
    "retries": 0,
    "chunk_elems": 4_194_304,
    "optimizer": "torch.optim.AdamW lr=1e-4, torch defaults elsewhere",
    "gradient_clipping": 1.0,
}


def device_groups() -> dict:
    """The frozen homogeneous device groups, with their provenance."""
    p = _groups_path()
    if not p.is_file():
        raise FileNotFoundError(
            f"frozen device groups not found at {p}; the matrix refuses to "
            f"invent device identities")
    return json.loads(p.read_text())


def _group_for(family: str, world_size: int) -> str:
    if family == "GTX 1080 Ti":
        return {1: "p1080_1", 2: "p1080_2", 4: "p1080_4", 6: "p1080_6"}[world_size]
    if family == "TITAN RTX":
        return {1: "rtx_1", 2: "rtx_2"}[world_size]
    raise ValueError(f"no homogeneous group defined for family {family!r}")


def _training_case(family: str, stage: int, world_size: int, precision: str) -> dict:
    g = _group_for(family, world_size)
    grp = device_groups()["groups"][g]
    if len(grp["device_indices"]) != world_size:
        raise ValueError(
            f"group {g} has {len(grp['device_indices'])} devices but world size "
            f"{world_size} was requested")
    names = set(grp["device_names"])
    if len(names) != 1:
        raise ValueError(f"group {g} is not homogeneous: {sorted(names)}")
    tag = {"GTX 1080 Ti": "p1080", "TITAN RTX": "rtx"}[family]
    pid = {"fp32": "fp32", "amp_fp16": "amp"}[precision]
    return dict(
        DIAG_DEFAULTS,
        case_id=f"D_{tag}_{world_size}_zero{stage}_{pid}",
        kind="training",
        family=family, group=g, stage=stage, world_size=world_size,
        precision=precision, amp=(precision == "amp_fp16"),
        device_indices=list(grp["device_indices"]),
        device_uuids=list(grp["device_uuids"]),
        device_names=list(grp["device_names"]),
        topology_worst_link=grp["topology_worst_link"],
        # Expected base-optimizer calls are NOT fixed at one here. DeepSpeed's
        # _optimizer_step(i) runs once per fp32 master group per rank, so the
        # expected count is derived at runtime from the installed engine's own
        # partitioning and checked against that. Gate-0's single call reflected
        # a single-group layout and is not generalised.
        expected_base_calls="derived from len(single_partition_of_fp32_groups)",
        expected_ranks=list(range(world_size)),
    )


GATE0_COMPLETED = {
    "case_id": "G0_1080ti_zero1_w1_fp32", "kind": "training",
    "family": "GTX 1080 Ti", "group": "p1080_1", "stage": 1, "world_size": 1,
    "precision": "fp32", "amp": False, "disposition": "completed",
    "authorization_id": "GATE0-20261020T041835Z",
    "evidence": ("titan_gate0fix3_20261020/out/attempt_gate0/run/"
                 "integration_evidence.json"),
    "evidence_sha256": ("f45b26062044bb08a560223524f23ca2d8a9be05ab926d25"
                        "fb870b36733508b1"),
    "not_re_executed": ("this context is complete; the consistency-criterion "
                        "correction does not re-open it"),
}


def training_cases() -> list[dict]:
    """The 15 remaining training contexts."""
    out = []
    for ws in (2, 4, 6):                       # Ti ZeRO-1 FP32
        out.append(_training_case("GTX 1080 Ti", 1, ws, "fp32"))
    for ws in (1, 2, 4, 6):                    # Ti ZeRO-2 FP32
        out.append(_training_case("GTX 1080 Ti", 2, ws, "fp32"))
    for stage in (1, 2):                       # RTX FP32
        for ws in (1, 2):
            out.append(_training_case("TITAN RTX", stage, ws, "fp32"))
    for stage in (1, 2):                       # RTX AMP
        for ws in (1, 2):
            out.append(_training_case("TITAN RTX", stage, ws, "amp_fp16"))
    return out


def reference_cases() -> list[dict]:
    """The two agreed singleton FP32 cross-family reference cases.

    The source plan also contains a TITAN Xp FP32 singleton. It is a third
    family and is deliberately NOT part of this two-singleton design; it is not
    substituted for either member.
    """
    g = device_groups()["groups"]
    cases = []
    for family, grp, src in (("TITAN RTX", "rtx_1", "H_rtx_1_tinyllama_float32"),
                             ("GTX 1080 Ti", "p1080_1",
                              "H_p1080_1_tinyllama_float32")):
        cases.append({
            "case_id": f"R_{grp}_tinyllama_fp32", "kind": "reference",
            "family": family, "group": grp, "world_size": 1,
            "precision": "fp32", "amp": False,
            "device_indices": list(g[grp]["device_indices"]),
            "device_uuids": list(g[grp]["device_uuids"]),
            "device_names": list(g[grp]["device_names"]),
            "plan_source_case": src,
            "recipe": "hf_generate, greedy, FP32",
            "design": ("two singletons, one per family, same frozen inputs and "
                       "initial state; a cross-family loss/logit agreement "
                       "control"),
            "establishes": ("that both families produce agreeing outputs on "
                            "identical realized inputs from an identical "
                            "initial state"),
            "does_not_establish": ("independent full-gradient verification; "
                                   "loss agreement is not gradient "
                                   "correctness"),
            "historical_status": ("the historical reference limitations are "
                                  "preserved; this control is INFORMED, not "
                                  "blind"),
            "worker_timeout_s": 300, "retries": 0,
        })
    return cases


def streaming_case() -> dict:
    """The prospective dedicated streaming case: 256 input / 64 output."""
    g = device_groups()["groups"]["rtx_1"]
    return {
        "case_id": "S_rtx_1_stream_256_64", "kind": "streaming",
        "family": "TITAN RTX", "group": "rtx_1", "world_size": 1,
        "device_indices": list(g["device_indices"]),
        "device_uuids": list(g["device_uuids"]),
        "device_names": list(g["device_names"]),
        "protocol": {
            "server": "one fresh loopback-only server, this session only",
            "requests": 100, "concurrency": 4,
            "prompt_len_tokens": 256, "output_tokens": 64,
            "model_revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
            "corpus": "bundle/inputs/prompt_corpus.json",
            "corpus_sha256": ("f2d11005cbce3e5419be77a3c3b05eff7a0b3e56b791e"
                              "0d3c13327d2ac611350"),
            "greedy": True,
        },
        "reporting": ("session variation and descriptive request summaries "
                      "only; no tail percentiles, no internal GPU TTFT, no "
                      "chunk-gap-as-ITL"),
        "token_timing_rule": ("per-token timing is reported ONLY if token "
                             "identity and timestamp attribution are genuinely "
                             "established; otherwise first-text and chunk "
                             "metrics are reported under those names"),
        "sessions": ("one new session. One session is NOT six independent "
                     "repeat sessions and is not reported as such."),
        "historical_untouched": ("the historical 278-token inputs and the "
                                 "preserved 64/128 protocol contradiction are "
                                 "not rewritten"),
        "worker_timeout_s": 900, "retries": 0,
    }


def full_matrix() -> dict:
    tr, rf, st = training_cases(), reference_cases(), streaming_case()
    cases = tr + rf + [st]
    return {
        "schema": "titan_diagnostic_matrix_v1",
        "completed_elsewhere": [GATE0_COMPLETED],
        "remaining_cases": cases,
        "counts": {"training": len(tr), "reference": len(rf), "streaming": 1,
                   "remaining_total": len(cases), "completed": 1,
                   "all_contexts": len(cases) + 1},
        "homogeneity": ("every group is single-model; TITAN Xp is never "
                        "substituted for a GTX 1080 Ti and never pooled with "
                        "the TITAN RTX pair"),
        "not_a_performance_campaign": ("these are diagnostics; the historical "
                                       "300-run campaign is not re-run"),
    }
