"""Strict admission of engine-aware DeepSpeed evidence.

What went wrong in the delivered gate
-------------------------------------
`deepspeed_engine_aware_evidence()` accepted an artifact with
`world_size = zero_stage = 999`, a fabricated version string and **no raw
steps**, because it only read a flag map. It also accepted every flag as the
*string* `"false"`, since a non-empty string is truthy. And because nothing
bound evidence to a precision, the 24 AMP performance rows were covered by
diagnostics that force `fp16.enabled=False` and `bf16.enabled=False` -- i.e. by
FP32 evidence. Finally the diagnostic instantiates `GPT2Config(n_positions=512)`
(124,046,592 parameters) while training recorded 124,439,808, the
`n_positions=1024` default: a different model.

What this module requires instead
---------------------------------
Evidence is admitted for a *cohort* only when the artifact is bound to that
cohort on every axis below, and when the booleans are recomputed from the raw
per-step payload rather than read from a summary flag:

  source identity, model parameter schema, precision, optimizer and engine
  config, gradient accumulation, initial state, input policy, GPU family,
  world size, ZeRO stage.

Any missing axis yields NOT ADMITTED with the axis named. Nothing is inferred.
CPU-only; imports nothing that touches a device.
"""
from __future__ import annotations

import hashlib, json, math, pathlib
from typing import Any

# Axes an artifact must pin before it may cover a performance cohort.
BINDING_AXES = ("source_sha256", "model_param_count", "model_n_positions",
                "precision", "optimizer", "gradient_accumulation_steps",
                "zero_stage", "world_size", "gpu_family", "initial_state_sha256",
                "input_policy")

# Checks that must be present AND recomputable from raw evidence.
RAW_RECOMPUTED = ("every_raw_step_shows_finite_gradients",
                  "every_raw_step_changed_the_master_partition",
                  "every_raw_step_advanced_base_optimizer_state",
                  "step_count_matches_raw",
                  "every_rank_passed_its_own_local_checks")

LIMITS = ("engine counters are boundary mechanics, not gradient coverage",
          "a changed master-partition norm shows the partition moved, not that "
          "it moved by the intended AdamW update",
          "no expected-gradient or expected-partition oracle was evaluated",
          "the DeepSpeed skip path was never exercised")


def _strict_bool(v: Any) -> bool | None:
    return v if isinstance(v, bool) else None


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def recompute_from_raw(ev: dict) -> dict[str, dict]:
    """Recompute every admissible boolean from the raw per-step payload."""
    out: dict[str, dict] = {}

    def add(name, good, detail):
        out[name] = {"ok": bool(good), "detail": detail}

    steps = ev.get("per_step_rank0")
    if not isinstance(steps, list) or not steps:
        add("raw_steps_present", False,
            f"per_step_rank0={type(steps).__name__}: no raw steps, so no boolean "
            f"can be recomputed and the artifact cannot be admitted")
        return out
    add("raw_steps_present", True, f"{len(steps)} raw steps")
    n = ev.get("steps")
    add("step_count_matches_raw", n == len(steps), f"steps={n!r} vs {len(steps)} rows")

    grads = [(s.get("grad_evidence") or {}).get("verdict") for s in steps]
    add("every_raw_step_shows_finite_gradients",
        bool(grads) and all(g == "observed_finite" for g in grads),
        f"verdicts={sorted(set(grads))}")
    srcs = [tuple((s.get("grad_evidence") or {}).get("evidence_sources_with_data") or [])
            for s in steps]
    add("every_raw_step_identified_a_gradient_source",
        all(bool(x) for x in srcs), f"sources={sorted({x for t in srcs for x in t})}")
    add("every_raw_step_changed_the_master_partition",
        all(_strict_bool(s.get("master_partition_changed")) is True for s in steps),
        "master_partition_changed must be a real boolean True on every step")
    add("every_raw_step_advanced_base_optimizer_state",
        all(_strict_bool(s.get("base_state_advanced")) is True for s in steps),
        "base_state_advanced must be a real boolean True on every step")
    gn = [(s.get("counters") or {}).get("global_grad_norm", {}).get("value") for s in steps]
    add("every_raw_step_reported_a_finite_positive_global_grad_norm",
        all(_finite(x) and x > 0 for x in gn),
        f"n={len(gn)} finite_positive={sum(1 for x in gn if _finite(x) and x > 0)}")
    ovf = [(s.get("counters") or {}).get("overflow", {}).get("value") for s in steps]
    add("no_raw_step_reported_overflow",
        all(_strict_bool(x) is False for x in ovf), f"overflow values={sorted(set(map(str, ovf)))}")

    per = ev.get("per_rank")
    ws = ev.get("world_size")
    if isinstance(per, list) and per:
        add("per_rank_count_matches_world_size", isinstance(ws, int) and len(per) == ws,
            f"{len(per)} rank records for world_size={ws!r}")
        failing = [p.get("rank") for p in per
                   if not all(_strict_bool(v) is True
                              for v in (p.get("local_checks") or {}).values())]
        add("every_rank_passed_its_own_local_checks", not failing,
            f"ranks failing locally: {failing}" if failing else f"{len(per)} ranks")
    else:
        add("per_rank_count_matches_world_size", False,
            f"per_rank={type(per).__name__}: no per-rank evidence")
        add("every_rank_passed_its_own_local_checks", False, "no per-rank evidence")

    agreed = ev.get("agreed_checks")
    if isinstance(agreed, dict) and agreed:
        nonbool = {k: v for k, v in agreed.items() if _strict_bool(v) is None}
        add("agreed_checks_are_real_booleans", not nonbool,
            f"non-boolean entries: {nonbool}" if nonbool else f"{len(agreed)} booleans")
        if not nonbool:
            add("no_agreed_check_is_false", all(agreed.values()),
                f"false: {[k for k, v in agreed.items() if v is False]}")
    else:
        add("agreed_checks_are_real_booleans", False, f"agreed_checks={type(agreed).__name__}")

    add("no_fault_injection", ev.get("fault_injection") in (None, "none"),
        f"fault_injection={ev.get('fault_injection')!r}")
    return out


def admit(artifact_path: str | pathlib.Path, cohort: dict,
          source_path: str | pathlib.Path | None = None) -> dict:
    """Decide whether one artifact may cover one cohort. Never defaults to yes."""
    res: dict[str, Any] = {
        "admitted": False, "artifact": str(artifact_path), "cohort": dict(cohort),
        "reasons": [], "unbound_axes": [], "raw_checks": {},
        "limits_of_this_evidence": list(LIMITS),
    }
    try:
        ev = json.loads(pathlib.Path(artifact_path).read_text())
    except (OSError, ValueError) as exc:
        res["reasons"].append(f"artifact unreadable: {type(exc).__name__}: {exc}")
        return res

    binding = dict(ev.get("evidence_binding") or {})
    # a legacy artifact can still be bound from fields it does happen to carry
    binding.setdefault("zero_stage", ev.get("zero_stage"))
    binding.setdefault("world_size", ev.get("world_size"))
    if source_path:
        try:
            binding.setdefault("source_sha256",
                               hashlib.sha256(pathlib.Path(source_path).read_bytes()).hexdigest())
        except OSError:
            pass

    missing = [a for a in BINDING_AXES if binding.get(a) in (None, "")]
    if missing:
        res["unbound_axes"] = missing
        res["reasons"].append(
            f"artifact does not pin {len(missing)} binding axes: {missing}; "
            f"evidence cannot be transferred to a cohort it never characterised")

    mism = []
    for axis, want in cohort.items():
        if axis not in binding:
            continue
        got = binding[axis]
        if got != want:
            mism.append((axis, got, want))
    if mism:
        res["reasons"].append(
            "artifact/cohort mismatch on " + ", ".join(
                f"{a}: artifact={g!r} cohort={w!r}" for a, g, w in mism))

    raw = recompute_from_raw(ev)
    res["raw_checks"] = raw
    failing = [k for k, v in raw.items() if not v["ok"]]
    if failing:
        res["reasons"].append(f"raw recomputation failed: {failing}")
    absent = [k for k in RAW_RECOMPUTED if k not in raw]
    if absent:
        res["reasons"].append(f"required raw checks not evaluated: {absent}")

    res["admitted"] = not res["reasons"]
    return res


# --------------------------------------------------------------- source binding
def gpt2_param_count(n_positions: int, vocab: int = 50257, n_embd: int = 768,
                     n_layer: int = 12) -> int:
    """Parameter count of a HF GPT2LMHeadModel for a given n_positions.

    Only the positional embedding depends on n_positions, so the difference
    between two configurations is exactly (n_pos_a - n_pos_b) * n_embd. The
    absolute count is anchored to the recorded training value for the 1024
    default, which keeps this independent of block-internal bookkeeping.
    """
    ANCHOR_1024 = 124_439_808
    return ANCHOR_1024 - (1024 - n_positions) * n_embd


def source_binding(source_path: str | pathlib.Path) -> dict:
    """Derive the binding a diagnostic SOURCE actually implies, by reading it."""
    import re
    p = pathlib.Path(source_path)
    try:
        src = p.read_text()
    except OSError as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    out: dict[str, Any] = {"source_sha256": hashlib.sha256(src.encode()).hexdigest()}
    m = re.search(r"n_positions\s*=\s*(\d+)", src)
    if m:
        out["model_n_positions"] = int(m.group(1))
        out["model_param_count"] = gpt2_param_count(int(m.group(1)))
    fp16_off = 'ds_cfg["fp16"] = {"enabled": False}' in src
    bf16_off = 'ds_cfg["bf16"] = {"enabled": False}' in src
    if fp16_off and bf16_off:
        out["precision"] = "fp32"
        out["precision_evidence"] = ("source forces fp16.enabled=False and "
                                     "bf16.enabled=False on every run")
    m = re.search(r"torch\.optim\.(\w+)", src)
    if m:
        out["optimizer"] = m.group(1)
    m = re.search(r'"gradient_accumulation_steps"\]\s*=\s*(\d+)', src)
    if m:
        out["gradient_accumulation_steps"] = int(m.group(1))
    if "torch.randint" in src and "generator=gen" in src:
        out["input_policy"] = ("synthetic randint from a CPU generator seeded "
                               "seed+rank, decoded per step")
    if "torch.manual_seed" in src:
        out["initial_state_sha256"] = "unrecorded: only a manual_seed is set"
    return out


def admit_with_source(artifact_path, cohort: dict, source_path) -> dict:
    """admit(), but the binding is additionally derived from the diagnostic source.

    This converts several 'axis unrecorded' outcomes into demonstrable
    mismatches, which is a stronger and more honest statement: the diagnostic
    did not merely fail to say what it ran, it provably ran something else.
    """
    sb = source_binding(source_path)
    res = admit(artifact_path, cohort, source_path=source_path)
    hard = []
    for axis in ("model_param_count", "model_n_positions", "precision",
                 "optimizer", "gradient_accumulation_steps"):
        if axis in sb and axis in cohort and sb[axis] != cohort[axis]:
            hard.append((axis, sb[axis], cohort[axis]))
    res["source_binding"] = sb
    res["demonstrable_mismatches"] = hard
    if hard:
        res["admitted"] = False
        res["reasons"].insert(0, "source-derived mismatch on " + ", ".join(
            f"{a}: diagnostic={g!r} cohort={w!r}" for a, g, w in hard))
    res["unrecorded_axes"] = [a for a in BINDING_AXES
                              if a not in sb and a not in (
                                  json.loads(pathlib.Path(artifact_path).read_text())
                                  .get("evidence_binding") or {})]
    return res
