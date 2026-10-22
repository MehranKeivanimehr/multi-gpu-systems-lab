"""Candidate 16: comparator for the pre-registered positive control C (R5 vs C5). CPU only: numpy + stdlib.

Pre-registration: v2_validation_20261014a/evidence/analysis/numerical_reference_preregistration.md, sha256
cd367fc553cb56d38cd7d17673c442253757ddc42923dfac85a8bcf67a668ce3. This R5/C5 follow-up is informed by prior results; it
is NOT a blind original validation.

Encoded exactly as written there:
  * G1 (primary): first-update gradient, relative L2 ||g - g_R|| / ||g_R|| and cosine, over the CONCATENATED FIXED SLICES
    (first 262,144 elements of model.layers.0.self_attn.q_proj.weight, model.layers.0.mlp.down_proj.weight,
    model.layers.<last>.mlp.up_proj.weight, model.norm.weight). Limits for C: rel <= 1e-4, cosine >= 0.99999.
  * L (primary): per-step global loss, RELATIVE |L_k - L_R,k| / L_R,k, for k = 0 (<= 1e-6) and max over k = 0..4
    (<= 1e-5). (The owner's request paraphrased these as absolute differences; the original is relative. Both are
    reported; only the original relative form is graded.)
  * U (secondary, descriptive in general; graded for C by its tolerance row): Delta-theta = theta_5 - theta_0 after FIVE
    updates, relative L2 <= 0.05 and cosine >= 0.999, on the same slices.
  * Verdict rule (verbatim): "Outcomes between the pass limit and 3x the limit are reported as marginal (investigate);
    beyond 3x as fail." Applied to upper-bound limits only. For cosine LOWER bounds "3x the limit" is undefined, so a
    cosine below its limit is reported as `not_pass (marginal/fail band undefined in the pre-registration)`.
  * Stopping rule (verbatim): "if C fails its tight tolerance, the harness (not the recipes) is at fault and recipe
    outcomes are not interpreted." Whether a 'marginal' C counts as 'fails' is not stated; conservatively, recipe
    interpretation is permitted only if EVERY C endpoint is `pass` (ambiguity exposed in the output).
Full-vector G1/U values are computed as SUPPLEMENTARY evidence (not the pre-registered endpoint) and not graded.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .fb_summary import compare_snapshots, read_index

PREREG_SHA256 = "cd367fc553cb56d38cd7d17673c442253757ddc42923dfac85a8bcf67a668ce3"
SLICE_ELEMENTS = 262_144
LIMITS = {"G1_rel_l2": ("upper", 1e-4), "G1_cosine": ("lower", 0.99999), "L_k0_rel": ("upper", 1e-6),
          "L_max_k0_4_rel": ("upper", 1e-5), "U_rel_l2": ("upper", 0.05), "U_cosine": ("lower", 0.999)}
AMBIGUITIES = [
    "cosine marginal/fail band: '3x the limit' has no meaning for a lower bound; not resolved, reported as not_pass",
    "stopping rule: whether 'marginal (investigate)' counts as 'fails its tight tolerance' is not stated; recipe "
    "interpretation is permitted here only if every endpoint passes",
    "L is defined as RELATIVE |L_k - L_R,k| / L_R,k in the pre-registration (the follow-up request said absolute); "
    "the relative form is graded, the absolute difference is reported"]


def slice_names(n_layers: int) -> list[str]:
    return ["model.layers.0.self_attn.q_proj.weight", "model.layers.0.mlp.down_proj.weight",
            f"model.layers.{n_layers - 1}.mlp.up_proj.weight", "model.norm.weight"]


def verdict(metric: str, value) -> str:
    kind, limit = LIMITS[metric]
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return "missing"
    if kind == "upper":
        return "pass" if value <= limit else "marginal (investigate)" if value <= 3 * limit else "fail"
    return "pass" if value >= limit else "not_pass (marginal/fail band undefined in the pre-registration)"


def _read_head(path: Path, entry: dict, n: int) -> np.ndarray:
    with open(path, "rb") as f:
        f.seek(entry["offset"])
        raw = f.read(4 * min(n, entry["numel"]))
    return np.frombuffer(raw, dtype="<f4").astype(np.float64)


def sliced_metrics(obs: Path, ref: Path, names: list[str]) -> dict:
    """Relative L2 and cosine over the concatenated fixed slices (pre-registered endpoint). The slice tensors are
    sha256-verified in full by compare_snapshots(); here only their first SLICE_ELEMENTS are read."""
    mo, _ = read_index(obs)
    mr, _ = read_index(ref)
    eo, er = {e["name"]: e for e in mo["index"]}, {e["name"]: e for e in mr["index"]}
    missing = [n for n in names if n not in eo or n not in er]
    if missing:
        return {"problems": [f"slice tensors missing: {missing}"], "rel_l2": None, "cosine": None, "elements": 0}
    a = np.concatenate([_read_head(obs, eo[n], SLICE_ELEMENTS) for n in names])
    b = np.concatenate([_read_head(ref, er[n], SLICE_ELEMENTS) for n in names])
    if a.size != b.size or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return {"problems": ["slice sizes differ or non-finite values"], "rel_l2": None, "cosine": None, "elements": int(a.size)}
    d = a - b
    rr, oo = float(b @ b), float(a @ a)
    return {"problems": [], "elements": int(a.size),
            "rel_l2": math.sqrt(float(d @ d) / rr) if rr > 0 else None,
            "cosine": float(a @ b) / math.sqrt(oo * rr) if oo > 0 and rr > 0 else None}


def run_contract_problems(rec: dict, *, world: int, grad_accum: int, frozen: dict) -> list[str]:
    """Evidence contract for one full-control run record (R5: world 1, GA 4; C5: world 2, GA 2)."""
    out = []
    if not isinstance(rec, dict):
        return ["run record missing"]
    if rec.get("result_kind") != "full_control_5update":
        out.append(f"result kind {rec.get('result_kind')!r}")
    spec = rec.get("spec") or {}
    for k, v in dict(frozen["recipe"], n_gpus=world, grad_accum=grad_accum, measure_steps=5, warmup_steps=0,
                     fp32_reference=True).items():
        if spec.get(k) != v:
            out.append(f"spec.{k} = {spec.get(k)!r}, required {v!r}")
    ranks = rec.get("per_rank") or []
    if sorted(r.get("rank") for r in ranks) != list(range(world)):
        out.append(f"per-rank evidence for ranks {[r.get('rank') for r in ranks]}, required {list(range(world))}")
    for r in ranks:
        m = r.get("measured") or {}
        if (m.get("logical_attempts"), m.get("microbatches"), m.get("counter_updates")) != (5, 5 * grad_accum, 5) or \
                m.get("base_calls_per_attempt") != [1] * 5 or m.get("boundary_coverage") != ["complete"] * 5 or \
                m.get("microbatch_losses_all_finite") is not True or (r.get("warmup") or {}).get("logical_attempts") != 0:
            out.append(f"rank {r.get('rank')}: attempts/microbatches/updates/base calls/coverage not "
                       f"(5, {5 * grad_accum}, 5, [1]*5, complete) or non-finite losses")
        if r.get("update_state") != "verified_sampled_every_base_call":
            out.append(f"rank {r.get('rank')}: update state {r.get('update_state')!r} (applicable here; must be verified)")
        if r.get("identity_ok") is not True:
            out.append(f"rank {r.get('rank')}: in-process device identity not verified")
    gm = rec.get("global_sample_map") or []
    rows = sorted((s.get("step"), s.get("global_row")) for s in gm)
    if rows != [(k, j) for k in range(5) for j in range(4)]:
        out.append("global sample map does not cover exactly global rows 0..3 at each of the 5 steps")
    for s in gm:
        exp = frozen["global_row_digests"][s.get("step", 0)][s.get("global_row", 0)] if \
            isinstance(s.get("step"), int) and isinstance(s.get("global_row"), int) and s["step"] < 5 and s["global_row"] < 4 else None
        if s.get("digest") != exp:
            out.append(f"step {s.get('step')} row {s.get('global_row')}: input digest differs from the frozen map")
            break
        if s.get("global_row") != s.get("microstep", -1) * world + s.get("rank", -1):
            out.append(f"step {s.get('step')}: rank {s.get('rank')} microstep {s.get('microstep')} took row "
                       f"{s.get('global_row')}, required (i*world + r)")
            break
    if rec.get("canonical_state_matches") is not True:
        out.append("realized canonical start not shown")
    snaps = rec.get("snapshots") or {}
    if snaps.get("writer_rank") != 0 or set(snaps.get("files", [])) != {"grad_b1.f32", "dtheta_final.f32"}:
        out.append(f"snapshots {snaps!r}: required grad_b1 and dtheta_final written by rank 0 only")
    if snaps.get("dtheta_after_update") != 5:
        out.append(f"delta-theta taken after update {snaps.get('dtheta_after_update')!r}, required 5")
    return out


def global_step_losses(gm: list[dict]) -> list[float | None]:
    """L_k = mean over the 4 global samples of each sample's mean token loss (weight 1/4 each, identity-checked)."""
    out = []
    for k in range(5):
        xs = {s["global_row"]: s["raw_loss"] for s in gm if s.get("step") == k}
        out.append(math.fsum(xs[j] for j in range(4)) / 4 if sorted(xs) == [0, 1, 2, 3] and
                   all(isinstance(v, (int, float)) and math.isfinite(v) for v in xs.values()) else None)
    return out


def compare_control(r5_dir: Path, c5_dir: Path, frozen: dict, *, chunk: int = 1 << 22) -> dict:
    r5 = json.loads((Path(r5_dir) / "control_record.json").read_text()) if (Path(r5_dir) / "control_record.json").exists() else None
    c5 = json.loads((Path(c5_dir) / "control_record.json").read_text()) if (Path(c5_dir) / "control_record.json").exists() else None
    problems = [f"R5: {p}" for p in run_contract_problems(r5, world=1, grad_accum=4, frozen=frozen)] + \
               [f"C5: {p}" for p in run_contract_problems(c5, world=2, grad_accum=2, frozen=frozen)]
    names = slice_names(frozen["n_layers"])
    ends, supp = {}, {}
    for snap, key in (("grad_b1.f32", "G1"), ("dtheta_final.f32", "U")):
        o, r = Path(c5_dir) / snap, Path(r5_dir) / snap
        if not (o.exists() and r.exists()):
            problems.append(f"{snap} missing")
            continue
        full = compare_snapshots(o, r, chunk=chunk, tied_aliases=frozen.get("tied_aliases"))
        problems += [f"{snap}: {p}" for p in full["problems"]]
        if full["degenerate_endpoint"]:
            problems.append(f"{snap}: degenerate (zero) endpoint")
        sl = sliced_metrics(o, r, names)
        problems += [f"{snap} slices: {p}" for p in sl["problems"]]
        ends[f"{key}_rel_l2"], ends[f"{key}_cosine"] = sl["rel_l2"], sl["cosine"]
        supp[key] = {"full_vector": {k: full["global"][k] for k in ("rel_l2", "cosine", "norm_ratio")},
                     "tensors": full["tensors"], "elements": full["elements"], "slice_elements": sl["elements"],
                     "worst_tensor": full["worst_tensor_by_rel_l2"], "largest_abs_difference": full["largest_abs_difference"]}
    lr = global_step_losses((r5 or {}).get("global_sample_map") or [])
    lc = global_step_losses((c5 or {}).get("global_sample_map") or [])
    rel = [abs(a - b) / abs(b) if a is not None and b not in (None, 0) else None for a, b in zip(lc, lr)]
    ends["L_k0_rel"] = rel[0] if rel else None
    ends["L_max_k0_4_rel"] = max(rel) if rel and None not in rel else None
    supp["L"] = {"R5": lr, "C5": lc, "relative": rel,
                 "absolute": [abs(a - b) if a is not None and b is not None else None for a, b in zip(lc, lr)]}
    verdicts = {k: verdict(k, ends.get(k)) for k in LIMITS}
    all_pass = not problems and all(v == "pass" for v in verdicts.values())
    return {"preregistration_sha256": PREREG_SHA256, "informed_by_prior_results": True,
            "endpoints_preregistered": ends, "verdicts": verdicts, "limits": {k: v[1] for k, v in LIMITS.items()},
            "supplementary_not_graded": supp, "problems": problems, "ambiguities": AMBIGUITIES,
            "control_C_met": all_pass,
            "recipe_interpretation": "permitted (all C endpoints pass)" if all_pass else
            "NOT interpreted: control C did not meet its tight tolerance or its evidence is incomplete",
            "scope_of_claim": "calibrates only the declared harness/control claim; not historical training, quality "
                              "or distributed performance"}
