"""Post-window summarizer for the first-boundary result kind (Candidate 15). CPU only: numpy + stdlib, no torch.

Reads (never writes) a window's per-case records and full first-boundary gradient snapshots plus the sealed S1a r0
reference; writes one new summary file into an explicit, fresh output root. Every tensor is streamed in bounded chunks
from both files, its per-tensor sha256 re-verified, and statistics accumulated in float64.

Validation (any failure makes the case `refused`; numbers are then reported only as unverified):
  * snapshot index: unique names, contiguous offsets, numel*4 bytes, shapes, no tied alias, the reference's exact set
    of tensors/shapes/elements (no missing/extra/duplicate);
  * case record: result kind, one boundary / four microbatches / zero optimizer calls / zero updates, update state
    `not_applicable`, parameters unchanged, the four frozen input digests, canonical start, recipe fields, extraction
    phase (DeepSpeed: engine.step() not reached at the boundary), finite raw losses;
  * data: every value finite; a degenerate endpoint (global norm 0 on either side) is never certifiable, while a
    legitimate zero tensor inside a nonzero endpoint is allowed and listed (relative error undefined for it).
No equivalence margin is applied: errors are reported, not graded.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

CHUNK = 1 << 22
TINY_REL = 1e-12          # tensors with ref norm < TINY_REL x global ref norm are listed as "tiny" (relative error
#                           reported but not interpretable on its own); exactly zero gives no relative value at all


def read_index(path: Path) -> tuple[dict, list[str]]:
    meta = json.loads(Path(str(path) + ".index.json").read_text())
    problems, off, names = [], 0, []
    size = Path(path).stat().st_size
    for e in meta.get("index", []):
        if e["offset"] != off:
            problems.append(f"{e['name']}: offset {e['offset']} != contiguous {off}")
        if int(np.prod(e["shape"])) != e["numel"]:
            problems.append(f"{e['name']}: shape {e['shape']} != numel {e['numel']}")
        off += 4 * e["numel"]
        names.append(e["name"])
    if len(set(names)) != len(names):
        problems.append("duplicate tensor names in the index")
    if off != size or meta.get("bytes") != size or meta.get("elements") != off // 4:
        problems.append(f"index totals {meta.get('bytes')}/{meta.get('elements')} disagree with file size {size}")
    return meta, problems


def _stream(fa, ea, fb, eb, chunk):
    """float64 sums over one tensor pair read in chunks; per-tensor sha256 of both sides verified."""
    ha, hb = hashlib.sha256(), hashlib.sha256()
    fa.seek(ea["offset"]); fb.seek(eb["offset"])
    dd = aa = bb = ab = 0.0
    best = (-1.0, None, None, None)
    finite_a = finite_b = True
    left, base = ea["numel"], 0
    while left:
        n = min(left, chunk)
        ra, rb = fa.read(4 * n), fb.read(4 * n)
        ha.update(ra); hb.update(rb)
        a = np.frombuffer(ra, dtype="<f4").astype(np.float64)
        b = np.frombuffer(rb, dtype="<f4").astype(np.float64)
        finite_a &= bool(np.isfinite(a).all()); finite_b &= bool(np.isfinite(b).all())
        if finite_a and finite_b:
            d = a - b
            dd += float(d @ d); aa += float(a @ a); bb += float(b @ b); ab += float(a @ b)
            j = int(np.abs(d).argmax()) if n else 0
            if n and abs(d[j]) > best[0]:
                best = (float(abs(d[j])), base + j, float(a[j]), float(b[j]))
        left -= n; base += n
    return {"sha_ok_obs": ha.hexdigest() == ea["sha256"], "sha_ok_ref": hb.hexdigest() == eb["sha256"],
            "finite_obs": finite_a, "finite_ref": finite_b, "dd": dd, "oo": aa, "rr": bb, "or": ab, "witness": best}


def compare_snapshots(obs: Path, ref: Path, *, chunk: int = CHUNK, tied_aliases: dict | None = None) -> dict:
    mo, po = read_index(obs)
    mr, pr = read_index(ref)
    problems = [f"observed: {p}" for p in po] + [f"reference: {p}" for p in pr]
    eo = {e["name"]: e for e in mo.get("index", [])}
    er = {e["name"]: e for e in mr.get("index", [])}
    for alias in (a for v in (tied_aliases or {}).values() for a in v):
        if alias in eo:
            problems.append(f"tied alias {alias} stored separately (would double-count)")
    missing, extra = sorted(set(er) - set(eo)), sorted(set(eo) - set(er))
    if missing or extra:
        problems.append(f"tensor sets differ: missing {missing[:5]} extra {extra[:5]}")
    per = []
    with open(obs, "rb") as fa, open(ref, "rb") as fb:
        for name in sorted(set(eo) & set(er)):
            a, b = eo[name], er[name]
            if list(a["shape"]) != list(b["shape"]):
                problems.append(f"{name}: shape {a['shape']} != reference {b['shape']}")
                continue
            s = _stream(fa, a, fb, b, chunk)
            if not (s["sha_ok_obs"] and s["sha_ok_ref"]):
                problems.append(f"{name}: sha256 mismatch (observed {s['sha_ok_obs']}, reference {s['sha_ok_ref']})")
            if not (s["finite_obs"] and s["finite_ref"]):
                problems.append(f"{name}: non-finite values (observed finite {s['finite_obs']}, reference {s['finite_ref']})")
            ro, rr = math.sqrt(s["oo"]), math.sqrt(s["rr"])
            per.append({"name": name, "numel": a["numel"], "ref_l2": rr, "obs_l2": ro,
                        "rel_l2": math.sqrt(s["dd"]) / rr if rr > 0 else None,
                        "cosine": s["or"] / (ro * rr) if ro > 0 and rr > 0 else None,
                        "norm_ratio": ro / rr if rr > 0 else None, "max_abs_diff": s["witness"][0],
                        "witness": {"flat_index": s["witness"][1], "observed": s["witness"][2], "reference": s["witness"][3]},
                        "_s": s})
    dd = math.fsum(p["_s"]["dd"] for p in per); oo = math.fsum(p["_s"]["oo"] for p in per)
    rr = math.fsum(p["_s"]["rr"] for p in per); orr = math.fsum(p["_s"]["or"] for p in per)
    for p in per:
        del p["_s"]
    gref = math.sqrt(rr)
    degenerate = not (rr > 0 and oo > 0)
    worst = max((p for p in per if p["rel_l2"] is not None), key=lambda p: p["rel_l2"], default=None)
    wmax = max(per, key=lambda p: p["max_abs_diff"], default=None)
    out = {"observed_file": str(obs), "reference_file": str(ref), "tensors": len(per),
           "elements": sum(p["numel"] for p in per), "problems": problems,
           "global": {"rel_l2": math.sqrt(dd / rr) if rr > 0 else None,
                      "cosine": orr / math.sqrt(oo * rr) if not degenerate else None,
                      "norm_ratio": math.sqrt(oo / rr) if rr > 0 else None,
                      "identity_check_abs": (abs(dd / rr - (oo / rr + 1 - 2 * orr / rr)) if rr > 0 else None)},
           "degenerate_endpoint": degenerate,
           "zero_reference_tensors": [p["name"] for p in per if p["ref_l2"] == 0],
           "zero_observed_tensors": [p["name"] for p in per if p["obs_l2"] == 0],
           "tiny_reference_tensors": [p["name"] for p in per if 0 < p["ref_l2"] < TINY_REL * gref],
           "worst_tensor_by_rel_l2": {k: worst[k] for k in ("name", "numel", "rel_l2", "cosine", "norm_ratio")} if worst else None,
           "largest_abs_difference": {"name": wmax["name"], **wmax["witness"], "abs": wmax["max_abs_diff"]} if wmax else None,
           "per_tensor": per}
    out["certifiable"] = not problems and not degenerate
    return out


def validate_case(result: dict | None, frozen: dict) -> list[str]:
    """Record-level evidence for one first-boundary case. Any attempted update claim is refused."""
    if not isinstance(result, dict):
        return ["result.json missing or unreadable"]
    out = []
    if result.get("result_kind") != "first_boundary_gradient":
        out.append(f"result kind {result.get('result_kind')!r}")
    a = result.get("assessment") or {}
    led = result.get("boundary_ledger") or {}
    m, w = led.get("measured") or {}, led.get("warmup") or {}
    if (m.get("logical_attempts"), m.get("microbatches"), w.get("logical_attempts")) != (1, 4, 0):
        out.append(f"attempts/microbatches {(m.get('logical_attempts'), m.get('microbatches'))}, warmup "
                   f"{w.get('logical_attempts')}; expected (1, 4), 0")
    if m.get("base_optimizer_calls") != 0 or m.get("counter_updates") != 0 or \
            (result.get("update_audit") or {}).get("optimizer_calls", 0) != 0:
        out.append("an optimizer call or update is recorded (attempted update claim)")
    if (a.get("update_state") or {}).get("status") != "not_applicable":
        out.append(f"update state {(a.get('update_state') or {}).get('status')!r} (must stay not_applicable)")
    if result.get("parameters_unchanged") is not True or a.get("parameters_unchanged") is not True:
        out.append("parameters/masters not shown unchanged")
    if a.get("execution") != "ok" or (a.get("gradient") or {}).get("status") != "complete_at_the_boundary":
        out.append(f"execution {a.get('execution')!r} / coverage {(a.get('gradient') or {}).get('status')!r}")
    losses = m.get("microbatch_losses") or []
    if len(losses) != 4 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in losses):
        out.append(f"raw microbatch losses {losses!r} (need 4 finite)")
    dig = (result.get("input_digests") or {}).get("per_step_microbatch_sha256")
    if dig != frozen["input_per_step_microbatch_sha256"][:1]:
        out.append("input/label digests differ from the first frozen step")
    if (result.get("input_digests") or {}).get("attention_mask", "missing") is not None:
        out.append("attention mask is not the frozen 'none'")
    cs = result.get("canonical_state_check") or {}
    spec = result.get("spec") or {}
    if cs.get("module_state_matches_canonical") is not True or \
            (spec.get("framework", "").startswith("zero") and cs.get("optimizer_master_matches_canonical") is not True):
        out.append("realized canonical start not shown")
    for k, v in frozen["recipe"].items():
        if spec.get(k) != v:
            out.append(f"spec.{k} = {spec.get(k)!r}, frozen {v!r}")
    order = result.get("microbatch_order")
    if order is not None and (sorted(order) != [0, 1, 2, 3] or order != (spec.get("microbatch_order") or [0, 1, 2, 3])):
        out.append(f"microbatch order {order!r} is not the declared permutation {spec.get('microbatch_order')!r}")
    steps = m.get("steps") or [{}]
    src = str(steps[0].get("counter_source", ""))
    if "first-boundary" not in src:
        out.append(f"extraction phase not recorded as first-boundary ({src!r})")
    if spec.get("framework", "").startswith("zero") and (steps[0].get("engine_global_steps"),
                                                        steps[0].get("engine_micro_steps")) != (0, 3):
        out.append(f"DeepSpeed bookkeeping {(steps[0].get('engine_global_steps'), steps[0].get('engine_micro_steps'))}, "
                   "expected global_steps 0 and micro_steps 3 (boundary reached by the 4th backward, step not called)")
    if spec.get("framework", "").startswith("zero") and steps[0].get("boundary_flags_before_backward") != [False] * 3 + [True]:
        out.append(f"DeepSpeed boundary predicate {steps[0].get('boundary_flags_before_backward')!r}, "
                   "expected [False, False, False, True]")
    return out


def summarize(window: Path, reference: Path, out_root: Path, frozen: dict, *, pairs=(), chunk: int = CHUNK) -> dict:
    """window/cases/NN_<id>/{result.json, first_boundary_grad.f32(.index.json)}; reference = sealed r0 directory.
    Writes out_root/fb_summary.json (out_root must not exist yet)."""
    out_root = Path(out_root)
    out_root.mkdir(mode=0o700)                                    # fresh output root; never reused
    scal = json.loads((Path(reference) / "reference_scalars.json").read_text())
    ref_losses = scal["microbatch_losses_measured"][:4]
    cases = {}
    for d in sorted(p for p in (Path(window) / "cases").iterdir() if p.is_dir()):
        cid = d.name.split("_", 1)[1]
        rp = d / "result.json"
        result = json.loads(rp.read_text()) if rp.exists() else None
        rec = {"dir": str(d), "record_problems": validate_case(result, frozen)}
        snap = d / "first_boundary_grad.f32"
        if snap.exists() and Path(str(snap) + ".index.json").exists():
            rec["vs_sealed_r0"] = compare_snapshots(snap, Path(reference) / "grad_b1.f32", chunk=chunk,
                                                    tied_aliases=frozen.get("tied_aliases"))
        else:
            rec["record_problems"].append("first-boundary snapshot missing")
        losses = ((result or {}).get("boundary_ledger") or {}).get("measured", {}).get("microbatch_losses") or []
        order = (result or {}).get("microbatch_order") or list(range(len(losses)))
        by_gen = [None] * len(losses)                                  # applied order -> generation order
        for j, i in enumerate(order[:len(losses)]):
            by_gen[i] = losses[j]
        rec["microbatch_order"] = order
        rec["raw_loss_differences"] = [{"microbatch": k, "observed": a, "reference": b, "abs": abs(a - b),
                                        "rel": abs(a - b) / abs(b) if b else None}
                                       for k, (a, b) in enumerate(zip(by_gen, ref_losses)) if a is not None]
        ok = not rec["record_problems"] and rec.get("vs_sealed_r0", {}).get("certifiable") is True
        rec["status"] = "summarized" if ok else "refused (numbers unverified)"
        rec["update_verification"] = "not_applicable"
        cases[cid] = rec
    pw = {}
    for a, b in pairs:
        if a in cases and b in cases and all(cases[x]["status"] == "summarized" for x in (a, b)):
            da, db = Path(cases[a]["dir"]), Path(cases[b]["dir"])
            pw[f"{a}_vs_{b}"] = compare_snapshots(da / "first_boundary_grad.f32", db / "first_boundary_grad.f32",
                                                  chunk=chunk, tied_aliases=frozen.get("tied_aliases"))
        else:
            pw[f"{a}_vs_{b}"] = {"status": "not computed: a case is missing or refused"}
    summary = {"kind": "first_boundary_gradient summary", "window": str(window), "reference": str(reference),
               "chunk_elements": chunk, "grading": "none (no justified equivalence margin); errors reported",
               "cases": cases, "pairwise": pw}
    with open(out_root / "fb_summary.json", "x") as f:
        json.dump(summary, f, indent=1)
    return summary
