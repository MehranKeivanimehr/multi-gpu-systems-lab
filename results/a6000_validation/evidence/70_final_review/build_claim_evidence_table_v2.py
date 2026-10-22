#!/usr/bin/env python3
"""Per model x framework/recipe x GPU count x evidence source: what was OBSERVED, kept in separate columns:
optimizer calls, gradient coverage, sampled update checks, and reference agreement. Built only from retained records
(stdlib only). v2: pilot rows per experiment (the first training_pilot run predates base-call recording), and the S1a
ledger is read from optimizer_update_audit_per_rank[0].boundary_ledger (draft1 read a key S1a does not have). Usage: build_claim_evidence_table.py <evidence root> <pilot analysis dir> <output csv>"""
import csv
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

E, PA, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rows = []
COLS = ["model", "recipe", "n_gpus", "evidence_source", "optimizer_calls", "gradient_coverage", "sampled_update_checks",
        "reference_agreement", "scope_note"]

# 1. 2026-10-14 pilot per-run diagnostics (all ranks), training_pilot / rerun1 / recipe_diagnostics
diag = defaultdict(list)
for r in csv.DictReader(open(PA / "training_diagnostics_table.csv")):
    diag[(r["run"].split("__")[0], r["framework"], r["gpus"], r["experiment"])].append(r)
for (model, fw, g, exp), rs in sorted(diag.items()):
    recorded = all(r["base_calls_recorded"] not in ("", "0") for r in rs)
    calls = all(r["logical_updates"] == r["expected_updates"] and r["base_calls_recorded"] == r["expected_updates"]
                and r["skipped"] == "0" for r in rs)
    grads = all(r["gradient_checks"] == r["expected_updates"] and r["nonfinite_gradient_steps"] == "0" for r in rs)
    state = all(r["adam_step_deltas_all_1"] == "True" for r in rs)
    rows.append({"model": model, "recipe": fw + " (retained recipe)", "n_gpus": g,
                 "evidence_source": f"pilot 2026-10-14 {exp} ({len(rs)} rank-records)",
                 "optimizer_calls": (f"{'observed: logical = expected = base calls, 0 skipped' if calls else 'MISMATCH'} "
                                     f"({rs[0]['expected_updates']} updates)") if recorded else
                                    (f"logical {rs[0]['logical_updates']}/{rs[0]['expected_updates']}, 0 skipped; base-call "
                                     "recording not yet available in this first run (added for the pilot rerun)"),
                 "gradient_coverage": "finite nonempty gradients every step (no per-unit coverage accounting)" if grads
                                      else "INCOMPLETE",
                 "sampled_update_checks": "optimizer-state step counter advanced by 1 per update (no sampled AdamW check)"
                                          if state else ("not recorded in this first run" if not recorded else "NOT SHOWN"),
                 "reference_agreement": "-",
                 "scope_note": "4-GPU: historical pilot run, outside the current 2-GPU guard" if g == "4" else ""})

# 2. pilot numerical reference (pre-registered, recipes at world 2)
for r in csv.DictReader(open(PA / "numref_final/numerical_reference_results.csv")):
    if r["category"] in ("", "reference"):
        continue
    model = {"qwen05b": "qwen2.5-0.5b", "qwen15b": "qwen2.5-1.5b", "resnet50a2": "resnet50"}.get(r["model"], r["model"])
    fw = r["run"].split("__")[1] if "__" in r["run"] else r["run"]
    post = "POST-HOC" in r["category"]
    rows.append({"model": model, "recipe": fw + (" fp32 positive control" if r["category"] == "positive_control" else
                                                 " (gacc32 post-hoc)" if post else " (retained recipe)"),
                 "n_gpus": "2", "evidence_source": "pilot 2026-10-14 numerical reference (pre-registered cd367fc5...)",
                 "optimizer_calls": "see pilot diagnostics row", "gradient_coverage": "-", "sampled_update_checks": "-",
                 "reference_agreement": f"G1 rel L2 {float(r['G1_rel_l2']):.3g}, cosine {float(r['G1_cosine']):.6f} "
                                        f"vs FP32 world-1 reference: {r['outcome']}",
                 "scope_note": "post-hoc, not pre-registered" if post else ""})

# 3. S1a (world 1, 3 updates, Qwen2.5-1.5B), graded against the sealed FP32 control r0
S = E / "42_s1a_window/w1a_20261016T061814Z/cases"
for d in sorted(S.iterdir()):
    r = json.loads((d / "result.json").read_text())
    m = (((r.get("optimizer_update_audit_per_rank") or [{}])[0].get("boundary_ledger")) or {}).get("measured") or {}
    rc = r.get("reference_comparison") or {}
    g = ((rc.get("snapshots") or {}).get("grad_b1") or {})
    ver = m.get("update_evidence_verdicts") or []
    rows.append({"model": "qwen2.5-1.5b", "recipe": d.name.split("_", 1)[1], "n_gpus": "1",
                 "evidence_source": "S1a window w1a_20261016T061814Z (3 updates, production precision settings)",
                 "optimizer_calls": f"{m.get('logical_attempts')} attempts / {m.get('counter_updates')} updates / base calls "
                                    f"{m.get('base_calls_per_attempt')}",
                 "gradient_coverage": f"per-unit accounting: {m.get('boundary_coverage')}",
                 "sampled_update_checks": f"sampled AdamW: {sum(m.get('discriminating_samples_per_call') or [])} discriminating "
                                          f"samples over {len(ver)} calls; verdicts {sorted(set(map(str, ver)))}",
                 "reference_agreement": ("reference (FP32 control)" if "r0" in d.name else
                                         f"G1 rel L2 {g.get('global_rel_l2', float('nan')):.4f}, cosine "
                                         f"{g.get('global_cosine', float('nan')):.6f} (full vector) vs r0"),
                 "scope_note": "S1a deviates from the pre-registered design: world 1 (not 2), 3 updates (not 5), "
                               "full-vector grading, informed tolerances (C18)"})

# 4. R5/C5 control C (window w18b, 5 updates, FP32 DDP)
W = E / "43_c18_window/w18b_20261020T122336Z/cases"
comp = json.loads((E / "44_c18_comparison/comparison_c18b.json").read_text())["result"]
for d, g in ((W / "01_R5_fp32_world1_ga4", "1"), (W / "02_C5_fp32_world2_ga2", "2")):
    c = json.loads((d / "control_record.json").read_text())
    pr = c["per_rank"]
    rows.append({"model": "qwen2.5-1.5b", "recipe": "ddp FP32 (control C " + ("R5)" if g == "1" else "C5)"), "n_gpus": g,
                 "evidence_source": "window w18b_20261020T122336Z (5 updates; pre-registered control C)",
                 "optimizer_calls": "; ".join(f"rank {p['rank']}: {p['measured']['logical_attempts']} attempts / "
                                              f"{p['measured']['counter_updates']} updates / base calls "
                                              f"{p['measured']['base_calls_per_attempt']}" for p in pr),
                 "gradient_coverage": "; ".join(f"rank {p['rank']}: {p['measured']['boundary_coverage']}" for p in pr),
                 "sampled_update_checks": "; ".join(f"rank {p['rank']}: {p['update_state']}" for p in pr),
                 "reference_agreement": "reference (R5)" if g == "1" else
                     f"control C vs R5: G1 {comp['endpoints_preregistered']['G1_rel_l2']:.3g}, U "
                     f"{comp['endpoints_preregistered']['U_rel_l2']:.3g}, all endpoints {sorted(set(comp['verdicts'].values()))}",
                 "scope_note": ""})

# 5. First-boundary attribution (window w20fb; matched precision policy; no optimizer update by design)
fb = json.loads((E / "48_c20_fb_summary/out/fb_summary.json").read_text())
F = E / "47_c20_window/w20fb_20261021T233546Z/cases"
for d in sorted(F.iterdir()):
    r = json.loads((d / "result.json").read_text())
    m = (r.get("boundary_ledger") or {}).get("measured") or {}
    cid = d.name.split("_", 1)[1]
    v = fb["cases"][cid]["vs_sealed_r0"]["global"]
    rows.append({"model": "qwen2.5-1.5b", "recipe": cid, "n_gpus": "1",
                 "evidence_source": "window w20fb_20261021T233546Z (one accumulation boundary; TF32 off, math SDPA)",
                 "optimizer_calls": f"{m.get('base_optimizer_calls')} by design (first-boundary result kind)",
                 "gradient_coverage": f"per-unit accounting: {m.get('boundary_coverage')}",
                 "sampled_update_checks": "not applicable (no update)",
                 "reference_agreement": f"G1 rel L2 {v['rel_l2']:.4g}, cosine {v['cosine']:.6f} vs sealed r0 (full vector; "
                                        "reported, not graded)",
                 "scope_note": "G4 vs G3 bitwise identical" if cid in ("G3_ddp_bf16_params", "G4_ds_bf16") else ""})

bad = [r for r in rows if "None" in r["optimizer_calls"] or "MISMATCH" in r["optimizer_calls"] or "NOT SHOWN" in
       r["sampled_update_checks"] or "INCOMPLETE" in r["gradient_coverage"]]
if bad:
    print("rows needing attention:", [(r["model"], r["recipe"], r["n_gpus"], r["evidence_source"]) for r in bad])
with open(OUT, "x", newline="") as f:
    w = csv.DictWriter(f, fieldnames=COLS)
    w.writeheader()
    w.writerows(rows)
print(len(rows), "rows")
