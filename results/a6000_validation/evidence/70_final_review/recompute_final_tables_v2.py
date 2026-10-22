#!/usr/bin/env python3
"""v2: recompute every number quoted in FINAL_REPORT_v2.md / CLAIM_LEDGER_v2.csv and check it (stdlib only, CPU only).
v2 applies corrections C34 (equal-cap DP vs TP) and C35 (true medians; scope split), replays control C from the exported
slices and rebuilds the claim-evidence table.
Usage: recompute_final_tables_v2.py <evidence root> <repository root> <pilot evidence dir>
  From the review ZIP: <pilot evidence dir> = <unzipped>/pilot_evidence
  On the node:        <evidence root> = a6000_evidence_root,   <repository root> = 20_candidate_c20/multigpu-bench
  From the review ZIP: <evidence root> = <unzipped>/evidence, <repository root> = <unzipped>/source/multigpu-bench
Exit 0 only if every check holds. Author of the repository: Mehran Keivanimehr."""
import csv
import json
import math
import sys
from pathlib import Path

E, REPO, PILOT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
fails, lines = [], []


def check(name, got, want, tol=None):
    ok = (got == want) if tol is None else (got is not None and math.isfinite(got) and abs(got - want) <= tol)
    lines.append(f"{'ok ' if ok else 'BAD'} {name}: {got!r} (expected {want!r}{'' if tol is None else f' +/- {tol:g}'})")
    if not ok:
        fails.append(name)


# 1. Control C (R5 vs C5), window w18b
c = json.loads((E / "44_c18_comparison/comparison_c18b.json").read_text())
r = c["result"]
check("control C met", r["control_C_met"], True)
check("control C verdicts all pass", sorted(set(r["verdicts"].values())), ["pass"])
ep = r["endpoints_preregistered"]
check("C: G1 rel L2", ep["G1_rel_l2"], 5.23e-8, 0.01e-8)
check("C: loss rel max k0-4", ep["L_max_k0_4_rel"], 1.95e-8, 0.01e-8)
check("C: U5 rel L2", ep["U_rel_l2"], 1.013e-5, 0.001e-5)
check("C: preregistration sha", r["preregistration_sha256"], "cd367fc553cb56d38cd7d17673c442253757ddc42923dfac85a8bcf67a668ce3")

# 2. ZeRO attribution, window w20fb
s = json.loads((E / "48_c20_fb_summary/out/fb_summary.json").read_text())
pw = s["pairwise"]
check("G4 vs G3 rel L2 (bitwise)", pw["G4_ds_bf16_vs_G3_ddp_bf16_params"]["global"]["rel_l2"], 0.0)
check("G3 vs G1 rel L2", pw["G3_ddp_bf16_params_vs_G1_ddp_autocast"]["global"]["rel_l2"], 0.2195, 0.0001)
check("G1 vs G0 rel L2", pw["G1_ddp_autocast_vs_G0_fp32_control"]["global"]["rel_l2"], 0.0362, 0.0001)
check("X1 vs G0 rel L2", pw["X1_fp32_order_permuted_vs_G0_fp32_control"]["global"]["rel_l2"], 5.35e-8, 0.01e-8)
check("G4 vs sealed r0 rel L2", s["cases"]["G4_ds_bf16"]["vs_sealed_r0"]["global"]["rel_l2"], 0.21754, 0.00001)
check("G0 vs sealed r0 rel L2", s["cases"]["G0_fp32_control"]["vs_sealed_r0"]["global"]["rel_l2"], 0.0)
check("all 5 fb cases summarized", sorted({v["status"] for v in s["cases"].values()}), ["summarized"])

# 3. S1a recipes interpreted after control C
i = json.loads((E / "61_recipe_interpretation/s1a_interpretation.json").read_text())
check("S1a reference == calibrated R5 grad_b1", i["s1a_reference_equals_calibrated_R5_grad_b1"], True)
for cid, want in (("a1_ddp_bf16_autocast", 0.0399), ("a2_fsdp2_bf16", 0.0496), ("a3_zero1_bf16", 0.2183), ("a4_zero2_bf16", 0.2184)):
    check(f"S1a {cid} full-vector G1", i["cases"][cid]["G1_full_vector"]["rel_l2"], want, 0.0001)

# 4. Inference DP vs TP at matched caps (historical records)
rows = list(csv.DictReader(open(REPO / "studies/a6000/results/rerun/vllm_tp_concurrency_comparison.csv")))


def mean_tok(model, tp, dp, cap):            # C34: equal TOTAL concurrency cap
    v = [float(x["output_tok_s"]) for x in rows if x["model"] == model and x["tp"] == str(tp) and x["dp"] == str(dp)
         and x["total_concurrency_cap"] == str(cap)]
    return sum(v) / len(v), (max(v) - min(v)) / (sum(v) / len(v))


for model, want in (("qwen2.5-1.5b", 33.23), ("llama3.1-8b", 14.99)):
    dp, sdp = mean_tok(model, 1, 2, 512)
    tp, stp = mean_tok(model, 2, 1, 512)
    check(f"{model} DP2 advantage over TP2 at equal cap 512 (%)", round(100 * (dp / tp - 1), 2), want)
    check(f"{model} repeat spread <= 0.4%", max(sdp, stp) <= 0.004, True)

# 5. Training repeat dispersion (historical)
rep = list(csv.DictReader(open(REPO / "results/corrections/a6000_repeats.csv")))
import statistics
check("training repeat groups", len(rep), 28)
hist = [x for x in rep if int(x["n_gpus"]) >= 4 or "8b" in x["model"].lower()]
cur = [x for x in rep if x not in hist]
for label, g, n, mcv, xcv, mrg, xrg in (("all", rep, 28, 0.15, 0.66, 0.29, 1.22), ("1-2 GPU non-8B", cur, 16, 0.12, 0.47, 0.23, 0.85),
                                        ("historical 4-GPU/8B", hist, 12, 0.19, 0.66, 0.36, 1.22)):
    cv = [float(x["sample_cv_pct"]) for x in g]; rg = [float(x["full_range_over_median_pct"]) for x in g]
    check(f"{label}: groups", len(g), n)
    check(f"{label}: median CV % (C35)", round(statistics.median(cv), 2), mcv)
    check(f"{label}: max CV %", round(max(cv), 2), xcv)
    check(f"{label}: median range/median % (C35)", round(statistics.median(rg), 2), mrg)
    check(f"{label}: max range/median %", round(max(rg), 2), xrg)

# 5b. control C replayed from the exported slices; claim-evidence table rebuilt; ledger v2 counts
import subprocess, tempfile, os
FR = E / "70_final_review"
p = subprocess.run([sys.executable, "-B", "-I", str(FR / "replay_control_c_slices.py"), str(FR / "calibration_slices"),
                    str(E / "44_c18_comparison/comparison_c18b.json")], capture_output=True, text=True)
check("control C replay from slices: all pass and agree with the on-node comparator", p.returncode == 0 and
      "ALL ENDPOINTS AGREE" in p.stdout, True)
with tempfile.TemporaryDirectory() as td:
    out = Path(td) / "t.csv"
    q = subprocess.run([sys.executable, "-B", "-I", str(FR / "build_claim_evidence_table_v2.py"), str(E),
                        str(PILOT / "analysis"), str(out)], capture_output=True, text=True)
    check("claim-evidence table rebuilt byte-identical", q.returncode == 0 and out.read_bytes() ==
          (FR / "CLAIM_EVIDENCE_TABLE.csv").read_bytes(), True)
led = list(csv.DictReader(open(FR / "CLAIM_LEDGER_v2.csv")))
from collections import Counter
check("ledger v2 claims", len(led), 28)
check("ledger v2 status counts", dict(sorted(Counter(x["status"] for x in led).items())),
      {"deferred": 2, "historical_qualified": 1, "not_established": 3, "supported": 8, "supported_qualified": 10, "withdrawn": 4})

# 6. Hardware windows: every cleanup confirmed
wins = ["40_smoke/w20261016T042911Z", "41_smoke_s2/w2_20261016T051201Z", "42_s1a_window/w1a_20261016T061814Z",
        "43_c18_window/w18_20261017T003855Z", "43_c18_window/w18b_20261020T122336Z",
        "45_c19_window/w19fb_20261021T224030Z", "47_c20_window/w20fb_20261021T233546Z"]
for w in wins:
    sm = json.loads((E / w / "window_summary.json").read_text())
    check(f"{w} all cleanups confirmed", sm.get("all_cleanups_confirmed"), True)

print("\n".join(lines))
print(f"\n{len(lines) - len(fails)} of {len(lines)} checks hold" + (f"; FAILED: {fails}" if fails else ""))
sys.exit(1 if fails else 0)
