#!/usr/bin/env python3
"""Replay the pre-registered control-C comparison (R5 vs C5) on any CPU from the exported slices (stdlib only).
Recomputes G1 (first-update gradient) and U (delta-theta after 5 updates) relative L2 and cosine over the fixed
slices, and the relative per-step loss differences from the control records, applies the frozen limits and verdict
rule, and checks the values against the on-node comparator output when it is given.
Usage: replay_control_c_slices.py <calibration_slices dir> [comparison_c18b.json]
Limits (pre-registration cd367fc5...): G1 rel L2 <= 1e-4, cosine >= 0.99999; L k0 <= 1e-6, max k0-4 <= 1e-5;
U rel L2 <= 0.05, cosine >= 0.999. Upper limits: between limit and 3x = marginal; beyond = fail. A cosine below its
limit = not_pass (no band). Recipe interpretation only if every endpoint passes."""
import array
import hashlib
import json
import math
import sys
from pathlib import Path

D = Path(sys.argv[1])
man = json.loads((D / "SLICES_MANIFEST.json").read_text())
LIMITS = {"G1_rel_l2": ("upper", 1e-4), "G1_cosine": ("lower", 0.99999), "L_k0_rel": ("upper", 1e-6),
          "L_max_k0_4_rel": ("upper", 1e-5), "U_rel_l2": ("upper", 0.05), "U_cosine": ("lower", 0.999)}


def load(name):
    raw = (D / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != man["files"][name]["sha256"]:
        sys.exit(f"{name}: sha256 differs from SLICES_MANIFEST.json")
    a = array.array("f")
    a.frombytes(raw)
    if sys.byteorder != "little":
        a.byteswap()
    return [float(x) for x in a]


def metrics(o, r):
    dd = math.fsum((x - y) ** 2 for x, y in zip(o, r))
    rr = math.fsum(y * y for y in r)
    oo = math.fsum(x * x for x in o)
    ab = math.fsum(x * y for x, y in zip(o, r))
    return math.sqrt(dd / rr), ab / math.sqrt(oo * rr)


def step_losses(rec):
    out = []
    for k in range(5):
        xs = {s["global_row"]: s["raw_loss"] for s in rec["global_sample_map"] if s["step"] == k}
        out.append(math.fsum(xs[j] for j in range(4)) / 4)
    return out


def verdict(m, v):
    kind, lim = LIMITS[m]
    if kind == "upper":
        return "pass" if v <= lim else "marginal (investigate)" if v <= 3 * lim else "fail"
    return "pass" if v >= lim else "not_pass (band undefined)"


ends = {}
ends["G1_rel_l2"], ends["G1_cosine"] = metrics(load("C5_grad_b1_slices.f32"), load("R5_grad_b1_slices.f32"))
ends["U_rel_l2"], ends["U_cosine"] = metrics(load("C5_dtheta_final_slices.f32"), load("R5_dtheta_final_slices.f32"))
lr = step_losses(json.loads((D / "R5_control_record.json").read_text()))
lc = step_losses(json.loads((D / "C5_control_record.json").read_text()))
rel = [abs(a - b) / abs(b) for a, b in zip(lc, lr)]
ends["L_k0_rel"], ends["L_max_k0_4_rel"] = rel[0], max(rel)
ver = {m: verdict(m, v) for m, v in ends.items()}
print(json.dumps({"endpoints": ends, "verdicts": ver, "control_C_met": all(v == "pass" for v in ver.values()),
                  "slice_elements": len(load("R5_grad_b1_slices.f32"))}, indent=1))
bad = []
if len(sys.argv) > 2:          # agreement with the on-node comparator (numpy float64 dot products; fsum here)
    onnode = json.loads(Path(sys.argv[2]).read_text())["result"]["endpoints_preregistered"]
    for m, v in ends.items():
        w = onnode[m]
        if not (abs(v - w) <= 1e-9 * max(abs(w), 1e-300) or (w == 0 and v == 0) or abs(v - w) <= 1e-15):
            bad.append((m, v, w))
    print("agreement with the on-node comparator:", "ALL ENDPOINTS AGREE" if not bad else bad)
sys.exit(1 if bad or not all(v == "pass" for v in ver.values()) else 0)
