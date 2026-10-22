#!/usr/bin/env python
"""Recompute the cross-family reference verdict from the shipped arrays.

Standard library only. No GPU, no model cache, no numpy, no torch: the shipped
evidence is raw little-endian binary plus a JSON schema, so this reads it with
`array` and `struct` and recomputes the frozen criterion directly.

The criterion is frozen in `protocols/REFERENCE_CRITERION_CLARIFICATION.md`:

    u    = finfo(dtype).eps / 2            # float32: 5.9604644775390625e-08
    M    = max(max|A|, max|B|)
    tol  = 64u * max(1.0, M) + 64u * max(M, 1e-300)
    pass iff max|A - B| <= tol

Exit codes: 0 PASS, 2 FAIL, 95 REFUSED (missing, corrupt, mismatched or
non-finite evidence). A refusal is NOT a numerical result, and a digest
comparison is never substituted for one.

Usage:
    python tools/recompute_reference_verdict.py --a <dir_A> --b <dir_B>
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
import sys

CRITERION = "titan_reference_logit_criterion_v1"
U = {"float32": 5.9604644775390625e-08,
     "float64": 1.1102230246251565e-16,
     "float16": 0.00048828125}
TYPECODE = {"float32": "f", "float64": "d", "int64": "q", "int32": "i"}
ITEMSIZE = {"float32": 4, "float64": 8, "int64": 8, "int32": 4}


class Refusal(Exception):
    pass


def read_array(d: str, name: str):
    """Read one shipped array, verifying the sidecar before trusting bytes."""
    sc = os.path.join(d, name + ".json")
    if not os.path.isfile(sc):
        raise Refusal(f"missing schema sidecar for {name!r}: {sc}")
    with open(sc) as f:
        rec = json.load(f)
    if rec.get("schema") != "titan_lossless_array_v1":
        raise Refusal(f"{name}: unsupported schema {rec.get('schema')!r}")
    payload = os.path.join(d, rec["file"])
    if not os.path.isfile(payload):
        raise Refusal(f"missing array payload for {name!r}: {payload}")
    with open(payload, "rb") as f:
        raw = f.read()
    if len(raw) != rec["n_bytes"]:
        raise Refusal(f"{name}: {len(raw)} bytes on disk, sidecar declares "
                      f"{rec['n_bytes']}")
    got = hashlib.sha256(raw).hexdigest()
    if got != rec["sha256"]:
        raise Refusal(f"{name}: checksum mismatch (sidecar {rec['sha256']}, "
                      f"payload {got})")
    dt = rec["dtype"]
    if dt not in TYPECODE:
        raise Refusal(f"{name}: dtype {dt!r} is not readable by this tool")
    if rec.get("byte_order") != "little":
        raise Refusal(f"{name}: byte order {rec.get('byte_order')!r}")
    a = array.array(TYPECODE[dt])
    if a.itemsize != ITEMSIZE[dt]:
        raise Refusal(f"{name}: this interpreter's {dt} is {a.itemsize} bytes")
    a.frombytes(raw)
    if sys.byteorder != "little":
        a.byteswap()
    if len(a) != rec["n_elements"]:
        raise Refusal(f"{name}: {len(a)} elements, sidecar declares "
                      f"{rec['n_elements']}")
    n = 1
    for s in rec["shape"]:
        n *= int(s)
    if n != len(a):
        raise Refusal(f"{name}: shape {rec['shape']} does not hold {len(a)} "
                      f"elements")
    return a, rec


def compare(dir_a: str, dir_b: str, name: str = "final_logits") -> dict:
    out = {"criterion": CRITERION, "arrays": name,
           "clarification": "protocols/REFERENCE_CRITERION_CLARIFICATION.md",
           "implementation": "standard library only (array/struct), no numpy"}
    try:
        A, ra = read_array(dir_a, name)
        B, rb = read_array(dir_b, name)
    except Refusal as exc:
        return dict(out, verdict="REFUSED", reason=str(exc),
                    note=("missing or corrupt evidence is a refusal, not a "
                          "numerical outcome"))
    out["a"] = {k: ra[k] for k in ("dtype", "shape", "sha256", "n_elements")}
    out["b"] = {k: rb[k] for k in ("dtype", "shape", "sha256", "n_elements")}
    out["digests_identical"] = ra["sha256"] == rb["sha256"]
    if ra["dtype"] != rb["dtype"]:
        return dict(out, verdict="REFUSED",
                    reason=f"dtype mismatch {ra['dtype']} vs {rb['dtype']}")
    if list(ra["shape"]) != list(rb["shape"]):
        return dict(out, verdict="REFUSED",
                    reason=f"shape mismatch {ra['shape']} vs {rb['shape']}")
    if ra["dtype"] not in U:
        return dict(out, verdict="REFUSED",
                    reason=f"no unit roundoff defined for {ra['dtype']}")
    nfa = sum(1 for x in A if not math.isfinite(x))
    nfb = sum(1 for x in B if not math.isfinite(x))
    if nfa or nfb:
        return dict(out, verdict="REFUSED",
                    reason=("non-finite values present; a maximum absolute "
                            "deviation is not meaningful"),
                    n_non_finite_a=nfa, n_non_finite_b=nfb)
    u = U[ra["dtype"]]
    m = 0.0
    for x in A:
        ax = abs(x)
        if ax > m:
            m = ax
    for x in B:
        ax = abs(x)
        if ax > m:
            m = ax
    atol = 64.0 * u * max(1.0, m)
    rtol_term = 64.0 * u * max(m, 1e-300)
    tol = atol + rtol_term
    max_err, worst, n_fail = 0.0, -1, 0
    for i in range(len(A)):
        e = abs(A[i] - B[i])
        if e > tol:
            n_fail += 1
        if e > max_err:
            max_err, worst = e, i
    if worst < 0:
        worst = 0
    pos, rem = [], worst
    for s in reversed([int(s) for s in ra["shape"]]):
        pos.append(rem % s)
        rem //= s
    pos.reverse()
    return dict(out,
                verdict=("PASS" if max_err <= tol else "FAIL"),
                dtype=ra["dtype"], u=u, magnitude_scale_M=m,
                atol=atol, rtol_term=rtol_term, tolerance=tol,
                max_abs_error=max_err, failing_elements=n_fail,
                total_elements=len(A),
                worst_witness={"flat_index": worst, "position": pos,
                               "a": A[worst], "b": B[worst],
                               "abs_error": max_err},
                aggregation="maximum absolute deviation over all elements",
                threshold_widened=False)


def greedy_tokens(dir_a: str, dir_b: str, name: str = "generated_ids") -> dict:
    try:
        A, ra = read_array(dir_a, name)
        B, rb = read_array(dir_b, name)
    except Refusal as exc:
        return {"verdict": "REFUSED", "reason": str(exc)}
    if list(ra["shape"]) != list(rb["shape"]):
        return {"verdict": "REFUSED",
                "reason": f"shape mismatch {ra['shape']} vs {rb['shape']}"}
    diff = [i for i in range(len(A)) if A[i] != B[i]]
    return {"verdict": "IDENTICAL" if not diff else "DIFFERENT",
            "criterion": "exact equality required",
            "differing_tokens": len(diff), "total_tokens": len(A),
            "first_difference_flat_index": (diff[0] if diff else None),
            "shape": list(ra["shape"]), "dtype": ra["dtype"],
            "note": ("token agreement is reported separately from the logit "
                     "comparison and does not substitute for it")}


def inputs_match(ma: dict, mb: dict) -> dict:
    keys = ("realized_input_sha256", "attention_mask_sha256",
            "initial_state_sha256", "input_shape", "model_dtype",
            "resolved_revision", "dropout_active")
    checks = {k: (ma.get(k) is not None and ma.get(k) == mb.get(k))
              for k in keys}
    cross = (ma.get("child_device_name") != mb.get("child_device_name")
             and ma.get("child_device_name") is not None
             and mb.get("child_device_name") is not None)
    return {"matched": all(checks.values()) and cross, "checks": checks,
            "cross_family": cross,
            "devices": [ma.get("child_device_name"), mb.get("child_device_name")],
            "basis": "recorded digests, not seeds"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", required=True, help="evidence directory, singleton A")
    ap.add_argument("--b", required=True, help="evidence directory, singleton B")
    ap.add_argument("--arrays-subdir", default="arrays")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    meta = {}
    for tag, base in (("a", a.a), ("b", a.b)):
        p = os.path.join(base, "comparison_metadata.json")
        if os.path.isfile(p):
            with open(p) as f:
                meta[tag] = json.load(f)
        else:
            meta[tag] = {}
    da = os.path.join(a.a, a.arrays_subdir)
    db = os.path.join(a.b, a.arrays_subdir)
    res = {"inputs": inputs_match(meta["a"], meta["b"]),
           "logits": compare(da, db),
           "greedy_tokens": greedy_tokens(da, db),
           "metadata": meta}
    print(json.dumps(res, indent=2, sort_keys=True))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=2, sort_keys=True)
    v = res["logits"]["verdict"]
    if v == "PASS" and not res["inputs"]["matched"]:
        print("REFUSED: the inputs and initial states do not match across the "
              "two singletons, so a PASS would not be a controlled comparison.",
              file=sys.stderr)
        return 95
    return {"PASS": 0, "FAIL": 2, "REFUSED": 95}.get(v, 95)


if __name__ == "__main__":
    sys.exit(main())
