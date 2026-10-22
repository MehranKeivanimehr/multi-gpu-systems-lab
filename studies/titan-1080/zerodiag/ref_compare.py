"""Offline comparator for the cross-family reference control.

No GPU, no model cache, no torch required: it reads the shipped lossless
arrays and their sidecars and recomputes the frozen verdict.

The criterion is the one frozen in
`protocols/REFERENCE_CRITERION_CLARIFICATION.md`:

    u    = finfo(dtype).eps / 2
    M    = max(max|A|, max|B|)
    tol  = 64u * max(1.0, M) + 64u * max(M, 1e-300)
    pass iff max|A - B| <= tol

Missing arrays are a REFUSAL. A digest match or mismatch is never reported as
the numerical outcome.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

CRITERION = "titan_reference_logit_criterion_v1"
U = {"float32": 5.9604644775390625e-08,     # finfo(float32).eps / 2
     "float64": 1.1102230246251565e-16,
     "float16": 0.00048828125}
_CODE = {"float32": "<f4", "float64": "<f8", "float16": "<f2",
         "int64": "<i8", "int32": "<i4", "int16": "<i2",
         "uint8": "<u1", "bool": "<b1"}


def _read(dirpath, name):
    import numpy as np
    d = pathlib.Path(dirpath)
    sc = d / f"{name}.json"
    if not sc.is_file():
        raise FileNotFoundError(f"missing schema sidecar for {name!r}: {sc}")
    rec = json.loads(sc.read_text())
    payload = d / rec["file"]
    if not payload.is_file():
        raise FileNotFoundError(f"missing array payload for {name!r}: {payload}")
    raw = payload.read_bytes()
    if len(raw) != rec["n_bytes"]:
        raise ValueError(f"{name}: {len(raw)} bytes, sidecar says {rec['n_bytes']}")
    got = hashlib.sha256(raw).hexdigest()
    if got != rec["sha256"]:
        raise ValueError(f"{name}: checksum mismatch (sidecar {rec['sha256']}, "
                         f"payload {got})")
    if rec["dtype"] not in _CODE:
        raise ValueError(f"{name}: unsupported dtype {rec['dtype']!r}")
    a = np.frombuffer(raw, dtype=_CODE[rec["dtype"]])
    if a.size != rec["n_elements"]:
        raise ValueError(f"{name}: {a.size} elements, sidecar says {rec['n_elements']}")
    return a.reshape(rec["shape"]), rec


def compare(dir_a, dir_b, *, name="final_logits") -> dict:
    """Recompute the frozen verdict from two retained arrays."""
    import numpy as np
    out = {"criterion": CRITERION, "arrays": name,
           "clarification": "protocols/REFERENCE_CRITERION_CLARIFICATION.md"}
    try:
        A, ra = _read(dir_a, name)
        B, rb = _read(dir_b, name)
    except (FileNotFoundError, ValueError) as exc:
        return dict(out, verdict="REFUSED", reason=f"{type(exc).__name__}: {exc}",
                    note=("a missing or corrupt array is a refusal; a digest "
                          "comparison is never substituted for the numerical "
                          "outcome"))
    out["a"] = {k: ra[k] for k in ("dtype", "shape", "sha256", "n_elements")}
    out["b"] = {k: rb[k] for k in ("dtype", "shape", "sha256", "n_elements")}
    out["digests_identical"] = ra["sha256"] == rb["sha256"]
    if ra["dtype"] != rb["dtype"]:
        return dict(out, verdict="REFUSED",
                    reason=f"dtype mismatch {ra['dtype']} vs {rb['dtype']}")
    if list(ra["shape"]) != list(rb["shape"]):
        return dict(out, verdict="REFUSED",
                    reason=f"shape mismatch {ra['shape']} vs {rb['shape']}")
    if not (np.isfinite(A).all() and np.isfinite(B).all()):
        return dict(out, verdict="REFUSED",
                    reason=("non-finite values present; a maximum absolute "
                            "deviation is not meaningful"),
                    n_non_finite_a=int((~np.isfinite(A)).sum()),
                    n_non_finite_b=int((~np.isfinite(B)).sum()))
    dtype = ra["dtype"]
    u = U[dtype]
    A64, B64 = A.astype("float64"), B.astype("float64")
    M = float(max(np.abs(A64).max(), np.abs(B64).max()))
    atol = 64.0 * u * max(1.0, M)
    rtol_term = 64.0 * u * max(M, 1e-300)
    tol = atol + rtol_term
    diff = np.abs(A64 - B64)
    max_err = float(diff.max())
    failing = diff > tol
    n_fail = int(failing.sum())
    idx = int(np.argmax(diff))
    pos = np.unravel_index(idx, A.shape)
    return dict(out,
                verdict=("PASS" if max_err <= tol else "FAIL"),
                dtype=dtype, u=u, magnitude_scale_M=M,
                atol=atol, rtol_term=rtol_term, tolerance=tol,
                max_abs_error=max_err,
                failing_elements=n_fail,
                total_elements=int(A.size),
                worst_witness={"flat_index": idx,
                               "position": [int(x) for x in pos],
                               "a": float(A64[pos]), "b": float(B64[pos]),
                               "abs_error": max_err},
                aggregation="maximum absolute deviation over all elements",
                threshold_widened=False)


def greedy_tokens(dir_a, dir_b, *, name="generated_ids") -> dict:
    """Greedy output-token agreement, reported SEPARATELY from the logits."""
    import numpy as np
    try:
        A, ra = _read(dir_a, name)
        B, rb = _read(dir_b, name)
    except (FileNotFoundError, ValueError) as exc:
        return {"verdict": "REFUSED", "reason": f"{type(exc).__name__}: {exc}"}
    if list(ra["shape"]) != list(rb["shape"]):
        return {"verdict": "REFUSED",
                "reason": f"shape mismatch {ra['shape']} vs {rb['shape']}"}
    ne = A != B
    n = int(ne.sum())
    first = (int(np.argmax(ne)) if n else None)
    return {"verdict": "IDENTICAL" if n == 0 else "DIFFERENT",
            "criterion": "exact equality required",
            "differing_tokens": n, "total_tokens": int(A.size),
            "first_difference_flat_index": first,
            "shape": list(ra["shape"]), "dtype": ra["dtype"],
            "note": ("token agreement is reported separately from the logit "
                     "comparison and does not substitute for it")}


def verify_inputs_match(meta_a: dict, meta_b: dict) -> dict:
    """Input and initial-state matching, from the recorded metadata."""
    keys = ("realized_input_sha256", "attention_mask_sha256",
            "initial_state_sha256", "input_shape", "model_dtype",
            "resolved_revision", "dropout_active")
    checks = {k: (meta_a.get(k) is not None and meta_a.get(k) == meta_b.get(k))
              for k in keys}
    cross = meta_a.get("child_device_name") != meta_b.get("child_device_name")
    return {"matched": all(checks.values()) and cross, "checks": checks,
            "cross_family": cross,
            "devices": [meta_a.get("child_device_name"),
                        meta_b.get("child_device_name")],
            "basis": "recorded digests, not seeds"}


def resolve_arrays_dir(base, meta: dict, *, subdir: str = "arrays",
                       probe: str = "final_logits"):
    """Find the directory that actually holds the arrays.

    An evidence directory holds the metadata and an `arrays/` subdirectory; a
    bare array directory holds the sidecars directly. The metadata declares
    `arrays_dir`, so that is preferred, with the conventional subdirectory and
    then the directory itself as fallbacks. A wrong location must never be
    reported as a numerical outcome, so the chosen directory is recorded.
    """
    base = pathlib.Path(base)
    cands = []
    declared = (meta or {}).get("arrays_dir")
    if declared:
        cands.append(base / declared)
    cands += [base / subdir, base]
    for c in cands:
        if (c / f"{probe}.json").is_file():
            return c, {"resolved": str(c), "considered": [str(x) for x in cands],
                       "declared_arrays_dir": declared}
    return cands[0], {"resolved": None, "considered": [str(x) for x in cands],
                      "declared_arrays_dir": declared,
                      "note": (f"no {probe}.json was found in any candidate "
                               f"location; this is a LOCATION failure, not a "
                               f"numerical result")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Recompute the reference verdict offline from retained arrays.")
    ap.add_argument("--a", required=True, help="evidence directory for singleton A")
    ap.add_argument("--b", required=True, help="evidence directory for singleton B")
    ap.add_argument("--arrays-subdir", default="arrays")
    ap.add_argument("--json", default=None, help="write the full result here")
    a = ap.parse_args(argv)
    pa, pb = pathlib.Path(a.a), pathlib.Path(a.b)
    meta = {}
    for tag, p in (("a", pa), ("b", pb)):
        m = p / "comparison_metadata.json"
        meta[tag] = json.loads(m.read_text()) if m.is_file() else {}
    da, loc_a = resolve_arrays_dir(pa, meta["a"], subdir=a.arrays_subdir)
    db, loc_b = resolve_arrays_dir(pb, meta["b"], subdir=a.arrays_subdir)
    res = {"inputs": verify_inputs_match(meta["a"], meta["b"]),
           "array_locations": {"a": loc_a, "b": loc_b},
           "logits": compare(da, db),
           "greedy_tokens": greedy_tokens(da, db),
           "metadata": meta}
    print(json.dumps(res, indent=2, sort_keys=True))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(res, indent=2, sort_keys=True))
    v = res["logits"]["verdict"]
    return {"PASS": 0, "FAIL": 2, "REFUSED": 95}.get(v, 95)


if __name__ == "__main__":
    sys.exit(main())
