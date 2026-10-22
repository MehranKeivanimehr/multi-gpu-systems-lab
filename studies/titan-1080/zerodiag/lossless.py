"""Lossless array evidence: raw little-endian binary plus a JSON schema.

Deliberately NOT pickle and not `.npz`: a reviewer must be able to read these
arrays with `numpy.fromfile`, `array.array` or a few lines of `struct` in any
language, without executing anything this project wrote.

Layout, per array:

    <name>.bin    raw C-contiguous little-endian elements, no header
    <name>.json   {"schema": "...", "dtype", "byte_order", "itemsize",
                   "shape", "n_elements", "n_bytes", "sha256", ...}

`sha256` is over the exact bytes of the `.bin` file, so integrity is checkable
before anything is parsed.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

SCHEMA = "titan_lossless_array_v1"
_DTYPES = {"float32": 4, "float64": 8, "int64": 8, "int32": 4, "int16": 2,
           "float16": 2, "uint8": 1, "bool": 1}


def save_array(t, out_dir, name: str, *, note: str = "") -> dict:
    """Write one tensor losslessly. Returns its sidecar record."""
    import torch
    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = t.detach().to("cpu").contiguous()
    dtype = str(d.dtype).replace("torch.", "")
    if dtype not in _DTYPES:
        raise ValueError(f"dtype {dtype!r} is not in the explicit schema table; "
                         f"refusing to guess its byte layout")
    arr = d.numpy()
    if arr.dtype.byteorder not in ("=", "|", "<"):
        arr = arr.astype(arr.dtype.newbyteorder("<"))
    raw = arr.tobytes(order="C")
    bin_p = out_dir / f"{name}.bin"
    bin_p.write_bytes(raw)
    rec = {
        "schema": SCHEMA, "name": name, "file": bin_p.name,
        "dtype": dtype, "byte_order": "little", "itemsize": _DTYPES[dtype],
        "shape": list(d.shape), "n_elements": int(d.numel()),
        "n_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(),
        "c_contiguous": True, "container": "raw binary, no header, not pickle",
        "read_with": (f"numpy.fromfile('{bin_p.name}', dtype='<{ {'float32':'f4','float64':'f8','float16':'f2','int64':'i8','int32':'i4','int16':'i2','uint8':'u1','bool':'b1'}[dtype] }')"
                      f".reshape({list(d.shape)})"),
        "note": note,
    }
    (out_dir / f"{name}.json").write_text(json.dumps(rec, indent=2, sort_keys=True))
    return rec


def load_array(out_dir, name: str):
    """Read one array back, verifying the checksum, shape and size first."""
    import numpy as np
    out_dir = pathlib.Path(out_dir)
    sidecar = out_dir / f"{name}.json"
    if not sidecar.is_file():
        raise FileNotFoundError(f"missing schema sidecar {sidecar}")
    rec = json.loads(sidecar.read_text())
    if rec.get("schema") != SCHEMA:
        raise ValueError(f"unsupported array schema {rec.get('schema')!r}")
    bin_p = out_dir / rec["file"]
    if not bin_p.is_file():
        raise FileNotFoundError(f"missing array payload {bin_p}")
    raw = bin_p.read_bytes()
    if len(raw) != rec["n_bytes"]:
        raise ValueError(f"{name}: {len(raw)} bytes on disk, sidecar declares "
                         f"{rec['n_bytes']}")
    got = hashlib.sha256(raw).hexdigest()
    if got != rec["sha256"]:
        raise ValueError(f"{name}: checksum mismatch; sidecar {rec['sha256']}, "
                         f"payload {got}")
    code = {"float32": "<f4", "float64": "<f8", "float16": "<f2",
            "int64": "<i8", "int32": "<i4", "int16": "<i2",
            "uint8": "<u1", "bool": "<b1"}[rec["dtype"]]
    a = np.frombuffer(raw, dtype=code)
    if a.size != rec["n_elements"]:
        raise ValueError(f"{name}: {a.size} elements, sidecar declares "
                         f"{rec['n_elements']}")
    return a.reshape(rec["shape"]), rec
