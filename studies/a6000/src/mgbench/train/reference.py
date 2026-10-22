"""Streaming full-tensor reference snapshots and comparisons (A6000 stage S1a).

The FP32 control WRITES snapshots: for each named tensor, raw float32 little-endian bytes are appended to one binary
file, and an index entry (name, shape, numel, offset, sha256) is recorded. Recipe cases COMPARE: each tensor is read
back by offset, its sha256 is verified, and statistics are computed against the case's own tensor in float64. Only
one reference tensor is resident at a time (bounded memory).

This module works on raw tensors only. It does not use the boundary ledger or its summaries, so the expected
(reference) and observed (case) quantities are never derived from the same summary machinery.

Coverage is exact: every index name must be compared exactly once, no unexpected names, equal element counts.
Tied parameters appear once, under their canonical (first) name; aliases are listed separately.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

CHUNK = 1 << 24          # elements per float64 comparison chunk (~128 MiB per float64 buffer)


def _f32_cpu(t):
    import torch

    t = t.to_local() if hasattr(t, "to_local") else t
    return t.detach().to(device="cpu", dtype=torch.float32).contiguous().reshape(-1)


class SnapshotWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.f = open(self.path, "xb")
        self.index, self.offset, self.total = [], 0, hashlib.sha256()

    def add(self, name: str, tensor) -> None:
        if any(e["name"] == name for e in self.index):
            raise ValueError(f"duplicate snapshot tensor {name}")
        arr = _f32_cpu(tensor).numpy()
        raw = arr.astype("<f4", copy=False).tobytes()
        self.f.write(raw)
        self.index.append({"name": name, "shape": list(tensor.shape), "numel": int(arr.size), "offset": self.offset,
                           "sha256": hashlib.sha256(raw).hexdigest()})
        self.total.update(raw)
        self.offset += len(raw)

    def close(self) -> dict:
        self.f.flush()
        import os

        os.fsync(self.f.fileno())
        self.f.close()
        meta = {"file": self.path.name, "bytes": self.offset, "tensors": len(self.index),
                "elements": sum(e["numel"] for e in self.index), "data_sha256": self.total.hexdigest(),
                "dtype": "float32 little-endian", "index": self.index}
        with open(str(self.path) + ".index.json", "x") as f:
            json.dump(meta, f, indent=0)
        return {k: v for k, v in meta.items() if k != "index"}


def _load_ref(path: Path, entry: dict):
    with open(path, "rb") as f:
        f.seek(entry["offset"])
        raw = f.read(entry["numel"] * 4)
    if len(raw) != entry["numel"] * 4 or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError(f"reference tensor {entry['name']} failed its sha256/size check")
    return np.frombuffer(raw, dtype="<f4")


class SnapshotComparer:
    """compare(name, tensor) for every expected tensor, then summary(). Statistics in float64 on the tensor's
    device; the reference is copied there one tensor at a time."""

    def __init__(self, path: Path):
        self.path = Path(path)
        meta = json.loads(Path(str(self.path) + ".index.json").read_text())
        self.meta = meta
        self.entries = {e["name"]: e for e in meta["index"]}
        self.seen, self.per_tensor, self.problems = set(), [], []
        self.acc = {"diff2": 0.0, "ref2": 0.0, "obs2": 0.0, "dot": 0.0}
        self.max_abs = {"value": 0.0, "name": None, "flat_index": None, "observed": None, "reference": None}

    def compare(self, name: str, tensor, *, baseline=None) -> None:
        """Compare `tensor` (minus `baseline`, a CPU float32 tensor of the same size, if given) with the reference,
        in float64 chunks of CHUNK elements on the tensor's device (bounded transient memory)."""
        import torch

        if name in self.seen:
            self.problems.append(f"duplicate comparison of {name}")
            return
        e = self.entries.get(name)
        if e is None:
            self.problems.append(f"unexpected tensor {name} (not in the reference index)")
            return
        self.seen.add(name)
        x = tensor.to_local() if hasattr(tensor, "to_local") else tensor
        x = x.detach().reshape(-1)
        if x.numel() != e["numel"] or (baseline is not None and baseline.numel() != e["numel"]):
            self.problems.append(f"{name}: {x.numel()} elements (baseline "
                                 f"{None if baseline is None else baseline.numel()}), reference has {e['numel']}")
            return
        ref_cpu = torch.from_numpy(_load_ref(self.path, e).copy())
        base = baseline.reshape(-1) if baseline is not None else None
        diff2 = ref2 = obs2 = dot = 0.0
        mx, k_best, finite = -1.0, 0, True
        best = (None, None)
        for lo in range(0, e["numel"], CHUNK):
            hi = min(lo + CHUNK, e["numel"])
            if base is not None:
                # Same definition as the writer: one float32 subtraction (theta - theta0), then widened. Identical
                # inputs therefore give an exactly zero difference.
                xd = (x[lo:hi].to(torch.float32) - base[lo:hi].to(x.device, torch.float32)).to(torch.float64)
            else:
                xd = x[lo:hi].to(torch.float64)
            rd = ref_cpu[lo:hi].to(x.device, torch.float64)
            d = xd - rd
            finite = finite and bool(torch.isfinite(xd).all())
            diff2 += float(d.pow(2).sum()); ref2 += float(rd.pow(2).sum())
            obs2 += float(xd.pow(2).sum()); dot += float((xd * rd).sum())
            j = int(d.abs().argmax())
            v = float(d.abs()[j])
            if v > mx:
                mx, k_best, best = v, lo + j, (float(xd[j]), float(rd[j]))
            del xd, rd, d
        rec = {"name": name, "numel": e["numel"], "finite": finite,
               "rel_l2": (math.sqrt(diff2 / ref2) if ref2 > 0 else None), "abs_l2_diff": math.sqrt(diff2),
               "ref_l2": math.sqrt(ref2), "obs_l2": math.sqrt(obs2),
               "cosine": (dot / math.sqrt(ref2 * obs2) if ref2 > 0 and obs2 > 0 else None), "max_abs_diff": mx,
               "zero_reference_norm": ref2 == 0.0}
        self.per_tensor.append(rec)
        if not finite:
            self.problems.append(f"{name}: non-finite observed values")
        for key, v in (("diff2", diff2), ("ref2", ref2), ("obs2", obs2), ("dot", dot)):
            self.acc[key] += v
        if mx > self.max_abs["value"]:
            self.max_abs = {"value": mx, "name": name, "flat_index": k_best, "observed": best[0], "reference": best[1]}

    def summary(self) -> dict:
        missing = sorted(set(self.entries) - self.seen)
        a = self.acc
        rel = math.sqrt(a["diff2"] / a["ref2"]) if a["ref2"] > 0 else None
        cos = a["dot"] / math.sqrt(a["ref2"] * a["obs2"]) if a["ref2"] > 0 and a["obs2"] > 0 else None
        ratio = math.sqrt(a["obs2"] / a["ref2"]) if a["ref2"] > 0 else None
        worst = max((r for r in self.per_tensor if r["rel_l2"] is not None), key=lambda r: r["rel_l2"], default=None)
        return {"reference_file": self.path.name, "reference_data_sha256": self.meta["data_sha256"],
                "expected_tensors": len(self.entries), "compared_tensors": len(self.seen),
                "expected_elements": self.meta["elements"],
                "compared_elements": sum(r["numel"] for r in self.per_tensor),
                "missing": missing[:20], "n_missing": len(missing), "problems": self.problems,
                "coverage_complete": not missing and not self.problems and len(self.seen) == len(self.entries),
                "global_rel_l2": rel, "global_cosine": cos, "global_norm_ratio": ratio,
                "zero_reference_tensors": [r["name"] for r in self.per_tensor if r["zero_reference_norm"]],
                "worst_tensor_rel_l2": worst, "max_abs_diff": self.max_abs,
                "per_tensor": self.per_tensor}


def grade(value, limit, marginal, higher_is_better=False):
    """pass / marginal / fail against frozen limits; missing values never pass."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "missing"
    if higher_is_better:
        return "pass" if value >= limit else "marginal" if value >= marginal else "fail"
    return "pass" if value <= limit else "marginal" if value <= marginal else "fail"


def norm64(t) -> float:
    """L2 norm accumulated in float64, chunked (bounded transient memory)."""
    import torch

    x = (t.to_local() if hasattr(t, "to_local") else t).detach().reshape(-1)
    s = 0.0
    for lo in range(0, x.numel(), CHUNK):
        s += float(x[lo:lo + CHUNK].to(torch.float64).pow(2).sum())
    return math.sqrt(s)


class ReferenceSession:
    """Role 'write' (FP32 control) or 'compare' (recipe cases) for one run.

    Snapshots (full tensors, every element):  grad_b1.f32 - gradient at the FIRST measured accumulation boundary,
    before the optimizer call;  dtheta_final.f32 - parameter (fp32 master for DeepSpeed) minus the canonical start,
    after the last measured update.
    Scalars (full coverage, norm level): per-tensor gradient norms at every measured boundary, per-tensor Adam moment
    norms after the last update, every microbatch raw loss.
    """

    def __init__(self, role: str, ref_dir: Path):
        if role not in ("write", "compare"):
            raise ValueError(role)
        self.role, self.dir = role, Path(ref_dir)
        self.grad_norms, self.moment_norms, self.results = [], {}, {}
        self.ref_scalars = json.loads((self.dir / "reference_scalars.json").read_text()) if role == "compare" else None

    @classmethod
    def from_spec(cls, spec: dict):
        role = spec.get("reference_role")
        return cls(role, Path(spec["reference_dir"])) if role else None

    def at_boundary(self, k: int, items: list) -> None:
        """items: [(canonical name, gradient tensor)] for measured boundary k (0-based), every expected tensor."""
        self.grad_norms.append({name: norm64(g) for name, g in items})
        if k != 0:
            return
        if self.role == "write":
            w = SnapshotWriter(self.dir / "grad_b1.f32")
            for name, g in items:
                w.add(name, g)
            self.results["grad_b1"] = w.close()
        else:
            c = SnapshotComparer(self.dir / "grad_b1.f32")
            for name, g in items:
                c.compare(name, g)
            self.results["grad_b1"] = c.summary()

    def final(self, items: list, theta0: dict) -> None:
        """items: [(canonical name, parameter-or-master tensor, exp_avg, exp_avg_sq)] after the last update."""
        self.moment_norms = {name: {"m": norm64(m), "v": norm64(v)} for name, _, m, v in items}
        if self.role == "write":
            import torch

            w = SnapshotWriter(self.dir / "dtheta_final.f32")
            for name, theta, _, _ in items:
                t = (theta.to_local() if hasattr(theta, "to_local") else theta).detach().to("cpu", torch.float32)
                w.add(name, t.reshape(theta0[name].shape) - theta0[name])
            self.results["dtheta_final"] = w.close()
        else:
            c = SnapshotComparer(self.dir / "dtheta_final.f32")
            for name, theta, _, _ in items:
                c.compare(name, theta, baseline=theta0[name])
            self.results["dtheta_final"] = c.summary()

    def scalars(self, measured_losses: list) -> dict:
        return {"grad_norms_per_boundary": self.grad_norms, "moment_norms_final": self.moment_norms,
                "microbatch_losses_measured": measured_losses}

    def finish(self, measured_losses: list) -> dict:
        own = self.scalars(measured_losses)
        if self.role == "write":
            with open(self.dir / "reference_scalars.json", "x") as f:
                json.dump(own, f, indent=0)
            return {"role": "write", "snapshots": self.results, "scalars_written": True}
        ref = self.ref_scalars
        out = {"role": "compare", "snapshots": self.results}
        rl = ref["microbatch_losses_measured"]
        out["loss_rel"] = [abs(a - b) / abs(b) if b else None for a, b in zip(measured_losses, rl)]
        out["loss_count_match"] = len(measured_losses) == len(rl)
        gn = []
        for k, (mine, theirs) in enumerate(zip(self.grad_norms, ref["grad_norms_per_boundary"])):
            ratios = {n: (mine[n] / theirs[n] if theirs.get(n) else None) for n in theirs if n in mine}
            g_mine = math.sqrt(sum(mine[n] ** 2 for n in theirs if n in mine))
            g_ref = math.sqrt(sum(v ** 2 for v in theirs.values()))
            gn.append({"boundary": k, "tensors": len(ratios), "missing": sorted(set(theirs) - set(mine))[:10],
                       "global_norm_ratio": (g_mine / g_ref if g_ref else None),
                       "min_ratio": min((r for r in ratios.values() if r is not None), default=None),
                       "max_ratio": max((r for r in ratios.values() if r is not None), default=None)})
        out["grad_norm_ratios"] = gn
        mr = {}
        for n, rv in ref["moment_norms_final"].items():
            mine = self.moment_norms.get(n)
            mr[n] = None if mine is None else {k: (mine[k] / rv[k] if rv[k] else None) for k in ("m", "v")}
        vals = [r[k] for r in mr.values() if r for k in ("m", "v") if r[k] is not None]
        glob = {}
        for key in ("m", "v"):
            ref_tot = math.sqrt(sum(rv[key] ** 2 for rv in ref["moment_norms_final"].values()))
            mine_tot = math.sqrt(sum(self.moment_norms[n][key] ** 2 for n in ref["moment_norms_final"] if n in self.moment_norms))
            glob[key] = mine_tot / ref_tot if ref_tot else None
        out["moment_norm_ratios"] = {"global": glob, "tensors": len(mr), "missing": [n for n, r in mr.items() if r is None][:10],
                                     "min": min(vals, default=None), "max": max(vals, default=None),
                                     "zero_reference": [n for n, r in mr.items() if r and None in r.values()][:10]}
        return out


class FirstBoundarySession:
    """Candidate 14 result kind 'first_boundary_gradient': ONE accumulation boundary, no optimizer update.

    at_boundary(0, items) writes this case's full first-boundary gradient (own snapshot, for later CPU pairwise
    contrasts) and, in the same pass, compares it with the sealed reference grad_b1 (read-only, per-tensor sha256
    verified). finish(losses) compares the first-boundary raw microbatch losses and per-tensor gradient norms with the
    reference's first boundary. A zero observed or zero reference vector is never certified (zero == zero is refused).
    """

    def __init__(self, out_dir: Path, ref_dir: Path):
        self.out_dir, self.ref_dir = Path(out_dir), Path(ref_dir)
        self.ref_scalars = json.loads((self.ref_dir / "reference_scalars.json").read_text())
        self.grad_norms, self.results, self.calls = {}, {}, 0

    @classmethod
    def from_spec(cls, spec: dict, out_dir):
        return cls(Path(out_dir), Path(spec["reference_dir"])) if spec.get("result_kind") == "first_boundary_gradient" else None

    def at_boundary(self, k: int, items: list) -> None:
        self.calls += 1
        if k != 0 or self.calls != 1:
            raise RuntimeError(f"first-boundary session observed boundary {k} (call {self.calls}); exactly one is allowed")
        w = SnapshotWriter(self.out_dir / "first_boundary_grad.f32")
        c = SnapshotComparer(self.ref_dir / "grad_b1.f32")
        for name, g in items:
            self.grad_norms[name] = norm64(g)
            w.add(name, g)
            c.compare(name, g)
        self.results["own_snapshot"] = w.close()
        s = c.summary()
        zero_obs = [r["name"] for r in s["per_tensor"] if r["obs_l2"] == 0.0]
        s["zero_observed_tensors"] = zero_obs
        s["nonzero_certifiable"] = bool(s.get("global_norm_ratio")) and s["global_norm_ratio"] > 0 and not zero_obs \
            and not s["zero_reference_tensors"]
        self.results["vs_sealed_r0_grad_b1"] = s

    def finish(self, measured_losses: list) -> dict:
        if self.calls != 1:
            return {"role": "first_boundary", "problems": [f"boundary observed {self.calls} times, expected 1"]}
        ref_l = self.ref_scalars["microbatch_losses_measured"][:len(measured_losses)]
        ref_n = self.ref_scalars["grad_norms_per_boundary"][0]
        g_m = math.sqrt(sum(self.grad_norms[n] ** 2 for n in ref_n if n in self.grad_norms))
        g_r = math.sqrt(sum(v ** 2 for v in ref_n.values()))
        return {"role": "first_boundary", "snapshots": self.results,
                "loss_rel": [abs(a - b) / abs(b) if b else None for a, b in zip(measured_losses, ref_l)],
                "loss_count": len(measured_losses),
                "grad_norm_ratio_global": g_m / g_r if g_r else None,
                "grad_norm_missing": sorted(set(ref_n) - set(self.grad_norms))[:10]}
