"""Matched cross-hardware reference comparison, strictly gated.

Corrections to the delivered comparator
---------------------------------------
* Two **empty** loss arrays passed, because `same_number_of_losses` compared
  0 == 0, `all_losses_finite` returned `all([])` = True, and the three tolerance
  checks were skipped by `if la and lb`. Empty evidence is now a refusal.
* `all_losses_finite` accepted `inf`, because the test was `x == x`. Finiteness
  now means `math.isfinite`.
* A length mismatch still evaluated tolerances over the `zip`-truncated prefix.
  Tolerances are now gated on equal, non-zero lengths.
* Missing match keys only showed as "differing"; they are now named as missing.

Chronology, stated rather than implied
--------------------------------------
The tolerance file was frozen at 2026-10-16T04:26:50Z, **after** D04/D06/D07 ran
at 03:25-03:28Z. This comparison is therefore **informed/exploratory**, not a
blind outcome-unseen preregistration. That does not establish that thresholds
were fitted to the differences; it does mean the freeze cannot be cited as
preregistration. `losses[0]` is the first *measured* step, i.e. after the
configured warmup updates -- not an untrained forward pass.

CPU-only. No device access.
"""
from __future__ import annotations

import argparse, hashlib, json, math, pathlib, sys

REQUIRED_CONFIG = ("seed", "model", "seq_len", "micro_batch_per_gpu", "steps",
                   "warmup", "precision", "memory_fraction", "world_size", "mode")
MIN_LOSSES = 2


def finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def ok(d=""):
    return {"ok": True, "detail": d}


def no(d):
    return {"ok": False, "detail": d}


def limit(d):
    """A claim-narrowing limitation, not an integrity failure."""
    return {"ok": True, "limitation": True, "detail": d}


def load_record(root: pathlib.Path, case: str) -> dict | None:
    d = root / "runs" / case / "results"
    if not d.is_dir():
        return None
    hits = sorted(d.glob("*.json"))
    if not hits:
        return None
    try:
        return json.loads(hits[0].read_text())
    except (OSError, ValueError):
        return None


def compare(root: pathlib.Path, case_a: str, case_b: str, spec: dict,
            spec_sha: str) -> dict:
    tol = spec["tolerances"]
    checks: dict[str, dict] = {}
    a, b = load_record(root, case_a), load_record(root, case_b)
    for nm, r in ((case_a, a), (case_b, b)):
        checks[f"record_readable[{nm}]"] = ok() if r else no(f"no usable record for {nm}")
    if not (a and b):
        return {"verdict": "INVALID", "checks": checks}

    ca, cb = a.get("config") or {}, b.get("config") or {}
    missing_a = [k for k in REQUIRED_CONFIG if ca.get(k) is None]
    missing_b = [k for k in REQUIRED_CONFIG if cb.get(k) is None]
    checks["required_config_fields_present"] = (
        ok(f"{len(REQUIRED_CONFIG)} fields on both sides")
        if not missing_a and not missing_b
        else no(f"missing on {case_a}: {missing_a}; on {case_b}: {missing_b}"))

    diffs = {k: (ca.get(k), cb.get(k)) for k in REQUIRED_CONFIG if ca.get(k) != cb.get(k)}
    checks["configurations_are_matched"] = (
        ok("all required keys equal") if not diffs else no(f"differing: {diffs}"))

    da = (ca.get("runtime_device_identity_after_memory_cap") or {}).get("device_name")
    db = (cb.get("runtime_device_identity_after_memory_cap") or {}).get("device_name")
    checks["device_names_recorded"] = (
        ok(f"{da!r} vs {db!r}") if da and db else no(f"device names {da!r}/{db!r}"))
    checks["hardware_actually_differs"] = (
        ok(f"{da!r} ({ca.get('group')}) vs {db!r} ({cb.get('group')})")
        if (da and db and da != db) or (ca.get("group") != cb.get("group"))
        else no(f"same hardware on both sides: {da!r}/{db!r}"))
    checks["device_identity_is_not_certified"] = ok(
        "device_name comes from the CUDA runtime but no in-process UUID exists "
        "on torch 2.4.1; identity is asserted by name, not certified")

    la, lb = a.get("losses"), b.get("losses")
    for nm, L in ((case_a, la), (case_b, lb)):
        checks[f"losses_nonempty[{nm}]"] = (
            ok(f"{len(L)} values") if isinstance(L, list) and len(L) >= MIN_LOSSES
            else no(f"{nm} has {0 if not isinstance(L, list) else len(L)} losses; "
                    f"empty or near-empty evidence cannot support a comparison"))
        checks[f"losses_all_finite[{nm}]"] = (
            ok(f"{len(L)} finite") if isinstance(L, list) and L
            and all(finite(x) for x in L)
            else no(f"{nm} losses contain non-finite or non-numeric values "
                    f"(inf and NaN are both rejected)"))
    usable = (isinstance(la, list) and isinstance(lb, list)
              and len(la) >= MIN_LOSSES and len(lb) >= MIN_LOSSES
              and all(finite(x) for x in la) and all(finite(x) for x in lb))
    checks["loss_arrays_same_length"] = (
        ok(f"{len(la)}") if isinstance(la, list) and isinstance(lb, list)
        and len(la) == len(lb) and la
        else no(f"{len(la) if isinstance(la, list) else '?'} vs "
                f"{len(lb) if isinstance(lb, list) else '?'}"))
    equal_len = isinstance(la, list) and isinstance(lb, list) and len(la) == len(lb)

    if usable and equal_len:
        d0 = abs(la[0] - lb[0])
        checks["first_measured_step_within_tolerance"] = (
            ok(f"|{la[0]:.6f}-{lb[0]:.6f}|={d0:.3e} <= {tol['step0_loss_abs']}")
            if d0 <= tol["step0_loss_abs"] else no(f"{d0:.6e} > {tol['step0_loss_abs']}"))
        dn = abs(la[-1] - lb[-1])
        checks["final_loss_within_tolerance"] = (
            ok(f"{dn:.3e} <= {tol['final_loss_abs']}")
            if dn <= tol["final_loss_abs"] else no(f"{dn:.6e} > {tol['final_loss_abs']}"))
        worst = max(abs(x - y) for x, y in zip(la, lb))
        idx = max(range(len(la)), key=lambda i: abs(la[i] - lb[i]))
        checks["trajectory_within_tolerance"] = (
            ok(f"max|dL|={worst:.3e} at step {idx} <= {tol['loss_trajectory_max_abs']}")
            if worst <= tol["loss_trajectory_max_abs"]
            else no(f"max|dL|={worst:.6e} at step {idx}"))
    else:
        checks["tolerances_evaluated"] = no(
            "tolerances NOT evaluated: evidence is empty, non-finite or of "
            "unequal length. A truncated prefix is never compared.")

    warm = ca.get("warmup")
    checks["first_step_is_post_warmup_not_untrained"] = ok(
        f"losses[0] follows {warm!r} warmup updates; it is NOT an untrained "
        f"forward pass, so agreement there does not test initial weights alone")
    checks["input_policy_matches_the_frozen_spec"] = limit(
        "the frozen spec declares a CPU generator, but train_llm.py draws from "
        "torch.Generator(device=device), i.e. the CUDA generator; the realised "
        "input stream is therefore not the one the spec describes")
    checks["realised_input_digest_retained"] = (
        ok() if "input_ids_sha256" in json.dumps(a) and "input_ids_sha256" in json.dumps(b)
        else limit("no realised input digest was retained for either run, so "
                   "input identity cannot be verified after the fact"))
    checks["preregistration_not_claimed"] = ok(
        f"tolerance spec sha256 {spec_sha[:16]}… was frozen AFTER the earlier "
        f"D-series runs; this is an informed/exploratory comparison and is not "
        f"cited as preregistration")

    fails = [(k, v["detail"]) for k, v in checks.items() if not v["ok"]]
    lims = [(k, v["detail"]) for k, v in checks.items() if v.get("limitation")]
    verdict = "FAIL" if fails else ("QUALIFIED" if lims else "PASS")
    return {"verdict": verdict, "checks": checks,
            "failures": fails, "limitations": lims,
            "throughput_a": (a.get("perf") or {}).get("tokens_per_s"),
            "throughput_b": (b.get("perf") or {}).get("tokens_per_s"),
            "throughput_tolerance": tol.get("relative_throughput_bound")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--spec", default=None,
                    help="tolerance spec (default <root>/bundle/plan/reference_tolerances.json)")
    ap.add_argument("pairs", nargs="+", help="CASE_A:CASE_B")
    a = ap.parse_args(argv)
    root = pathlib.Path(a.root).resolve()
    out = pathlib.Path(a.output).resolve(); out.mkdir(parents=True, exist_ok=True)
    sp = pathlib.Path(a.spec) if a.spec else root / "bundle/plan/reference_tolerances.json"
    try:
        spec = json.loads(sp.read_text())
        spec_sha = hashlib.sha256(sp.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        print(f"cannot read tolerance spec {sp}: {exc}"); return 2

    results = {}
    worst = 0
    for pair in a.pairs:
        if ":" not in pair:
            print(f"bad pair {pair!r}"); return 2
        ca, cb = pair.split(":", 1)
        r = compare(root, ca, cb, spec, spec_sha)
        results[pair] = r
        print(f"\n=== {ca}  vs  {cb}  ->  {r['verdict']} ===")
        for k, v in r["checks"].items():
            tag = "NO " if not v["ok"] else ("LIM" if v.get("limitation") else "ok ")
            print(f"   [{tag}] {k}: {v['detail'][:110]}")
        if r["verdict"] == "FAIL":
            worst = 1
    (out / "reference_comparison.json").write_text(
        json.dumps({"spec": str(sp), "spec_sha256": spec_sha, "results": results},
                   indent=2, sort_keys=True, default=str))
    print(f"\n  -> {out/'reference_comparison.json'}")
    return worst


if __name__ == "__main__":
    sys.exit(main())
