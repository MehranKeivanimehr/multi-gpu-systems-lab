"""Judge matched short reference runs against PREDECLARED tolerances.

The tolerance file is hashed before any reference run exists, and this script
re-reads that hash and prints it with the verdict, so a reader can confirm the
thresholds were not tuned to the differences they judge. A failure is reported
as a failure.
"""
from __future__ import annotations
import os

import hashlib, json, pathlib, sys

W = pathlib.Path(os.environ.get("TITAN_CAMPAIGN_ROOT",
    str(pathlib.Path(__file__).resolve().parent.parent)))
SPEC = W / "bundle/plan/reference_tolerances.json"
MATCH_KEYS = ("seed", "model", "seq_len", "micro_batch_per_gpu", "warmup",
              "steps", "precision", "memory_fraction", "world_size", "mode")


def record(case: str) -> dict:
    hits = sorted((W / "runs" / case / "results").glob("*.json"))
    if not hits:
        raise SystemExit(f"no result json for {case}")
    return json.loads(hits[0].read_text())


def main(argv) -> int:
    if len(argv) != 2:
        raise SystemExit("usage: compare_reference.py CASE_A CASE_B")
    spec = json.loads(SPEC.read_text())
    tol = spec["tolerances"]
    spec_sha = hashlib.sha256(SPEC.read_bytes()).hexdigest()

    a, b = record(argv[0]), record(argv[1])
    ca, cb = a["config"], b["config"]
    la, lb = a.get("losses") or [], b.get("losses") or []

    mismatched = {k: (ca.get(k), cb.get(k)) for k in MATCH_KEYS
                  if ca.get(k) != cb.get(k)}
    checks = {}
    checks["configurations_are_matched"] = (not mismatched, f"differing: {mismatched}"
                                            if mismatched else "all matched keys equal")
    dev_a = (ca.get("runtime_device_identity_after_memory_cap") or {}).get("device_name")
    dev_b = (cb.get("runtime_device_identity_after_memory_cap") or {}).get("device_name")
    checks["hardware_actually_differs"] = (
        dev_a != dev_b or ca.get("group") != cb.get("group"),
        f"{dev_a!r} ({ca.get('group')}) vs {dev_b!r} ({cb.get('group')})")
    checks["same_number_of_losses"] = (len(la) == len(lb), f"{len(la)} vs {len(lb)}")
    finite = all(isinstance(x, (int, float)) and x == x for x in la + lb)
    checks["all_losses_finite"] = (finite is bool(tol["all_losses_finite"]),
                                   f"{len(la)+len(lb)} values")
    if la and lb:
        d0 = abs(la[0] - lb[0])
        checks["step0_loss_within_tolerance"] = (
            d0 <= tol["step0_loss_abs"],
            f"|{la[0]:.6f} - {lb[0]:.6f}| = {d0:.6f} <= {tol['step0_loss_abs']}")
        dn = abs(la[-1] - lb[-1])
        checks["final_loss_within_tolerance"] = (
            dn <= tol["final_loss_abs"],
            f"|{la[-1]:.6f} - {lb[-1]:.6f}| = {dn:.6f} <= {tol['final_loss_abs']}")
        if len(la) == len(lb):
            worst = max(abs(x - y) for x, y in zip(la, lb))
            idx = max(range(len(la)), key=lambda i: abs(la[i] - lb[i]))
            checks["trajectory_within_tolerance"] = (
                worst <= tol["loss_trajectory_max_abs"],
                f"max |dL| = {worst:.6f} at step {idx} "
                f"<= {tol['loss_trajectory_max_abs']}")

    ok = all(v[0] for v in checks.values())
    out = {
        "measurement_kind": "matched_reference_comparison_not_performance",
        "case_a": argv[0], "case_b": argv[1],
        "tolerance_spec": str(SPEC.relative_to(W)),
        "tolerance_spec_sha256": spec_sha,
        "tolerances_frozen_before_runs": spec["frozen_before_any_comparison_run"],
        "device_a": dev_a, "device_b": dev_b,
        "group_a": ca.get("group"), "group_b": cb.get("group"),
        "throughput_a": a["perf"]["tokens_per_s"], "throughput_b": b["perf"]["tokens_per_s"],
        "throughput_tolerance": tol["relative_throughput_bound"],
        "checks": {k: {"ok": v[0], "detail": v[1]} for k, v in checks.items()},
        "verdict": "PASS" if ok else "FAIL",
    }
    d = W / "results"; d.mkdir(parents=True, exist_ok=True)
    (d / f"reference_compare_{argv[0]}_vs_{argv[1]}.json").write_text(
        json.dumps(out, indent=2, sort_keys=True, default=str))
    print(f"tolerance spec sha256: {spec_sha}")
    print(f"frozen before runs:    {spec['frozen_before_any_comparison_run']}")
    for k, (good, detail) in checks.items():
        print(f"  [{'PASS' if good else 'FAIL'}] {k}: {detail}")
    print(f"verdict: {out['verdict']}  "
          f"({dev_a} {out['throughput_a']:,.0f} tok/s vs {dev_b} {out['throughput_b']:,.0f} tok/s; "
          f"throughput deliberately untested)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
