"""Regenerate the DeepSpeed cohort classification required by the ledger.

Reproducible from the records: counts the historical DeepSpeed performance rows
by (zero_stage, world_size, family, precision) and emits the disposition that
covers all 98 of them. Nothing is hard-coded to a desired total; the total is
counted and the ledger refuses it if it is not 98.
"""
from __future__ import annotations
import argparse, collections, json, pathlib, sys

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--plan", default=None,
                    help="run plan binding the EXPECTED record identities "
                         "(default: <root>/../protocols/run_plan.json, else "
                         "<root>/../bundle/plan/run_plan.json)")
    ap.add_argument("--blocks", default="1,2,3,4,5,6")
    a = ap.parse_args(argv)
    root = pathlib.Path(a.root).resolve()
    P, extra = collections.Counter(), []
    OBSERVED_P = collections.Counter()
    for p in sorted(root.glob("runs/*/results/*.json")):
        if p.name == "uuid_binding_receipt.json":
            continue
        try:
            r = json.loads(p.read_text())
        except (OSError, ValueError) as exc:
            print(f"REFUSED: unreadable input {p}: {exc}"); return 3
        c = r.get("config") or {}
        if c.get("mode") != "deepspeed" or r.get("status") != "ok":
            continue
        fam = "rtx" if str(c.get("group", "")).startswith("rtx") else "p1080"
        key = (c.get("zero_stage"), c.get("world_size"), fam, c.get("precision"))
        if p.parent.parent.name.startswith("P"):
            P.update([key]); OBSERVED_P.update([p.parent.parent.name])
        else:
            extra.append({"case": p.parent.parent.name, "cohort": list(key)})
    # ---- EXPECTED record identities, bound to the plan, not to a count ----
    plan_path = pathlib.Path(a.plan) if a.plan else None
    if plan_path is None:
        for cand in ("../protocols/run_plan.json", "../bundle/plan/run_plan.json",
                     "protocols/run_plan.json", "bundle/plan/run_plan.json"):
            q = (root / cand).resolve()
            if q.is_file():
                plan_path = q; break
    if plan_path is None or not plan_path.is_file():
        print("REFUSED: no run plan found to bind expected record identities; "
              "a row count alone cannot establish which records were observed")
        return 3
    try:
        plan = json.loads(plan_path.read_text())
    except (OSError, ValueError) as exc:
        print(f"REFUSED: plan unreadable: {plan_path}: {exc}"); return 3
    blocks = [int(x) for x in a.blocks.split(",")]
    expected_records, expected_contexts = [], []
    for cfg in plan.get("configurations", []):
        if cfg.get("recipe") not in ("zero1", "zero2"):
            continue
        fam = "rtx" if str(cfg.get("group", "")).startswith("rtx") else "p1080"
        prec = "amp_fp16" if cfg.get("dtype") == "amp_fp16" else "fp32"
        expected_contexts.append([int(cfg["recipe"][-1]), cfg.get("world_size"),
                                  fam, prec])
        for b in blocks:
            expected_records.append(f"P{b}_{cfg['case_id']}")
    observed_records = sorted(OBSERVED_P)
    exp_set = set(expected_records)
    dup_records = sorted({r for r in observed_records if OBSERVED_P[r] > 1})
    missing_records = sorted(exp_set - set(observed_records))
    unexpected_records = sorted(set(observed_records) - exp_set)
    obs_ctx = {tuple(k) for k in P}
    exp_ctx = {tuple(c) for c in expected_contexts}
    missing_contexts = sorted(exp_ctx - obs_ctx, key=str)
    unexpected_contexts = sorted(obs_ctx - exp_ctx, key=str)
    dup_contexts = sorted({tuple(k) for k, v in P.items()
                           if v != len(blocks)}, key=str)
    identities_reconciled = not (dup_records or missing_records
                                 or unexpected_records or missing_contexts
                                 or unexpected_contexts)
    out = pathlib.Path(a.output); out.mkdir(parents=True, exist_ok=True)
    total = sum(P.values()) + len(extra)
    doc = {
        "identities_reconciled": identities_reconciled,
        "identity_binding": {
            "plan": str(plan_path), "blocks": blocks,
            "note": ("expected record identities are derived from the bound run "
                     "plan, never from a target total. A preserved total is NOT "
                     "evidence: removing one expected record and duplicating "
                     "another keeps the total and still fails here.")},
        "expected_records": expected_records,
        "observed_records": observed_records,
        "missing_records": missing_records,
        "unexpected_records": unexpected_records,
        "duplicate_records": dup_records,
        "expected_contexts": expected_contexts,
        "missing_contexts": [list(c) for c in missing_contexts],
        "unexpected_contexts": [list(c) for c in unexpected_contexts],
        "duplicate_contexts": [list(c) for c in dup_contexts],
        "baseline_scope": ("98 rows is the expected baseline for THIS frozen "
                           "campaign only; it is not a universal assumption for "
                           "future datasets and is not sufficient evidence alone"),
        "p_cohort_contexts": len(P),
        "p_cohort_rows": sum(P.values()),
        "extra_rows_outside_p_cohort": extra,
        "totals": {"invalid-for-claim": total},
        "reason": ("the engine-aware diagnostic provably ran GPT2 n_positions=512 "
                   "(124,046,592 parameters) against a trained model of "
                   "124,439,808; every diagnostic forced fp16 and bf16 off, so "
                   "the AMP rows additionally lack precision-matched evidence"),
        "scope": (f"{len(P)} P-cohort contexts = {sum(P.values())} P rows, plus "
                  f"{len(extra)} rows outside the P cohort = {total} overall"),
        "per_cohort": [{"cohort": list(k), "rows": v} for k, v in sorted(P.items(), key=lambda kv: str(kv[0]))],
    }
    (out / "ds_classification.json").write_text(json.dumps(doc, indent=2, sort_keys=True))
    print(f"p_cohort_contexts={len(P)} p_rows={sum(P.values())} extra={len(extra)} total={total}")
    print(f"identities_reconciled={identities_reconciled}")
    if not identities_reconciled:
        print(f"  missing_records={missing_records[:4]}")
        print(f"  duplicate_records={dup_records[:4]}")
        print(f"  unexpected_records={unexpected_records[:4]}")
        print(f"  duplicate_contexts={[list(c) for c in dup_contexts][:4]}")
    print(f"  -> {out/'ds_classification.json'}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
