"""Aggregate the repeat blocks into per-configuration statistics.

Every repeat is reported, not only the summary: the points list carries all six
observations so a reader can see the spread that produced the median. Scaling
and ratio comparisons are computed per block and then reduced, never from
medians of different block sets, so a missing block cannot silently pair an
arm's good block with another arm's bad one.
"""
from __future__ import annotations
import os

import collections, csv, json, pathlib, re, sys

W = pathlib.Path(os.environ.get("TITAN_CAMPAIGN_ROOT",
    str(pathlib.Path(__file__).resolve().parent.parent)))
sys.path.insert(0, str(W / "src/multigpu-bench/tools"))
from titan_stats import matched_block_ratio, summarize  # noqa: E402

CASE_RE = re.compile(r"^P(\d+)_(T_.+)$")


def load() -> dict[str, dict[int, dict]]:
    """{case_id: {block: record}} for every completed repeat."""
    out: dict[str, dict[int, dict]] = collections.defaultdict(dict)
    for d in sorted((W / "runs").iterdir()):
        m = CASE_RE.match(d.name)
        if not m:
            continue
        block, case = int(m.group(1)), m.group(2)
        js = list((d / "results").glob("*.json")) if (d / "results").is_dir() else []
        if not js:
            continue
        r = json.loads(js[0].read_text())
        if r.get("status") != "ok":
            continue
        out[case][block] = r
    return out


def main() -> int:
    data = load()
    rows = []
    for case in sorted(data):
        by_block = data[case]
        cfg = next(iter(by_block.values()))["config"]
        tps = {b: r["perf"]["tokens_per_s"] for b, r in sorted(by_block.items())}
        ene = {b: r["telemetry"]["energy_total_j"] for b, r in sorted(by_block.items())}
        pk = {b: r["memory"]["peak_alloc_mib_rank0"] for b, r in sorted(by_block.items())}
        rows.append({
            "case_id": case, "group": cfg["group"], "recipe": cfg["mode"],
            "zero_stage": cfg.get("zero_stage"), "precision": cfg["precision"],
            "world_size": cfg["world_size"],
            "micro_batch_per_gpu": cfg["micro_batch_per_gpu"],
            "blocks_present": sorted(by_block),
            "tokens_per_s": summarize(tps.values()),
            "tokens_per_s_by_block": tps,
            "energy_total_j": summarize(ene.values()),
            "peak_alloc_mib": summarize(pk.values()),
        })

    # Weak-scaling: per-block speedup against the SAME block's N=1 baseline of
    # the same family, recipe and precision. Mixing blocks would manufacture
    # precision that the repeat design does not support.
    index = {r["case_id"]: r for r in rows}
    for r in rows:
        fam = "rtx" if r["group"].startswith("rtx") else "p1080"
        base = next((q for q in rows
                     if q["world_size"] == 1 and q["recipe"] == r["recipe"]
                     and q["precision"] == r["precision"]
                     and (q["zero_stage"] == r["zero_stage"])
                     and ("rtx" if q["group"].startswith("rtx") else "p1080") == fam), None)
        if base is None or base["case_id"] == r["case_id"]:
            r["speedup_vs_1gpu"] = None
            r["scaling_efficiency_pct"] = None
            r["scaling_baseline"] = base["case_id"] if base else None
            continue
        sp = matched_block_ratio(r["tokens_per_s_by_block"], base["tokens_per_s_by_block"])
        r["scaling_baseline"] = base["case_id"]
        r["speedup_vs_1gpu"] = sp
        eff = dict(sp)
        if sp.get("ratios"):
            eff = matched_block_ratio(
                {b: v / r["world_size"] for b, v in r["tokens_per_s_by_block"].items()},
                base["tokens_per_s_by_block"])
            eff["median_ratio_pct"] = (eff["median_ratio"] * 100.0
                                       if eff.get("median_ratio") is not None else None)
        r["scaling_efficiency_pct"] = eff

    out = W / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "block_statistics.json").write_text(
        json.dumps({"schema": "titan-block-stats-1",
                    "author_of_repository": "Mehran Keivanimehr",
                    "blocks_expected": 6, "configurations": rows},
                   indent=2, sort_keys=True, default=str))

    with open(out / "block_statistics.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["case_id", "group", "recipe", "zero_stage", "precision",
                     "world_size", "n_blocks", "tok_s_median", "tok_s_min",
                     "tok_s_max", "tok_s_cv_pct", "tok_s_range_over_median_pct",
                     "coverage", "precision_goal_met", "speedup_median",
                     "efficiency_pct_median", "energy_j_median", "peak_mib_median",
                     "tok_s_all_repeats"])
        for r in sorted(rows, key=lambda q: (q["group"], q["recipe"],
                                             str(q["zero_stage"]), q["precision"])):
            t = r["tokens_per_s"]
            sp = r.get("speedup_vs_1gpu") or {}
            ef = r.get("scaling_efficiency_pct") or {}
            wr.writerow([
                r["case_id"], r["group"], r["recipe"], r["zero_stage"],
                r["precision"], r["world_size"], t["n"],
                t["median"], t["min"], t["max"], t["sample_cv_pct"],
                t["range_over_median_pct"], t["minmax_median_coverage"],
                t["precision_goal_met"], sp.get("median_ratio"),
                ef.get("median_ratio_pct"), r["energy_total_j"]["median"],
                r["peak_alloc_mib"]["median"],
                ";".join(f"{v:.1f}" for v in r["tokens_per_s_by_block"].values()),
            ])

    complete = sum(1 for r in rows if r["tokens_per_s"]["blocks_complete"])
    print(f"configurations={len(rows)}  six-block complete={complete}  "
          f"partial={len(rows) - complete}")
    print(f"  -> {out/'block_statistics.json'}")
    print(f"  -> {out/'block_statistics.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
