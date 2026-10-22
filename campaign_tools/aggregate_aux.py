"""Per-configuration statistics for the non-training repeat blocks.

Each recipe has its own primary metric and its own sub-keys (batch size for CNN
and HF, concurrency for vLLM, payload bytes for collectives), so observations
are grouped by (case_id, sub_key) and summarised within each group. Mixing
sub-keys would average a batch-1 latency with a batch-16 throughput.
"""
from __future__ import annotations
import os

import collections, csv, json, pathlib, re, sys

W = pathlib.Path(os.environ.get("TITAN_CAMPAIGN_ROOT",
    str(pathlib.Path(__file__).resolve().parent.parent)))
sys.path.insert(0, str(W / "src/multigpu-bench/tools"))
from titan_stats import summarize  # noqa: E402

CASE_RE = re.compile(r"^Q(\d+)_([CHVNS]_.+)$")

# recipe prefix -> (sub_key field, metric field, metric label)
METRICS = {
    "C": ("batch_size", "throughput", "images_per_s"),
    "H": ("batch_size", "gen_tokens_per_s", "gen_tokens_per_s"),
    "V": ("concurrency", "output_tokens_per_s", "output_tokens_per_s"),
    "N": ("payload_bytes", "algbw_gb_s", "algbw_gb_s"),
}
ENERGY = {
    "C": "energy_per_image",
    "H": "energy_per_1k_tokens_j",
    "V": "energy_per_1k_output_tokens_j",
    "N": None,
}


def main() -> int:
    # {(case, sub): {block: value}}
    vals: dict[tuple, dict[int, float]] = collections.defaultdict(dict)
    ener: dict[tuple, dict[int, float]] = collections.defaultdict(dict)
    stream: dict[str, dict[int, dict]] = collections.defaultdict(dict)
    meta: dict[str, dict] = {}

    for d in sorted((W / "runs").iterdir()):
        m = CASE_RE.match(d.name)
        if not m or not (d / "results").is_dir():
            continue
        block, case = int(m.group(1)), m.group(2)
        js = sorted((d / "results").glob("*.json"))
        if not js:
            continue
        r = json.loads(js[0].read_text())
        kind = case[0]
        if kind == "S":
            if r.get("status") == "ok":
                stream[case][block] = r["perf"]
                meta[case] = {"recipe": "vllm_loopback_streaming",
                              "group": r["config"].get("group")}
            continue
        sub_f, met_f, _ = METRICS[kind]
        en_f = ENERGY[kind]
        meta[case] = {"recipe": kind, "group": (r.get("config") or {}).get("group")}
        for row in r.get("rows", []):
            if row.get("status") not in (None, "ok"):
                continue
            sub, v = row.get(sub_f), row.get(met_f)
            if sub is None or not isinstance(v, (int, float)):
                continue
            vals[(case, sub)][block] = v
            if en_f and isinstance(row.get(en_f), (int, float)):
                ener[(case, sub)][block] = row[en_f]

    rows = []
    for (case, sub), by_block in sorted(vals.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        kind = case[0]
        rows.append({
            "case_id": case, "recipe": METRICS[kind][2], "group": meta[case]["group"],
            "sub_key_name": METRICS[kind][0], "sub_key": sub,
            "metric": METRICS[kind][2],
            "value": summarize(by_block.values()),
            "value_by_block": dict(sorted(by_block.items())),
            "energy": summarize(ener[(case, sub)].values()) if ener.get((case, sub)) else None,
        })
    srows = []
    for case, by_block in sorted(stream.items()):
        for field in ("ttft_median_s", "itl_median_s", "requests_per_s"):
            pts = {b: p[field] for b, p in sorted(by_block.items())
                   if isinstance(p.get(field), (int, float))}
            if pts:
                srows.append({"case_id": case, "group": meta[case]["group"],
                              "metric": field, "value": summarize(pts.values()),
                              "value_by_block": pts})

    out = W / "results"; out.mkdir(parents=True, exist_ok=True)
    (out / "aux_block_statistics.json").write_text(json.dumps(
        {"schema": "titan-aux-block-stats-1",
         "author_of_repository": "Mehran Keivanimehr",
         "blocks_expected": 6, "configurations": rows, "streaming": srows},
        indent=2, sort_keys=True, default=str))

    with open(out / "aux_block_statistics.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["case_id", "group", "metric", "sub_key_name", "sub_key",
                     "n_blocks", "median", "min", "max", "cv_pct",
                     "range_over_median_pct", "coverage", "precision_goal_met",
                     "energy_median", "all_repeats"])
        for r in rows + srows:
            v = r["value"]
            wr.writerow([r["case_id"], r.get("group"), r["metric"],
                         r.get("sub_key_name", ""), r.get("sub_key", ""),
                         v["n"], v["median"], v["min"], v["max"],
                         v["sample_cv_pct"], v["range_over_median_pct"],
                         v["minmax_median_coverage"], v["precision_goal_met"],
                         (r.get("energy") or {}).get("median"),
                         ";".join(f"{x:.4g}" for x in r["value_by_block"].values())])

    done = sum(1 for r in rows + srows if r["value"]["blocks_complete"])
    print(f"aux groups={len(rows)+len(srows)}  six-block complete={done}  "
          f"partial={len(rows)+len(srows)-done}")
    print(f"  -> {out/'aux_block_statistics.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
