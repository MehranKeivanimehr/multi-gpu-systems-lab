"""Independent reanalysis from the repeat blocks, with explicit estimands.

Corrections this implements
---------------------------
* Central results come from the **Q/P repeat blocks**, never from the single
  E/F/G pilot records that several delivered chapter tables used.
* Every ratio states its **estimand**. The delivered chapter printed 0.90 for
  RTX HF generate at batch 16 and labelled it `fp16/fp32`; 0.90 is `fp32/fp16`.
  Both estimands are computed here and named:
    - ratio of medians            = median(fp16) / median(fp32)
    - paired median ratio         = median over blocks of (fp16_b / fp32_b)
  They differ, neither is "the" ratio, and all six per-block values plus the
  range and CV are reported so no single number stands alone.
* The CNN analytical constant is the one the code **executed** (8.2 GFLOP/image).
* Memory peaks are post-warmup, energy is uncalibrated GPU-board energy over the
  actual window, and the 0.6 s absolute support-gap rule is preserved.
* `31/32` pointwise median coverage is reported **conditionally**: it follows
  from continuity and exchangeability, which six fresh processes alone do not
  establish.

Portable: explicit --root/--output, no absolute working-root assumption.
CPU-only; imports nothing that touches a device.
"""
from __future__ import annotations

import argparse, collections, json, math, pathlib, statistics, sys
from fractions import Fraction

COVERAGE_NOTE = (
    "POINTWISE coverage of the median for ONE arm. Sufficient assumption: six "
    "IID continuous observations, giving P(all above m)=P(all below m)=(1/2)^6 "
    "and hence 1-2(1/2)^6 = 31/32. A conditional construction also suffices "
    "ONLY if the observations are independent GIVEN the latent parameter AND "
    "every conditional law is continuous with the SAME median. Exchangeability "
    "with continuous marginals is NOT sufficient (see COVERAGE_COUNTEREXAMPLE). "
    "Six fresh processes do not establish IID. This is pointwise for a single "
    "arm and is NOT simultaneous coverage over the 90 endpoint groups.")

COVERAGE_COUNTEREXAMPLE = (
    "Let S be one shared equiprobable sign in {-1,+1} and U_i be independent "
    "Uniform(0,2), i=1..6; set X_i = S*U_i. Each marginal is continuous "
    "Uniform(-2,2) with unique median 0 and the sample is exchangeable, yet all "
    "X_i share the sign of S, so the min-max interval NEVER contains 0. "
    "Coverage is 0, not 31/32.")

COVERAGE_NOT_SIMULTANEOUS = (
    "Each group's interval is pointwise. No simultaneous or family-wise "
    "coverage over the 90 endpoint groups is claimed or implied.")


class StrictInputError(RuntimeError):
    """A required input could not be read. Never silently skipped."""


DISPOSITIONS: dict[str, int] = {}
NON_OK: list[dict] = []


def finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def minmax_median_coverage(n: int) -> Fraction:
    if n < 1:
        return Fraction(0)
    return Fraction(1) - 2 * Fraction(1, 2) ** n


def summarise(values) -> dict:
    v = [x for x in values if finite(x)]
    n = len(v)
    cov = minmax_median_coverage(n)
    mean = sum(v) / n if n else None
    sd = statistics.stdev(v) if n >= 2 else None
    return {
        "n": n, "values": v,
        "median": statistics.median(v) if v else None,
        "min": min(v) if v else None, "max": max(v) if v else None,
        "mean": mean,
        "sample_cv_pct": (sd / mean * 100.0) if (sd is not None and mean) else None,
        "range_over_median_pct": ((max(v) - min(v)) / statistics.median(v) * 100.0
                                  if v and statistics.median(v) else None),
        "interval_min_max": [min(v), max(v)] if v else None,
        "minmax_median_coverage": f"{cov.numerator}/{cov.denominator}",
        "minmax_median_coverage_pct": float(cov) * 100.0,
        "coverage_sufficient_assumption": COVERAGE_NOTE,
        "coverage_counterexample_if_only_exchangeable": COVERAGE_COUNTEREXAMPLE,
        "coverage_is_pointwise_not_simultaneous": COVERAGE_NOT_SIMULTANEOUS,
        "blocks_complete": n == 6,
    }


def paired_ratio(num_by_block: dict, den_by_block: dict) -> dict:
    """Per-block ratios over blocks present in BOTH arms, then summarised."""
    common = sorted(set(num_by_block) & set(den_by_block))
    rs, used = [], []
    for b in common:
        a, d = num_by_block[b], den_by_block[b]
        if finite(a) and finite(d) and d != 0:
            rs.append(a / d); used.append(b)
    if not rs:
        return {"estimand": "paired_median_ratio", "n_paired": 0,
                "median_ratio": None, "ratios": [], "blocks_used": [],
                "note": "no shared usable blocks; comparison unestablished"}
    return {"estimand": "paired_median_ratio",
            "definition": "median over blocks of (numerator_b / denominator_b)",
            "n_paired": len(rs), "blocks_used": used, "ratios": rs,
            "median_ratio": statistics.median(rs),
            "min_ratio": min(rs), "max_ratio": max(rs)}


def ratio_of_medians(num: dict, den: dict) -> dict:
    a, d = num.get("median"), den.get("median")
    return {"estimand": "ratio_of_medians",
            "definition": "median(numerator) / median(denominator)",
            "value": (a / d) if (finite(a) and finite(d) and d) else None,
            "numerator_median": a, "denominator_median": d}


# --------------------------------------------------------------- loading
def load_training(root: pathlib.Path) -> dict:
    """{case_id: {block: record}} from P-blocks only."""
    import re
    out: dict[str, dict[int, dict]] = collections.defaultdict(dict)
    rx = re.compile(r"^P(\d+)_(T_.+)$")
    for d in sorted((root / "runs").iterdir()):
        m = rx.match(d.name)
        if not m or not (d / "results").is_dir():
            continue
        hits = sorted((d / "results").glob("*.json"))
        if not hits:
            continue
        try:
            r = json.loads(hits[0].read_text())
        except (OSError, ValueError) as exc:
            raise StrictInputError(
                f"required input is unreadable: {hits[0]}: "
                f"{type(exc).__name__}: {exc}") from exc
        disp = r.get("status")
        DISPOSITIONS[str(disp)] = DISPOSITIONS.get(str(disp), 0) + 1
        if disp == "ok":
            out[m.group(2)][int(m.group(1))] = r
        else:
            NON_OK.append({"case": d.name, "path": str(hits[0]), "status": disp})
    return out


def load_aux(root: pathlib.Path) -> dict:
    """{(case_id, metric, sub_key): {block: value}} from Q-blocks only."""
    import re
    rx = re.compile(r"^Q(\d+)_([CHVNS]_.+)$")
    METRIC = {"C": ("batch_size", "throughput"), "H": ("batch_size", "gen_tokens_per_s"),
              "V": ("concurrency", "output_tokens_per_s"),
              "N": ("payload_bytes", "algbw_gb_s")}
    out: dict[tuple, dict[int, float]] = collections.defaultdict(dict)
    stream: dict[tuple, dict[int, float]] = collections.defaultdict(dict)
    for d in sorted((root / "runs").iterdir()):
        m = rx.match(d.name)
        if not m or not (d / "results").is_dir():
            continue
        block, case = int(m.group(1)), m.group(2)
        hits = sorted((d / "results").glob("*.json"))
        hits = [h for h in hits if h.name != "uuid_binding_receipt.json"]
        if not hits:
            continue
        try:
            r = json.loads(hits[0].read_text())
        except (OSError, ValueError) as exc:
            raise StrictInputError(
                f"required input is unreadable: {hits[0]}: "
                f"{type(exc).__name__}: {exc}") from exc
        kind = case[0]
        if kind == "S":
            if r.get("status") == "ok":
                for f in ("ttft_median_s", "itl_median_s", "requests_per_s"):
                    v = (r.get("perf") or {}).get(f)
                    if finite(v):
                        stream[(case, f, None)][block] = v
            continue
        if kind not in METRIC:
            continue
        subf, metf = METRIC[kind]
        for row in (r.get("rows") or []):
            if row.get("status") not in (None, "ok"):
                continue
            sub, v = row.get(subf), row.get(metf)
            if sub is not None and finite(v):
                out[(case, metf, sub)][block] = v
    out.update(stream)
    return out


# --------------------------------------------------------------- energy recompute
def recompute_energy_rows(root: pathlib.Path) -> dict:
    """Independently re-integrate EVERY board-energy window in the archive."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from titan_certify import integrate_trapezoid, GAP_CUTOFF_S
    rows_total = ok_n = mismatch = unbracketed = gapfail = 0
    worst_dev = 0.0
    worst_where = None
    gaps: list[float] = []
    for p in sorted((root / "runs").glob("*/results/*.json")):
        if p.name == "uuid_binding_receipt.json":
            continue
        try:
            r = json.loads(p.read_text())
        except (OSError, ValueError) as exc:
            raise StrictInputError(
                f"required input is unreadable: {p}: "
                f"{type(exc).__name__}: {exc}") from exc
        tels = []
        if isinstance(r.get("telemetry"), dict):
            tels.append(("top", r["telemetry"]))
        for i, row in enumerate(r.get("rows") or []):
            if isinstance(row, dict) and isinstance(row.get("telemetry"), dict):
                tels.append((f"row{i}", row["telemetry"]))
        for tag, tel in tels:
            raw = tel.get("raw_samples") or []
            t0, t1 = tel.get("window_start_s"), tel.get("window_stop_s")
            per = tel.get("energy_per_gpu_j") or {}
            if not raw or not finite(t0) or not finite(t1):
                continue
            idx = sorted({str(g.get("index")) for s in raw for g in (s.get("gpus") or [])})
            for gi in idx:
                ts = [s["time_s"] for s in raw for g in (s.get("gpus") or [])
                      if str(g.get("index")) == gi and finite(g.get("power.draw"))]
                ps = [g["power.draw"] for s in raw for g in (s.get("gpus") or [])
                      if str(g.get("index")) == gi and finite(g.get("power.draw"))]
                e, gap, n, st = integrate_trapezoid(ts, ps, t0, t1)
                rows_total += 1
                if st == "window_not_bracketed":
                    unbracketed += 1
                    continue
                if st == "gap_exceeds_cutoff":
                    gapfail += 1
                if finite(gap):
                    gaps.append(gap)
                rep = per.get(gi)
                if e is None or not finite(rep) or not rep:
                    mismatch += 1
                    continue
                dev = abs(e - rep) / rep * 100
                if dev > worst_dev:
                    worst_dev, worst_where = dev, f"{p.parent.parent.name}:{tag}:gpu{gi}"
                if dev <= 0.5:
                    ok_n += 1
                else:
                    mismatch += 1
    return {"board_energy_integrals_recomputed": rows_total,
            "agreeing_within_0_5_pct": ok_n, "disagreeing": mismatch,
            "windows_not_bracketed": unbracketed,
            "support_gap_exceeding_cutoff": gapfail,
            "absolute_support_gap_rule_s": GAP_CUTOFF_S,
            "max_observed_support_gap_s": max(gaps) if gaps else None,
            "worst_relative_deviation_pct": worst_dev,
            "worst_deviation_at": worst_where,
            "energy_scope": ("uncalibrated GPU-board energy integrated over the "
                             "actual measured window; NOT PSU or wall energy")}


# --------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Independent reanalysis (CPU only).")
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args(argv)
    root = pathlib.Path(a.root).resolve()
    out = pathlib.Path(a.output).resolve()
    if not (root / "runs").is_dir():
        print(f"--root has no runs/ directory: {root}"); return 2
    out.mkdir(parents=True, exist_ok=True)

    try:
        tr = load_training(root)
        aux = load_aux(root)
    except StrictInputError as exc:
        print(f"REFUSED: {exc}")
        return 3
    if not tr and not aux:
        print(f"REFUSED: no usable observations under {root}; refusing to emit "
              f"an empty analysis that would read as a completed one")
        return 3
    report: dict = {"schema": "titan-reanalysis-1", "root": str(root),
                    "estimand_policy": (
                        "every ratio names its estimand; ratio_of_medians and "
                        "paired_median_ratio are both reported and are not "
                        "interchangeable"),
                    "source_to_result_map": []}

    # ---- training cohorts ----
    tcohorts = {}
    for case, by_block in sorted(tr.items()):
        cfg = next(iter(by_block.values()))["config"]
        tps = {b: (r.get("perf") or {}).get("tokens_per_s") for b, r in sorted(by_block.items())}
        ene = {b: (r.get("telemetry") or {}).get("energy_total_j") for b, r in sorted(by_block.items())}
        mem = {b: (r.get("memory") or {}).get("peak_alloc_mib_rank0") for b, r in sorted(by_block.items())}
        tcohorts[case] = {
            "group": cfg.get("group"), "recipe": cfg.get("mode"),
            "zero_stage": cfg.get("zero_stage"), "precision": cfg.get("precision"),
            "world_size": cfg.get("world_size"),
            "blocks": sorted(by_block),
            "tokens_per_s": summarise(tps.values()),
            "tokens_per_s_by_block": tps,
            "energy_total_j": summarise(ene.values()),
            "peak_alloc_mib_post_warmup": summarise(mem.values()),
            "memory_scope": ("post-warmup peak: reset_peak_memory_stats is called "
                             "after warmup, so this is NOT a whole-run peak"),
        }
        report["source_to_result_map"] += [
            {"result": f"training/{case}", "from": f"runs/P{b}_{case}/results"}
            for b in sorted(by_block)]
    report["training_cohorts"] = tcohorts

    # ---- paired scaling, within blocks only ----
    scaling = {}
    for case, c in tcohorts.items():
        if c["world_size"] in (None, 1):
            continue
        base = next((k for k, q in tcohorts.items()
                     if q["world_size"] == 1 and q["recipe"] == c["recipe"]
                     and q["precision"] == c["precision"]
                     and q["zero_stage"] == c["zero_stage"]
                     and (str(q["group"]).startswith("rtx") ==
                          str(c["group"]).startswith("rtx"))), None)
        if not base:
            scaling[case] = {"baseline": None,
                             "note": "no world_size-1 baseline of the same wrapper"}
            continue
        sp = paired_ratio(c["tokens_per_s_by_block"], tcohorts[base]["tokens_per_s_by_block"])
        eff = paired_ratio({b: v / c["world_size"] for b, v in c["tokens_per_s_by_block"].items()},
                           tcohorts[base]["tokens_per_s_by_block"])
        scaling[case] = {"baseline": base, "speedup": sp,
                         "efficiency_fraction": eff,
                         "efficiency_pct_median": (eff["median_ratio"] * 100.0
                                                   if eff.get("median_ratio") is not None else None),
                         "also_ratio_of_medians": ratio_of_medians(
                             c["tokens_per_s"], tcohorts[base]["tokens_per_s"])}
    report["paired_scaling"] = scaling

    # ---- aux cohorts ----
    acohorts = {}
    for (case, metric, sub), by_block in sorted(aux.items(), key=lambda kv: str(kv[0])):
        acohorts[f"{case}|{metric}|{sub}"] = {
            "case_id": case, "metric": metric, "sub_key": sub,
            "blocks": sorted(by_block), "by_block": by_block,
            "summary": summarise(by_block.values()),
        }
        report["source_to_result_map"] += [
            {"result": f"aux/{case}|{metric}|{sub}", "from": f"runs/Q{b}_{case}/results"}
            for b in sorted(by_block)]
    report["aux_cohorts"] = acohorts

    # ---- the HF dtype comparison, with BOTH estimands ----
    dtype = {}
    for fam in ("rtx_1", "p1080_1", "xp_1"):
        for bs in (1, 4, 16):
            k32 = f"H_{fam}_tinyllama_float32|gen_tokens_per_s|{bs}"
            k16 = f"H_{fam}_tinyllama_float16|gen_tokens_per_s|{bs}"
            if k32 not in acohorts or k16 not in acohorts:
                continue
            s32, s16 = acohorts[k32]["summary"], acohorts[k16]["summary"]
            dtype[f"{fam}|bs{bs}"] = {
                "fp32": s32, "fp16": s16,
                "fp16_over_fp32_ratio_of_medians": ratio_of_medians(s16, s32),
                "fp16_over_fp32_paired": paired_ratio(acohorts[k16]["by_block"],
                                                      acohorts[k32]["by_block"]),
                "direction": ("fp16 faster" if (s16["median"] or 0) > (s32["median"] or 0)
                              else "fp32 faster"),
            }
    report["hf_dtype_comparison"] = dtype
    try:
        report["energy_recomputation"] = recompute_energy_rows(root)
    except StrictInputError as exc:
        print(f"REFUSED: {exc}")
        return 3
    report["input_inventory"] = {
        "status_dispositions_seen": dict(DISPOSITIONS),
        "non_ok_attempts_retained": NON_OK,
        "note": ("failed, interrupted and superseded attempts remain visible "
                 "with their actual dispositions; valid slow observations are "
                 "never excluded for being slow"),
    }
    partial = [k for k, c in tcohorts.items() if not c["tokens_per_s"]["blocks_complete"]]
    partial += [k for k, c in acohorts.items() if not c["summary"]["blocks_complete"]]
    report["partial_evidence"] = {
        "groups_with_fewer_than_six_observations": sorted(partial),
        "eligibility": ("explicitly PARTIAL: cannot acquire six-block "
                        "completeness or full scientific eligibility"),
    }

    # Both criteria, generated from the observations, with explicit provenance.
    named = ([(c_id, c["tokens_per_s"], list(c["tokens_per_s_by_block"].values()))
              for c_id, c in tcohorts.items()]
             + [(k, c["summary"], list(c["by_block"].values()))
                for k, c in acohorts.items()])
    def endpoints_ok(v):
        v = sorted(x for x in v if finite(x))
        if not v: return None
        m = statistics.median(v)
        if m == 0: return False
        return (abs(max(v)-m)/m*100 <= 5.0) and (abs(m-min(v))/m*100 <= 5.0)
    def span_ok(v):
        v = sorted(x for x in v if finite(x))
        if not v: return None
        m = statistics.median(v)
        if m == 0: return False
        return (max(v)-min(v))/m*100 <= 5.0
    orig_miss = [k for k, _s, v in named if endpoints_ok(v) is False]
    strict_miss = [k for k, _s, v in named if span_ok(v) is False]
    extra = [k for k in strict_miss if k not in orig_miss]
    report["precision_criteria"] = {
        "schema_version": 2,
        "schema_note": ("v1 placed the STRICTER span rule under the key "
                        "'precision_goal'. That was wrong: the frozen plan's goal "
                        "is the ENDPOINT rule. v2 publishes both under explicit "
                        "names and keeps 'precision_goal' as a documented "
                        "compatibility alias for the ORIGINAL frozen rule only."),
        "groups_total": len(named),
        "groups_six_block": sum(1 for _k, s, _v in named if s["blocks_complete"]),
        "original_frozen_endpoint_rule": {
            "authority": "ORIGINAL, frozen",
            "provenance": ("bundle/plan/run_plan.json -> frozen_defaults."
                           "statistics.precision_goal, verbatim: 'endpoints "
                           "within +/-5% of the median'"),
            "definition": "max(|max-median|, |median-min|)/median <= 0.05",
            "met": len(named) - len(orig_miss),
            "misses": len(orig_miss),
            "missing_groups": sorted(orig_miss),
        },
        "additional_stricter_span_rule": {
            "authority": "POST-HOC, additional; NOT in the frozen plan",
            "definition": "(max-min)/median <= 0.05",
            "met": len(named) - len(strict_miss),
            "misses": len(strict_miss),
            "missing_groups": sorted(strict_miss),
            "stricter_only_misses": sorted(extra),
            "relationship": ("the stricter rule implies the original, so its "
                             "miss set strictly contains the original's"),
        },
        "precision_goal": {
            "compatibility_alias_for": "original_frozen_endpoint_rule",
            "definition": "max(|max-median|, |median-min|)/median <= 0.05",
            "met": len(named) - len(orig_miss),
            "misses": len(orig_miss),
        },
        "retained_because": "missed goals are retained, never discarded or tuned",
        "frozen_plan_unchanged": True,
    }
    (out / "reanalysis.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str))

    print(f"training cohorts: {len(tcohorts)}   aux cohorts: {len(acohorts)}")
    print(f"paired scaling comparisons: {sum(1 for v in scaling.values() if v.get('speedup'))}")
    e = report["energy_recomputation"]
    print(f"board-energy integrals recomputed: {e['board_energy_integrals_recomputed']}  "
          f"agree={e['agreeing_within_0_5_pct']}  disagree={e['disagreeing']}  "
          f"unbracketed={e['windows_not_bracketed']}  gap>{e['absolute_support_gap_rule_s']}s="
          f"{e['support_gap_exceeding_cutoff']}")
    print(f"max observed support gap: {e['max_observed_support_gap_s']}  "
          f"worst energy deviation: {e['worst_relative_deviation_pct']:.6f}% at {e['worst_deviation_at']}")
    pc = report["precision_criteria"]
    o, st = pc["original_frozen_endpoint_rule"], pc["additional_stricter_span_rule"]
    print(f"precision (ORIGINAL frozen endpoint rule): {o['met']}/{pc['groups_total']} "
          f"met, {o['misses']} misses")
    print(f"precision (POST-HOC stricter span rule)  : {st['met']}/{pc['groups_total']} "
          f"met, {st['misses']} misses, {len(st['stricter_only_misses'])} stricter-only")
    print("\nHF dtype, both estimands:")
    for k, v in sorted(dtype.items()):
        rm = v["fp16_over_fp32_ratio_of_medians"]["value"]
        pm = v["fp16_over_fp32_paired"]["median_ratio"]
        lo, hi = v["fp16_over_fp32_paired"].get("min_ratio"), v["fp16_over_fp32_paired"].get("max_ratio")
        print(f"  {k:14s} ratio_of_medians={rm:.5f}  paired_median={pm:.5f}  "
              f"paired_range=[{lo:.5f},{hi:.5f}]  -> {v['direction']}")
    print(f"  -> {out/'reanalysis.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
