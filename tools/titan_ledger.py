"""Per-claim / per-cohort disposition ledger.

Dispositions (exactly these, never collapsed into pass/fail):
  supported            - observed, recomputed, and within its stated scope
  qualified            - observed, but a stated limitation narrows the claim
  unverified           - not contradicted, but the evidence to check it is absent
  invalid-for-claim    - the evidence provably characterises something else
  not-tested           - no attempt was made
  needs-fresh-evidence - only a new measurement can settle it

Portable: explicit --root/--output. CPU-only.
"""
from __future__ import annotations

import argparse, json, pathlib, sys

SUPPORTED, QUALIFIED, UNVERIFIED = "supported", "qualified", "unverified"
INVALID, NOT_TESTED, FRESH = "invalid-for-claim", "not-tested", "needs-fresh-evidence"


def build(root: pathlib.Path, cert: dict | None, rean: dict | None,
          dsclass: dict | None) -> dict:
    L: list[dict] = []

    def add(claim, cohort, disp, basis, scope=None):
        L.append({"claim": claim, "cohort": cohort, "disposition": disp,
                  "basis": basis, "scope_limit": scope})

    # ---- record integrity -------------------------------------------------
    if cert:
        t = cert.get("tally", {})
        add("Every delivered record is internally consistent and recomputable",
            f"{cert.get('records')} records",
            QUALIFIED,
            f"strict certification: {t}. 0 records with zero checks (the "
            f"delivered audit had 46, all passing).",
            "7 FAIL are superseded pre-fix records plus D08/D22, whose own "
            "payloads record failure; 28 QUALIFIED carry stated limitations")
        add("Certification is non-vacuous", "tooling", SUPPORTED,
            "unknown schema -> UNSUPPORTED; missing mandatory check -> INVALID; "
            "zero checks can never PASS; every row keyed by a unique id",
            None)

    # ---- energy -----------------------------------------------------------
    if rean and rean.get("energy_recomputation"):
        e = rean["energy_recomputation"]
        add("Reported board energy equals an independent re-integration",
            f"{e['board_energy_integrals_recomputed']} per-device windows",
            SUPPORTED,
            f"agree={e['agreeing_within_0_5_pct']} disagree={e['disagreeing']}, "
            f"worst deviation {e['worst_relative_deviation_pct']:.6f}%, "
            f"0 unbracketed, 0 support gaps above {e['absolute_support_gap_rule_s']}s "
            f"(max observed {e['max_observed_support_gap_s']:.4f}s)",
            e["energy_scope"])
        add("Energy figures are wall or PSU energy", "all energy rows", INVALID,
            "the sampler reads GPU board power only", 
            "uncalibrated GPU-board energy over the actual window")

    # ---- repeat statistics ------------------------------------------------
    if rean:
        pg = rean.get("precision_goal", {})
        add("Six-block repeat statistics are complete and reported in full",
            f"{pg.get('groups_six_block')} groups", SUPPORTED,
            "every repeat value is published beside each median; "
            f"{pg.get('groups_meeting')} meet the span<=5% rule, "
            f"{pg.get('groups_missing')} retained as missing (the delivered "
            f"two-sided-endpoint rule counted 13; the tighter span rule counts "
            f"16 and strictly contains them)", None)
        add("[min,max] gives 31/32 = 96.875% pointwise median coverage",
            "each six-block arm", QUALIFIED,
            "algebraically correct for 6 exchangeable continuous draws",
            "conditional on exchangeability and continuity; six fresh processes "
            "do not by themselves establish IID")
        add("Paired scaling ratios are computed within blocks",
            f"{sum(1 for v in rean.get('paired_scaling', {}).values() if v.get('speedup'))} comparisons",
            SUPPORTED, "per-block ratio then median; never a ratio of medians "
            "drawn from different block sets", None)

    # ---- dtype / inference ------------------------------------------------
    if rean and rean.get("hf_dtype_comparison"):
        d = rean["hf_dtype_comparison"]
        r16 = d.get("rtx_1|bs16", {})
        if r16:
            rm = r16["fp16_over_fp32_ratio_of_medians"]["value"]
            pm = r16["fp16_over_fp32_paired"]["median_ratio"]
            lo = r16["fp16_over_fp32_paired"]["min_ratio"]
            hi = r16["fp16_over_fp32_paired"]["max_ratio"]
            add("On TITAN RTX, fp16 HF generate is slower than fp32 (0.90x)",
                "rtx_1 batch 16", INVALID,
                f"0.90 is fp32/fp16, mislabelled as fp16/fp32. fp16 is FASTER: "
                f"ratio_of_medians={rm:.5f}, paired_median={pm:.5f}", None)
            add("fp16 is faster than fp32 on TITAN RTX at every batch size",
                "rtx_1 all batches", INVALID,
                f"only batch 16 favours fp16 (paired {pm:.5f}); batch 1 "
                f"({d['rtx_1|bs1']['fp16_over_fp32_paired']['median_ratio']:.5f}) and "
                f"batch 4 ({d['rtx_1|bs4']['fp16_over_fp32_paired']['median_ratio']:.5f}) "
                f"favour fp32", None)
            add("fp16 at batch 16 on TITAN RTX is faster than fp32",
                "rtx_1 batch 16", QUALIFIED,
                f"ratio_of_medians={rm:.5f}, paired_median={pm:.5f}",
                f"paired per-block range [{lo:.5f},{hi:.5f}] spans 1.0, so the "
                f"direction is not consistent across all six blocks")
        for fam, label in (("p1080_1", "GTX 1080 Ti"), ("xp_1", "TITAN Xp")):
            k = f"{fam}|bs16"
            if k in d:
                pm = d[k]["fp16_over_fp32_paired"]["median_ratio"]
                lo = d[k]["fp16_over_fp32_paired"]["min_ratio"]
                hi = d[k]["fp16_over_fp32_paired"]["max_ratio"]
                add(f"On {label}, fp16 HF generate is far slower than fp32",
                    f"{fam} batch 16", SUPPORTED,
                    f"paired median fp16/fp32 = {pm:.5f} (i.e. fp32 is "
                    f"{1/pm:.2f}x faster), paired range [{lo:.5f},{hi:.5f}] "
                    f"entirely below 1", "HF generate only; not a claim about "
                    "all fp16 work on Pascal")

    # ---- CNN --------------------------------------------------------------
    add("ResNet-50 achieved TFLOP/s uses 8.178 analytical GFLOP/image",
        "all CNN rows", INVALID,
        "the executed constant is 8.2 GFLOP/image; recomputation from the "
        "records returns 8.2002. 8.178 was never executed.",
        "any 8.178-based figure must be labelled a separate derived recalculation")

    # ---- memory -----------------------------------------------------------
    add("Reported peak memory is a whole-run peak",
        "training and HF rows", INVALID,
        "train_llm.py and infer_hf.py both call reset_peak_memory_stats after "
        "warmup", "figures are POST-WARMUP peaks")

    # ---- DeepSpeed --------------------------------------------------------
    if dsclass:
        for disp, n in dsclass.get("totals", {}).items():
            add("ZeRO performance rows are covered by engine-aware numerical evidence",
                f"{n} DeepSpeed performance rows", disp,
                dsclass.get("reason", ""), dsclass.get("scope"))
    add("The DeepSpeed native numerical path was validated on hardware",
        "all ZeRO rows", INVALID,
        "nothing brackets engine.step(); deepspeed_hardware_validated is false "
        "in every record", "structural, not a measurement gap")
    add("The DeepSpeed skip path was exercised", "all ZeRO rows", NOT_TESTED,
        "no overflow was induced and no skip injection was authorised", None)
    add("Engine counters and changed master norms prove the intended AdamW update",
        "all ZeRO rows", INVALID,
        "counters are boundary mechanics and a changed partition norm shows "
        "movement, not the intended update; no oracle was evaluated", None)

    # ---- streaming --------------------------------------------------------
    add("The streaming arm measures per-token inter-token latency",
        "7 streaming records", INVALID,
        "arrivals are non-empty TEXT CHUNKS; no token ids, usage or "
        "finish_reason retained. A chunk is not a token.",
        "field name inter_token_latencies_s is a misnomer")
    add("The streaming arm measures first non-empty text arrival and chunk cadence",
        "7 streaming records", QUALIFIED,
        "every row carries a finite positive first-arrival time and >=1 chunk; "
        "medians recomputed from rows",
        "no raw SSE, HTTP status, usage or completion evidence; cleanup not "
        "proven by an integer pid")
    add("The streaming arm ran the frozen offline workload",
        "7 streaming records", INVALID,
        "frozen config declares prompt_len=256 and max_new_tokens=128 and "
        "claims it applies identically to TP1/TP2/streaming; the arm ran "
        "output_len=64 with a realised prompt of 278 tokens", None)
    add("Streaming tail percentiles", "7 streaming records", NOT_TESTED,
        "100 requests support a median only; no tail percentile was claimed", None)

    # ---- references -------------------------------------------------------
    add("The cross-hardware loss comparison was preregistered",
        "REF2 pairs", INVALID,
        "the tolerance file was frozen 04:26:50Z, after D04/D06/D07 at "
        "03:25-03:28Z; the comparison was informed/exploratory",
        "this does not establish that thresholds were fitted")
    add("Matched FP32 single-GPU loss trajectories agree across families "
        "within the declared tolerances", "REF2 rtx vs p1080 and rtx vs xp",
        QUALIFIED,
        "first measured step agrees to 4.43e-3 and 5.03e-4 against a 5e-2 "
        "bound; max trajectory divergence 9.16e-3 against 5e-1",
        "losses[0] follows 5 warmup updates so it is not an untrained forward "
        "pass; the frozen input policy names a CPU generator while training "
        "drew from the CUDA generator; no realised input digest was retained")
    add("Matched controls exist for every recipe, precision and world size",
        "all recipes", NOT_TESTED,
        "only singleton FP32 cross-hardware references were run", None)

    # ---- topology ---------------------------------------------------------
    add("Disabled CUDA P2P causes the TITAN RTX pair's lower scaling efficiency",
        "rtx_2 cohorts", INVALID,
        "no intervention isolated P2P; the association is confounded with "
        "family, clocks, driver path and kernel selection",
        "retained as an association between a capability flag, measured "
        "bandwidth and training throughput")
    add("Measured all-reduce bandwidth and DDP throughput differ between the "
        "RTX pair and the Pascal groups", "collective + training cohorts",
        SUPPORTED, "both recomputed independently from raw rows", 
        "association only; no causal mechanism established")

    # ---- identity / provenance -------------------------------------------
    add("Device identity was certified in-process by UUID",
        "all records", INVALID,
        "torch 2.4.1 exposes neither uuid nor pci_bus_id and pynvml is absent; "
        "numeric visibility plus nvidia-smi UUIDs and runtime device NAMES is "
        "an external correspondence argument", None)
    add("The pyairports distribution was never executed",
        "env provenance", UNVERIFIED,
        "absence of __pycache__ and of test.txt is consistent with never "
        "running, but neither is proof: bytecode writing can be disabled and a "
        "sentinel file can be removed", None)
    add("The pyairports wheel proves no source build occurred",
        "env provenance", INVALID,
        "a py3-none-any wheel shows the artifact installed was pure; it does "
        "not establish that no source build happened anywhere upstream", None)

    # ---- execution path ---------------------------------------------------
    add("The audited Python launcher gated every campaign launch",
        "300 block launches", INVALID,
        "run_blocks.py and run_blocks_aux.py invoke tools/gpu_attempt.sh "
        "directly, bypassing bench/launch.py and bench/launch_gate.py", None)
    add("The shell launch path fails closed on failed, empty or malformed "
        "telemetry", "gpu_attempt.sh", FRESH,
        "the shell path parses nvidia-smi text without validating exit status, "
        "emptiness, NaN/inf or unrequested rows, holds no reservation, does not "
        "fully revalidate each settle poll, and confirms no ownership-scoped "
        "descendant cleanup", "requires the routed launcher plus lifecycle "
        "acceptance on the real path")

    return {"schema": "titan-claim-ledger-1", "root": str(root),
            "dispositions_used": [SUPPORTED, QUALIFIED, UNVERIFIED, INVALID,
                                  NOT_TESTED, FRESH],
            "entries": L}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--certification", default=None)
    ap.add_argument("--reanalysis", default=None)
    ap.add_argument("--ds-classification", required=True,
                    help="REQUIRED: DeepSpeed cohort classification. Omitting it "
                         "previously produced 31 entries instead of 32, silently "
                         "dropping the disposition covering all 98 historical rows.")
    a = ap.parse_args(argv)
    root = pathlib.Path(a.root).resolve()
    out = pathlib.Path(a.output).resolve(); out.mkdir(parents=True, exist_ok=True)

    def rd(p, *, required=False):
        """An explicitly supplied but unreadable input is an ERROR, never
        silently converted into absent optional data."""
        if not p:
            if required:
                print(f"REFUSED: a required input was not supplied")
                raise SystemExit(3)
            return None
        try:
            return json.loads(pathlib.Path(p).read_text())
        except (OSError, ValueError) as exc:
            print(f"REFUSED: explicitly supplied input is unreadable: {p}: "
                  f"{type(exc).__name__}: {exc}")
            raise SystemExit(3)

    ds = rd(a.ds_classification, required=True)
    total = sum(ds.get("totals", {}).values())
    if total != 98:
        print(f"REFUSED: the DeepSpeed classification covers {total} rows; the "
              f"disposition must cover all 98 historical rows (96 P-cohort + "
              f"D33 + D34)")
        raise SystemExit(3)
    led = build(root, rd(a.certification, required=True),
                rd(a.reanalysis, required=True), ds)
    counts: dict[str, int] = {}
    for e in led["entries"]:
        counts[e["disposition"]] = counts.get(e["disposition"], 0) + 1
    led["disposition_counts"] = counts
    led["counting_unit_note"] = (
        "entries are CLAIMS, not records or rows. Record-level certification "
        "counts (384 artifacts) and historical row counts (98 DeepSpeed rows) "
        "are different units and are reported separately; a structurally "
        "consistent record does not become numerically certified.")
    led["deepspeed_rows_covered"] = total
    (out / "claim_ledger.json").write_text(json.dumps(led, indent=2, sort_keys=True, default=str))
    counts: dict[str, int] = {}
    for e in led["entries"]:
        counts[e["disposition"]] = counts.get(e["disposition"], 0) + 1
    print(f"ledger entries: {len(led['entries'])}")
    for k in led["dispositions_used"]:
        print(f"  {k:22s} {counts.get(k, 0)}")
    print()
    for e in led["entries"]:
        print(f"[{e['disposition']:20s}] {e['claim'][:72]}")
        print(f"      cohort: {e['cohort']}")
        if e["scope_limit"]:
            print(f"      scope : {e['scope_limit'][:100]}")
    print(f"\n  -> {out/'claim_ledger.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
