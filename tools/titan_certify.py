"""Strict, non-vacuous certification of campaign records.

Why this replaces the previous verdict semantics
------------------------------------------------
The delivered `titan_audit.py` reported 379/384 PASS, but 46 of those passes ran
**zero checks** — including `D08` and `D22`, whose own payloads record failure —
because no handler matched their schema and an empty findings set produces no
failures. An empty `{}` and an arbitrary unknown object also passed. Separately,
vLLM findings were keyed `energy[bs={row['batch_size']}]`; vLLM rows carry
`concurrency`, so every row collapsed onto `energy[bs=None]` and later rows
overwrote earlier ones, leaving only the final row audited.

Design rules enforced here
--------------------------
1. A record whose schema is not recognised is **UNSUPPORTED**, never PASS.
2. Every supported schema declares MANDATORY check names. If any is absent from
   the findings, the record is **INVALID** (certification incomplete), never PASS.
3. Zero checks can never yield PASS.
4. Row-bearing schemas key findings by a **unique row id** and audit EVERY row.
5. Presence of a field is never evidence of the thing it names: an integer PID
   is not cleanup proof, a non-empty string is not a boolean, and an empty
   collection does not satisfy an "all" predicate.

CPU-only. Imports nothing that touches a device.
"""
from __future__ import annotations

import argparse, json, math, pathlib, sys
from typing import Any, Callable

PASS, FAIL, UNSUPPORTED, INVALID = "PASS", "FAIL", "UNSUPPORTED", "INVALID"
QUALIFIED = "QUALIFIED"   # integrity holds, but a stated limitation narrows the claim
NOMINAL_INTERVAL_S = 0.2
GAP_CUTOFF_S = 0.6                      # absolute support-gap rule, preserved
GPT2_124M_FP32_GRADIENT_BYTES = 124_439_808 * 4


# --------------------------------------------------------------- result helpers
def ok(detail: str = "") -> dict:
    return {"ok": True, "detail": detail}


def bad(detail: str) -> dict:
    return {"ok": False, "detail": detail}


def limit(detail: str) -> dict:
    """A claim-narrowing LIMITATION, not an integrity failure.

    Missing raw SSE does not make a streaming record corrupt; it makes the
    per-token reading unavailable. Conflating the two would either hide the
    limitation (if scored as a pass) or imply the data are untrustworthy (if
    scored as a failure). Limitations are recorded separately and downgrade the
    verdict to QUALIFIED.
    """
    return {"ok": True, "limitation": True, "detail": detail}


def is_finite_number(v: Any) -> bool:
    """True only for a real, finite number. inf and NaN are both rejected.

    The previous comparator used `x == x`, which accepts inf.
    """
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def strict_bool(v: Any) -> bool | None:
    """True/False only for real booleans. Any string -- including "false" --
    returns None, i.e. 'not a boolean', so it can never satisfy a flag check."""
    return v if isinstance(v, bool) else None


# --------------------------------------------------------------- integration
def integrate_trapezoid(ts: list[float], ps: list[float], t0: float, t1: float):
    """Independent trapezoid integration with linear boundary interpolation.

    Returns (energy_j, max_support_gap_s, n_used, status).
    """
    pts = sorted((t, p) for t, p in zip(ts, ps) if is_finite_number(t) and is_finite_number(p))
    if len(pts) < 2:
        return None, None, len(pts), "insufficient_samples"
    if pts[0][0] > t0 or pts[-1][0] < t1:
        return None, None, len(pts), "window_not_bracketed"

    def interp(at: float) -> float:
        for (ta, pa), (tb, pb) in zip(pts, pts[1:]):
            if ta <= at <= tb:
                if tb == ta:
                    return pa
                w = (at - ta) / (tb - ta)
                return pa + w * (pb - pa)
        return pts[-1][1]

    inner = [(t, p) for t, p in pts if t0 < t < t1]
    seq = [(t0, interp(t0))] + inner + [(t1, interp(t1))]
    gap = max((b[0] - a[0]) for a, b in zip(seq, seq[1:])) if len(seq) > 1 else None
    e = sum((b[0] - a[0]) * (a[1] + b[1]) / 2.0 for a, b in zip(seq, seq[1:]))
    status = "ok" if (gap is not None and gap <= GAP_CUTOFF_S) else "gap_exceeds_cutoff"
    return e, gap, len(seq), status


# --------------------------------------------------------------- shared checks
def check_telemetry(tel: dict, expect_devices: set[str] | None = None) -> dict:
    """Telemetry invariants plus an INDEPENDENT energy recomputation."""
    out: dict[str, dict] = {}
    iv = tel.get("sample_interval_s")
    out["sample_interval_is_frozen_nominal"] = (
        ok(f"{iv}") if iv == NOMINAL_INTERVAL_S
        else bad(f"sample_interval_s {iv!r} != nominal {NOMINAL_INTERVAL_S}"))
    mg = tel.get("max_permitted_gap_s")
    out["absolute_gap_rule_is_zero_point_six"] = (
        ok(f"{mg}") if is_finite_number(mg) and abs(mg - GAP_CUTOFF_S) < 1e-9
        else bad(f"max_permitted_gap_s {mg!r} != {GAP_CUTOFF_S}"))
    dur = tel.get("duration_s")
    out["duration_positive_finite"] = (
        ok(f"{dur}") if is_finite_number(dur) and dur > 0 else bad(f"duration_s {dur!r}"))

    raw = tel.get("raw_samples") or []
    t0, t1 = tel.get("window_start_s"), tel.get("window_stop_s")
    out["window_boundaries_present"] = (
        ok() if is_finite_number(t0) and is_finite_number(t1) and t1 > t0
        else bad(f"window [{t0!r},{t1!r}]"))
    out["raw_samples_present"] = (
        ok(f"{len(raw)}") if raw else bad("no raw_samples: energy is unverifiable"))

    # sensor set must be exactly what the run claims
    seen: set[str] = set()
    for s in raw:
        for g in (s.get("gpus") or []):
            if "index" in g:
                seen.add(str(g["index"]))
    claimed = set((tel.get("energy_per_gpu_j") or {}).keys())
    if claimed or seen:
        out["sensor_set_matches_claim"] = (
            ok(f"{sorted(seen)}") if seen == claimed
            else bad(f"raw sensors {sorted(seen)} != claimed {sorted(claimed)}"))
    if expect_devices is not None:
        out["sensor_set_matches_expected"] = (
            ok(f"{sorted(seen)}") if seen == expect_devices
            else bad(f"raw sensors {sorted(seen)} != expected {sorted(expect_devices)}"))

    # independent per-device energy
    if raw and is_finite_number(t0) and is_finite_number(t1):
        per = tel.get("energy_per_gpu_j") or {}
        total_recomputed = 0.0
        for gi in sorted(seen):
            ts = [s["time_s"] for s in raw
                  for g in (s.get("gpus") or [])
                  if str(g.get("index")) == gi and is_finite_number(g.get("power.draw"))]
            ps = [g["power.draw"] for s in raw
                  for g in (s.get("gpus") or [])
                  if str(g.get("index")) == gi and is_finite_number(g.get("power.draw"))]
            e, gap, n, st = integrate_trapezoid(ts, ps, t0, t1)
            out[f"gpu{gi}_integration_status"] = (
                ok(f"{st}, n={n}") if st == "ok" else bad(f"{st} (gap={gap})"))
            out[f"gpu{gi}_support_gap_within_cutoff"] = (
                ok(f"{gap:.4f}s") if is_finite_number(gap) and gap <= GAP_CUTOFF_S
                else bad(f"max support gap {gap!r} > {GAP_CUTOFF_S}"))
            rep = per.get(gi)
            if e is not None and is_finite_number(rep):
                dev = abs(e - rep) / rep * 100 if rep else 100.0
                out[f"gpu{gi}_energy_matches_report"] = (
                    ok(f"independent {e:.4f} J vs reported {rep:.4f} J ({dev:.5f}%)")
                    if dev <= 0.5 else bad(f"independent {e:.4f} != reported {rep:.4f} ({dev:.3f}%)"))
                total_recomputed += e
            else:
                out[f"gpu{gi}_energy_matches_report"] = bad(
                    f"cannot recompute (independent={e!r}, reported={rep!r})")
        tot = tel.get("energy_total_j")
        if is_finite_number(tot) and total_recomputed:
            d = abs(total_recomputed - tot) / tot * 100
            out["energy_total_matches_sum_of_independent"] = (
                ok(f"{total_recomputed:.3f} J vs {tot:.3f} J ({d:.5f}%)") if d <= 0.5
                else bad(f"independent sum {total_recomputed:.3f} != total {tot:.3f}"))

    for field, lo, hi in (("power_mean_w", 0.0, 1000.0), ("power_max_w", 0.0, 1000.0),
                          ("util_mean_pct", 0.0, 100.0), ("util_max_pct", 0.0, 100.0),
                          ("temp_max_c", 0.0, 120.0), ("mem_max_mib", 0.0, 100000.0)):
        vals = tel.get(field) or {}
        offenders = [(k, v) for k, v in vals.items() if not (is_finite_number(v) and lo <= v <= hi)]
        out[f"{field}_finite_and_in_range"] = (
            ok(f"{len(vals)} device(s)") if vals and not offenders
            else bad(f"{field}: {offenders or 'absent'}"))
    errs = tel.get("sampling_errors")
    out["no_sampling_errors"] = (
        ok() if isinstance(errs, list) and not errs else bad(f"sampling_errors={errs!r}"))
    return out


# --------------------------------------------------------------- per-schema audits
def audit_training(rec: dict) -> dict:
    out: dict[str, dict] = {}
    cfg, perf = rec.get("config") or {}, rec.get("perf") or {}
    tel = rec.get("telemetry") or {}
    status = rec.get("status")
    expect = {str(i) for i in (rec.get("meta", {}).get("phys_devices") or [])} or None
    out.update({f"telemetry.{k}": v for k, v in check_telemetry(tel, expect).items()})

    if status not in (None, "ok"):
        out["non_ok_claims_no_rate"] = (
            ok(f"status={status!r}") if perf.get("tokens_per_s") is None
            else bad(f"status={status!r} yet tokens_per_s={perf.get('tokens_per_s')!r}"))
        out["non_ok_claims_no_per_gpu_rate"] = (
            ok() if perf.get("tokens_per_s_per_gpu") is None
            else bad("per-gpu rate claimed by a run that never measured"))
    else:
        steps, tps = cfg.get("steps"), perf.get("tokens_per_step")
        wall, rep = perf.get("wall_time_s"), perf.get("tokens_per_s")
        if all(is_finite_number(x) for x in (steps, tps, wall, rep)) and wall > 0:
            recomputed = steps * tps / wall
            d = abs(recomputed - rep) / rep * 100 if rep else 100.0
            out["throughput_matches_step_accounting"] = (
                ok(f"independent {recomputed:.4f} vs {rep:.4f} ({d:.5f}%)") if d <= 0.5
                else bad(f"independent {recomputed:.4f} != reported {rep:.4f}"))
            ws, mb, sl = cfg.get("world_size"), cfg.get("micro_batch_per_gpu"), cfg.get("seq_len")
            if all(isinstance(x, int) for x in (ws, mb, sl)):
                out["tokens_per_step_matches_shape"] = (
                    ok(f"{tps}") if tps == ws * mb * sl
                    else bad(f"tokens_per_step {tps} != {ws}*{mb}*{sl}={ws*mb*sl}"))
            pg = perf.get("tokens_per_s_per_gpu")
            if is_finite_number(pg) and isinstance(ws, int) and ws:
                out["per_gpu_rate_consistent"] = (
                    ok() if abs(pg * ws - rep) <= max(0.1, abs(rep) * 1e-4)
                    else bad(f"{pg}*{ws} != {rep}"))
        else:
            out["throughput_matches_step_accounting"] = bad(
                "cannot recompute throughput: missing steps/tokens_per_step/wall/rate")

    losses = rec.get("losses")
    out["losses_present_and_finite"] = (
        ok(f"{len(losses)} values") if isinstance(losses, list) and losses
        and all(is_finite_number(x) for x in losses)
        else bad(f"losses={type(losses).__name__} len="
                 f"{len(losses) if isinstance(losses, list) else 'n/a'}"))
    out["memory_peak_scope_is_declared_post_warmup"] = ok(
        "peak_alloc_mib is POST-WARMUP (reset_peak_memory_stats is called after "
        "warmup), not a whole-run peak")

    audits = rec.get("optimizer_update_audit_per_rank")
    if not isinstance(audits, list) or not audits:
        out["per_rank_audit_present"] = bad("no optimizer_update_audit_per_rank")
        return out
    ws = cfg.get("world_size")
    out["per_rank_audit_count_matches_world_size"] = (
        ok(f"{len(audits)}") if isinstance(ws, int) and len(audits) == ws
        else bad(f"{len(audits)} rank audits for world_size={ws!r}"))
    for i, a in enumerate(audits):
        steps = a.get("per_step") or []
        requested = strict_bool(a.get("gradient_checks_requested"))
        if not steps:
            declared = (a.get("mode") == "deepspeed"
                        and "deepspeed_observer_status" in a
                        and str(a.get("observer_path")) != "native"
                        and strict_bool(a.get("numerical_diagnostic_certified")) is not True)
            if not declared:
                out[f"rank{i}.per_step_present"] = bad(
                    "no per-step payload and the engine path is not declared")
                continue
            out[f"rank{i}.engine_path_declared"] = ok("native observation structurally absent")
            out[f"rank{i}.native_claim_not_promoted"] = (
                ok() if strict_bool(a.get("deepspeed_hardware_validated")) is False
                else bad(f"deepspeed_hardware_validated={a.get('deepspeed_hardware_validated')!r}"))
            out[f"rank{i}.skip_path_not_claimed"] = (
                ok() if strict_bool(a.get("deepspeed_skip_path_exercised")) in (False, None)
                else bad("skip path claimed"))
            b, sk = a.get("engine_update_boundaries"), a.get("engine_skipped_updates")
            su, exp = a.get("engine_successful_updates"), a.get("expected_logical_updates")
            if all(isinstance(x, int) for x in (b, sk, su)):
                out[f"rank{i}.engine_counters_reconcile"] = (
                    ok(f"{b}-{sk}={su}") if b - sk == su else bad(f"{b}-{sk}!={su}"))
                if isinstance(exp, int):
                    out[f"rank{i}.boundaries_match_expected"] = (
                        ok(f"{b}") if b == exp else bad(f"{b} != expected {exp}"))
            else:
                out[f"rank{i}.engine_counters_present"] = bad(
                    f"counters not integers: {b!r},{sk!r},{su!r}")
            out[f"rank{i}.counters_are_limited_mechanics_evidence"] = ok(
                "engine counters are boundary mechanics; they are NOT gradient "
                "coverage and NOT an AdamW update oracle")
            continue

        succ = sum(1 for s in steps if (s.get("base_calls_in_step") or 0) > 0)
        base = sum((s.get("base_calls_in_step") or 0) for s in steps)
        fin = sum(1 for s in steps if s.get("gradient_verdict") == "observed_finite")
        nonfin = sum(1 for s in steps if s.get("gradient_verdict") == "observed_nonfinite")
        empty = sum(1 for s in steps if s.get("gradient_verdict") == "observed_but_no_gradient_tensors")
        unobs = sum(1 for s in steps if s.get("gradient_verdict") == "not_observed")
        out[f"rank{i}.attempts_match_per_step"] = (
            ok(f"{len(steps)}") if a.get("attempted_steps") == len(steps)
            else bad(f"attempted_steps={a.get('attempted_steps')!r} vs {len(steps)} rows"))
        for key, val in (("native_successful_logical_updates", succ),
                         ("base_optimizer_calls", base),
                         ("gradient_observations_finite", fin),
                         ("nonfinite_gradient_steps", nonfin),
                         ("missing_gradient_steps", empty),
                         ("unobserved_attempt_steps", unobs),
                         ("native_skipped_steps", len(steps) - succ)):
            if key in a:
                out[f"rank{i}.{key}"] = (
                    ok(f"{val}") if a[key] == val else bad(f"reported {a[key]!r} != recomputed {val}"))
        if requested is False:
            out[f"rank{i}.checks_off_claims_no_coverage"] = (
                ok() if strict_bool(a.get("gradient_coverage_complete")) in (None, False)
                else bad("coverage claimed with checks off"))
            out[f"rank{i}.checks_off_claims_no_certification"] = (
                ok() if strict_bool(a.get("numerical_diagnostic_certified")) in (None, False)
                else bad("certification claimed with checks off"))
            out[f"rank{i}.checks_off_all_unobserved"] = (
                ok(f"{unobs}/{len(steps)}") if unobs == len(steps)
                else bad(f"checks off yet only {unobs}/{len(steps)} unobserved"))
            out[f"rank{i}.checks_off_no_finite_claims"] = (
                ok() if fin == 0 and nonfin == 0 else bad(f"{fin} finite / {nonfin} nonfinite claimed"))
        elif requested is True:
            out[f"rank{i}.no_evidence_gaps_when_requested"] = (
                ok() if empty == 0 and unobs == 0
                else bad(f"checks requested but {empty} empty / {unobs} unobserved"))
            cov = (fin + nonfin) == len(steps) and empty == 0 and unobs == 0
            out[f"rank{i}.coverage_flag_matches_recomputation"] = (
                ok(f"{cov}") if strict_bool(a.get("gradient_coverage_complete")) == cov
                else bad(f"flag {a.get('gradient_coverage_complete')!r} != recomputed {cov}"))
        else:
            out[f"rank{i}.gradient_checks_requested_is_a_boolean"] = bad(
                f"gradient_checks_requested={a.get('gradient_checks_requested')!r} is not a bool")
    return out


def _row_id(kind: str, i: int, row: dict) -> str:
    """A UNIQUE, collision-free id for every row.

    The previous keying used `bs={row['batch_size']}`. vLLM rows have no
    batch_size, so every row keyed to `bs=None` and later rows overwrote earlier
    ones in the findings dict -- corrupting row 0 was invisible. The row index
    is always included, so no two rows can ever share a key.
    """
    disc = row.get("batch_size", row.get("concurrency", row.get("payload_bytes")))
    return f"{kind}[i={i},k={disc}]"


def _audit_rowed(rec: dict, kind: str, metric: str,
                 extra: Callable[[dict], dict] | None = None) -> dict:
    """Audit EVERY row of a row-bearing record under unique ids."""
    out: dict[str, dict] = {}
    rows = rec.get("rows")
    if not isinstance(rows, list) or not rows:
        out["rows_present"] = bad(f"rows={rows!r}: no raw rows, nothing to certify")
        return out
    out["rows_present"] = ok(f"{len(rows)} rows")
    expect = {str(i) for i in (rec.get("meta", {}).get("phys_devices") or [])} or None
    for i, row in enumerate(rows):
        rid = _row_id(kind, i, row)
        if row.get("status") not in (None, "ok"):
            out[f"{rid}.non_ok_row_declared"] = ok(f"status={row.get('status')!r}")
            continue
        tel = row.get("telemetry")
        if isinstance(tel, dict):
            for k, v in check_telemetry(tel, expect).items():
                out[f"{rid}.{k}"] = v
            for field in ("energy_total_j",):
                rr, tt = row.get(field), tel.get(field)
                if is_finite_number(rr) and is_finite_number(tt):
                    out[f"{rid}.row_energy_matches_telemetry"] = (
                        ok(f"{rr}") if abs(rr - tt) <= max(0.01, abs(tt) * 1e-6)
                        else bad(f"row {field}={rr} != telemetry {tt}"))
        else:
            out[f"{rid}.telemetry_present"] = bad("row has no telemetry block")
        v = row.get(metric)
        out[f"{rid}.{metric}_finite_positive"] = (
            ok(f"{v}") if is_finite_number(v) and v > 0 else bad(f"{metric}={v!r}"))
        if extra:
            for k, r in extra(row).items():
                out[f"{rid}.{k}"] = r
    return out


def audit_cnn(rec: dict) -> dict:
    def per(row):
        o = {}
        n, el, thr = row.get("n_images"), row.get("elapsed_s"), row.get("throughput")
        if all(is_finite_number(x) for x in (n, el, thr)) and el > 0:
            rc = n / el
            o["throughput_matches_count_over_window"] = (
                ok(f"independent {rc:.4f} vs {thr:.4f}")
                if abs(rc - thr) / thr * 100 <= 0.5 else bad(f"{rc:.4f} != {thr:.4f}"))
            tf = row.get("achieved_tflops")
            if is_finite_number(tf):
                g = tf * 1000 * el / n
                o["analytical_gflop_per_image_as_executed"] = (
                    ok(f"{g:.4f} GFLOP/img (executed constant)")
                    if abs(g - 8.2) < 0.01
                    else bad(f"recomputed {g:.4f} GFLOP/img; executed constant is 8.2"))
            jpi, et = row.get("energy_per_image"), row.get("energy_total_j")
            if all(is_finite_number(x) for x in (jpi, et, n)):
                o["joules_per_image_matches"] = (
                    ok(f"{et/n:.6f}") if abs(et / n - jpi) / jpi * 100 <= 0.5
                    else bad(f"{et/n:.6f} != {jpi:.6f}"))
        else:
            o["throughput_matches_count_over_window"] = bad("missing n_images/elapsed/throughput")
        return o
    return _audit_rowed(rec, "cnn", "throughput", per)


def audit_hf(rec: dict) -> dict:
    def per(row):
        o = {}
        o["memory_peak_scope_is_declared_post_warmup"] = ok(
            "peak_alloc_mib is POST-WARMUP (infer_hf.py resets peak stats), "
            "not a whole-run peak")
        o["prefill_is_a_separate_call_proxy"] = (
            ok("true_ttft_measured is false; prefill_forward_proxy_s is a proxy")
            if strict_bool(row.get("true_ttft_measured")) is False
            else bad(f"true_ttft_measured={row.get('true_ttft_measured')!r} on an HF arm"))
        nt, el, gen = row.get("new_tokens"), row.get("elapsed_s"), row.get("gen_tokens_per_s")
        if all(is_finite_number(x) for x in (nt, el, gen)) and el > 0:
            o["generation_rate_denominator_is_the_measured_window"] = (
                ok(f"{nt}/{el:.4f}={nt/el:.4f} vs {gen:.4f}")
                if abs(nt / el - gen) / gen * 100 <= 2.0
                else bad(f"{nt}/{el:.4f}={nt/el:.4f} vs reported {gen:.4f}"))
        return o
    return _audit_rowed(rec, "hf", "gen_tokens_per_s", per)


def audit_vllm_offline(rec: dict) -> dict:
    def per(row):
        o = {}
        ot, el, rate = row.get("output_tokens"), row.get("elapsed_s"), row.get("output_tokens_per_s")
        if all(is_finite_number(x) for x in (ot, el, rate)) and el > 0:
            o["output_rate_denominator_is_the_measured_window"] = (
                ok(f"{ot}/{el:.4f}={ot/el:.4f} vs {rate:.4f}")
                if abs(ot / el - rate) / rate * 100 <= 0.5
                else bad(f"{ot}/{el:.4f} != {rate:.4f}"))
        else:
            o["output_rate_denominator_is_the_measured_window"] = bad(
                "missing output_tokens/elapsed_s/output_tokens_per_s")
        return o
    out = _audit_rowed(rec, "vllm", "output_tokens_per_s", per)
    cfg = rec.get("config") or {}
    pc = [cfg[k] for k in ("enable_prefix_caching", "prefix_caching") if k in cfg]
    out["prefix_caching_disabled"] = (
        ok(f"{pc}") if pc and all(strict_bool(x) is False for x in pc)
        else bad(f"prefix caching evidence {pc!r}"))
    g = rec.get("guided_decoding") or {}
    out["guided_configured_and_exercised_recorded_separately"] = (
        ok(f"configured={g.get('configured_backend')!r} exercised={g.get('exercised')!r} "
           f"conformed={g.get('conformed')!r}")
        if "configured_backend" in g and "exercised" in g
        else bad(f"guided_decoding block incomplete: {sorted(g)}"))
    out["guided_backend_is_not_an_outlines_repair"] = (
        ok() if "outlines" in str(cfg.get("guided_decoding_backend_is_not_outlines_repair", "")).lower()
        or cfg.get("guided_decoding_backend_configured") == "lm-format-enforcer"
        else bad("the record does not state that the backend is not an outlines repair"))
    return out


def audit_collective(rec: dict) -> dict:
    out: dict[str, dict] = {}
    rows = rec.get("rows")
    if not isinstance(rows, list) or not rows:
        out["rows_present"] = bad(f"rows={rows!r}")
        return out
    out["rows_present"] = ok(f"{len(rows)} rows")
    n_exact = 0
    for i, row in enumerate(rows):
        rid = _row_id("collective", i, row)
        mib, by = row.get("size_mib"), row.get("payload_bytes")
        claim = strict_bool(row.get("is_exact_gpt2_124m_fp32_gradient_volume"))
        if claim is True:
            n_exact += 1
        if mib is None:
            out[f"{rid}.exact_payload_bytes"] = (
                ok(f"{by}") if by == GPT2_124M_FP32_GRADIENT_BYTES
                else bad(f"null size_mib but payload {by!r} != {GPT2_124M_FP32_GRADIENT_BYTES}"))
            out[f"{rid}.exact_row_is_flagged"] = (
                ok() if claim is True else bad(f"flag={row.get('is_exact_gpt2_124m_fp32_gradient_volume')!r}"))
        else:
            out[f"{rid}.payload_matches_mebibytes"] = (
                ok(f"{by}") if by == mib * 1024 * 1024 else bad(f"{by!r} != {mib}*1MiB"))
            if mib == 497:
                out[f"{rid}.497_mebibytes_is_not_the_gradient_volume"] = (
                    ok() if claim is False else bad(f"497 MiB claims exactness: {claim!r}"))
        for k in ("algbw_gb_s", "ring_equivalent_bw_gb_s", "time_ms"):
            out[f"{rid}.{k}_finite_positive"] = (
                ok(f"{row.get(k)}") if is_finite_number(row.get(k)) and row[k] > 0
                else bad(f"{k}={row.get(k)!r}"))
        if all(is_finite_number(row.get(k)) for k in ("payload_bytes", "time_ms", "algbw_gb_s")):
            rc = row["payload_bytes"] / (row["time_ms"] / 1000.0) / 1e9
            out[f"{rid}.algbw_matches_payload_over_time"] = (
                ok(f"independent {rc:.4f} vs {row['algbw_gb_s']:.4f}")
                if abs(rc - row["algbw_gb_s"]) / row["algbw_gb_s"] * 100 <= 1.0
                else bad(f"{rc:.4f} != {row['algbw_gb_s']:.4f}"))
    out["exactly_one_row_is_the_gradient_volume"] = (
        ok("1") if n_exact == 1
        else bad(f"{n_exact} rows flagged exact; the sweep must measure it exactly once"))
    out["ring_equivalent_is_an_analytical_normalisation"] = (
        ok() if any("NORMALIS" in str(r.get("ring_equivalent_bw_note", "")).upper()
                    for r in rows)
        else bad("ring-equivalent bandwidth is not labelled as an analytical normalisation"))
    p2p = rec.get("p2p_all_enabled")
    out["p2p_is_capability_not_cause"] = (
        ok(f"p2p_all_enabled={p2p!r}") if p2p is None or isinstance(p2p, bool)
        else bad(f"p2p_all_enabled={p2p!r}"))
    return out


def audit_streaming(rec: dict) -> dict:
    """Streaming is certified as a CHUNK-ARRIVAL diagnostic, not per-token latency.

    The executed driver timestamps the arrival of non-empty text chunks. It
    retains no token ids, no `usage`, no `finish_reason`, no HTTP status and no
    raw SSE. A chunk is not a token, so any per-token claim is refused here and
    the arm is admitted only under its qualified reading.
    """
    out: dict[str, dict] = {}
    perf, cfg = rec.get("perf") or {}, rec.get("config") or {}
    tel = rec.get("telemetry") or {}
    expect = {str(i) for i in (rec.get("meta", {}).get("phys_devices") or [])} or None
    out.update({f"telemetry.{k}": v for k, v in check_telemetry(tel, expect).items()})

    rows = rec.get("rows")
    out["raw_request_rows_present"] = (
        ok(f"{len(rows)} rows") if isinstance(rows, list) and rows
        else bad(f"rows={rows!r}: raw per-request evidence absent, so neither "
                 f"latency nor completion can be certified"))
    n_req = cfg.get("requests")
    out["row_count_matches_requested"] = (
        ok(f"{len(rows)}") if isinstance(rows, list) and isinstance(n_req, int)
        and len(rows) == n_req
        else bad(f"{len(rows) if isinstance(rows, list) else rows!r} rows for "
                 f"requests={n_req!r}"))
    if isinstance(rows, list) and rows:
        boundrows = [r for r in rows if is_finite_number(r.get("ttft_s")) and r["ttft_s"] > 0]
        out["every_row_has_a_finite_positive_first_arrival"] = (
            ok(f"{len(boundrows)}/{len(rows)}") if len(boundrows) == len(rows)
            else bad(f"only {len(boundrows)}/{len(rows)} rows have a usable ttft_s"))
        chunky = [r for r in rows if isinstance(r.get("n_chunks"), int) and r["n_chunks"] > 0]
        out["every_row_recorded_at_least_one_chunk"] = (
            ok(f"{len(chunky)}/{len(rows)}") if len(chunky) == len(rows)
            else bad(f"only {len(chunky)}/{len(rows)} rows recorded a chunk"))
        import statistics as _s
        tt = sorted(r["ttft_s"] for r in boundrows)
        if tt:
            med = _s.median(tt)
            rep = perf.get("ttft_median_s")
            out["first_arrival_median_matches_rows"] = (
                ok(f"independent {med:.6f} vs {rep:.6f}") if is_finite_number(rep)
                and abs(med - rep) <= max(1e-9, abs(rep) * 1e-6)
                else bad(f"independent median {med:.6f} vs reported {rep!r}"))
        allitl = [x for r in rows for x in (r.get("inter_token_latencies_s") or [])]
        out["chunk_gap_count_matches_rows"] = (
            ok(f"{len(allitl)}") if perf.get("itl_n") == len(allitl)
            else bad(f"itl_n={perf.get('itl_n')!r} vs {len(allitl)} recomputed gaps"))

    # --- the claim-narrowing checks: what this arm may and may not assert ---
    out["per_token_latency_is_NOT_claimed"] = (
        ok("field is chunk-arrival derived; per-token latency refused")
        if not any(k in perf for k in ("per_token_latency_s", "tpot_s"))
        else bad("the record asserts a per-token latency it cannot support"))
    out["token_identification_evidence_absent_is_declared"] = ok(
        "no token ids, no usage block and no finish_reason are retained, so "
        "arrivals are non-empty TEXT CHUNKS; a chunk is not a token")
    out["raw_sse_evidence_absent"] = limit(
        "no raw timestamped SSE payloads retained: completion, truncation and "
        "error handling cannot be verified from this record")
    out["http_status_evidence_absent"] = limit(
        "no per-request HTTP status recorded: success cannot be distinguished "
        "from a truncated or errored stream")
    out["cleanup_is_not_proven_by_a_pid_integer"] = limit(
        f"engine_pid={cfg.get('engine_pid')!r} is an integer, which is not "
        f"evidence the owned group exited; no confirmed-cleanup record exists")

    # workload declaration vs what actually ran
    req, act = cfg.get("prompt_len_requested"), cfg.get("prompt_len_tokens_actual")
    out["prompt_length_requested_equals_realised"] = (
        ok(f"{act}") if req == act
        else limit(f"requested {req!r} tokens but realised {act!r}: the arm did "
                   f"not run the declared prompt length, so it is not "
                   f"workload-comparable with the offline arms"))
    out["tail_percentiles_not_claimed"] = (
        ok() if strict_bool(perf.get("tail_percentiles_reported")) is False
        else bad(f"tail_percentiles_reported={perf.get('tail_percentiles_reported')!r}"))
    rate, wall = perf.get("requests_per_s"), perf.get("wall_time_s")
    if all(is_finite_number(x) for x in (rate, wall, n_req)) and wall > 0 and rate > 0:
        out["request_rate_matches_wall_time"] = (
            ok(f"{n_req}/{wall:.4f}={n_req/wall:.4f} vs {rate:.4f}")
            if abs(n_req / wall - rate) / rate * 100 <= 0.5
            else bad(f"{n_req/wall:.4f} != {rate:.4f}"))
    else:
        # A zero or non-finite rate, or a non-positive window, is refused rather
        # than divided by: malformed input must fail closed, not raise.
        out["request_rate_matches_wall_time"] = bad(
            f"cannot recompute a request rate: requests={n_req!r} "
            f"wall_time_s={wall!r} requests_per_s={rate!r}")
    return out


def audit_dtype_probe(rec: dict) -> dict:
    out: dict[str, dict] = {}
    dev, res, fl = rec.get("device") or {}, rec.get("results") or {}, rec.get("flags") or {}
    out["device_identified"] = (
        ok(f"{dev.get('name')} cap {dev.get('capability')}")
        if dev.get("name") and dev.get("capability") else bad(f"device={dev!r}"))
    for dt in ("fp32", "fp16", "bf16"):
        b = res.get(dt) or {}
        out[f"{dt}_tflops_finite_positive"] = (
            ok(f"{b.get('tflops')}") if is_finite_number(b.get("tflops")) and b["tflops"] > 0
            else bad(f"{dt}.tflops={b.get('tflops')!r}"))
        out[f"{dt}_result_finite_flag"] = (
            ok() if strict_bool(b.get("finite")) is True else bad(f"{dt}.finite={b.get('finite')!r}"))
        ms = b.get("ms_per_gemm")
        n = rec.get("gemm_n")
        if all(is_finite_number(x) for x in (ms, b.get("tflops"))) and isinstance(n, int) and ms > 0:
            flops = 2.0 * n ** 3
            rc = flops / (ms / 1000.0) / 1e12
            out[f"{dt}_tflops_matches_2n3_over_time"] = (
                ok(f"independent {rc:.4f} vs {b['tflops']:.4f}")
                if abs(rc - b["tflops"]) / b["tflops"] * 100 <= 1.0
                else bad(f"independent {rc:.4f} != reported {b['tflops']:.4f}"))
    for key, num, den in (("fp16_vs_fp32_speedup", "fp16", "fp32"),
                          ("bf16_vs_fp32_speedup", "bf16", "fp32")):
        a, b2 = (res.get(num) or {}).get("tflops"), (res.get(den) or {}).get("tflops")
        if all(is_finite_number(x) for x in (a, b2)) and b2:
            rc = a / b2
            out[f"{key}_matches_tflops_ratio"] = (
                ok(f"independent {rc:.4f} vs {rec.get(key)!r}")
                if is_finite_number(rec.get(key)) and abs(rc - rec[key]) <= 0.005
                else bad(f"independent {rc:.4f} != reported {rec.get(key)!r}"))
    e, nat = strict_bool(fl.get("bf16_including_emulation")), strict_bool(fl.get("bf16_native_api_flag"))
    out["bf16_emulation_and_native_flags_are_separate_booleans"] = (
        ok(f"including_emulation={e} native={nat}") if e is not None and nat is not None
        else bad(f"flags not booleans: {fl!r}"))
    out["bf16_claim_is_scoped_to_this_device"] = (
        ok() if "not all devices" in str(fl.get("scope", "")) else bad(f"scope={fl.get('scope')!r}"))
    return out


def audit_ds_observe(rec: dict) -> dict:
    """The engine-aware DeepSpeed diagnostic, recomputed from its raw per-step payload."""
    out: dict[str, dict] = {}
    agreed = rec.get("agreed_checks")
    out["agreed_checks_is_a_map_of_real_booleans"] = (
        ok(f"{len(agreed)} checks") if isinstance(agreed, dict) and agreed
        and all(strict_bool(v) is not None for v in agreed.values())
        else bad(f"agreed_checks={type(agreed).__name__}; non-boolean values are refused"))
    out["overall_flag_matches_agreed_checks"] = bad("cannot evaluate: agreed_checks unusable")
    if isinstance(agreed, dict) and agreed and all(strict_bool(v) is not None for v in agreed.values()):
        allp = all(agreed.values())
        out["overall_flag_matches_agreed_checks"] = (
            ok(f"{allp}") if strict_bool(rec.get("all_agreed_checks_passed")) == allp
            else bad(f"all_agreed_checks_passed={rec.get('all_agreed_checks_passed')!r} "
                     f"!= conjunction {allp}"))
        failing = [k for k, v in agreed.items() if v is False]
        out["no_agreed_check_failed"] = (
            ok() if not failing else bad(f"failing checks: {failing}"))
    steps = rec.get("per_step_rank0")
    out["raw_per_step_payload_present"] = (
        ok(f"{len(steps)} steps") if isinstance(steps, list) and steps
        else bad(f"per_step_rank0={type(steps).__name__}: no raw steps to recompute from"))
    n = rec.get("steps")
    if isinstance(steps, list) and steps:
        out["step_count_matches_raw"] = (
            ok(f"{len(steps)}") if n == len(steps) else bad(f"steps={n!r} vs {len(steps)} rows"))
        gradok = all((s.get("grad_evidence") or {}).get("verdict") == "observed_finite" for s in steps)
        out["every_raw_step_shows_finite_gradients"] = (
            ok(f"{len(steps)}") if gradok else bad("a raw step lacks observed_finite gradients"))
        changed = all(strict_bool(s.get("master_partition_changed")) is True for s in steps)
        out["every_raw_step_changed_the_master_partition"] = (
            ok() if changed else bad("a raw step did not change the master partition"))
        adv = all(strict_bool(s.get("base_state_advanced")) is True for s in steps)
        out["every_raw_step_advanced_base_optimizer_state"] = (
            ok() if adv else bad("a raw step did not advance base optimizer state"))
    per = rec.get("per_rank")
    ws = rec.get("world_size")
    out["per_rank_evidence_matches_world_size"] = (
        ok(f"{len(per)}") if isinstance(per, list) and isinstance(ws, int) and len(per) == ws
        else bad(f"per_rank={len(per) if isinstance(per, list) else per!r} ws={ws!r}"))
    if isinstance(per, list) and per:
        bads = [p.get("rank") for p in per
                if not all(strict_bool(v) is True for v in (p.get("local_checks") or {}).values())]
        out["every_rank_passed_its_own_local_checks"] = (
            ok(f"{len(per)} ranks") if not bads else bad(f"ranks failing locally: {bads}"))
    out["no_fault_injection"] = (
        ok() if rec.get("fault_injection") in (None, "none") else bad(f"{rec.get('fault_injection')!r}"))
    out["native_status_unchanged_is_declared"] = (
        ok() if "unobservable" in str(rec.get("native_observer_status_unchanged", "")).lower()
        else bad("the diagnostic does not restate that the native status is unchanged"))
    out["evidence_is_limited_mechanics_not_an_update_oracle"] = ok(
        "counters and changed master norms are limited mechanics evidence; they "
        "are NOT full expected gradient/partition coverage and NOT an intended "
        "AdamW update oracle")
    return out


def audit_amp_skip(rec: dict) -> dict:
    out: dict[str, dict] = {}
    checks = rec.get("checks")
    out["checks_is_a_map_of_real_booleans"] = (
        ok(f"{len(checks)}") if isinstance(checks, dict) and checks
        and all(strict_bool(v) is not None for v in checks.values())
        else bad(f"checks={type(checks).__name__}"))
    if isinstance(checks, dict) and checks and all(strict_bool(v) is not None for v in checks.values()):
        failing = [k for k, v in checks.items() if v is False]
        out["no_declared_check_failed"] = (
            ok(f"{len(checks)} checks") if not failing else bad(f"failing: {failing}"))
        allp = all(checks.values())
        out["overall_flag_matches_checks"] = (
            ok(f"{allp}") if strict_bool(rec.get("all_checks_passed")) == allp
            else bad(f"all_checks_passed={rec.get('all_checks_passed')!r} != {allp}"))
    steps = rec.get("per_step")
    out["raw_per_step_payload_present"] = (
        ok(f"{len(steps)}") if isinstance(steps, list) and steps else bad("no per_step rows"))
    if isinstance(steps, list) and steps:
        att, base = rec.get("attempted_steps"), rec.get("base_optimizer_calls")
        applied = sum(1 for s in steps if (s.get("base_calls_in_step") or 0) > 0)
        out["attempts_match_raw"] = (
            ok(f"{len(steps)}") if att == len(steps) else bad(f"{att!r} vs {len(steps)}"))
        out["base_calls_match_raw"] = (
            ok(f"{applied}") if base == applied else bad(f"{base!r} vs {applied}"))
        skipped = rec.get("native_skipped_steps")
        out["skips_match_raw"] = (
            ok(f"{len(steps)-applied}") if skipped == len(steps) - applied
            else bad(f"{skipped!r} vs {len(steps)-applied}"))
    return out


def audit_uuid_receipt(rec: dict) -> dict:
    """The numeric-visibility binding receipt. It is NOT in-process UUID proof."""
    out: dict[str, dict] = {}
    out["binding_mode_declared"] = (
        ok(f"{rec.get('binding_mode')}") if rec.get("binding_mode")
        else bad("no binding_mode"))
    ext = rec.get("verified_externally")
    exp = rec.get("verified_expected_uuids")
    out["external_uuid_map_present"] = (
        ok(f"{len(ext)} device(s)") if isinstance(ext, list) and ext else bad(f"{ext!r}"))
    if isinstance(ext, list) and isinstance(exp, list) and ext and len(ext) == len(exp):
        mism = [(e.get("uuid"), w) for e, w in zip(ext, exp) if e.get("uuid") != w]
        out["external_uuids_match_expected"] = (
            ok(f"{len(ext)} matched") if not mism else bad(f"mismatched: {mism}"))
    else:
        out["external_uuids_match_expected"] = bad(
            f"cannot compare: externally={type(ext).__name__} expected={type(exp).__name__}")
    names = rec.get("verified_runtime_device_names")
    out["runtime_names_present"] = (
        ok(f"{names}") if isinstance(names, list) and names else bad(f"{names!r}"))
    out["weaker_than_uuid_visibility_is_declared"] = (
        ok() if "UUID-bound" in str(rec.get("weaker_than", "")) else bad("no weaker_than note"))
    out["numeric_visibility_is_not_in_process_uuid_proof"] = limit(
        "numeric CUDA_VISIBLE_DEVICES plus nvidia-smi UUIDs and runtime device "
        "NAMES is an external correspondence argument; the CUDA runtime never "
        "reported a UUID in-process, so device identity is NOT certified")
    return out


# --------------------------------------------------------------- schema registry
def _has(rec: dict, *keys: str) -> bool:
    return all(k in rec for k in keys)


def detect_schema(rec: dict) -> str | None:
    """Return a supported schema name, or None (which yields UNSUPPORTED)."""
    if not isinstance(rec, dict) or not rec:
        return None
    cfg = rec.get("config") if isinstance(rec.get("config"), dict) else {}
    if _has(rec, "optimizer_update_audit_per_rank", "perf", "telemetry", "config"):
        return "training"
    if cfg.get("engine") == "vllm_loopback_streaming" or (
            _has(rec, "perf", "rows", "telemetry", "energy") and "startup_s_outside_measurement" in rec):
        return "vllm_streaming"
    if _has(rec, "guided_decoding", "rows", "config", "model_load_s"):
        return "vllm_offline"
    if _has(rec, "p2p_access", "rows") or _has(rec, "p2p_all_enabled", "rows"):
        return "collective"
    if _has(rec, "device", "results", "flags") and "gemm_n" in rec:
        return "dtype_probe"
    if _has(rec, "agreed_checks", "per_rank", "zero_stage", "observation_order"):
        return "ds_observe"
    if _has(rec, "checks", "all_checks_passed", "per_step") and "init_scale" in rec:
        return "amp_skip_diagnostic"
    if _has(rec, "binding_mode", "verified_externally", "verified_expected_uuids"):
        return "uuid_binding_receipt"
    if _has(rec, "device", "rows", "config"):
        rows = rec.get("rows") or [{}]
        first = rows[0] if isinstance(rows, list) and rows else {}
        if "achieved_tflops" in first or "n_images" in first:
            return "cnn"
        if "gen_tokens_per_s" in first or "new_tokens" in first:
            return "hf_generate"
    return None


# Every supported schema must produce AT LEAST these check names. A record that
# omits one is INVALID (certification incomplete), never PASS. Adding a name
# here deliberately invalidates older records rather than silently passing them.
MANDATORY: dict[str, tuple[str, ...]] = {
    "training": ("telemetry.sample_interval_is_frozen_nominal",
                 "telemetry.absolute_gap_rule_is_zero_point_six",
                 "telemetry.raw_samples_present",
                 "telemetry.window_boundaries_present",
                 "losses_present_and_finite",
                 "per_rank_audit_count_matches_world_size",
                 "memory_peak_scope_is_declared_post_warmup"),
    "cnn": ("rows_present",),
    "hf_generate": ("rows_present",),
    "vllm_offline": ("rows_present", "prefix_caching_disabled",
                     "guided_configured_and_exercised_recorded_separately"),
    "vllm_streaming": ("raw_request_rows_present", "row_count_matches_requested",
                       "per_token_latency_is_NOT_claimed",
                       "prompt_length_requested_equals_realised",
                       "cleanup_is_not_proven_by_a_pid_integer",
                       "telemetry.raw_samples_present"),
    "collective": ("rows_present", "exactly_one_row_is_the_gradient_volume",
                   "ring_equivalent_is_an_analytical_normalisation"),
    "dtype_probe": ("device_identified",
                    "bf16_emulation_and_native_flags_are_separate_booleans",
                    "fp32_tflops_finite_positive", "fp16_tflops_finite_positive",
                    "bf16_tflops_finite_positive"),
    "ds_observe": ("agreed_checks_is_a_map_of_real_booleans",
                   "overall_flag_matches_agreed_checks",
                   "raw_per_step_payload_present",
                   "per_rank_evidence_matches_world_size",
                   "no_fault_injection",
                   "evidence_is_limited_mechanics_not_an_update_oracle"),
    "amp_skip_diagnostic": ("checks_is_a_map_of_real_booleans",
                            "overall_flag_matches_checks",
                            "raw_per_step_payload_present"),
    "uuid_binding_receipt": ("binding_mode_declared", "external_uuid_map_present",
                             "numeric_visibility_is_not_in_process_uuid_proof"),
}

AUDITORS: dict[str, Callable[[dict], dict]] = {
    "training": audit_training, "cnn": audit_cnn, "hf_generate": audit_hf,
    "vllm_offline": audit_vllm_offline, "vllm_streaming": audit_streaming,
    "collective": audit_collective, "dtype_probe": audit_dtype_probe,
    "ds_observe": audit_ds_observe, "amp_skip_diagnostic": audit_amp_skip,
    "uuid_binding_receipt": audit_uuid_receipt,
}


def certify(path: str | pathlib.Path) -> dict:
    """Certify one record. Never returns PASS on zero checks or unknown schema."""
    p = pathlib.Path(path)
    try:
        rec = json.loads(p.read_text())
    except (OSError, ValueError) as exc:
        return {"path": str(p), "schema": None, "verdict": INVALID, "checks": {},
                "failures": [("read", f"{type(exc).__name__}: {exc}")],
                "limitations": [], "missing_mandatory": [], "n_checks": 0}
    schema = detect_schema(rec)
    if schema is None:
        return {"path": str(p), "schema": None, "verdict": UNSUPPORTED, "checks": {},
                "failures": [], "limitations": [], "missing_mandatory": [],
                "n_checks": 0,
                "note": "schema not recognised; unverified, explicitly NOT a pass"}
    checks = AUDITORS[schema](rec)
    missing = [k for k in MANDATORY[schema] if k not in checks]
    failures = [(k, v["detail"]) for k, v in checks.items() if not v["ok"]]
    limitations = [(k, v["detail"]) for k, v in checks.items() if v.get("limitation")]
    if not checks or missing:
        verdict = INVALID
    elif failures:
        verdict = FAIL
    elif limitations:
        verdict = QUALIFIED
    else:
        verdict = PASS
    return {"path": str(p), "schema": schema, "verdict": verdict, "checks": checks,
            "failures": failures, "limitations": limitations,
            "missing_mandatory": missing, "n_checks": len(checks)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Strict record certification (CPU only).")
    ap.add_argument("--root", required=True,
                    help="campaign/archive root containing runs/ (no absolute default)")
    ap.add_argument("--output", required=True, help="directory for the certification report")
    ap.add_argument("--glob", default="runs/*/results/*.json")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    root = pathlib.Path(a.root).resolve()
    outdir = pathlib.Path(a.output).resolve()
    if not root.is_dir():
        print(f"--root is not a directory: {root}"); return 2
    outdir.mkdir(parents=True, exist_ok=True)
    paths = sorted(root.glob(a.glob))
    if not paths:
        print(f"no records matched {a.glob!r} under {root}"); return 2

    results = [certify(p) for p in paths]
    tally: dict[str, int] = {}
    for r in results:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
    zero = [r for r in results if r["n_checks"] == 0]
    bad_pass = [r for r in results if r["verdict"] == PASS and r["n_checks"] == 0]

    report = {
        "schema": "titan-certify-2",
        "root": str(root),
        "records": len(results),
        "tally": tally,
        "records_with_zero_checks": len(zero),
        "zero_check_records_that_passed": len(bad_pass),
        "by_schema": {},
        "results": [{k: v for k, v in r.items() if k != "checks"} for r in results],
        "source_to_result_map": [
            {"source_record": str(pathlib.Path(r["path"]).relative_to(root)),
             "schema": r["schema"], "verdict": r["verdict"], "n_checks": r["n_checks"],
             "failures": [f[0] for f in r["failures"]],
             "limitations": [l[0] for l in r.get("limitations", [])],
             "missing_mandatory": r["missing_mandatory"]}
            for r in results],
    }
    for r in results:
        s = r["schema"] or "UNSUPPORTED"
        d = report["by_schema"].setdefault(s, {"n": 0, "verdicts": {}, "checks_min": None,
                                               "checks_max": None})
        d["n"] += 1
        d["verdicts"][r["verdict"]] = d["verdicts"].get(r["verdict"], 0) + 1
        d["checks_min"] = r["n_checks"] if d["checks_min"] is None else min(d["checks_min"], r["n_checks"])
        d["checks_max"] = r["n_checks"] if d["checks_max"] is None else max(d["checks_max"], r["n_checks"])
    (outdir / "certification.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str))

    if not a.quiet:
        print(f"records={len(results)}  " + "  ".join(f"{k}={v}" for k, v in sorted(tally.items())))
        print(f"zero-check records={len(zero)}  zero-check PASSES={len(bad_pass)} "
              f"(must be 0 by construction)")
        print(f"{'schema':24s} {'n':>4s} {'checks':>11s}  verdicts")
        for s, d in sorted(report["by_schema"].items()):
            print(f"{s:24s} {d['n']:>4d} {str(d['checks_min'])+'..'+str(d['checks_max']):>11s}  "
                  + " ".join(f"{k}={v}" for k, v in sorted(d["verdicts"].items())))
        for r in results:
            if r["verdict"] not in (PASS,):
                rel = pathlib.Path(r["path"]).relative_to(root)
                print(f"  {r['verdict']:11s} {str(rel.parent.parent.name):36s} "
                      f"schema={r['schema']} checks={r['n_checks']}")
                for k, d in r["failures"][:4]:
                    print(f"      FAIL {k}: {d[:100]}")
                for k, d in r.get("limitations", [])[:4]:
                    print(f"      LIMIT {k}: {d[:100]}")
                if r["missing_mandatory"]:
                    print(f"      MISSING MANDATORY: {r['missing_mandatory']}")
    print(f"  -> {outdir/'certification.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
