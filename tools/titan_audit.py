"""Independent recomputation of Titan campaign records.

Deliberately does NOT trust any summary field. Every check recomputes the
quantity from the RAW payload (raw power samples, per-step verdicts, explicit
counts) and then compares against what the record claims. A mismatch is a
finding, not a rounding note.

`--self-test` builds synthetic valid and deliberately corrupted records in a
scratch directory and asserts that each corruption is rejected. It never touches
real results.
"""
from __future__ import annotations
import argparse, json, math, os, sys, tempfile

NOMINAL_INTERVAL_S = 0.2
GAP_CUTOFF_S = 3 * NOMINAL_INTERVAL_S          # 0.6 s, declared before data
GPT2_124M_FP32_GRADIENT_BYTES = 124_439_808 * 4
PINNED_REVISION = "fe8a4ea1ffedaf415f4da2f062534de366a451e6"


class Finding(Exception):
    pass


def _f(msg):
    return {"ok": False, "detail": msg}


def _ok(detail=""):
    return {"ok": True, "detail": detail}


# ---------------------------------------------------------------- energy
def integrate(ts, ps, t0, t1):
    """Trapezoid integral with boundary interpolation. Returns (J, covered)."""
    pts = sorted(zip(ts, ps))
    if not pts or pts[0][0] > t0 or pts[-1][0] < t1:
        return None, False
    def interp(t):
        for i in range(1, len(pts)):
            if pts[i][0] >= t:
                (ta, pa), (tb, pb) = pts[i - 1], pts[i]
                return pa if tb == ta else pa + (pb - pa) * (t - ta) / (tb - ta)
        return pts[-1][1]
    nodes = [(t0, interp(t0))] + [(t, p) for t, p in pts if t0 < t < t1] + [(t1, interp(t1))]
    return sum(0.5 * (nodes[i-1][1] + nodes[i][1]) * (nodes[i][0] - nodes[i-1][0])
               for i in range(1, len(nodes))), True


def audit_energy(row, tel):
    out = {}
    t0, t1 = tel.get("window_start_s"), tel.get("window_stop_s")
    raw = tel.get("raw_samples") or []
    if t0 is None or t1 is None or not raw:
        return {"window": _f("missing window bounds or raw samples")}
    out["sampling_interval_declared"] = (
        _ok(f"{tel.get('sample_interval_s')}") if tel.get("sample_interval_s") == NOMINAL_INTERVAL_S
        else _f(f"declared interval {tel.get('sample_interval_s')!r} != frozen {NOMINAL_INTERVAL_S}"))
    if tel.get("sampling_errors"):
        out["no_query_failures"] = _f(f"{len(tel['sampling_errors'])} sampling errors recorded")
    else:
        out["no_query_failures"] = _ok()
    idx = sorted({g["index"] for s in raw for g in s["gpus"]})
    total = 0.0
    per_dev = {}
    for gi in idx:
        ts = [s["time_s"] for s in raw for g in s["gpus"]
              if g["index"] == gi and g.get("power.draw") is not None]
        ps = [g["power.draw"] for s in raw for g in s["gpus"]
              if g["index"] == gi and g.get("power.draw") is not None]
        if not ts:
            out[f"gpu{gi}_readings"] = _f("no power readings"); continue
        brackets = min(ts) <= t0 and max(ts) >= t1
        gaps = [b - a for a, b in zip(ts, ts[1:])] or [0.0]
        j, covered = integrate(ts, ps, t0, t1)
        per_dev[gi] = j
        out[f"gpu{gi}_brackets_boundaries"] = _ok() if brackets else _f("readings do not bracket both boundaries")
        out[f"gpu{gi}_covered"] = _ok() if covered else _f("window not fully covered; refusing to extrapolate")
        out[f"gpu{gi}_max_gap"] = (_ok(f"{max(gaps):.4f}s") if max(gaps) <= GAP_CUTOFF_S
                                   else _f(f"max gap {max(gaps):.4f}s > cutoff {GAP_CUTOFF_S}s"))
        if covered:
            total += j
    rep = tel.get("energy_total_j")
    if rep is None:
        out["energy_matches_report"] = _f("no reported energy_total_j")
    else:
        dev = abs(total - rep) / rep * 100 if rep else float("inf")
        out["energy_matches_report"] = (_ok(f"independent {total:.3f} J vs reported {rep:.3f} J ({dev:.4f}%)")
                                        if dev <= 0.5 else _f(f"energy mismatch {dev:.4f}%"))
    # joules per image recomputed from EXPLICIT counts over the SAME interval
    n_img, elapsed = row.get("n_images"), (t1 - t0)
    if n_img:
        jpi = total / n_img
        rep_jpi = row.get("energy_per_image")
        out["j_per_image_matches"] = (_ok(f"{jpi:.6f}")
            if rep_jpi and abs(jpi - rep_jpi) / rep_jpi * 100 <= 0.5
            else _f(f"J/image independent {jpi:.6f} vs reported {rep_jpi!r}"))
        tp = n_img / elapsed
        rep_tp = row.get("throughput")
        out["throughput_matches"] = (_ok(f"{tp:.3f} img/s")
            if rep_tp and abs(tp - rep_tp) / rep_tp * 100 <= 0.5
            else _f(f"throughput independent {tp:.3f} vs reported {rep_tp!r}"))
        out["energy_and_count_share_interval"] = _ok(
            f"both over [{t0:.3f},{t1:.3f}] span {elapsed:.4f}s (not an assumed 60s)")
    return out


# ------------------------------------------------------- numerical validity
def audit_numerics(rec):
    out = {}
    audits = rec.get("optimizer_update_audit_per_rank") or []
    if not audits:
        return {"present": _f("no per-rank update audit")}
    for i, a in enumerate(audits):
        steps = a.get("per_step") or []
        if not steps:
            # A ZeRO arm legitimately has no native per-step payload: nothing
            # brackets engine.step(). That is a DECLARED structural condition,
            # not a corrupt record, and must not be scored the same way. It is
            # only acceptable when the record says so and carries the engine
            # counters plus the separately-evidenced engine-aware fields.
            declared = (a.get("mode") == "deepspeed"
                        and "deepspeed_observer_status" in a
                        and str(a.get("observer_path")) != "native"
                        and a.get("numerical_diagnostic_certified") is not True)
            if not declared:
                out[f"rank{i}_per_step"] = _f(
                    "no per-step payload to recompute from, and the record does "
                    "not declare the engine path")
                continue
            out[f"rank{i}_engine_path_declared"] = _ok(
                "native per-step observation is structurally absent and declared")
            out[f"rank{i}_native_claim_not_promoted"] = (
                _ok() if a.get("deepspeed_hardware_validated") is False
                else _f(f"deepspeed_hardware_validated "
                        f"{a.get('deepspeed_hardware_validated')!r} must be False"))
            out[f"rank{i}_skip_path_not_claimed"] = (
                _ok() if a.get("deepspeed_skip_path_exercised") in (False, None)
                else _f("deepspeed_skip_path_exercised must not be claimed"))
            b, sk = a.get("engine_update_boundaries"), a.get("engine_skipped_updates")
            su = a.get("engine_successful_updates")
            if all(isinstance(x, int) for x in (b, sk, su)):
                out[f"rank{i}_engine_counters_reconcile"] = (
                    _ok(f"{b} - {sk} = {su}") if b - sk == su
                    else _f(f"engine counters do not reconcile: {b} - {sk} != {su}"))
                exp = a.get("expected_logical_updates")
                if isinstance(exp, int):
                    out[f"rank{i}_boundaries_match_expected"] = (
                        _ok(f"{b}") if b == exp
                        else _f(f"engine_update_boundaries {b} != expected {exp}"))
            else:
                out[f"rank{i}_engine_counters_present"] = _f(
                    f"engine counters missing/non-integer: "
                    f"boundaries={b!r} skipped={sk!r} successful={su!r}")
            ev = a.get("deepspeed_engine_aware_validation")
            if isinstance(ev, dict):
                out[f"rank{i}_engine_aware_evidence_consistent"] = (
                    _ok(f"validated={ev.get('validated')} "
                        f"stage={ev.get('zero_stage')} ws={ev.get('world_size')}")
                    if bool(ev.get("validated")) == bool(
                        a.get("deepspeed_engine_aware_validated"))
                    else _f("engine-aware validation flag contradicts its artifact"))
            continue
        succ = sum(1 for s in steps if s.get("base_calls_in_step", 0) > 0)
        finite = sum(1 for s in steps if s.get("gradient_verdict") == "observed_finite")
        nonfin = sum(1 for s in steps if s.get("gradient_verdict") == "observed_nonfinite")
        empty = sum(1 for s in steps if s.get("gradient_verdict") == "observed_but_no_gradient_tensors")
        unobs = sum(1 for s in steps if s.get("gradient_verdict") == "not_observed")
        out[f"rank{i}_successful_updates"] = (
            _ok(f"{succ}") if succ == a.get("native_successful_logical_updates")
            else _f(f"recomputed {succ} != reported {a.get('native_successful_logical_updates')!r}"))
        out[f"rank{i}_attempts"] = (
            _ok(f"{len(steps)}") if len(steps) == a.get("attempted_steps")
            else _f(f"recomputed {len(steps)} attempts != reported {a.get('attempted_steps')!r}"))
        cov = (finite + nonfin) == len(steps) and empty == 0 and unobs == 0
        out[f"rank{i}_gradient_coverage"] = (
            _ok("every attempt validated") if cov == bool(a.get("gradient_coverage_complete"))
            else _f(f"recomputed coverage {cov} != reported {a.get('gradient_coverage_complete')!r}"))
        requested = a.get("gradient_checks_requested")
        if requested is False:
            # A performance run deliberately disables gradient checks, so every
            # attempt is legitimately unobserved. That is acceptable ONLY if the
            # record claims nothing it did not observe: no coverage, no
            # certification. Accepting the gap without that condition would let
            # a run assert verified numerics it never looked at.
            out[f"rank{i}_checks_declared_off"] = _ok(
                f"gradient checks not requested; {unobs} unobserved attempts expected")
            out[f"rank{i}_claims_no_coverage_when_checks_off"] = (
                _ok() if a.get("gradient_coverage_complete") in (None, False)
                else _f(f"checks were off but gradient_coverage_complete="
                        f"{a.get('gradient_coverage_complete')!r}"))
            out[f"rank{i}_claims_no_certification_when_checks_off"] = (
                _ok() if a.get("numerical_diagnostic_certified") in (None, False)
                else _f(f"checks were off but numerical_diagnostic_certified="
                        f"{a.get('numerical_diagnostic_certified')!r}"))
            out[f"rank{i}_all_attempts_unobserved_when_checks_off"] = (
                _ok(f"{unobs}/{len(steps)}") if unobs == len(steps)
                else _f(f"checks were off yet only {unobs} of {len(steps)} "
                        f"attempts are unobserved"))
            out[f"rank{i}_no_finite_claims_when_checks_off"] = (
                _ok() if finite == 0 and nonfin == 0
                else _f(f"checks were off but {finite} finite / {nonfin} "
                        f"nonfinite gradient observations are claimed"))
        elif empty or unobs:
            out[f"rank{i}_evidence_gaps"] = _f(
                f"gradient checks WERE requested but {empty} empty-gradient and "
                f"{unobs} unobserved attempts were recorded")
        # The scalar counters are what readers quote, so each must be
        # reconciled against per_step independently. Checking only the
        # coverage boolean let a forged missing_gradient_steps pass.
        for key, recomputed in (("gradient_observations_finite", finite),
                                ("nonfinite_gradient_steps", nonfin),
                                ("missing_gradient_steps", empty),
                                ("unobserved_attempt_steps", unobs),
                                ("base_optimizer_calls",
                                 sum(s.get("base_calls_in_step", 0) for s in steps)),
                                ("native_skipped_steps", len(steps) - succ)):
            if key in a:
                out[f"rank{i}_{key}"] = (
                    _ok(f"{recomputed}") if a[key] == recomputed
                    else _f(f"reported {key}={a[key]!r} != recomputed {recomputed}"))
    return out


# --------------------------------------------------------------- units etc
def audit_units(rec):
    out = {}
    for row in rec.get("rows", []):
        if "size_mib" not in row:
            continue
        mib, by = row["size_mib"], row.get("payload_bytes")
        claim = row.get("is_exact_gpt2_124m_fp32_gradient_volume")
        if mib is None:
            # The exact-gradient-volume point is addressed in bytes, not MiB,
            # precisely because it is not a whole number of mebibytes.
            out["exact_payload_row_bytes"] = (
                _ok(f"{by}") if by == GPT2_124M_FP32_GRADIENT_BYTES
                else _f(f"size_mib is null but payload {by!r} != "
                        f"{GPT2_124M_FP32_GRADIENT_BYTES}"))
            out["exact_payload_row_is_flagged"] = (
                _ok() if claim is True
                else _f(f"the exact-volume row must be flagged, got {claim!r}"))
            continue
        out[f"payload_{mib}MiB_bytes"] = (_ok(f"{by}") if by == mib * 1024 * 1024
                                          else _f(f"{by!r} != {mib*1024*1024}"))
        if mib == 497:
            out["497MiB_not_gradient_volume"] = (
                _ok("correctly flagged not exact") if claim is False
                else _f(f"claims exactness: {claim!r}"))
        exact_mib = row.get("size_mib_exact")
        if isinstance(exact_mib, (int, float)) and isinstance(by, int):
            out[f"payload_{mib}MiB_exact_mib_consistent"] = (
                _ok(f"{exact_mib}") if abs(exact_mib - by / 1024 / 1024) < 1e-3
                else _f(f"size_mib_exact {exact_mib} != {by/1024/1024:.3f}"))
    rows = [r for r in rec.get("rows", []) if "size_mib" in r]
    if rows:
        n_exact = sum(1 for r in rows
                      if r.get("is_exact_gpt2_124m_fp32_gradient_volume") is True)
        out["exactly_one_row_is_the_gradient_volume"] = (
            _ok("1") if n_exact == 1
            else _f(f"{n_exact} rows flagged as the exact gradient volume; "
                    f"the sweep must measure it exactly once"))
    return out


def audit_inputs(rec, corpus_sha=None):
    out = {}
    cfg = rec.get("config", {})
    rev = cfg.get("model_revision") or cfg.get("revision")
    if cfg.get("model", "").startswith("TinyLlama") or "tinyllama" in str(cfg.get("model", "")).lower():
        out["pinned_revision"] = (_ok(rev) if rev == PINNED_REVISION
                                  else _f(f"revision {rev!r} != pinned {PINNED_REVISION}"))
        if corpus_sha:
            out["corpus_hash"] = (_ok() if cfg.get("corpus_sha256") == corpus_sha
                                  else _f(f"corpus {cfg.get('corpus_sha256')!r} != {corpus_sha}"))
        # Prefix caching is a vLLM concept. HF `generate` has no equivalent, so
        # requiring the key there reported a failure for a setting that does not
        # exist. The two arms also spelled the key differently.
        engine = str(cfg.get("engine", ""))
        if engine.startswith("vllm"):
            present = [k for k in ("enable_prefix_caching", "prefix_caching")
                       if k in cfg]
            if not present:
                out["prefix_caching_disabled"] = _f(
                    "no prefix-caching field recorded on a vLLM arm")
            else:
                vals = {k: cfg[k] for k in present}
                out["prefix_caching_disabled"] = (
                    _ok(f"{vals}") if all(v is False for v in vals.values())
                    else _f(f"prefix caching {vals}"))
        elif engine:
            out["prefix_caching_not_applicable"] = _ok(
                f"{engine} has no prefix-caching setting; not checked")
    return out


def audit_telemetry_invariants(tel):
    """Schema-independent telemetry invariants.

    These ran on CNN-shaped records only, because audit_record looked for
    telemetry under rec["rows"]. Training records carry it at the top level, so
    none of their energy, interval or sampling fields were ever checked. A
    corrupted-copy trial passed 9 of 10 mutations before this was added.
    """
    out = {}
    cutoff = round(3 * NOMINAL_INTERVAL_S, 9)

    iv = tel.get("sample_interval_s")
    out["sample_interval_is_frozen_value"] = (
        _ok(f"{iv}") if iv == NOMINAL_INTERVAL_S
        else _f(f"sample_interval_s {iv!r} != frozen {NOMINAL_INTERVAL_S}"))

    mg = tel.get("max_permitted_gap_s")
    out["gap_cutoff_is_three_intervals"] = (
        _ok(f"{mg}") if mg is not None and abs(mg - cutoff) < 1e-9
        else _f(f"max_permitted_gap_s {mg!r} != 3 x {NOMINAL_INTERVAL_S} = {cutoff}"))

    mf = tel.get("sample_interval_matches_frozen")
    if mf is not None:
        out["interval_match_flag_is_consistent"] = (
            _ok() if bool(mf) == (iv == NOMINAL_INTERVAL_S)
            else _f(f"flag {mf!r} contradicts interval {iv!r}"))

    dur = tel.get("duration_s")
    out["duration_positive"] = (
        _ok(f"{dur:.4f} s") if isinstance(dur, (int, float)) and dur > 0
        else _f(f"duration_s {dur!r} is not positive"))

    for field, lo, hi in (("power_mean_w", 0.0, 1000.0),
                          ("power_max_w", 0.0, 1000.0),
                          ("util_mean_pct", 0.0, 100.0),
                          ("util_max_pct", 0.0, 100.0),
                          ("temp_max_c", 0.0, 120.0),
                          ("mem_max_mib", 0.0, 100000.0)):
        vals = tel.get(field) or {}
        bad = [(k, v) for k, v in vals.items()
               if not isinstance(v, (int, float)) or v != v or not lo <= v <= hi]
        out[f"{field}_within_bounds"] = (
            _ok(f"{len(vals)} device(s)") if not bad
            else _f(f"{field} out of [{lo},{hi}]: {bad}"))

    ig = tel.get("integration") or {}
    bad_status = [k for k, v in ig.items() if isinstance(v, dict) and v.get("status") != "ok"]
    out["integration_status_ok"] = (
        _ok(f"{len(ig)} device(s)") if not bad_status
        else _f(f"non-ok integration for {bad_status}"))
    over = [(k, v.get("max_support_gap_s")) for k, v in ig.items()
            if isinstance(v, dict) and isinstance(v.get("max_support_gap_s"), (int, float))
            and v["max_support_gap_s"] > cutoff]
    out["observed_gaps_within_cutoff"] = (
        _ok() if not over
        else _f(f"max_support_gap_s exceeds {cutoff}s: {over}"))

    # Energy must equal the sum of its per-device parts, and each part must
    # equal what that device's own integration reported.
    per = tel.get("energy_per_gpu_j") or {}
    tot = tel.get("energy_total_j")
    if per and isinstance(tot, (int, float)):
        ssum = sum(per.values())
        out["energy_total_is_sum_of_parts"] = (
            _ok(f"{ssum:.3f} J") if abs(ssum - tot) <= max(0.01, abs(tot) * 1e-6)
            else _f(f"energy_total_j {tot} != sum of parts {ssum}"))
    mismatched = [(k, per.get(k), ig[k].get("energy_j")) for k in per
                  if k in ig and isinstance(ig[k], dict)
                  and isinstance(ig[k].get("energy_j"), (int, float))
                  and abs(per[k] - ig[k]["energy_j"]) > max(0.01, abs(per[k]) * 1e-6)]
    out["per_device_energy_matches_integration"] = (
        _ok() if not mismatched else _f(f"energy disagrees with integration: {mismatched}"))

    errs = tel.get("sampling_errors") or []
    out["no_sampling_errors"] = _ok() if not errs else _f(f"{len(errs)} sampling errors")
    return out


def audit_training_perf(rec):
    """Recompute a training record's throughput from its own step accounting."""
    out = {}
    perf, cfg = rec.get("perf") or {}, rec.get("config") or {}
    steps, tps = cfg.get("steps"), perf.get("tokens_per_step")
    wall, rep = perf.get("wall_time_s"), perf.get("tokens_per_s")
    status = rec.get("status")
    if status not in (None, "ok"):
        # A run that refused to enter measurement -- for instance a ZeRO arm
        # asked for numerical checks it cannot provide -- has no throughput by
        # design. The requirement then INVERTS: it must not report one.
        # Recording elapsed time before refusing is legitimate and useful; it
        # is the derived RATE that must be absent.
        out["non_ok_run_claims_no_rate"] = (
            _ok(f"status={status!r}; elapsed recorded, no rate claimed")
            if rep is None
            else _f(f"status={status!r} yet tokens_per_s={rep!r} is claimed; "
                    f"a run that never entered measurement must not report a rate"))
        out["non_ok_run_claims_no_per_gpu_rate"] = (
            _ok() if perf.get("tokens_per_s_per_gpu") is None
            else _f(f"status={status!r} yet tokens_per_s_per_gpu="
                    f"{perf.get('tokens_per_s_per_gpu')!r} is claimed"))
        return out
    if not all(isinstance(x, (int, float)) for x in (steps, tps, wall, rep)) or wall <= 0:
        return {"recomputable": _f("missing steps/tokens_per_step/wall_time_s/tokens_per_s")}
    recomputed = steps * tps / wall
    dev = abs(recomputed - rep) / rep * 100 if rep else 100.0
    out["tokens_per_s_matches_step_accounting"] = (
        _ok(f"independent {recomputed:.1f} vs reported {rep:.1f} ({dev:.4f}%)")
        if dev <= 0.5 else
        _f(f"recomputed {recomputed:.1f} tok/s != reported {rep:.1f} ({dev:.2f}%)"))
    ws, mb = cfg.get("world_size"), cfg.get("micro_batch_per_gpu")
    sl = cfg.get("seq_len")
    if all(isinstance(x, int) for x in (ws, mb, sl)):
        out["tokens_per_step_matches_shape"] = (
            _ok(f"{tps}") if tps == ws * mb * sl
            else _f(f"tokens_per_step {tps} != world_size x batch x seq_len "
                    f"= {ws}x{mb}x{sl} = {ws*mb*sl}"))
    pg = perf.get("tokens_per_s_per_gpu")
    if isinstance(pg, (int, float)) and isinstance(ws, int) and ws:
        out["per_gpu_throughput_consistent"] = (
            _ok() if abs(pg * ws - rep) <= max(0.1, abs(rep) * 1e-4)
            else _f(f"tokens_per_s_per_gpu {pg} x {ws} != {rep}"))
    return out


def audit_streaming_perf(rec):
    """A streaming latency arm reports TTFT and inter-token latency, not a
    token-throughput rate, so the training recomputation does not apply. What
    must hold is that the latency claim is real and bounded."""
    out = {}
    perf, cfg = rec.get("perf") or {}, rec.get("config") or {}
    out["true_ttft_is_claimed_and_measured"] = (
        _ok() if perf.get("true_ttft_measured") is True
        else _f(f"true_ttft_measured={perf.get('true_ttft_measured')!r} on an "
                f"arm whose whole purpose is measuring it"))
    for k, lo, hi in (("ttft_median_s", 0.0, 60.0),
                      ("ttft_min_s", 0.0, 60.0),
                      ("ttft_max_s", 0.0, 60.0),
                      ("itl_median_s", 0.0, 10.0)):
        v = perf.get(k)
        out[f"{k}_positive_and_bounded"] = (
            _ok(f"{v:.6f} s") if isinstance(v, (int, float)) and lo < v <= hi
            else _f(f"{k}={v!r} outside ({lo},{hi}]"))
    lo_, med, hi_ = perf.get("ttft_min_s"), perf.get("ttft_median_s"), perf.get("ttft_max_s")
    if all(isinstance(x, (int, float)) for x in (lo_, med, hi_)):
        out["ttft_order_is_consistent"] = (
            _ok(f"{lo_:.4f} <= {med:.4f} <= {hi_:.4f}") if lo_ <= med <= hi_
            else _f(f"min/median/max out of order: {lo_}, {med}, {hi_}"))
    out["inter_token_samples_present"] = (
        _ok(f"{perf.get('itl_n')}") if isinstance(perf.get("itl_n"), int)
        and perf["itl_n"] > 0 else _f(f"itl_n={perf.get('itl_n')!r}"))
    out["tail_percentiles_not_claimed"] = (
        _ok("median only, as the sample size supports")
        if perf.get("tail_percentiles_reported") is False
        else _f(f"tail_percentiles_reported="
                f"{perf.get('tail_percentiles_reported')!r} at "
                f"{cfg.get('requests')} requests"))
    reqs, wall = cfg.get("requests"), perf.get("wall_time_s")
    rate = perf.get("requests_per_s")
    if all(isinstance(x, (int, float)) for x in (reqs, wall, rate)) and wall > 0:
        recomputed = reqs / wall
        out["request_rate_matches_wall_time"] = (
            _ok(f"independent {recomputed:.3f} vs reported {rate:.3f}")
            if abs(recomputed - rate) / rate * 100 <= 0.5
            else _f(f"recomputed {recomputed:.3f} != reported {rate:.3f}"))
    out["engine_was_reaped"] = (
        _ok(f"engine_pid={cfg.get('engine_pid')} recorded")
        if isinstance(cfg.get("engine_pid"), int)
        else _f("no engine pid recorded, so cleanup cannot be checked"))
    out["guided_exercise_state_is_explicit"] = (
        _ok(f"configured={cfg.get('guided_decoding_backend_configured')!r}, "
            f"exercised={cfg.get('guided_backend_exercised_in_this_arm')!r}")
        if cfg.get("guided_backend_exercised_in_this_arm") is not None
        else _f("the arm does not say whether the guided backend was exercised"))
    return out


def audit_record(path, corpus_sha=None):
    rec = json.load(open(path))
    findings = {}
    for row in rec.get("rows", []):
        tel = row.get("telemetry")
        if tel and tel.get("raw_samples"):
            findings[f"energy[bs={row.get('batch_size')}]"] = audit_energy(row, tel)
        if tel:
            findings[f"telemetry[bs={row.get('batch_size')}]"] = \
                audit_telemetry_invariants(tel)
    # Training records keep telemetry at the top level, not under "rows".
    top_tel = rec.get("telemetry")
    if isinstance(top_tel, dict):
        findings["telemetry"] = audit_telemetry_invariants(top_tel)
        if top_tel.get("raw_samples"):
            # The image-count checks inside audit_energy self-skip when
            # n_images is absent, which it is for a training record.
            findings["energy"] = audit_energy(rec, top_tel)
    cfg_mode = (rec.get("config") or {}).get("mode")
    engine = str((rec.get("config") or {}).get("engine", ""))
    if rec.get("perf") and cfg_mode in ("single", "ddp", "fsdp", "deepspeed"):
        findings["perf"] = audit_training_perf(rec)
    elif rec.get("perf") and engine == "vllm_loopback_streaming":
        findings["streaming"] = audit_streaming_perf(rec)
    if rec.get("optimizer_update_audit_per_rank"):
        findings["numerics"] = audit_numerics(rec)
    u = audit_units(rec)
    if u:
        findings["units"] = u
    i = audit_inputs(rec, corpus_sha)
    if i:
        findings["inputs"] = i
    failures = [(g, k, v["detail"]) for g, checks in findings.items()
                for k, v in checks.items() if not v["ok"]]
    return {"path": path, "groups": findings, "failures": failures,
            "verdict": "PASS" if not failures else "FAIL"}


# --------------------------------------------------------------- self-test
def _synthetic_energy_record(**corrupt):
    t0, t1 = 1000.0, 1060.0
    n = int((t1 - t0) / NOMINAL_INTERVAL_S) + 3
    raw = []
    for k in range(n):
        t = t0 - NOMINAL_INTERVAL_S + k * NOMINAL_INTERVAL_S
        raw.append({"time_s": t, "query_start_s": t - 0.01, "query_stop_s": t + 0.01,
                    "gpus": [{"index": 0, "power.draw": 100.0, "pstate": "P2"}]})
    if corrupt.get("drop_tail"):
        raw = [s for s in raw if s["time_s"] < t1 - 5]
    if corrupt.get("big_gap"):
        raw = [s for s in raw if not (t0 + 10 < s["time_s"] < t0 + 12)]
    j, _ = integrate([s["time_s"] for s in raw],
                     [s["gpus"][0]["power.draw"] for s in raw], t0, t1)
    tel = {"window_start_s": t0, "window_stop_s": t1, "raw_samples": raw,
           "sample_interval_s": corrupt.get("interval", NOMINAL_INTERVAL_S),
           "max_permitted_gap_s": corrupt.get("cutoff", round(3 * NOMINAL_INTERVAL_S, 9)),
           "duration_s": corrupt.get("duration", t1 - t0),
           "sampling_errors": corrupt.get("errors", []),
           "energy_total_j": corrupt.get("energy", j if j else 0.0)}
    n_img = 6000
    row = {"batch_size": 32, "n_images": n_img, "telemetry": tel,
           "energy_per_image": (tel["energy_total_j"] / n_img) if not corrupt.get("jpi") else corrupt["jpi"],
           "throughput": corrupt.get("throughput", n_img / (t1 - t0))}
    return {"rows": [row], "config": {}}


def _synthetic_numeric_record(**corrupt):
    steps = []
    for i in range(5):
        steps.append({"step": i, "gradient_verdict": "observed_finite",
                      "base_calls_in_step": 1, "successful_update": True})
    if corrupt.get("empty_grads"):
        for s in steps:
            s["gradient_verdict"] = "observed_but_no_gradient_tensors"
    a = {"per_step": steps, "attempted_steps": corrupt.get("attempts", len(steps)),
         "native_successful_logical_updates": corrupt.get("succ", 5),
         "gradient_coverage_complete": corrupt.get("cov", not corrupt.get("empty_grads")),
         "base_optimizer_calls": 5}
    return {"optimizer_update_audit_per_rank": [a], "config": {}}


def self_test() -> int:
    scratch = tempfile.mkdtemp(prefix="titan_audit_selftest_")
    cases = []

    def write(name, obj):
        p = os.path.join(scratch, name)
        json.dump(obj, open(p, "w"))
        return p

    # integrator sanity on constant and linear traces
    ts = [i * 0.1 for i in range(101)]
    cj, _ = integrate(ts, [100.0] * 101, 0.0, 10.0)
    lj, _ = integrate(ts, [10.0 * t for t in ts], 0.0, 10.0)
    _, cov = integrate([1.0, 2.0], [50.0, 50.0], 0.5, 2.5)
    cases.append(("integrator_constant_1000J", abs(cj - 1000) < 1e-6))
    cases.append(("integrator_linear_500J", abs(lj - 500) < 1e-9))
    cases.append(("integrator_rejects_uncovered_window", cov is False))

    cases.append(("valid_energy_passes",
                  audit_record(write("good_e.json", _synthetic_energy_record()))["verdict"] == "PASS"))
    for label, kw in (("corrupt_energy_value", {"energy": 1.0}),
                      ("corrupt_jpi", {"jpi": 99.0}),
                      ("corrupt_throughput", {"throughput": 1.0}),
                      ("truncated_coverage", {"drop_tail": True}),
                      ("oversized_gap", {"big_gap": True}),
                      ("wrong_interval", {"interval": 0.1}),
                      ("silent_query_failures", {"errors": ["timeout"]})):
        r = audit_record(write(f"bad_{label}.json", _synthetic_energy_record(**kw)))
        cases.append((f"rejects_{label}", r["verdict"] == "FAIL"))

    cases.append(("valid_numerics_passes",
                  audit_record(write("good_n.json", _synthetic_numeric_record()))["verdict"] == "PASS"))
    for label, kw in (("inflated_successful_updates", {"succ": 9}),
                      ("inflated_attempts", {"attempts": 99}),
                      ("empty_gradients_claiming_coverage", {"empty_grads": True, "cov": True})):
        r = audit_record(write(f"bad_{label}.json", _synthetic_numeric_record(**kw)))
        cases.append((f"rejects_{label}", r["verdict"] == "FAIL"))

    # A valid sweep now carries BOTH the historical 497 MiB point and the exact
    # gradient volume, the latter addressed in bytes because it is not a whole
    # number of mebibytes.
    _exact_row = {"size_mib": None, "payload_bytes": GPT2_124M_FP32_GRADIENT_BYTES,
                  "size_mib_exact": round(GPT2_124M_FP32_GRADIENT_BYTES / 1024 / 1024, 3),
                  "is_exact_gpt2_124m_fp32_gradient_volume": True}
    _497_row = {"size_mib": 497, "payload_bytes": 497 * 1024 * 1024,
                "size_mib_exact": 497.0,
                "is_exact_gpt2_124m_fp32_gradient_volume": False}
    good_u = {"rows": [dict(_497_row), dict(_exact_row)], "config": {}}
    # the original conflation: the 497 MiB point claiming to be the gradient volume
    bad_u = {"rows": [{"size_mib": 497, "payload_bytes": GPT2_124M_FP32_GRADIENT_BYTES,
                       "is_exact_gpt2_124m_fp32_gradient_volume": True}], "config": {}}
    # a sweep that never measures the exact volume at all
    missing_u = {"rows": [dict(_497_row)], "config": {}}
    # a sweep that flags the exact volume twice
    double_u = {"rows": [dict(_exact_row), dict(_exact_row)], "config": {}}
    cases.append(("valid_units_pass", audit_record(write("good_u.json", good_u))["verdict"] == "PASS"))
    cases.append(("rejects_bad_units", audit_record(write("bad_u.json", bad_u))["verdict"] == "FAIL"))
    cases.append(("rejects_sweep_missing_exact_payload",
                  audit_record(write("missing_u.json", missing_u))["verdict"] == "FAIL"))
    cases.append(("rejects_sweep_with_two_exact_payloads",
                  audit_record(write("double_u.json", double_u))["verdict"] == "FAIL"))

    width = max(len(n) for n, _ in cases)
    for name, ok in cases:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}")
    bad = [n for n, ok in cases if not ok]
    print(f"\n  self-test: {len(cases) - len(bad)}/{len(cases)} checks behaved correctly")
    print(f"  scratch (synthetic only, no real results touched): {scratch}")
    return 0 if not bad else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--corpus-sha256", default=None)
    p.add_argument("records", nargs="*")
    a = p.parse_args(argv)
    if a.self_test:
        return self_test()
    if not a.records:
        p.error("give record paths or --self-test")
    rc = 0
    for path in a.records:
        r = audit_record(path, a.corpus_sha256)
        print(f"{r['verdict']}  {path}")
        for g, k, d in r["failures"]:
            print(f"   FAIL {g}.{k}: {d}")
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
