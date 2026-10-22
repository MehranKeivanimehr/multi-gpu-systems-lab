"""Isolated native-AMP known-skip diagnostic. NOT a performance measurement.

The AMP arms all reported native_skipped_steps == 0, so the observer's
skipped-step accounting was never exercised on real hardware. A GradScaler
started at an absurdly high scale overflows its first unscale, which is the
documented scaler-warmup behaviour, not injected corruption: the scaler itself
decides to omit optimizer.step(). This run checks that when that happens the
observer reports it rather than silently counting an update.

Falsifiable: if the observer were counting attempts as updates, skipped would be
0 and base calls would equal attempts even though the scaler omitted steps.
"""
import json, os, pathlib, sys

sys.path.insert(0, os.environ["BENCH"])
import torch
from update_observer import UpdateObserver, OBSERVED_NONFINITE, OBSERVED_FINITE

STEPS = 16
FP16_MAX = 65504.0
TARGET_OVERFLOWS = 4  # the scaler halves per overflow, so this many then recovery


def main() -> int:
    if not torch.cuda.is_available():
        print("REFUSED: no CUDA device visible"); return 90
    dev = torch.device("cuda", 0)
    torch.manual_seed(1234)
    model = torch.nn.Sequential(
        torch.nn.Linear(256, 512), torch.nn.GELU(), torch.nn.Linear(512, 256),
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(32, 256, device=dev)
    y = torch.randn(32, 256, device=dev)

    # Calibrate: a fixed huge scale either never overflows or never recovers in a
    # bounded run. Measure this model's actual gradient magnitude, then choose a
    # scale exactly TARGET_OVERFLOWS halvings above its overflow threshold, so
    # the run must contain both skipped and applied steps.
    with torch.autocast("cuda", dtype=torch.float16):
        cal = torch.nn.functional.mse_loss(model(x), y)
    cal.backward()
    peak = max(float(q.grad.abs().max()) for q in model.parameters() if q.grad is not None)
    opt.zero_grad(set_to_none=True)
    threshold = FP16_MAX / peak
    init_scale = threshold * (2.0 ** TARGET_OVERFLOWS)

    obs = UpdateObserver(numerical_checks=True)
    obs.attach(opt)
    scaler = torch.cuda.amp.GradScaler(init_scale=init_scale, growth_interval=10**9)
    for _ in range(STEPS):
        obs.begin_step()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = torch.nn.functional.mse_loss(model(x), y)
        scaler.scale(loss).backward()
        before = scaler.get_scale()
        scaler.unscale_(opt)
        obs.observe_gradients(opt, scaling="unscaled")
        scaler.step(opt)           # conditionally omits opt.step()
        scaler.update()
        obs.record_scale(before, scaler.get_scale())
        obs.end_step()

    s = obs.summary()
    per = s["per_step"]
    skipped = [r for r in per if r["base_calls_in_step"] == 0]
    applied = [r for r in per if r["base_calls_in_step"] > 0]
    checks = {
        "observer_reported_at_least_one_skip": s["native_skipped_steps"] > 0,
        "base_calls_below_attempts": s["base_optimizer_calls"] < s["attempted_steps"],
        "every_skip_had_a_nonfinite_verdict":
            all(r["gradient_verdict"] == OBSERVED_NONFINITE for r in skipped),
        "every_applied_step_had_a_finite_verdict":
            all(r["gradient_verdict"] == OBSERVED_FINITE for r in applied),
        "attempts_reconcile":
            s["native_successful_logical_updates"] + s["native_skipped_steps"]
            == s["attempted_steps"],
        "logical_updates_match_base_calls":
            s["native_successful_logical_updates"] == s["base_optimizer_calls"],
        "coverage_still_complete": s["gradient_coverage_complete"] is True,
        "recovered_after_skips": len(applied) > 0,
        "both_branches_exercised": len(skipped) > 0 and len(applied) > 0,
        "skips_came_first":
            not applied or not skipped
            or max(r["step"] for r in skipped) < min(r["step"] for r in applied),
    }
    out = {
        "measurement_kind": "observer_skip_diagnostic_not_performance",
        "calibrated_peak_abs_grad": peak,
        "overflow_threshold_scale": threshold,
        "init_scale": init_scale, "target_overflows": TARGET_OVERFLOWS,
        "steps": STEPS,
        "device_name": torch.cuda.get_device_name(0),
        "visible_devices_env": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "attempted_steps": s["attempted_steps"],
        "base_optimizer_calls": s["base_optimizer_calls"],
        "native_successful_logical_updates": s["native_successful_logical_updates"],
        "native_skipped_steps": s["native_skipped_steps"],
        "nonfinite_gradient_steps": s["nonfinite_gradient_steps"],
        "gradient_coverage_complete": s["gradient_coverage_complete"],
        "scale_first": per[0]["scale_before"], "scale_last": per[-1]["scale_after"],
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "per_step": [{k: r[k] for k in ("step", "gradient_verdict",
                                        "base_calls_in_step", "scale_before")} for r in per],
    }
    d = pathlib.Path(os.environ["BENCH_RESULTS_DIR"])
    d.mkdir(parents=True, exist_ok=True)
    (d / "amp_skip_diagnostic.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"attempts={out['attempted_steps']} base_calls={out['base_optimizer_calls']} "
          f"skipped={out['native_skipped_steps']} nonfinite={out['nonfinite_gradient_steps']} "
          f"scale {out['scale_first']:.3g} -> {out['scale_last']:.3g}")
    return 0 if out["all_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
