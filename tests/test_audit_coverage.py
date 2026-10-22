"""The validator must actually reject corrupted TRAINING records.

A corrupted-copy trial found the validator passing 9 of 10 mutations of a real
six-GPU DDP record while its own self-test reported 17/17. The cause: the self
test built CNN-shaped fixtures, whose telemetry sits under rec["rows"], and
`audit_record` looked only there. Training records carry telemetry at the top
level, so none of their energy, interval, bounds or throughput fields were ever
checked. These tests pin the training shape so that gap cannot reopen.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def module_at(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def training_record(interval=0.2, steps=30, world=2, mb=4, seq=512, dur=10.0):
    """A minimally complete record in the shape train_llm.py actually writes."""
    n = int(dur / interval) + 3
    raw, power = [], 120.0
    for k in range(n):
        t = 1000.0 - interval + k * interval
        raw.append({"time_s": t, "query_start_s": t - 0.01, "query_stop_s": t + 0.01,
                    "gpus": [{"index": i, "power.draw": power, "pstate": "P2"}
                             for i in range(world)]})
    t0, t1 = 1000.0, 1000.0 + dur
    per = {str(i): power * dur for i in range(world)}
    tokens_per_step = world * mb * seq
    return {
        "status": "ok",
        "config": {"mode": "ddp", "precision": "fp32", "world_size": world,
                   "micro_batch_per_gpu": mb, "seq_len": seq, "steps": steps,
                   "group": "p1080_2", "model": "gpt2-124m"},
        "perf": {"tokens_per_step": tokens_per_step, "wall_time_s": dur,
                 "tokens_per_s": steps * tokens_per_step / dur,
                 "tokens_per_s_per_gpu": steps * tokens_per_step / dur / world},
        "memory": {"peak_alloc_mib_rank0": 5000.0},
        "telemetry": {
            "window_start_s": t0, "window_stop_s": t1, "duration_s": dur,
            "raw_samples": raw, "sample_interval_s": interval,
            "sample_interval_frozen_default_s": 0.2,
            "sample_interval_matches_frozen": interval == 0.2,
            "max_permitted_gap_s": 0.6, "sampling_errors": [],
            "energy_total_j": sum(per.values()), "energy_per_gpu_j": per,
            "power_mean_w": {str(i): power for i in range(world)},
            "power_max_w": {str(i): power for i in range(world)},
            "util_mean_pct": {str(i): 95.0 for i in range(world)},
            "util_max_pct": {str(i): 98.0 for i in range(world)},
            "mem_max_mib": {str(i): 5400.0 for i in range(world)},
            "temp_max_c": {str(i): 60.0 for i in range(world)},
            "integration": {str(i): {"status": "ok", "energy_j": per[str(i)],
                                     "max_support_gap_s": 0.25,
                                     "duration_s": dur, "n_samples": n}
                            for i in range(world)},
        },
        "optimizer_update_audit_per_rank": [
            {"mode": "ddp", "observer_path": "native", "attempted_steps": steps,
             "native_successful_logical_updates": steps,
             "base_optimizer_calls": steps, "native_skipped_steps": 0,
             "gradient_observations_finite": steps, "nonfinite_gradient_steps": 0,
             "missing_gradient_steps": 0, "unobserved_attempt_steps": 0,
             "gradient_coverage_complete": True, "status_verdict": "ok",
             "numerical_diagnostic_certified": True, "certification_reasons": [],
             "per_step": [{"step": k, "gradient_verdict": "observed_finite",
                           "base_calls_in_step": 1, "n_grad_tensors": 148,
                           "successful_update": True} for k in range(steps)]}
            for _ in range(world)],
    }


class TrainingRecordAuditCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ta = module_at("tac_audit", "tools/titan_audit.py")
        cls.dir = Path(tempfile.mkdtemp())

    def _verdict(self, rec):
        p = self.dir / "r.json"
        p.write_text(json.dumps(rec, default=str))
        return self.ta.audit_record(str(p))["verdict"]

    def test_an_unmodified_training_record_passes(self):
        self.assertEqual(self._verdict(training_record()), "PASS")

    def test_the_fixture_exercises_the_top_level_telemetry_path(self):
        """Guards the actual cause: if audit_record stopped inspecting top-level
        telemetry, this record would carry no telemetry findings at all."""
        p = self.dir / "shape.json"
        p.write_text(json.dumps(training_record(), default=str))
        groups = self.ta.audit_record(str(p))["groups"]
        self.assertIn("telemetry", groups)
        self.assertIn("energy", groups)
        self.assertIn("perf", groups)
        self.assertTrue(groups["telemetry"], "telemetry group must not be empty")

    def _mutations(self):
        return {
            "energy_inflated":
                lambda r: r["telemetry"].update(energy_total_j=r["telemetry"]["energy_total_j"] * 1.5),
            "throughput_inflated":
                lambda r: r["perf"].update(tokens_per_s=r["perf"]["tokens_per_s"] * 2),
            "tokens_per_step_wrong":
                lambda r: r["perf"].update(tokens_per_step=999),
            "interval_mislabelled":
                lambda r: r["telemetry"].update(sample_interval_s=0.1),
            "gap_cutoff_widened":
                lambda r: r["telemetry"].update(max_permitted_gap_s=5.0),
            "interval_flag_contradicts":
                lambda r: r["telemetry"].update(sample_interval_matches_frozen=False),
            "negative_power":
                lambda r: r["telemetry"]["power_mean_w"].update({"0": -20.0}),
            "util_above_100":
                lambda r: r["telemetry"]["util_mean_pct"].update({"0": 140.0}),
            "duration_zeroed":
                lambda r: r["telemetry"].update(duration_s=0.0),
            "integration_status_forged":
                lambda r: r["telemetry"]["integration"]["0"].update(
                    status="ok", max_support_gap_s=99.0),
            "integration_status_not_ok":
                lambda r: r["telemetry"]["integration"]["0"].update(status="big_gap"),
            "energy_parts_do_not_sum":
                lambda r: r["telemetry"]["energy_per_gpu_j"].update({"0": 1.0}),
            "sampling_errors_present":
                lambda r: r["telemetry"].update(sampling_errors=[{"e": "query failed"}]),
            "updates_exceed_attempts":
                lambda r: r["optimizer_update_audit_per_rank"][0].update(
                    native_successful_logical_updates=999),
            "missing_gradients_forged":
                lambda r: r["optimizer_update_audit_per_rank"][0].update(
                    missing_gradient_steps=7),
            "nonfinite_count_forged":
                lambda r: r["optimizer_update_audit_per_rank"][0].update(
                    nonfinite_gradient_steps=5),
            "base_calls_forged":
                lambda r: r["optimizer_update_audit_per_rank"][0].update(
                    base_optimizer_calls=1),
            "skipped_count_forged":
                lambda r: r["optimizer_update_audit_per_rank"][0].update(
                    native_skipped_steps=4),
            "per_gpu_throughput_inconsistent":
                lambda r: r["perf"].update(tokens_per_s_per_gpu=1.0),
            "per_step_emptied_while_claiming_native":
                lambda r: r["optimizer_update_audit_per_rank"][0].update(per_step=[]),
        }

    def test_every_mutation_is_detected(self):
        undetected = []
        for name, fn in self._mutations().items():
            rec = copy.deepcopy(training_record())
            fn(rec)
            if self._verdict(rec) != "FAIL":
                undetected.append(name)
        self.assertEqual(undetected, [], f"validator missed: {undetected}")


class EngineePathAuditTests(unittest.TestCase):
    """A ZeRO arm has no native per-step payload by construction. That must be
    accepted when declared and refused when the record claims native results."""

    @classmethod
    def setUpClass(cls):
        cls.ta = module_at("tac_audit2", "tools/titan_audit.py")
        cls.dir = Path(tempfile.mkdtemp())

    def _engine_record(self, **over):
        r = training_record()
        r["config"]["mode"] = "deepspeed"
        a = {
            "mode": "deepspeed", "observer_path": "engine_internal", "per_step": [],
            "deepspeed_observer_status": "engine counters only; NOT a native validation",
            "deepspeed_hardware_validated": False,
            "deepspeed_skip_path_exercised": False,
            "deepspeed_engine_aware_validated": True,
            "deepspeed_engine_aware_validation": {"validated": True, "zero_stage": 1,
                                                  "world_size": 2},
            "engine_update_boundaries": 30, "engine_skipped_updates": 0,
            "engine_successful_updates": 30, "expected_logical_updates": 30,
            "numerical_diagnostic_certified": None, "status_verdict": "ok",
        }
        a.update(over)
        r["optimizer_update_audit_per_rank"] = [dict(a), dict(a)]
        return r

    def _verdict(self, rec):
        p = self.dir / "e.json"
        p.write_text(json.dumps(rec, default=str))
        return self.ta.audit_record(str(p))["verdict"]

    def test_a_declared_engine_path_record_passes(self):
        self.assertEqual(self._verdict(self._engine_record()), "PASS")

    def test_an_undeclared_empty_per_step_fails(self):
        rec = self._engine_record()
        for a in rec["optimizer_update_audit_per_rank"]:
            a.pop("deepspeed_observer_status")
        self.assertEqual(self._verdict(rec), "FAIL")

    def test_claiming_native_certification_without_per_step_fails(self):
        self.assertEqual(
            self._verdict(self._engine_record(numerical_diagnostic_certified=True)),
            "FAIL", "a record cannot certify native numerics with no per-step data")

    def test_claiming_native_path_without_per_step_fails(self):
        self.assertEqual(self._verdict(self._engine_record(observer_path="native")),
                         "FAIL")

    def test_promoting_the_native_hardware_claim_fails(self):
        self.assertEqual(
            self._verdict(self._engine_record(deepspeed_hardware_validated=True)),
            "FAIL")

    def test_claiming_the_skip_path_was_exercised_fails(self):
        self.assertEqual(
            self._verdict(self._engine_record(deepspeed_skip_path_exercised=True)),
            "FAIL")

    def test_non_reconciling_engine_counters_fail(self):
        self.assertEqual(self._verdict(self._engine_record(engine_skipped_updates=3)),
                         "FAIL")

    def test_boundaries_disagreeing_with_expected_updates_fail(self):
        self.assertEqual(self._verdict(self._engine_record(engine_update_boundaries=12,
                                                           engine_successful_updates=12)),
                         "FAIL")

    def test_a_contradicted_engine_aware_flag_fails(self):
        self.assertEqual(
            self._verdict(self._engine_record(deepspeed_engine_aware_validated=False)),
            "FAIL")


if __name__ == "__main__":
    unittest.main()


class ChecksDisabledAuditTests(unittest.TestCase):
    """A performance run disables gradient checks on purpose, so every attempt
    is unobserved. Accepting that is correct. Accepting it *unconditionally* is
    not: the record must claim nothing it never looked at."""

    @classmethod
    def setUpClass(cls):
        cls.ta = module_at("tac_audit3", "tools/titan_audit.py")
        cls.dir = Path(tempfile.mkdtemp())

    def _perf_record(self, **over):
        """A record shaped like a real performance block run."""
        r = training_record()
        r["config"]["numerical_checks"] = False
        for a in r["optimizer_update_audit_per_rank"]:
            a.update({
                "gradient_checks_requested": False,
                "gradient_checks_enabled": False,
                "gradient_coverage_complete": None,
                "numerical_diagnostic_certified": None,
                "gradient_observations_finite": 0,
                "nonfinite_gradient_steps": 0,
                "missing_gradient_steps": 0,
                "unobserved_attempt_steps": len(a["per_step"]),
            })
            for st in a["per_step"]:
                st["gradient_verdict"] = "not_observed"
                st["n_grad_tensors"] = None
            a.update(over)
        return r

    def _verdict(self, rec):
        p = self.dir / "p.json"
        p.write_text(json.dumps(rec, default=str))
        return self.ta.audit_record(str(p))["verdict"]

    def test_a_performance_run_with_checks_off_passes(self):
        self.assertEqual(self._verdict(self._perf_record()), "PASS")

    def test_update_accounting_is_still_verified_with_checks_off(self):
        """Attempts, logical updates and base calls are observable without
        gradient checks, so forging them must still fail."""
        self.assertEqual(
            self._verdict(self._perf_record(native_successful_logical_updates=999)),
            "FAIL")
        self.assertEqual(self._verdict(self._perf_record(base_optimizer_calls=1)),
                         "FAIL")

    def test_claiming_coverage_with_checks_off_fails(self):
        self.assertEqual(self._verdict(self._perf_record(gradient_coverage_complete=True)),
                         "FAIL", "coverage cannot be claimed without observation")

    def test_claiming_certification_with_checks_off_fails(self):
        self.assertEqual(
            self._verdict(self._perf_record(numerical_diagnostic_certified=True)),
            "FAIL", "certification cannot be claimed without observation")

    def test_claiming_finite_observations_with_checks_off_fails(self):
        self.assertEqual(self._verdict(self._perf_record(gradient_observations_finite=30)),
                         "FAIL")

    def test_claiming_nonfinite_observations_with_checks_off_fails(self):
        self.assertEqual(self._verdict(self._perf_record(nonfinite_gradient_steps=3)),
                         "FAIL")

    def test_a_partial_unobserved_count_with_checks_off_fails(self):
        """Checks off means ALL attempts unobserved. A partial count means the
        record and its per-step payload disagree."""
        self.assertEqual(self._verdict(self._perf_record(unobserved_attempt_steps=5)),
                         "FAIL")

    def test_checks_requested_but_nothing_observed_still_fails(self):
        """The original rule must survive: if the observer WAS asked and
        produced nothing, that is a real evidence gap."""
        rec = self._perf_record(gradient_checks_requested=True)
        self.assertEqual(self._verdict(rec), "FAIL")

    def test_telemetry_is_still_audited_with_checks_off(self):
        rec = self._perf_record()
        rec["telemetry"]["sample_interval_s"] = 0.1
        self.assertEqual(self._verdict(rec), "FAIL")
