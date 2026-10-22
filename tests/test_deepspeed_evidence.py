"""The DeepSpeed engine-aware evidence gate must never default to validated.

ZeRO performance arms are eligible only once engine-aware evidence passes, so
this gate is load-bearing: if it could be satisfied by a missing, truncated or
failing artifact, the arms would be promoted on no evidence at all.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def module_at(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class DeepSpeedEvidenceGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rr = module_at("dse_run_report", "studies/titan-1080/bench/run_report.py")

    def _artifact(self, **over):
        ev = {
            "agreed_checks": {k: True for k in self.rr.DS_REQUIRED_CHECKS},
            "fault_injection": "none", "zero_stage": 1, "world_size": 2,
            "deepspeed_version": "0.15.1",
        }
        ev.update(over)
        d = Path(tempfile.mkdtemp())
        p = d / "ev.json"
        p.write_text(json.dumps(ev))
        return str(p)

    # -- the only accepting case ---------------------------------------------
    def test_a_complete_passing_artifact_validates(self):
        r = self.rr.deepspeed_engine_aware_evidence(self._artifact())
        self.assertTrue(r["validated"])
        self.assertIsNone(r["reason"])
        self.assertEqual(r["zero_stage"], 1)
        self.assertEqual(r["world_size"], 2)

    # -- refusals ------------------------------------------------------------
    def test_no_artifact_does_not_validate(self):
        r = self.rr.deepspeed_engine_aware_evidence("")
        self.assertFalse(r["validated"])
        self.assertIn("no engine-aware observation artifact", r["reason"])

    def test_a_missing_file_does_not_validate(self):
        r = self.rr.deepspeed_engine_aware_evidence("/nonexistent/ev.json")
        self.assertFalse(r["validated"])
        self.assertIn("unreadable", r["reason"])

    def test_malformed_json_does_not_validate(self):
        d = Path(tempfile.mkdtemp()); p = d / "bad.json"; p.write_text("{not json")
        r = self.rr.deepspeed_engine_aware_evidence(str(p))
        self.assertFalse(r["validated"])
        self.assertIn("unreadable", r["reason"])

    def test_an_artifact_without_agreed_checks_does_not_validate(self):
        r = self.rr.deepspeed_engine_aware_evidence(self._artifact(agreed_checks=None))
        self.assertFalse(r["validated"])
        self.assertIn("no agreed_checks", r["reason"])

    def test_an_artifact_omitting_one_required_check_does_not_validate(self):
        """A truncated artifact must fail, not pass on the checks it happens to
        carry. This is what makes adding a check invalidate older artifacts."""
        for drop in self.rr.DS_REQUIRED_CHECKS:
            checks = {k: True for k in self.rr.DS_REQUIRED_CHECKS if k != drop}
            r = self.rr.deepspeed_engine_aware_evidence(self._artifact(agreed_checks=checks))
            self.assertFalse(r["validated"], f"omitting {drop} must not validate")
            self.assertIn(drop, r["reason"])

    def test_any_single_failing_check_does_not_validate(self):
        for fail in self.rr.DS_REQUIRED_CHECKS:
            checks = {k: (k != fail) for k in self.rr.DS_REQUIRED_CHECKS}
            r = self.rr.deepspeed_engine_aware_evidence(self._artifact(agreed_checks=checks))
            self.assertFalse(r["validated"], f"{fail}=False must not validate")
            self.assertIn(fail, r["reason"])

    def test_an_artifact_reporting_fault_injection_does_not_validate(self):
        r = self.rr.deepspeed_engine_aware_evidence(self._artifact(fault_injection="grad_nan"))
        self.assertFalse(r["validated"])
        self.assertIn("fault injection", r["reason"])

    # -- the native claim is never promoted ----------------------------------
    def test_native_validation_stays_false_even_with_passing_evidence(self):
        rec = self.rr.build_update_record(
            mode="deepspeed", native_summary={}, engine_counters={},
            losses=[1.0], expected_logical_updates=8, numerical_checks=True)
        self.assertIs(rec["deepspeed_hardware_validated"], False)
        self.assertIs(rec["deepspeed_skip_path_exercised"], False)
        self.assertIn("NOT a native hardware validation", rec["deepspeed_observer_status"])

    def test_with_no_artifact_in_the_environment_nothing_is_validated(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(self.rr.DS_EVIDENCE_ENV, None)
            rec = self.rr.build_update_record(
                mode="deepspeed", native_summary={}, engine_counters={},
                losses=[1.0], expected_logical_updates=8, numerical_checks=True)
        self.assertIn("deepspeed_engine_aware_validation", rec)
        self.assertFalse(rec["deepspeed_engine_aware_validated"])
        self.assertFalse(rec["deepspeed_hardware_validated"])

    def test_passing_evidence_separates_the_two_fields(self):
        """The whole point of the split: engine-aware evidence can be true while
        the native claim stays false, and a reader can tell which is which."""
        with mock.patch.dict(os.environ, {self.rr.DS_EVIDENCE_ENV: self._artifact()}):
            rec = self.rr.build_update_record(
                mode="deepspeed", native_summary={}, engine_counters={},
                losses=[1.0], expected_logical_updates=8, numerical_checks=True)
        self.assertTrue(rec["deepspeed_engine_aware_validated"])
        self.assertFalse(rec["deepspeed_hardware_validated"],
                         "engine-aware evidence must never promote the native claim")
        self.assertIs(rec["deepspeed_skip_path_exercised"], False)

    def test_non_deepspeed_modes_carry_no_deepspeed_fields(self):
        with mock.patch.dict(os.environ, {self.rr.DS_EVIDENCE_ENV: self._artifact()}):
            rec = self.rr.build_update_record(
                mode="ddp", native_summary={}, engine_counters=None,
                losses=[1.0], expected_logical_updates=8, numerical_checks=True)
        for k in ("deepspeed_engine_aware_validated", "deepspeed_hardware_validated",
                  "deepspeed_engine_aware_validation"):
            self.assertNotIn(k, rec, f"{k} must not appear on a non-engine mode")


if __name__ == "__main__":
    unittest.main()
