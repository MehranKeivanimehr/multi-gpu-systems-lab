"""Engine-aware DeepSpeed evidence must be bound, recomputed and refusable.

Counterexamples the delivered gate accepted (reproduced in
out/repro/item4_ds_admission_FIRST_FAILING.*):
  * world_size = zero_stage = 999, fabricated version, NO raw steps
  * every required flag present as the STRING "false"
No device access anywhere in this file.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


def _good_steps(n=8):
    return [{"step": i,
             "grad_evidence": {"verdict": "observed_finite",
                               "evidence_sources_with_data": ["averaged_gradients"]},
             "master_partition_changed": True,
             "base_state_advanced": True,
             "counters": {"global_grad_norm": {"value": 1.5, "available": True},
                          "overflow": {"value": False, "available": True}}}
            for i in range(n)]


def _artifact(**over):
    checks = ("gradients_observed_finite_every_step",
              "master_partition_changed_every_step",
              "base_optimizer_state_advanced_every_step",
              "engine_global_steps_reconcile",
              "every_rank_drove_the_same_step_count")
    a = {
        "zero_stage": 1, "world_size": 2, "steps": 8,
        "deepspeed_version": "0.15.1", "fault_injection": "none",
        "agreed_checks": {k: True for k in checks},
        "all_agreed_checks_passed": True,
        "per_step_rank0": _good_steps(8),
        "per_rank": [{"rank": r, "local_checks": {k: True for k in checks}}
                     for r in range(2)],
        "evidence_binding": {
            "source_sha256": "a" * 64, "model_param_count": 124_439_808,
            "model_n_positions": 1024, "precision": "fp32", "optimizer": "AdamW",
            "gradient_accumulation_steps": 1, "zero_stage": 1, "world_size": 2,
            "gpu_family": "p1080", "initial_state_sha256": "b" * 64,
            "input_policy": "declared",
        },
    }
    a.update(over)
    return a


COHORT = {"zero_stage": 1, "world_size": 2, "gpu_family": "p1080",
          "precision": "fp32", "model_param_count": 124_439_808}


class DsAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dse = load("dse_t", "tools/ds_evidence.py")
        cls.dir = pathlib.Path(tempfile.mkdtemp())
        cls._n = 0

    def admit(self, art, cohort=None):
        type(self)._n += 1
        p = self.dir / f"a{self._n}.json"
        p.write_text(json.dumps(art, default=str))
        return self.dse.admit(p, cohort or COHORT)

    # -- the only accepting case -------------------------------------------
    def test_a_fully_bound_recomputable_artifact_is_admitted(self):
        r = self.admit(_artifact())
        self.assertTrue(r["admitted"], r["reasons"])

    # -- the two counterexamples from the review ---------------------------
    def test_world_size_and_stage_999_with_no_raw_steps_is_refused(self):
        bogus = {"zero_stage": 999, "world_size": 999,
                 "deepspeed_version": "0.0.0-not-real",
                 "fault_injection": "none",
                 "agreed_checks": {k: True for k in
                                   _artifact()["agreed_checks"]},
                 "all_agreed_checks_passed": True}
        r = self.admit(bogus)
        self.assertFalse(r["admitted"])
        self.assertTrue(any("no raw steps" in x or "raw" in x for x in r["reasons"]),
                        r["reasons"])

    def test_flags_as_the_string_false_are_refused(self):
        art = _artifact(agreed_checks={k: "false" for k in
                                       _artifact()["agreed_checks"]})
        r = self.admit(art)
        self.assertFalse(r["admitted"])
        self.assertFalse(r["raw_checks"]["agreed_checks_are_real_booleans"]["ok"])

    def test_flags_as_the_string_true_are_also_refused(self):
        art = _artifact(agreed_checks={k: "true" for k in
                                       _artifact()["agreed_checks"]})
        self.assertFalse(self.admit(art)["admitted"])

    # -- binding ------------------------------------------------------------
    def test_every_binding_axis_is_required(self):
        for axis in self.dse.BINDING_AXES:
            art = _artifact()
            art["evidence_binding"].pop(axis, None)
            r = self.admit(art)
            if axis in ("zero_stage", "world_size"):
                continue      # these are recoverable from top-level fields
            self.assertFalse(r["admitted"], f"{axis} missing must refuse")
            self.assertIn(axis, r["unbound_axes"])

    def test_precision_mismatch_is_refused(self):
        art = _artifact()
        art["evidence_binding"]["precision"] = "fp32"
        r = self.admit(art, dict(COHORT, precision="amp_fp16"))
        self.assertFalse(r["admitted"])
        self.assertTrue(any("precision" in x for x in r["reasons"]))

    def test_model_parameter_count_mismatch_is_refused(self):
        art = _artifact()
        art["evidence_binding"]["model_param_count"] = 124_046_592
        r = self.admit(art)
        self.assertFalse(r["admitted"])
        self.assertTrue(any("model_param_count" in x for x in r["reasons"]))

    def test_world_size_mismatch_is_refused(self):
        r = self.admit(_artifact(), dict(COHORT, world_size=6))
        self.assertFalse(r["admitted"])

    # -- raw recomputation --------------------------------------------------
    def test_a_single_nonfinite_gradient_step_refuses(self):
        art = _artifact()
        art["per_step_rank0"][3]["grad_evidence"]["verdict"] = "observed_nonfinite"
        self.assertFalse(self.admit(art)["admitted"])

    def test_a_single_unchanged_master_partition_refuses(self):
        art = _artifact()
        art["per_step_rank0"][5]["master_partition_changed"] = False
        self.assertFalse(self.admit(art)["admitted"])

    def test_a_single_unadvanced_optimizer_state_refuses(self):
        art = _artifact()
        art["per_step_rank0"][0]["base_state_advanced"] = False
        self.assertFalse(self.admit(art)["admitted"])

    def test_a_nonfinite_grad_norm_refuses(self):
        for v in (float("inf"), float("nan"), 0.0, -1.0, "1.5"):
            art = _artifact()
            art["per_step_rank0"][2]["counters"]["global_grad_norm"]["value"] = v
            self.assertFalse(self.admit(art)["admitted"], f"value {v!r} must refuse")

    def test_overflow_true_or_nonboolean_refuses(self):
        for v in (True, "False", None):
            art = _artifact()
            art["per_step_rank0"][1]["counters"]["overflow"]["value"] = v
            self.assertFalse(self.admit(art)["admitted"], f"overflow {v!r} must refuse")

    def test_step_count_disagreeing_with_raw_refuses(self):
        self.assertFalse(self.admit(_artifact(steps=99))["admitted"])

    def test_a_rank_failing_locally_refuses(self):
        art = _artifact()
        art["per_rank"][1]["local_checks"]["engine_global_steps_reconcile"] = False
        self.assertFalse(self.admit(art)["admitted"])

    def test_per_rank_count_must_match_world_size(self):
        art = _artifact()
        art["per_rank"] = art["per_rank"][:1]
        self.assertFalse(self.admit(art)["admitted"])

    def test_fault_injection_refuses(self):
        self.assertFalse(self.admit(_artifact(fault_injection="grad_nan"))["admitted"])

    def test_overall_flag_is_never_trusted_over_raw(self):
        """all_agreed_checks_passed=True must not rescue a failing raw step."""
        art = _artifact(all_agreed_checks_passed=True)
        art["per_step_rank0"][4]["base_state_advanced"] = False
        self.assertFalse(self.admit(art)["admitted"])

    def test_limits_are_always_reported(self):
        r = self.admit(_artifact())
        joined = " ".join(r["limits_of_this_evidence"]).lower()
        self.assertIn("not gradient coverage", joined)
        self.assertIn("skip path", joined)


class SourceBindingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dse = load("dse_t2", "tools/ds_evidence.py")

    def test_param_count_formula_matches_the_recorded_training_value(self):
        self.assertEqual(self.dse.gpt2_param_count(1024), 124_439_808)
        self.assertEqual(self.dse.gpt2_param_count(512), 124_046_592)

    def test_source_derived_mismatch_is_reported_as_demonstrable(self):
        d = pathlib.Path(tempfile.mkdtemp())
        src = d / "diag.py"
        src.write_text('cfg = GPT2Config(n_positions=512)\n'
                       'ds_cfg["fp16"] = {"enabled": False}\n'
                       'ds_cfg["bf16"] = {"enabled": False}\n'
                       'opt = torch.optim.AdamW(p)\n')
        art = d / "a.json"
        art.write_text(json.dumps(_artifact()))
        r = self.dse.admit_with_source(art, COHORT, src)
        self.assertFalse(r["admitted"])
        axes = [a for a, _, _ in r["demonstrable_mismatches"]]
        self.assertIn("model_param_count", axes)
        self.assertEqual(r["source_binding"]["precision"], "fp32")

    def test_amp_cohort_is_a_demonstrable_precision_mismatch(self):
        d = pathlib.Path(tempfile.mkdtemp())
        src = d / "diag.py"
        src.write_text('GPT2Config(n_positions=1024)\n'
                       'ds_cfg["fp16"] = {"enabled": False}\n'
                       'ds_cfg["bf16"] = {"enabled": False}\n')
        art = d / "a.json"
        art.write_text(json.dumps(_artifact()))
        r = self.dse.admit_with_source(art, dict(COHORT, precision="amp_fp16"), src)
        self.assertFalse(r["admitted"])
        self.assertIn("precision", [a for a, _, _ in r["demonstrable_mismatches"]])


if __name__ == "__main__":
    unittest.main()
