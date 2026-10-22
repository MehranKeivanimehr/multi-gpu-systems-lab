"""The comparator must refuse empty, non-finite, mismatched and unidentified evidence.

Reproduced in out/repro/item5c_comparator_detail.*: the delivered comparator
returned PASS for two EMPTY loss arrays when the rest of the record was
well-formed, and accepted `inf` as finite. No device access in this file.
"""
from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import shutil
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPEC = {
    "frozen_before_any_comparison_run": True,
    "comparison_design": {"input_policy": "CPU generator"},
    "tolerances": {"step0_loss_abs": 0.05, "final_loss_abs": 0.5,
                   "loss_trajectory_max_abs": 0.5, "all_losses_finite": True,
                   "relative_throughput_bound": None},
}


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


def record(device="A", group="g1", losses=None, drop=()):
    cfg = {"seed": 1, "model": "gpt2-124m", "seq_len": 512,
           "micro_batch_per_gpu": 4, "steps": 30, "warmup": 5,
           "precision": "fp32", "memory_fraction": 0.85, "world_size": 1,
           "mode": "single", "group": group,
           "runtime_device_identity_after_memory_cap": {"device_name": device}}
    for k in drop:
        cfg.pop(k, None)
    return {"status": "ok", "config": cfg,
            "losses": [10.0 + i * 0.001 for i in range(30)] if losses is None else losses,
            "perf": {"tokens_per_s": 100.0}}


class ComparatorStrictTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = load("crs", "tools/compare_reference_strict.py")
        cls.root = pathlib.Path(tempfile.mkdtemp())
        cls.sha = "deadbeef" * 8

    def write(self, case, rec):
        d = self.root / "runs" / case / "results"
        d.mkdir(parents=True, exist_ok=True)
        (d / "r.json").write_text(json.dumps(rec, default=str))

    def run_pair(self, a_rec, b_rec, tag):
        ca, cb = f"A_{tag}", f"B_{tag}"
        self.write(ca, a_rec); self.write(cb, b_rec)
        return self.m.compare(self.root, ca, cb, SPEC, self.sha)

    # -- the accepting case is QUALIFIED, never an unqualified PASS ----------
    def test_a_wellformed_pair_is_qualified_by_input_provenance(self):
        r = self.run_pair(record("A", "g1"), record("B", "g2"), "ok")
        self.assertEqual(r["verdict"], "QUALIFIED")
        names = [k for k, _ in r["limitations"]]
        self.assertIn("input_policy_matches_the_frozen_spec", names)
        self.assertIn("realised_input_digest_retained", names)

    # -- the reproduced counterexample --------------------------------------
    def test_two_empty_loss_arrays_are_refused(self):
        r = self.run_pair(record("A", "g1", losses=[]),
                          record("B", "g2", losses=[]), "empty")
        self.assertEqual(r["verdict"], "FAIL")
        self.assertTrue(any(k.startswith("losses_nonempty") for k, _ in r["failures"]))
        self.assertTrue(any(k == "tolerances_evaluated" for k, _ in r["failures"]),
                        "tolerances must be explicitly marked not evaluated")

    def test_one_empty_loss_array_is_refused(self):
        r = self.run_pair(record("A", "g1"), record("B", "g2", losses=[]), "one_empty")
        self.assertEqual(r["verdict"], "FAIL")

    def test_a_single_value_array_is_refused(self):
        r = self.run_pair(record("A", "g1", losses=[1.0]),
                          record("B", "g2", losses=[1.0]), "single")
        self.assertEqual(r["verdict"], "FAIL")

    def test_inf_is_not_finite(self):
        r = self.run_pair(record("A", "g1"),
                          record("B", "g2", losses=[float("inf")] * 30), "inf")
        self.assertEqual(r["verdict"], "FAIL")
        self.assertTrue(any(k.startswith("losses_all_finite") for k, _ in r["failures"]))

    def test_nan_is_not_finite(self):
        r = self.run_pair(record("A", "g1"),
                          record("B", "g2", losses=[float("nan")] * 30), "nan")
        self.assertEqual(r["verdict"], "FAIL")

    def test_non_numeric_losses_are_refused(self):
        r = self.run_pair(record("A", "g1"),
                          record("B", "g2", losses=["x"] * 30), "str")
        self.assertEqual(r["verdict"], "FAIL")

    def test_mismatched_lengths_do_not_evaluate_a_truncated_prefix(self):
        r = self.run_pair(record("A", "g1"),
                          record("B", "g2", losses=[10.0] * 7), "len")
        self.assertEqual(r["verdict"], "FAIL")
        self.assertTrue(any(k == "tolerances_evaluated" for k, _ in r["failures"]))
        self.assertNotIn("trajectory_within_tolerance", r["checks"])

    def test_missing_config_fields_are_named_as_missing(self):
        r = self.run_pair(record("A", "g1"),
                          record("B", "g2", drop=("seed", "steps")), "miss")
        self.assertEqual(r["verdict"], "FAIL")
        detail = dict(r["failures"]).get("required_config_fields_present", "")
        self.assertIn("seed", detail)

    def test_identical_hardware_is_refused(self):
        r = self.run_pair(record("A", "g1"), record("A", "g1"), "same")
        self.assertEqual(r["verdict"], "FAIL")
        self.assertTrue(any(k == "hardware_actually_differs" for k, _ in r["failures"]))

    def test_absent_device_names_are_refused(self):
        a = record("A", "g1"); b = record("B", "g2")
        a["config"]["runtime_device_identity_after_memory_cap"] = {}
        r = self.run_pair(a, b, "nodev")
        self.assertEqual(r["verdict"], "FAIL")

    def test_a_missing_record_is_invalid(self):
        self.write("ONLY_A", record("A", "g1"))
        r = self.m.compare(self.root, "ONLY_A", "DOES_NOT_EXIST", SPEC, self.sha)
        self.assertEqual(r["verdict"], "INVALID")

    def test_a_malformed_record_is_invalid(self):
        d = self.root / "runs" / "BROKEN" / "results"; d.mkdir(parents=True, exist_ok=True)
        (d / "r.json").write_text("{not json")
        self.write("GOOD", record("A", "g1"))
        r = self.m.compare(self.root, "GOOD", "BROKEN", SPEC, self.sha)
        self.assertEqual(r["verdict"], "INVALID")

    # -- claims the comparator must always state ---------------------------
    def test_warmup_caveat_is_always_reported(self):
        r = self.run_pair(record("A", "g1"), record("B", "g2"), "warm")
        self.assertIn("first_step_is_post_warmup_not_untrained", r["checks"])
        self.assertIn("warmup updates",
                      r["checks"]["first_step_is_post_warmup_not_untrained"]["detail"])

    def test_preregistration_is_explicitly_not_claimed(self):
        r = self.run_pair(record("A", "g1"), record("B", "g2"), "prereg")
        d = r["checks"]["preregistration_not_claimed"]["detail"]
        self.assertIn("not cited as preregistration", d)

    def test_identity_is_declared_uncertified(self):
        r = self.run_pair(record("A", "g1"), record("B", "g2"), "ident")
        self.assertIn("device_identity_is_not_certified", r["checks"])

    def test_throughput_has_no_declared_tolerance(self):
        r = self.run_pair(record("A", "g1"), record("B", "g2"), "thr")
        self.assertIsNone(r["throughput_tolerance"])


if __name__ == "__main__":
    unittest.main()


class BothSidedOmissionTests(ComparatorStrictTests):
    """The immutable comparator passed when a required MATCH_KEYS field was
    absent from BOTH configs, because ca.get(k) == cb.get(k) == None. All 10
    required fields behaved this way, not only `model`. One-sided omission and
    identical hardware were already refused there; those narrower refusals do
    not establish missing-field rejection, so they are kept as separate cases.
    """

    def test_both_sided_omission_is_rejected_for_every_required_field(self):
        for k in self.m.REQUIRED_CONFIG:
            with self.subTest(field=k):
                r = self.run_pair(record("A", "g1", drop=(k,)),
                                  record("B", "g2", drop=(k,)), f"both_{k}")
                self.assertEqual(r["verdict"], "FAIL",
                                 f"both-sided omission of {k!r} must be refused")
                detail = dict(r["failures"]).get("required_config_fields_present", "")
                self.assertIn(k, detail, f"the refusal must name {k!r}")

    def test_one_sided_omission_is_still_rejected(self):
        r = self.run_pair(record("A", "g1", drop=("model",)),
                          record("B", "g2"), "one_sided_model")
        self.assertEqual(r["verdict"], "FAIL")

    def test_identical_hardware_remains_a_distinct_refusal(self):
        r = self.run_pair(record("A", "g1"), record("A", "g1"), "same_hw_distinct")
        self.assertEqual(r["verdict"], "FAIL")
        names = [k for k, _ in r["failures"]]
        self.assertIn("hardware_actually_differs", names)
        self.assertNotIn("required_config_fields_present", names,
                         "identical hardware must not be conflated with a "
                         "missing-field refusal")

    def test_inf_fails_the_finite_predicate_itself_not_only_the_tolerances(self):
        """The delivered finite predicate accepted inf and the overall verdict
        was refused by other checks. Here the predicate itself must fail."""
        r = self.run_pair(record("A", "g1"),
                          record("B", "g2", losses=[float("inf")] * 30), "inf_pred")
        self.assertEqual(r["verdict"], "FAIL")
        names = [k for k, _ in r["failures"]]
        self.assertTrue(any(n.startswith("losses_all_finite") for n in names),
                        f"the finiteness check must itself fail: {names}")
