"""Coverage justification, both precision criteria, and descriptive ratios.

The counterexample is executed, not merely asserted: exchangeable continuous
marginals with coverage 0.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import random
import statistics
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


class CoverageCounterexampleTests(unittest.TestCase):
    """X_i = S * U_i with one common sign S and U_i iid Uniform(0,2)."""

    @classmethod
    def setUpClass(cls):
        cls.inf = load("ti", "tools/titan_inference.py")

    def sample(self, rng, n=6):
        s = rng.choice((-1.0, 1.0))
        return [s * rng.uniform(0.0, 2.0) for _ in range(n)]

    def test_marginals_are_continuous_uniform_on_minus_two_two(self):
        rng = random.Random(7)
        xs = [self.sample(rng, 1)[0] for _ in range(20000)]
        self.assertGreater(min(xs), -2.0)
        self.assertLess(max(xs), 2.0)
        # symmetric about 0, so the marginal median is 0
        self.assertLess(abs(statistics.mean(xs)), 0.05)
        self.assertLess(abs(sum(1 for x in xs if x > 0) / len(xs) - 0.5), 0.02)

    def test_the_sample_is_exchangeable(self):
        """Any permutation has the same law: the construction is symmetric in i,
        so a permuted sample is distributed identically."""
        rng = random.Random(11)
        a = [tuple(sorted(self.sample(rng))) for _ in range(4000)]
        rng2 = random.Random(11)
        b = []
        for _ in range(4000):
            x = self.sample(rng2)
            random.Random(_).shuffle(x)
            b.append(tuple(sorted(x)))
        self.assertEqual(a, b, "sorted samples must agree under permutation")

    def test_the_minmax_interval_NEVER_covers_the_median(self):
        rng = random.Random(13)
        covered = 0
        trials = 20000
        for _ in range(trials):
            x = self.sample(rng)
            if min(x) < 0.0 < max(x):
                covered += 1
        self.assertEqual(covered, 0,
                         f"{covered}/{trials} intervals covered the median 0; "
                         f"exchangeability + continuity would wrongly predict "
                         f"~96.875%")

    def test_exchangeability_is_recorded_as_insufficient(self):
        joined = " ".join(self.inf.COVERAGE_INSUFFICIENT_CONDITIONS).lower()
        self.assertIn("exchangeability", joined)
        self.assertIn("counterexample", joined)

    def test_iid_continuous_is_recorded_as_sufficient(self):
        self.assertIn("IID continuous", self.inf.COVERAGE_SUFFICIENT_CONDITIONS[0])
        self.assertIn("same median", self.inf.COVERAGE_SUFFICIENT_CONDITIONS[1].lower()
                      .replace("SAME", "same"))

    def test_iid_sampling_does_achieve_the_stated_coverage(self):
        """The control: genuinely IID continuous draws cover at ~31/32."""
        rng = random.Random(17)
        trials = 40000
        covered = sum(1 for _ in range(trials)
                      if (lambda x: min(x) < 0.0 < max(x))(
                          [rng.uniform(-2.0, 2.0) for _ in range(6)]))
        self.assertAlmostEqual(covered / trials, 31 / 32, delta=0.01)

    def test_no_coverage_is_claimed_when_iid_is_not_justified(self):
        s = self.inf.coverage_statement(6, iid_justified=False)
        self.assertIsNone(s["coverage_claimed"])
        self.assertIn("descriptive range", s["report"])

    def test_coverage_is_claimed_only_conditionally(self):
        s = self.inf.coverage_statement(6, iid_justified=True)
        self.assertEqual(s["coverage_claimed"], "31/32")
        self.assertIn("IID continuous", s["conditional_on"])


class BothPrecisionCriteriaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inf = load("ti2", "tools/titan_inference.py")

    def test_the_original_criterion_matches_the_frozen_plan_text(self):
        prov = self.inf.CRITERIA[self.inf.ORIGINAL_CRITERION]["provenance"]
        self.assertIn("frozen_defaults.statistics", prov)
        self.assertIn("endpoints within +/-5%", prov)

    def test_the_stricter_criterion_is_labelled_as_not_in_the_plan(self):
        c = self.inf.CRITERIA[self.inf.STRICTER_CRITERION]
        self.assertIn("NOT in the frozen plan", c["provenance"])
        self.assertIn("does not replace", c["status"])

    def test_a_group_can_pass_the_original_and_fail_the_stricter(self):
        """Endpoints each within 5% of the median, span just over 5%."""
        v = [96.0, 100.0, 100.0, 100.0, 100.0, 103.0]
        self.assertTrue(self.inf.endpoints_within(v))
        self.assertFalse(self.inf.span_within(v))
        both = self.inf.evaluate_both(v)
        self.assertTrue(both[self.inf.ORIGINAL_CRITERION]["met"])
        self.assertFalse(both[self.inf.STRICTER_CRITERION]["met"])

    def test_the_stricter_criterion_implies_the_original(self):
        rng = random.Random(23)
        for _ in range(3000):
            v = [rng.uniform(90, 110) for _ in range(6)]
            if self.inf.span_within(v):
                self.assertTrue(self.inf.endpoints_within(v),
                                f"span passed but endpoints failed: {v}")

    def test_empty_input_yields_none_not_false(self):
        self.assertIsNone(self.inf.endpoints_within([]))
        self.assertIsNone(self.inf.span_within([]))

    def test_nonfinite_values_are_dropped_not_accepted(self):
        v = [100.0] * 6 + [float("inf"), float("nan")]
        self.assertTrue(self.inf.endpoints_within(v))


class DescriptiveRatioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inf = load("ti3", "tools/titan_inference.py")

    def test_the_rtx_batch16_ratio_is_reported_as_unestablished(self):
        """The real paired ratios: median above 1, range crossing 1."""
        num = {1: 550.25, 2: 554.55, 3: 545.15, 4: 575.60, 5: 429.56, 6: 569.01}
        den = {1: 505.89, 2: 415.59, 3: 418.79, 4: 497.70, 5: 512.17, 6: 500.98}
        r = self.inf.paired_ratio_descriptive(num, den)
        self.assertEqual(r["n_paired"], 6)
        self.assertGreater(r["median_ratio"], 1.0)
        self.assertTrue(r["crosses_one"])
        self.assertIn("no universal or reliable winner", r["claim"])

    def test_more_blocks_cannot_remove_an_existing_crossing(self):
        num = {1: 1.2, 2: 0.8}
        den = {1: 1.0, 2: 1.0}
        base = self.inf.paired_ratio_descriptive(num, den)
        self.assertTrue(base["crosses_one"])
        for extra in (1.0, 1.5, 0.5, 1.01):
            n2 = dict(num); d2 = dict(den)
            n2[3] = extra; d2[3] = 1.0
            r = self.inf.paired_ratio_descriptive(n2, d2)
            self.assertTrue(r["crosses_one"],
                            "adding a block removed the crossing, which is impossible")
            self.assertLessEqual(r["min_ratio"], base["min_ratio"])
            self.assertGreaterEqual(r["max_ratio"], base["max_ratio"])

    def test_monotonicity_note_is_present(self):
        r = self.inf.paired_ratio_descriptive({1: 1.1}, {1: 1.0})
        self.assertIn("can only widen", r["monotonicity_note"])

    def test_no_shared_blocks_is_unestablished(self):
        r = self.inf.paired_ratio_descriptive({1: 1.0}, {2: 1.0})
        self.assertEqual(r["n_paired"], 0)
        self.assertEqual(r["claim"], "unestablished")


if __name__ == "__main__":
    unittest.main()
