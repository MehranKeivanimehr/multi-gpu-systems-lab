"""Analytical proofs and adversarial boundary tests; these do not run CUDA."""
import json
import math
import statistics
import unittest
from decimal import Decimal, localcontext
from fractions import Fraction
from unittest.mock import patch

from tools import scientific_corrections
from tools.scientific_metrics import cnn_compute, decode_proxy, dispersion, integrate_power, resnet50_conv_linear_macs
from tools.scientific_metrics import _sqrt_fraction_binary64


class ScientificMetricTests(unittest.TestCase):
    def test_resnet_architecture_count(self):
        self.assertEqual(resnet50_conv_linear_macs(), 4_089_184_256)
        self.assertEqual(2 * resnet50_conv_linear_macs(), 8_178_368_512)
        self.assertLess(abs(8.2e9 / (2 * resnet50_conv_linear_macs()) - 1), 0.003)

    def test_cnn_fma_convention(self):
        result = cnn_compute(1000, "fp32", 20)
        self.assertAlmostEqual(result["analytical_tflops_fma2"], 8.2)
        self.assertAlmostEqual(result["fp32_peak_fraction_pct"], 41)

    def test_amp_has_no_fp32_peak_percentage(self):
        self.assertIsNone(cnn_compute(2000, "amp_fp16", 20)["fp32_peak_fraction_pct"])

    def test_constant_power_exact_boundaries(self):
        result = integrate_power([(0, 100), (1, 100), (2, 100)], 0.25, 1.75)
        self.assertEqual(result["status"], "ok")
        self.assertAlmostEqual(result["energy_j"], 150)

    def test_linear_power_analytic_integral(self):
        result = integrate_power([(0, 10), (1, 30), (2, 50)], 0.25, 1.75)
        self.assertAlmostEqual(result["energy_j"], 45)

    def test_boundary_sample_does_not_require_extrapolation(self):
        result = integrate_power([(0, 10), (1, 20)], 0, 1)
        self.assertAlmostEqual(result["energy_j"], 15)

    def test_uncovered_window_is_not_shortened(self):
        result = integrate_power([(0.2, 100), (0.8, 100)], 0, 1)
        self.assertEqual(result["status"], "unbracketed_window")
        self.assertIsNone(result["energy_j"])

    def test_missing_reading_is_not_silently_zero(self):
        result = integrate_power([(0, 100), (0.5, None), (1, 100)], 0, 1)
        self.assertEqual(result["status"], "missing_or_invalid_power")

    def test_duplicates_and_invalid_intervals(self):
        self.assertEqual(integrate_power([(0, 1), (0, 2), (1, 2)], 0, 1)["status"], "duplicate_timestamps")
        self.assertEqual(integrate_power([(0, 1), (1, 2)], 1, 0)["status"], "invalid_window")

    def test_decode_excludes_first_token_per_request(self):
        self.assertEqual(decode_proxy(512, 4, 10, 2), 508 / 8)
        self.assertIsNone(decode_proxy(4, 4, 10, 2))
        self.assertIsNone(decode_proxy(512, 4, 2, 2))

    def test_dispersion_exposes_range_not_only_cv(self):
        result = dispersion([99.4, 100, 100.6])
        self.assertLess(result["sample_cv_pct"], 0.7)
        self.assertGreater(result["full_range_over_median_pct"], 1)
        self.assertIsNone(dispersion([100])["sample_cv_pct"])

    def test_corrections_are_reproducible(self):
        for relative, content in scientific_corrections.artifacts().items():
            self.assertEqual((scientific_corrections.ROOT / relative).read_bytes(), content.encode("utf-8"), relative)

    def test_cluster_reported_one_ulp_regression(self):
        values = [4067.8526683780683, 4071.2049695865194, 4071.239699282067]
        actual = dispersion(values)["sample_cv_pct"]
        self.assertEqual(actual, 0.047801172563880445)
        mean = sum(map(Fraction, values)) / 3
        variance = sum((Fraction(value) - mean) ** 2 for value in values) / 2
        # Reproduce the 3.10 two-rounding path on any interpreter, independently
        # of which statistics implementation happens to be installed.
        legacy = 100 * math.sqrt(float(variance)) / statistics.fmean(values)
        self.assertEqual(legacy, 0.04780117256388044)
        self.assertEqual(math.nextafter(legacy, math.inf), actual)

    def test_dispersion_does_not_depend_on_stdlib_stdev(self):
        with patch.object(statistics, "stdev", side_effect=AssertionError("version-dependent path")):
            self.assertEqual(dispersion([1.0, 2.0, 3.0])["sample_cv_pct"], 50.0)

    def test_rational_sqrt_midpoints_round_ties_to_even(self):
        for lower in (1.0, math.nextafter(1.0, math.inf)):
            upper = math.nextafter(lower, math.inf)
            midpoint = (Fraction(lower) + Fraction(upper)) / 2
            squared = midpoint**2
            epsilon = Fraction(1, 10**200)
            self.assertEqual(_sqrt_fraction_binary64(squared - epsilon), lower)
            self.assertEqual(_sqrt_fraction_binary64(squared + epsilon), upper)
        self.assertEqual(_sqrt_fraction_binary64(((Fraction(1.0) + Fraction(math.nextafter(1.0, math.inf))) / 2)**2), 1.0)
        lower = math.nextafter(1.0, math.inf)
        upper = math.nextafter(lower, math.inf)
        self.assertEqual(_sqrt_fraction_binary64(((Fraction(lower) + Fraction(upper)) / 2)**2), upper)

    def test_rational_sqrt_subnormal_and_invalid_domain(self):
        smallest = math.nextafter(0.0, 1.0)
        midpoint = Fraction(smallest) / 2
        self.assertEqual(_sqrt_fraction_binary64(midpoint**2), 0.0)
        self.assertEqual(_sqrt_fraction_binary64(Fraction(smallest)**2), smallest)
        self.assertEqual(_sqrt_fraction_binary64(Fraction(0)), 0.0)
        with self.assertRaises(ValueError):
            _sqrt_fraction_binary64(Fraction(-1))

    def test_all_recorded_cv_values_have_correct_rounding_bounds(self):
        root = scientific_corrections.ROOT
        table = scientific_corrections.read_csv(root / "results/corrections/a6000_repeats.csv")
        for row in table:
            metric = "throughput_samples_s" if row["model"] == "resnet50" else "throughput_tokens_s"
            values = [json.loads((root / relative).read_text())[metric] for relative in row["sources"].split(";")]
            xs = list(map(Fraction, values))
            mean = sum(xs) / len(xs)
            variance = sum((value - mean)**2 for value in xs) / (len(xs) - 1)
            squared = 10000 * variance / mean**2
            result = dispersion(values)["sample_cv_pct"]
            left = (Fraction(math.nextafter(result, 0.0)) + Fraction(result)) / 2
            right = (Fraction(math.nextafter(result, math.inf)) + Fraction(result)) / 2
            self.assertGreater(squared, left**2, row["sources"])
            self.assertLess(squared, right**2, row["sources"])
            with localcontext() as context:
                context.prec = 100
                reference = (Decimal(squared.numerator) / Decimal(squared.denominator)).sqrt()
                self.assertEqual(result, float(reference))

    def test_dispersion_handles_equal_values_and_extreme_scale(self):
        self.assertEqual(dispersion([1.0, 1.0, 1.0])["sample_cv_pct"], 0.0)
        self.assertAlmostEqual(dispersion([1e-300, 1e300])["sample_cv_pct"], 100 * math.sqrt(2))
        for values in ([], [0.0], [-1.0], [math.inf], [math.nan]):
            with self.assertRaises(ValueError):
                dispersion(values)


if __name__ == "__main__":
    unittest.main()
