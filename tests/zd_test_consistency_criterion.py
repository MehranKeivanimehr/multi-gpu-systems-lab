"""Discriminating tests for the module/master consistency criterion.

Gate-0 (GATE0-20261020T041835Z) ran with a generic 1e-3 tolerance. The maximum
master update in that step was 1.0103e-04, i.e. BELOW the tolerance, so a total
failure to propagate would still have passed. The measured difference happened
to be exactly 0.0, which is what carried the conclusion there - not the
criterion. These tests exercise the repaired criterion against exactly that
failure mode.

The new checker has NOT been executed on hardware.
"""
import pathlib, sys, unittest

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent / "studies" / "titan-1080"
sys.path.insert(0, str(ROOT))
from zerodiag import engine_probe as EP          # noqa: E402

GATE0_UPDATE = 1.0103e-04      # the actual max |master update| at Gate-0


class Layout:
    """Minimal stand-in exposing the attribute names the probe reads."""

    def __init__(self, numels=(64, 32), master_dtype=torch.float32,
                 working_dtype=torch.float32, propagate=True, seed=0,
                 working_numels=None, n_working=None, contiguous=True):
        torch.manual_seed(seed)
        self.single_partition_of_fp32_groups = [
            torch.nn.Parameter(torch.randn(n, dtype=master_dtype))
            for n in numels]
        wn = working_numels or numels
        flat = []
        for m, n in zip(self.single_partition_of_fp32_groups, wn):
            src = m.detach() if propagate else m.detach() - GATE0_UPDATE
            t = src[:n].to(working_dtype).clone() if n <= m.numel() \
                else torch.zeros(n, dtype=working_dtype)
            if not contiguous:
                t = torch.stack([t, t], dim=1)[:, 0]
            flat.append(t)
        if n_working is not None:
            flat = flat[:n_working]
        self.bit16_groups_flat = flat


class Fp32ExactCriterion(unittest.TestCase):

    def test_correct_propagation_passes(self):
        r = EP.module_reflects_masters(Layout(), chunk_elems=16)
        self.assertTrue(r["checked"])
        self.assertEqual(r["criterion"], "fp32_exact_equality")
        self.assertTrue(r["verdict_successful"])
        self.assertTrue(r["numerically_equal"])
        self.assertTrue(r["bitwise_identical"])
        self.assertEqual(r["max_abs_difference"], 0.0)
        self.assertIsNone(r["tolerance_used"])

    def test_stale_working_parameters_fail_at_the_gate0_update_size(self):
        """The exact case the old 1e-3 criterion could not detect."""
        r = EP.module_reflects_masters(Layout(propagate=False), chunk_elems=16)
        self.assertLess(GATE0_UPDATE, 1e-3,
                        "fixture must sit below the withdrawn tolerance")
        self.assertFalse(r["verdict_successful"])
        self.assertFalse(r["numerically_equal"])
        d = r["max_abs_difference"]
        # the fp32 subtraction carries a representation error of order
        # eps*|master| ~ 1e-7, so compare at fp32 resolution, not to 1e-9
        self.assertLess(abs(d - GATE0_UPDATE), 1e-6,
                        f"difference {d:.6g} is not the fixture's update size")
        # the decisive point: the deviation sits BELOW the withdrawn tolerance,
        # so the old criterion would have passed this stale state
        self.assertLess(d, 1e-3)
        self.assertIn("exact equality", r["per_group"][0]["reason"])

    def test_a_single_representable_element_mismatch_fails(self):
        lay = Layout(numels=(64,))
        # perturb one element by one representable fp32 step
        w = lay.bit16_groups_flat[0]
        w[7] = torch.nextafter(w[7], torch.tensor(float("inf")))
        r = EP.module_reflects_masters(lay, chunk_elems=16)
        self.assertFalse(r["verdict_successful"])
        self.assertEqual(r["per_group"][0]["n_differing_elements"], 1)
        self.assertEqual(r["per_group"][0]["first_difference"]["global_index"], 7)
        self.assertGreater(r["max_abs_difference"], 0.0)
        self.assertFalse(r["bitwise_identical"])

    def test_non_finite_values_refuse_even_when_equal(self):
        lay = Layout(numels=(32,))
        lay.single_partition_of_fp32_groups[0].data[3] = float("nan")
        lay.bit16_groups_flat[0][3] = float("nan")
        r = EP.module_reflects_masters(lay, chunk_elems=8)
        g = r["per_group"][0]
        self.assertFalse(g["all_finite"])
        self.assertFalse(r["verdict_successful"])
        # NaN != NaN numerically, yet the bit patterns are identical here
        self.assertFalse(g["numerically_equal"])
        self.assertTrue(g["bitwise_identical"])

    def test_coverage_is_complete_and_counted(self):
        r = EP.module_reflects_masters(Layout(numels=(64, 32)), chunk_elems=16)
        c = r["coverage"]
        self.assertEqual(c["expected_groups"], 2)
        self.assertEqual(c["compared_groups"], 2)
        self.assertEqual(c["expected_elements"], 96)
        self.assertEqual(c["compared_elements"], 96)
        self.assertTrue(c["complete"])

    def test_chunking_does_not_change_the_verdict(self):
        for ce in (1, 7, 16, 95, 96, 1000):
            r = EP.module_reflects_masters(Layout(numels=(64, 32)), chunk_elems=ce)
            self.assertTrue(r["verdict_successful"], f"failed at chunk {ce}")
            self.assertEqual(r["coverage"]["compared_elements"], 96)


class Refusals(unittest.TestCase):

    def test_missing_working_view_refuses(self):
        lay = Layout()
        del lay.bit16_groups_flat
        r = EP.module_reflects_masters(lay, chunk_elems=16)
        self.assertFalse(r["checked"])
        self.assertFalse(r["verdict_successful"])
        self.assertIn("NOT established", r["reason"])

    def test_partial_group_count_refuses_rather_than_comparing_a_prefix(self):
        r = EP.module_reflects_masters(Layout(numels=(64, 32), n_working=1),
                                       chunk_elems=16)
        self.assertFalse(r["checked"])
        self.assertFalse(r["verdict_successful"])
        self.assertIn("refused rather than compared as a prefix", r["reason"])

    def test_shape_mismatch_refuses_and_records_no_broadcasting(self):
        r = EP.module_reflects_masters(
            Layout(numels=(64, 32), working_numels=(60, 32)), chunk_elems=16)
        g = r["per_group"][0]
        self.assertEqual(g["criterion"], "unchecked")
        self.assertFalse(g["shapes_match"])
        self.assertFalse(g["broadcasting_used"])
        self.assertIn("no broadcasting is permitted", g["reason"])
        self.assertFalse(r["verdict_successful"])

    def test_non_contiguous_view_refuses(self):
        r = EP.module_reflects_masters(Layout(numels=(64,), contiguous=False),
                                       chunk_elems=16)
        g = r["per_group"][0]
        self.assertIn(g["criterion"], ("unchecked",))
        self.assertFalse(r["verdict_successful"])

    def test_unresolved_dtype_pair_is_unverified_not_passed(self):
        """A non-floating working dtype has no resolved conversion semantics."""
        lay = Layout(numels=(32,))
        lay.bit16_groups_flat = [lay.single_partition_of_fp32_groups[0]
                                 .detach().to(torch.int32)]
        r = EP.module_reflects_masters(lay, chunk_elems=8)
        g = r["per_group"][0]
        self.assertEqual(g["criterion"], "unchecked")
        self.assertFalse(g["verdict_successful"])
        self.assertIn("not resolved", g["reason"])
        self.assertIn("rather than", g["reason"])
        self.assertFalse(r["checked"])

    def test_the_withdrawn_tolerance_is_gone_from_the_result(self):
        r = EP.module_reflects_masters(Layout(), chunk_elems=16)
        self.assertNotIn("within_tolerance", r)
        self.assertNotIn("tolerance", [k for k in r if k == "tolerance"])
        self.assertIsNone(r["tolerance_used"])
        self.assertIn("withdrawn", r["tolerance_note"])
        self.assertIn("GATE0-20261020T041835Z", r["supersedes"])


class MixedPrecisionCriterion(unittest.TestCase):

    def test_expected_cast_passes_and_does_not_inherit_the_fp32_rule(self):
        r = EP.module_reflects_masters(
            Layout(numels=(64,), working_dtype=torch.float16), chunk_elems=16)
        g = r["per_group"][0]
        self.assertEqual(g["criterion"], "mixed_precision_expected_cast")
        self.assertEqual(g["expected_cast"], "master.to(torch.float16)")
        self.assertTrue(g["bitwise_identical"])
        self.assertTrue(r["verdict_successful"])
        self.assertIn("not inherited", g["note"])
        self.assertIsNone(r["tolerance_used"])

    def test_a_working_copy_that_is_not_the_expected_cast_fails(self):
        lay = Layout(numels=(64,), working_dtype=torch.float16)
        w = lay.bit16_groups_flat[0]
        w[5] = torch.nextafter(w[5], torch.tensor(float("inf"),
                                                  dtype=torch.float16))
        r = EP.module_reflects_masters(lay, chunk_elems=16)
        self.assertFalse(r["verdict_successful"])
        self.assertFalse(r["per_group"][0]["bitwise_identical"])

    def test_a_widening_cast_is_checked_not_refused(self):
        """fp32 -> fp64 is lossless; it is compared, not declared unresolved."""
        lay = Layout(numels=(32,))
        lay.bit16_groups_flat = [lay.single_partition_of_fp32_groups[0]
                                 .detach().to(torch.float64)]
        r = EP.module_reflects_masters(lay, chunk_elems=8)
        self.assertEqual(r["per_group"][0]["criterion"],
                         "mixed_precision_expected_cast")
        self.assertTrue(r["verdict_successful"])

    def test_the_expected_cast_criterion_cannot_see_sub_resolution_updates(self):
        """A documented limitation, asserted rather than left implicit.

        An update smaller than the working dtype's resolution rounds to the same
        value, so the cast criterion passes whether or not it was applied. This
        is a property of the working precision, not a tolerance.
        """
        lay = Layout(numels=(8,), working_dtype=torch.float16)
        m = lay.single_partition_of_fp32_groups[0]
        with torch.no_grad():
            m.data.add_(1e-6)          # far below fp16 resolution near |m|~1
        r = EP.module_reflects_masters(lay, chunk_elems=8)
        self.assertTrue(r["verdict_successful"],
                        "a sub-resolution update is invisible to the cast check")
        self.assertTrue(r["bitwise_identical"])

    def test_bfloat16_uses_the_same_separate_criterion(self):
        r = EP.module_reflects_masters(
            Layout(numels=(32,), working_dtype=torch.bfloat16), chunk_elems=8)
        self.assertEqual(r["per_group"][0]["criterion"],
                         "mixed_precision_expected_cast")
        self.assertTrue(r["verdict_successful"])


class EqualityNamesAreDistinct(unittest.TestCase):
    """Numerical equality and bit-pattern equality must not be conflated."""

    def test_signed_zero_is_numerically_equal_but_not_bitwise(self):
        a = torch.tensor([0.0, 1.0]); b = torch.tensor([-0.0, 1.0])
        self.assertTrue(bool((a == b).all()))                  # numerical
        bw = EP.bitwise_identical(a, b, chunk_elems=2)
        self.assertTrue(bw["comparable"])
        self.assertFalse(bw["bitwise_identical"])              # bit pattern
        self.assertEqual(bw["first_differing_index"], 0)

    def test_nan_is_bitwise_identical_but_not_numerically_equal(self):
        a = torch.tensor([float("nan")]); b = a.clone()
        self.assertFalse(bool((a == b).all()))
        self.assertTrue(EP.bitwise_identical(a, b, chunk_elems=1)["bitwise_identical"])

    def test_dtype_mismatch_is_not_comparable_rather_than_false(self):
        bw = EP.bitwise_identical(torch.zeros(4),
                                  torch.zeros(4, dtype=torch.float16),
                                  chunk_elems=4)
        self.assertFalse(bw["comparable"])
        self.assertIsNone(bw["bitwise_identical"])
        self.assertIn("NOT established", bw["reason"])

    def test_bit_compare_is_chunk_bounded(self):
        a = torch.randn(1000); b = a.clone()
        bw = EP.bitwise_identical(a, b, chunk_elems=64)
        self.assertTrue(bw["bitwise_identical"])
        self.assertEqual(bw["chunk_elems"], 64)
        self.assertEqual(bw["integer_view"], "torch.int32")


class ObservationPhase(unittest.TestCase):

    def test_comparison_after_simulated_deferred_propagation(self):
        """Mimics DeepSpeed copying masters into the working view inside step()."""
        lay = Layout(numels=(64,), propagate=False)
        pre = EP.module_reflects_masters(lay, chunk_elems=16)
        self.assertFalse(pre["verdict_successful"],
                         "stale state must fail before propagation")
        # deferred propagation, as stage_1_and_2.py:1915 performs it
        lay.bit16_groups_flat[0].copy_(
            lay.single_partition_of_fp32_groups[0].detach())
        post = EP.module_reflects_masters(lay, chunk_elems=16)
        self.assertTrue(post["verdict_successful"],
                        "comparison after propagation must pass")
        self.assertEqual(post["max_abs_difference"], 0.0)

    def test_the_result_states_its_observation_phase(self):
        r = EP.module_reflects_masters(Layout(), chunk_elems=16)
        self.assertIn("after engine.step() returned", r["observation_phase"])
        self.assertIn("propagated the updated masters", r["observation_phase"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
