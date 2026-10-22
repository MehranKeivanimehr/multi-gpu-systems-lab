"""Discriminating fixtures for the offline reference comparator.

Every case here would be indistinguishable to a digest comparison: the arrays
differ bitwise in all of them. Only the numerical criterion separates them.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import struct
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
ROOT = REPO / "studies" / "titan-1080"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from zerodiag import ref_compare as RC            # noqa: E402

U32 = 5.9604644775390625e-08                      # finfo(float32).eps / 2


def write(dirpath, name, values, shape, dtype="float32"):
    """Write a lossless array WITHOUT torch, so fixtures need no GPU stack."""
    d = pathlib.Path(dirpath); d.mkdir(parents=True, exist_ok=True)
    code = {"float32": "<f", "float64": "<d", "int64": "<q"}[dtype]
    raw = b"".join(struct.pack(code, v) for v in values)
    (d / f"{name}.bin").write_bytes(raw)
    rec = {"schema": "titan_lossless_array_v1", "name": name,
           "file": f"{name}.bin", "dtype": dtype, "byte_order": "little",
           "itemsize": struct.calcsize(code), "shape": list(shape),
           "n_elements": len(values), "n_bytes": len(raw),
           "sha256": hashlib.sha256(raw).hexdigest(), "c_contiguous": True}
    (d / f"{name}.json").write_text(json.dumps(rec, indent=2, sort_keys=True))
    return rec


def f32(x: float) -> float:
    """Round a python float to the nearest float32 value."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


class Fixtures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.a = pathlib.Path(self.tmp.name) / "a"
        self.b = pathlib.Path(self.tmp.name) / "b"

    def tearDown(self):
        self.tmp.cleanup()

    # -- in tolerance, different digests -------------------------------
    def test_differing_arrays_within_tolerance_pass(self):
        base = [1.0, -2.5, 3.25, 0.0]
        pert = [f32(v + 1e-7 * (1 if i % 2 else -1))
                for i, v in enumerate(base)]
        ra = write(self.a, "final_logits", base, [2, 2])
        rb = write(self.b, "final_logits", pert, [2, 2])
        self.assertNotEqual(ra["sha256"], rb["sha256"],
                            "fixture must differ bitwise or it proves nothing")
        r = RC.compare(self.a, self.b)
        self.assertEqual(r["verdict"], "PASS", r)
        self.assertFalse(r["digests_identical"])
        self.assertEqual(r["failing_elements"], 0)
        self.assertGreater(r["max_abs_error"], 0.0)

    # -- out of tolerance ----------------------------------------------
    def test_out_of_tolerance_fails(self):
        base = [1.0, 2.0, 3.0, 4.0]
        bad = [1.0, 2.0, 3.0, f32(4.0 + 1e-3)]
        write(self.a, "final_logits", base, [4])
        write(self.b, "final_logits", bad, [4])
        r = RC.compare(self.a, self.b)
        self.assertEqual(r["verdict"], "FAIL", r)
        self.assertEqual(r["failing_elements"], 1)
        self.assertEqual(r["worst_witness"]["position"], [3])
        self.assertAlmostEqual(r["worst_witness"]["abs_error"],
                               abs(bad[3] - 4.0), places=12)

    # -- boundary ------------------------------------------------------
    def test_boundary_just_inside_and_just_outside(self):
        M = 4.0
        tol = 64.0 * U32 * max(1.0, M) + 64.0 * U32 * M        # frozen formula
        inside = [0.0, 0.0, 0.0, f32(M - tol * 0.5)]
        write(self.a, "final_logits", [0.0, 0.0, 0.0, M], [4])
        write(self.b, "final_logits", inside, [4])
        self.assertEqual(RC.compare(self.a, self.b)["verdict"], "PASS")
        outside = [0.0, 0.0, 0.0, f32(M - tol * 4.0)]
        write(self.b, "final_logits", outside, [4])
        r = RC.compare(self.a, self.b)
        self.assertEqual(r["verdict"], "FAIL", r)
        self.assertGreater(r["max_abs_error"], r["tolerance"])

    def test_boundary_is_exact_to_one_float32_ulp(self):
        """The decisive element sits one float32 step either side of the bound.

        With A = [0, 0] and B = [x, 0] the frozen formula gives
        tol = 64u*max(1, x) + 64u*max(x, 1e-300) = 64u*(1 + x) for x < 1, so
        the criterion |x| <= tol flips at x* = 64u / (1 - 64u). No slack, no
        multiplier: the two fixtures are adjacent float32 values.
        """
        import numpy as np
        xstar = 64.0 * U32 / (1.0 - 64.0 * U32)
        lo = np.nextafter(np.float32(xstar), np.float32(0.0))
        hi = np.nextafter(np.float32(xstar), np.float32(1.0))
        self.assertLessEqual(float(lo), xstar)
        self.assertGreater(float(hi), xstar)
        self.assertEqual(int(np.float32(hi).view("uint32"))
                         - int(np.float32(lo).view("uint32")), 2,
                         "fixtures must straddle the bound within two ULPs")
        write(self.a, "final_logits", [0.0, 0.0], [2])
        write(self.b, "final_logits", [float(lo), 0.0], [2])
        r_in = RC.compare(self.a, self.b)
        self.assertEqual(r_in["verdict"], "PASS", r_in)
        self.assertLessEqual(r_in["max_abs_error"], r_in["tolerance"])
        write(self.b, "final_logits", [float(hi), 0.0], [2])
        r_out = RC.compare(self.a, self.b)
        self.assertEqual(r_out["verdict"], "FAIL", r_out)
        self.assertGreater(r_out["max_abs_error"], r_out["tolerance"])
        self.assertEqual(r_out["failing_elements"], 1)

    def test_tolerance_matches_the_frozen_clarification(self):
        write(self.a, "final_logits", [10.0, -1.0], [2])
        write(self.b, "final_logits", [10.0, -1.0], [2])
        r = RC.compare(self.a, self.b)
        self.assertEqual(r["u"], U32)
        self.assertEqual(r["magnitude_scale_M"], 10.0)
        self.assertAlmostEqual(r["atol"], 64 * U32 * 10.0, places=18)
        self.assertAlmostEqual(r["rtol_term"], 64 * U32 * 10.0, places=18)
        self.assertAlmostEqual(r["tolerance"], r["atol"] + r["rtol_term"],
                               places=18)
        self.assertFalse(r["threshold_widened"])

    def test_zero_arrays_use_the_unit_floor(self):
        write(self.a, "final_logits", [0.0, 0.0], [2])
        write(self.b, "final_logits", [0.0, 0.0], [2])
        r = RC.compare(self.a, self.b)
        self.assertEqual(r["magnitude_scale_M"], 0.0)
        self.assertAlmostEqual(r["atol"], 64 * U32, places=18)
        self.assertEqual(r["verdict"], "PASS")

    # -- non-finite ----------------------------------------------------
    def test_non_finite_is_refused_not_failed(self):
        write(self.a, "final_logits", [1.0, float("nan")], [2])
        write(self.b, "final_logits", [1.0, 1.0], [2])
        r = RC.compare(self.a, self.b)
        self.assertEqual(r["verdict"], "REFUSED", r)
        self.assertIn("non-finite", r["reason"])
        write(self.a, "final_logits", [1.0, float("inf")], [2])
        self.assertEqual(RC.compare(self.a, self.b)["verdict"], "REFUSED")

    # -- missing / corrupt data ----------------------------------------
    def test_missing_array_is_refused_not_a_verdict(self):
        write(self.a, "final_logits", [1.0], [1])
        r = RC.compare(self.a, self.b)
        self.assertEqual(r["verdict"], "REFUSED")
        self.assertIn("FileNotFoundError", r["reason"])
        self.assertNotIn("max_abs_error", r)

    def test_missing_payload_with_present_sidecar_is_refused(self):
        write(self.a, "final_logits", [1.0], [1])
        write(self.b, "final_logits", [1.0], [1])
        (self.b / "final_logits.bin").unlink()
        self.assertEqual(RC.compare(self.a, self.b)["verdict"], "REFUSED")

    def test_checksum_mismatch_is_refused(self):
        write(self.a, "final_logits", [1.0, 2.0], [2])
        write(self.b, "final_logits", [1.0, 2.0], [2])
        p = self.b / "final_logits.bin"
        p.write_bytes(b"\x01" + p.read_bytes()[1:])
        r = RC.compare(self.a, self.b)
        self.assertEqual(r["verdict"], "REFUSED")
        self.assertIn("checksum", r["reason"])

    def test_truncated_payload_is_refused(self):
        write(self.a, "final_logits", [1.0, 2.0], [2])
        write(self.b, "final_logits", [1.0, 2.0], [2])
        p = self.b / "final_logits.bin"
        p.write_bytes(p.read_bytes()[:-2])
        self.assertEqual(RC.compare(self.a, self.b)["verdict"], "REFUSED")

    def test_shape_and_dtype_mismatch_are_refused(self):
        write(self.a, "final_logits", [1.0, 2.0, 3.0, 4.0], [2, 2])
        write(self.b, "final_logits", [1.0, 2.0, 3.0, 4.0], [4])
        self.assertEqual(RC.compare(self.a, self.b)["verdict"], "REFUSED")
        write(self.b, "final_logits", [1.0, 2.0, 3.0, 4.0], [2, 2],
              dtype="float64")
        r = RC.compare(self.a, self.b)
        self.assertEqual(r["verdict"], "REFUSED")
        self.assertIn("dtype", r["reason"])

    # -- identical digests still produce a numerical verdict -----------
    def test_identical_arrays_pass_with_zero_error(self):
        write(self.a, "final_logits", [1.0, 2.0], [2])
        write(self.b, "final_logits", [1.0, 2.0], [2])
        r = RC.compare(self.a, self.b)
        self.assertTrue(r["digests_identical"])
        self.assertEqual(r["verdict"], "PASS")
        self.assertEqual(r["max_abs_error"], 0.0)

    # -- greedy tokens, reported separately ----------------------------
    def test_greedy_tokens_separate_from_logits(self):
        write(self.a, "generated_ids", [1, 2, 3, 4], [1, 4], dtype="int64")
        write(self.b, "generated_ids", [1, 2, 3, 4], [1, 4], dtype="int64")
        self.assertEqual(RC.greedy_tokens(self.a, self.b)["verdict"],
                         "IDENTICAL")
        write(self.b, "generated_ids", [1, 2, 9, 4], [1, 4], dtype="int64")
        g = RC.greedy_tokens(self.a, self.b)
        self.assertEqual(g["verdict"], "DIFFERENT")
        self.assertEqual(g["differing_tokens"], 1)
        self.assertEqual(g["first_difference_flat_index"], 2)

    def test_greedy_tokens_missing_is_refused(self):
        self.assertEqual(RC.greedy_tokens(self.a, self.b)["verdict"], "REFUSED")

    # -- mismatched inputs ---------------------------------------------
    def test_mismatched_inputs_are_detected(self):
        m1 = {"realized_input_sha256": "aa", "attention_mask_sha256": "bb",
              "initial_state_sha256": "cc", "input_shape": [2, 256],
              "model_dtype": "torch.float32", "resolved_revision": "rev",
              "dropout_active": False, "child_device_name": "NVIDIA TITAN RTX"}
        m2 = dict(m1, child_device_name="NVIDIA GeForce GTX 1080 Ti")
        self.assertTrue(RC.verify_inputs_match(m1, m2)["matched"])
        bad = dict(m2, realized_input_sha256="zz")
        r = RC.verify_inputs_match(m1, bad)
        self.assertFalse(r["matched"])
        self.assertFalse(r["checks"]["realized_input_sha256"])
        same_device = dict(m1)
        self.assertFalse(RC.verify_inputs_match(m1, same_device)["cross_family"])
        self.assertFalse(RC.verify_inputs_match({}, {})["matched"])

    # -- array location resolution -------------------------------------
    def test_cli_finds_arrays_in_the_evidence_layout(self):
        """The runner writes <evidence>/arrays/, not <evidence>/ directly."""
        for base in (self.a, self.b):
            write(base / "arrays", "final_logits", [1.0, 2.0], [2])
            write(base / "arrays", "generated_ids", [7, 8], [2], dtype="int64")
            (base / "comparison_metadata.json").write_text(json.dumps(
                {"arrays_dir": "arrays",
                 "realized_input_sha256": "x", "attention_mask_sha256": "y",
                 "initial_state_sha256": "z", "input_shape": [2, 256],
                 "model_dtype": "torch.float32", "resolved_revision": "r",
                 "dropout_active": False,
                 "child_device_name": ("NVIDIA TITAN RTX" if base is self.a
                                       else "NVIDIA GeForce GTX 1080 Ti")}))
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = RC.main(["--a", str(self.a), "--b", str(self.b)])
        out = json.loads(buf.getvalue())
        self.assertEqual(rc, 0, out["logits"])
        self.assertEqual(out["logits"]["verdict"], "PASS")
        self.assertTrue(out["inputs"]["matched"])
        self.assertEqual(out["greedy_tokens"]["verdict"], "IDENTICAL")
        self.assertTrue(out["array_locations"]["a"]["resolved"].endswith("arrays"))

    def test_a_wrong_location_is_named_as_a_location_failure(self):
        d, loc = RC.resolve_arrays_dir(self.a, {"arrays_dir": "arrays"})
        self.assertIsNone(loc["resolved"])
        self.assertIn("LOCATION failure", loc["note"])
        self.assertEqual(loc["declared_arrays_dir"], "arrays")

    def test_a_bare_array_directory_still_works(self):
        write(self.a, "final_logits", [1.0], [1])
        d, loc = RC.resolve_arrays_dir(self.a, {})
        self.assertEqual(str(d), str(self.a))
        self.assertEqual(loc["resolved"], str(self.a))

    # -- the CLI contract ----------------------------------------------
    def test_cli_exit_codes_distinguish_fail_from_refusal(self):
        write(self.a, "final_logits", [1.0], [1])
        write(self.b, "final_logits", [1.0], [1])
        write(self.a, "generated_ids", [1], [1], dtype="int64")
        write(self.b, "generated_ids", [1], [1], dtype="int64")
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = RC.main(["--a", str(self.a), "--b", str(self.b)])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(buf.getvalue())["logits"]["verdict"], "PASS")
        write(self.b, "final_logits", [2.0], [1])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(RC.main(["--a", str(self.a), "--b", str(self.b)]), 2)
        (self.b / "final_logits.bin").unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(RC.main(["--a", str(self.a), "--b", str(self.b)]), 95)


if __name__ == "__main__":
    unittest.main(verbosity=2)
