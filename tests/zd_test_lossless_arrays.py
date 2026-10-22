"""Lossless array evidence, and the reference completion gate that needs it."""
from __future__ import annotations

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

from zerodiag import lossless as LL               # noqa: E402
from zerodiag import ref_compare as RC            # noqa: E402
from zerodiag import ref_runner as RR             # noqa: E402


class Roundtrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.tmp.name) / "arrays"

    def tearDown(self):
        self.tmp.cleanup()

    def test_float32_roundtrip_is_bitwise_exact(self):
        import torch
        t = torch.tensor([[1.0, -2.5, 3.0625], [0.0, 1e-30, -1e30]],
                         dtype=torch.float32)
        rec = LL.save_array(t, self.d, "final_logits", note="unit test")
        self.assertEqual(rec["dtype"], "float32")
        self.assertEqual(rec["shape"], [2, 3])
        self.assertEqual(rec["n_bytes"], 24)
        back, rec2 = LL.load_array(self.d, "final_logits")
        self.assertEqual(rec2["sha256"], rec["sha256"])
        self.assertEqual(back.tobytes(), t.numpy().tobytes())
        self.assertEqual(back.shape, (2, 3))

    def test_payload_is_not_pickle_and_reads_with_plain_struct(self):
        import torch
        t = torch.tensor([1.5, -0.25], dtype=torch.float32)
        rec = LL.save_array(t, self.d, "x")
        raw = (self.d / rec["file"]).read_bytes()
        self.assertEqual(len(raw), 8)
        self.assertEqual(list(struct.unpack("<2f", raw)), [1.5, -0.25])
        self.assertNotIn(b"numpy", raw)
        self.assertFalse(raw.startswith(b"\x80"), "must not be a pickle")
        self.assertIn("numpy.fromfile", rec["read_with"])

    def test_int64_token_ids_roundtrip(self):
        import torch
        t = torch.tensor([[1, 2, 3]], dtype=torch.long)
        LL.save_array(t, self.d, "generated_ids")
        back, rec = LL.load_array(self.d, "generated_ids")
        self.assertEqual(rec["dtype"], "int64")
        self.assertEqual(back.tolist(), [[1, 2, 3]])

    def test_unsupported_dtype_is_refused_not_guessed(self):
        import torch
        t = torch.tensor([1 + 2j], dtype=torch.complex64)
        with self.assertRaises(ValueError):
            LL.save_array(t, self.d, "bad")

    def test_corruption_and_truncation_are_detected(self):
        import torch
        LL.save_array(torch.tensor([1.0, 2.0]), self.d, "x")
        p = self.d / "x.bin"
        good = p.read_bytes()
        p.write_bytes(b"\xff" + good[1:])
        with self.assertRaises(ValueError):
            LL.load_array(self.d, "x")
        p.write_bytes(good[:-1])
        with self.assertRaises(ValueError):
            LL.load_array(self.d, "x")
        p.write_bytes(good)
        LL.load_array(self.d, "x")

    def test_schema_change_is_refused(self):
        import torch
        LL.save_array(torch.tensor([1.0]), self.d, "x")
        sc = self.d / "x.json"
        rec = json.loads(sc.read_text()); rec["schema"] = "something_else"
        sc.write_text(json.dumps(rec))
        with self.assertRaises(ValueError):
            LL.load_array(self.d, "x")

    def test_the_comparator_reads_what_the_runner_writes(self):
        import torch
        a = pathlib.Path(self.tmp.name) / "a"
        b = pathlib.Path(self.tmp.name) / "b"
        t = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
        LL.save_array(t, a, "final_logits")
        LL.save_array(t, b, "final_logits")
        r = RC.compare(a, b)
        self.assertEqual(r["verdict"], "PASS")
        self.assertEqual(r["total_elements"], 3)


class CompletionGate(unittest.TestCase):
    """A reference case cannot be reported complete without the arrays."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.tmp.name) / "arrays"

    def tearDown(self):
        self.tmp.cleanup()

    def write_all(self, skip=()):
        import torch
        made = {}
        for n in RR.REQUIRED_ARRAYS:
            if n in skip:
                continue
            t = (torch.tensor([[1, 2]], dtype=torch.long)
                 if n in ("input_ids", "attention_mask", "generated_ids")
                 else torch.tensor([[1.0, 2.0]], dtype=torch.float32))
            made[n] = LL.save_array(t, self.d, n)
        return made

    def test_all_arrays_present_allows_completion(self):
        self.write_all()
        ev = {}
        self.assertEqual(RR.gate_completed(ev, self.d), "completed")
        self.assertTrue(ev["reference_case_complete"])
        self.assertTrue(ev["retained_arrays"]["ok"])

    def test_a_missing_logit_array_blocks_completion(self):
        self.write_all(skip=("final_logits",))
        ev = {}
        self.assertEqual(RR.gate_completed(ev, self.d), "refused")
        self.assertFalse(ev["reference_case_complete"])
        self.assertIn("UNAVAILABLE",
                      ev["retained_arrays"]["verified"]["final_logits"])
        self.assertIn("cannot be recomputed", ev["reason"])

    def test_a_corrupt_logit_array_blocks_completion(self):
        self.write_all()
        p = self.d / "final_logits.bin"
        raw = bytearray(p.read_bytes())
        raw[-1] ^= 0xFF                    # a byte that genuinely changes
        self.assertNotEqual(bytes(raw), p.read_bytes())
        p.write_bytes(bytes(raw))
        ev = {}
        self.assertEqual(RR.gate_completed(ev, self.d), "refused")

    def test_each_required_array_is_individually_load_bearing(self):
        for name in RR.REQUIRED_ARRAYS:
            with self.subTest(missing=name):
                for f in list(self.d.glob("*")) if self.d.is_dir() else []:
                    f.unlink()
                self.write_all(skip=(name,))
                ev = {}
                self.assertEqual(RR.gate_completed(ev, self.d), "refused")

    def test_empty_directory_blocks_completion(self):
        ev = {}
        self.assertEqual(RR.gate_completed(ev, self.d), "refused")
        self.assertFalse(ev["retained_arrays"]["ok"])


class MetadataContract(unittest.TestCase):
    """The comparator's input contract, built without touching a device."""

    EV = {
        "retained": {
            "final_logits": {"file": "final_logits.bin", "sha256": "a" * 64,
                             "shape": [2, 32000], "dtype": "float32"},
            "input_ids": {"file": "input_ids.bin", "sha256": "b" * 64,
                          "shape": [2, 256], "dtype": "int64"},
            "attention_mask": {"file": "attention_mask.bin",
                               "sha256": "c" * 64, "shape": [2, 256],
                               "dtype": "int64"},
            "generated_ids": {"file": "generated_ids.bin", "sha256": "d" * 64,
                              "shape": [2, 64], "dtype": "int64"}},
        "realized_input": {"realized_input_sha256": "r" * 64,
                           "attention_mask_sha256": "m" * 64,
                           "shape": [2, 256]},
        "initial_state": {"parameter_state_sha256": "s" * 64},
        "resolved_revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
        "model_dtype": "torch.float32", "dropout_active": False,
        "child_device_name": "NVIDIA TITAN RTX",
        "child_device_capability": "7.5",
        "child_visible_devices": "TITAN_RTX#1",
        "torch_version": "2.4.0",
    }

    def test_every_key_the_comparator_reads_is_present(self):
        meta = RR.build_comparison_metadata(self.EV, case_id="R_rtx_1",
                                            pad_token_id=2)
        for k in ("realized_input_sha256", "attention_mask_sha256",
                  "initial_state_sha256", "input_shape", "model_dtype",
                  "resolved_revision", "dropout_active", "child_device_name"):
            self.assertIn(k, meta, k)
        other = dict(meta, child_device_name="NVIDIA GeForce GTX 1080 Ti")
        self.assertTrue(RC.verify_inputs_match(meta, other)["matched"])

    def test_metadata_is_json_serializable_and_records_the_criterion(self):
        meta = RR.build_comparison_metadata(self.EV, case_id="R_p1080_1",
                                            pad_token_id=2)
        text = json.dumps(meta, indent=2, sort_keys=True)
        self.assertIn("titan_reference_logit_criterion_v1", text)
        self.assertIn("REFERENCE_CRITERION_CLARIFICATION.md", text)
        self.assertEqual(meta["case_id"], "R_p1080_1")
        self.assertEqual(meta["eval_settings"]["pad_token_id"], 2)
        self.assertEqual(meta["eval_settings"]["max_new_tokens"], 64)
        self.assertFalse(meta["eval_settings"]["do_sample"])
        self.assertEqual(meta["eval_settings"]["num_beams"], 1)
        self.assertEqual(sorted(meta["arrays"]), sorted(RR.REQUIRED_ARRAYS))

    def test_a_missing_retained_array_raises_before_any_claim(self):
        ev = json.loads(json.dumps(self.EV))
        ev["retained"].pop("final_logits")
        meta = RR.build_comparison_metadata(ev, case_id="x", pad_token_id=2)
        self.assertNotIn("final_logits", meta["arrays"])
        ev2 = json.loads(json.dumps(self.EV))
        ev2.pop("initial_state")
        with self.assertRaises(KeyError):
            RR.build_comparison_metadata(ev2, case_id="x", pad_token_id=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
