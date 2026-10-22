"""The advertised public offline commands, exercised as published.

Both tools are run as subprocesses with `-I -S`, so site-packages is not even
on the path: if either needed numpy, torch or anything this project wrote, the
test would fail. That is the published contract -- a reviewer with a fresh
copy, no GPU and no model cache can recompute both results.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
ROOT = REPO / "studies" / "titan-1080"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from zd_test_ref_compare import f32, write                   # noqa: E402
from zerodiag import ref_compare as RC                        # noqa: E402

REF_TOOL = REPO / "tools/recompute_reference_verdict.py"
STREAM_TOOL = REPO / "tools/recompute_streaming_metrics.py"
U32 = 5.9604644775390625e-08


def run(tool, args):
    """Run a published tool with site-packages disabled."""
    p = subprocess.run([sys.executable, "-I", "-S", str(tool)] + args,
                       capture_output=True, text=True, timeout=600)
    body = None
    if p.stdout.strip().startswith("{"):
        body = json.loads(p.stdout)
    return p.returncode, body, p.stderr


def meta(dirpath, device):
    (pathlib.Path(dirpath) / "comparison_metadata.json").write_text(json.dumps({
        "arrays_dir": "arrays", "realized_input_sha256": "r" * 64,
        "attention_mask_sha256": "m" * 64, "initial_state_sha256": "s" * 64,
        "input_shape": [2, 256], "model_dtype": "torch.float32",
        "resolved_revision": "rev", "dropout_active": False,
        "child_device_name": device}, indent=2, sort_keys=True))


class ReferenceTool(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.a = pathlib.Path(self.tmp.name) / "a"
        self.b = pathlib.Path(self.tmp.name) / "b"
        for p, dev in ((self.a, "DEVICE_FAMILY_X"), (self.b, "DEVICE_FAMILY_Y")):
            p.mkdir(parents=True)
            meta(p, dev)

    def tearDown(self):
        self.tmp.cleanup()

    def arrays(self, va, vb, shape, tokens=(1, 2)):
        write(self.a / "arrays", "final_logits", va, shape)
        write(self.b / "arrays", "final_logits", vb, shape)
        write(self.a / "arrays", "generated_ids", list(tokens),
              [1, len(tokens)], dtype="int64")
        write(self.b / "arrays", "generated_ids", list(tokens),
              [1, len(tokens)], dtype="int64")

    def test_in_tolerance_passes_without_site_packages(self):
        base = [1.0, -2.5, 3.25, 0.0]
        pert = [f32(v + 1e-7 * (1 if i % 2 else -1)) for i, v in enumerate(base)]
        self.arrays(base, pert, [2, 2])
        rc, out, err = run(REF_TOOL, ["--a", str(self.a), "--b", str(self.b)])
        self.assertEqual(rc, 0, err)
        self.assertEqual(out["logits"]["verdict"], "PASS")
        self.assertFalse(out["logits"]["digests_identical"])
        self.assertTrue(out["inputs"]["matched"])
        self.assertIn("standard library only", out["logits"]["implementation"])

    def test_out_of_tolerance_exits_two(self):
        self.arrays([1.0, 2.0], [1.0, f32(2.0 + 1e-3)], [2])
        rc, out, err = run(REF_TOOL, ["--a", str(self.a), "--b", str(self.b)])
        self.assertEqual(rc, 2, err)
        self.assertEqual(out["logits"]["verdict"], "FAIL")
        self.assertEqual(out["logits"]["failing_elements"], 1)

    def test_missing_evidence_refuses_cleanly(self):
        rc, out, err = run(REF_TOOL, ["--a", str(self.a), "--b", str(self.b)])
        self.assertEqual(rc, 95, err)
        self.assertEqual(out["logits"]["verdict"], "REFUSED")
        self.assertNotIn("max_abs_error", out["logits"])
        self.assertEqual(err, "", "a clean refusal writes no traceback")

    def test_corrupted_evidence_refuses_cleanly(self):
        self.arrays([1.0, 2.0], [1.0, 2.0], [2])
        p = self.b / "arrays/final_logits.bin"
        raw = bytearray(p.read_bytes()); raw[-1] ^= 0xFF
        p.write_bytes(bytes(raw))
        rc, out, err = run(REF_TOOL, ["--a", str(self.a), "--b", str(self.b)])
        self.assertEqual(rc, 95, err)
        self.assertIn("checksum mismatch", out["logits"]["reason"])
        self.assertEqual(err, "")

    def test_mismatched_inputs_cannot_report_a_pass(self):
        self.arrays([1.0], [1.0], [1], tokens=(5,))
        meta(self.b, "DEVICE_FAMILY_X")                 # same device => no control
        rc, out, err = run(REF_TOOL, ["--a", str(self.a), "--b", str(self.b)])
        self.assertEqual(out["logits"]["verdict"], "PASS")
        self.assertFalse(out["inputs"]["matched"])
        self.assertEqual(rc, 95, "a PASS on unmatched inputs must not exit 0")
        self.assertIn("not be a controlled comparison", err)

    def test_the_two_implementations_agree(self):
        """The stdlib tool and the numpy library must give the same numbers."""
        import importlib.util
        if importlib.util.find_spec("numpy") is None:
            self.skipTest("optional NumPy cross-implementation comparison; standalone stdlib command tested separately")
        cases = ([1.0, -2.5, 3.25, 0.0], [0.0, 0.0, 0.0, 0.0],
                 [16.0, -11.3, 0.5, 2.25])
        for base in cases:
            pert = [f32(v + 3e-7) for v in base]
            self.arrays(base, pert, [2, 2])
            rc, out, err = run(REF_TOOL, ["--a", str(self.a), "--b", str(self.b)])
            lib = RC.compare(self.a / "arrays", self.b / "arrays")
            std = out["logits"]
            for k in ("verdict", "u", "magnitude_scale_M", "atol", "rtol_term",
                      "tolerance", "max_abs_error", "failing_elements",
                      "total_elements"):
                self.assertEqual(std[k], lib[k], f"{k} disagrees on {base}")
            self.assertEqual(std["worst_witness"]["flat_index"],
                             lib["worst_witness"]["flat_index"])

    def test_the_tool_reads_a_hand_written_array(self):
        """A reviewer can write the payload with struct alone."""
        d = self.a / "arrays"; d.mkdir(parents=True, exist_ok=True)
        raw = struct.pack("<3f", 1.0, 2.0, 3.0)
        (d / "final_logits.bin").write_bytes(raw)
        (d / "final_logits.json").write_text(json.dumps({
            "schema": "titan_lossless_array_v1", "file": "final_logits.bin",
            "dtype": "float32", "byte_order": "little", "itemsize": 4,
            "shape": [3], "n_elements": 3, "n_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest()}))
        write(self.b / "arrays", "final_logits", [1.0, 2.0, 3.0], [3])
        rc, out, err = run(REF_TOOL, ["--a", str(self.a), "--b", str(self.b)])
        self.assertEqual(out["logits"]["verdict"], "PASS", err)
        self.assertEqual(out["logits"]["max_abs_error"], 0.0)


class StreamingTool(unittest.TestCase):
    """The published streaming tool, run as published, with `-I -S`."""

    OUT = 64

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.tmp.name) / "session"
        self.d.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def session(self, events, status, usage, *, expected=None, stage="completed",
                workload=True, evidence=True):
        (self.d / "raw_events.json").write_text(json.dumps(
            {"events": events, "status": status, "usage": usage}))
        if not evidence:
            return
        from zerodiag import streaming as ST
        acc = ST.account_session(events, status, usage,
                                 requests_expected=expected or len(events),
                                 output_tokens=self.OUT)
        (self.d / "streaming_evidence.json").write_text(json.dumps({
            "stage": stage, "protocol": ST.PROTOCOL, "accounting": acc,
            "clock_definitions": {"event_t": "monotonic seconds from t_origin"},
            "frozen_workload": {
                "requests_expected": expected or len(events),
                "input_length_verified": workload,
                "consumed_prompt_lengths": [256],
                "issued_as_frozen": workload},
            "server_shutdown": {"confirmed_absent": True},
            "guided_decoding_backend": {"selected": "lm-format-enforcer"},
        }, default=str))

    def good(self, rid="r000", tokens=64, t0=0.10):
        return [{"t": t0, "text": "a"}, {"t": t0 + 0.15, "text": "b"},
                {"t": t0 + 0.30, "text": "", "finish_reason": "length"},
                {"t": t0 + 0.31, "done": True}]

    def test_a_valid_session_is_observed_and_measured(self):
        self.session({"r000": self.good(), "r001": self.good(t0=0.2)},
                     {"r000": "ok", "r001": "ok"},
                     {"r000": {"completion_tokens": 64, "prompt_tokens": 256},
                      "r001": {"completion_tokens": 64, "prompt_tokens": 256}},
                     expected=2)
        rc, out, err = run(STREAM_TOOL, ["--session", str(self.d)])
        self.assertEqual(rc, 0, err)
        self.assertEqual(out["verdict"], "SESSION_OBSERVED")
        a = out["accounting"]
        self.assertEqual(a["counts"], {"successful": 2, "length_mismatch": 0,
                                       "incomplete": 0, "failed": 0})
        self.assertTrue(a["accounting_complete"])
        self.assertEqual(a["successful_without_length_evidence"], 0)
        cm = out["chunk_measurements"]
        self.assertEqual(cm["session_relative_first_event_s"]["n"], 2)
        self.assertAlmostEqual(cm["session_relative_first_event_s"]["min"], 0.10)
        self.assertIn("not time to first token", cm["note"])
        self.assertEqual(out["lengths_supported_by_evidence"]
                         ["server_reported_completion_tokens"], [64])
        self.assertTrue(out["lengths_supported_by_evidence"]
                        ["all_successful_match_protocol"])
        self.assertFalse(out["token_level"]["reported"])
        self.assertTrue(out["cross_check"]["per_request_classes_agree"])
        self.assertEqual(out["session_scope"]["sessions"], 1)

    def test_an_empty_body_200_session_is_not_a_pass(self):
        self.session({f"r{i:03d}": [] for i in range(5)},
                     {f"r{i:03d}": "ok" for i in range(5)}, {}, expected=5)
        rc, out, err = run(STREAM_TOOL, ["--session", str(self.d)])
        self.assertEqual(rc, 2, err)
        self.assertEqual(out["verdict"], "NO_SUCCESSFUL_REQUEST")
        self.assertEqual(out["accounting"]["counts"]["incomplete"], 5)
        self.assertEqual(out["accounting"]["counts"]["successful"], 0)
        self.assertIsNone(out["chunk_measurements"]["first_text_latency_s"])

    def test_missing_completion_is_incomplete_not_successful(self):
        self.session({"r000": [{"t": 0.1, "text": "a"}]}, {"r000": "ok"}, {},
                     expected=1)
        rc, out, _ = run(STREAM_TOOL, ["--session", str(self.d)])
        self.assertEqual(rc, 2)
        self.assertEqual(out["per_request"]["r000"]["class"], "incomplete")
        self.assertFalse(out["per_request"]["r000"]["checks"]
                         ["has_completion_marker"])

    def test_malformed_events_are_counted_not_hidden(self):
        evs = self.good()
        evs.insert(1, {"t": 0.12, "unparsed": "{not json"})
        self.session({"r000": evs}, {"r000": "ok"},
                     {"r000": {"completion_tokens": 64}}, expected=1)
        rc, out, _ = run(STREAM_TOOL, ["--session", str(self.d)])
        self.assertEqual(rc, 0)
        self.assertEqual(out["per_request"]["r000"]["n_unparsed_events"], 1)
        self.assertEqual(out["per_request"]["r000"]["class"], "successful")

    def test_an_only_malformed_session_is_not_a_pass(self):
        self.session({"r000": [{"t": 0.1, "unparsed": "{not json"}]},
                     {"r000": "ok"}, {}, expected=1)
        rc, out, _ = run(STREAM_TOOL, ["--session", str(self.d)])
        self.assertEqual(rc, 2)
        self.assertEqual(out["per_request"]["r000"]["class"], "incomplete")

    def test_a_length_mismatch_is_not_counted_as_successful(self):
        self.session({"r000": self.good()}, {"r000": "ok"},
                     {"r000": {"completion_tokens": 12}}, expected=1)
        rc, out, _ = run(STREAM_TOOL, ["--session", str(self.d)])
        self.assertEqual(rc, 2)
        self.assertEqual(out["per_request"]["r000"]["class"], "length_mismatch")
        self.assertEqual(out["accounting"]["counts"]["successful"], 0)
        L = out["lengths_supported_by_evidence"]
        self.assertEqual(L["length_mismatch_reported_tokens"], [12],
                         "the mismatched length must be visible at session "
                         "level, not only per request")
        self.assertEqual(L["successful_requests_with_length_evidence"], 0)
        self.assertEqual(L["server_reported_completion_tokens"], [],
                         "lengths are summarised over successful requests")

    def test_http_errors_are_failed_not_discarded(self):
        self.session({"r000": [{"t": 0.1, "error": "HTTPError 500"}]},
                     {"r000": "http_500"}, {}, expected=1)
        rc, out, _ = run(STREAM_TOOL, ["--session", str(self.d)])
        self.assertEqual(rc, 2)
        self.assertEqual(out["accounting"]["counts"]["failed"], 1)
        self.assertEqual(out["accounting"]["classes_sum"], 1)

    def test_a_short_session_against_the_frozen_count_is_incomplete(self):
        self.session({"r000": self.good()}, {"r000": "ok"},
                     {"r000": {"completion_tokens": 64}}, expected=1)
        rc, out, _ = run(STREAM_TOOL, ["--session", str(self.d),
                                       "--expected", "100"])
        self.assertEqual(rc, 95)
        self.assertEqual(out["verdict"], "REFUSED")
        self.assertIn("disagrees", out["reason"])

    def test_missing_evidence_refuses_cleanly(self):
        rc, out, err = run(STREAM_TOOL, ["--session", str(self.d)])
        self.assertEqual(rc, 95)
        self.assertEqual(out["verdict"], "REFUSED")
        self.assertIn("missing raw events", out["reason"])
        self.assertEqual(err, "")

    def test_corrupt_evidence_refuses_cleanly(self):
        (self.d / "raw_events.json").write_text("{not json")
        rc, out, err = run(STREAM_TOOL, ["--session", str(self.d)])
        self.assertEqual(rc, 95)
        self.assertEqual(out["verdict"], "REFUSED")
        self.assertIn("not valid JSON", out["reason"])
        self.assertEqual(err, "")

    def test_a_truncated_events_file_refuses(self):
        (self.d / "raw_events.json").write_text(json.dumps({"status": {}}))
        rc, out, _ = run(STREAM_TOOL, ["--session", str(self.d)])
        self.assertEqual(rc, 95)
        self.assertIn("no 'events' mapping", out["reason"])

    def test_corrupt_runner_evidence_refuses(self):
        self.session({"r000": self.good()}, {"r000": "ok"},
                     {"r000": {"completion_tokens": 64}}, expected=1)
        (self.d / "streaming_evidence.json").write_text("{nope")
        rc, out, _ = run(STREAM_TOOL, ["--session", str(self.d)])
        self.assertEqual(rc, 95)
        self.assertEqual(out["verdict"], "REFUSED")

    def test_no_token_timing_is_invented(self):
        self.session({"r000": self.good()}, {"r000": "ok"},
                     {"r000": {"completion_tokens": 64}}, expected=1)
        rc, out, _ = run(STREAM_TOOL, ["--session", str(self.d)])
        self.assertFalse(out["token_level"]["reported"])
        self.assertIn("were not divided", out["token_level"]["reason"])
        d = out["per_request"]["r000"]
        self.assertIsNone(d["per_token_timing"])
        self.assertNotEqual(d["n_content_events"], 64)

    def test_token_timing_is_reported_only_with_real_payload_evidence(self):
        evs = [{"t": 0.1, "text": "a", "token_ids": [5]},
               {"t": 0.2, "text": "b", "token_ids": [6]},
               {"t": 0.3, "text": "", "finish_reason": "length",
                "token_ids": []},
               {"t": 0.31, "done": True, "token_ids": []}]
        self.session({"r000": evs}, {"r000": "ok"},
                     {"r000": {"completion_tokens": 64}}, expected=1)
        rc, out, _ = run(STREAM_TOOL, ["--session", str(self.d)])
        self.assertTrue(out["per_request"]["r000"]["token_identity_in_payload"])
        self.assertFalse(out["token_level"]["reported"], "token IDs alone do not establish individual token emission times")

    # -- the two implementations of the frozen rule must agree ----------
    def test_the_tool_and_the_library_classify_identically(self):
        from zerodiag import streaming as ST
        matrix = {
            "valid": (self.good(), "ok", {"completion_tokens": 64}),
            "empty200": ([], "ok", None),
            "no_marker": ([{"t": 0.1, "text": "a"}], "ok", None),
            "marker_no_text": ([{"t": 0.1, "text": ""},
                                {"t": 0.2, "finish_reason": "length"}],
                               "ok", {"completion_tokens": 64}),
            "short": (self.good(), "ok", {"completion_tokens": 1}),
            "no_usage": (self.good(), "ok", None),
            "http500": ([{"t": 0.1, "error": "HTTPError 500"}], "http_500",
                        None),
            "mid_error": ([{"t": 0.1, "text": "a"},
                           {"t": 0.2, "error": "IncompleteRead"}],
                          "error:IncompleteRead", None),
            "unparsed_only": ([{"t": 0.1, "unparsed": "x"}], "ok", None),
            "done_only": ([{"t": 0.1, "text": "a"}, {"t": 0.2, "done": True}],
                          "ok", {"completion_tokens": 64}),
        }
        events = {k: v[0] for k, v in matrix.items()}
        status = {k: v[1] for k, v in matrix.items()}
        usage = {k: v[2] for k, v in matrix.items() if v[2] is not None}
        self.session(events, status, usage, expected=len(matrix))
        rc, out, err = run(STREAM_TOOL, ["--session", str(self.d)])
        self.assertIsNotNone(out, err)
        for k in matrix:
            lib = ST.classify_request(events[k], status[k], usage.get(k),
                                      output_tokens=self.OUT)
            tool = out["per_request"][k]
            self.assertEqual(tool["class"], lib["class"],
                             f"{k}: tool {tool['class']} vs library "
                             f"{lib['class']}")
            self.assertEqual(tool["checks"], lib["checks"], k)
        seen = {out["per_request"][k]["class"] for k in matrix}
        self.assertEqual(seen, {"successful", "incomplete", "failed",
                                "length_mismatch"},
                         "the matrix must exercise every class")

    def test_recorded_corrections_are_surfaced_by_the_command(self):
        self.session({"r000": self.good()}, {"r000": "ok"},
                     {"r000": {"completion_tokens": 64}}, expected=1)
        (self.d / "evidence_corrections.json").write_text(json.dumps({
            "schema": "titan_evidence_corrections_v1",
            "applies_to": "streaming_evidence.json",
            "corrections": [{"id": "X1", "field": "server.loopback_only",
                             "status": "WRONG as recorded"}]}))
        rc, out, err = run(STREAM_TOOL, ["--session", str(self.d)])
        self.assertEqual(rc, 0, err)
        c = out["evidence_corrections"]
        self.assertTrue(c["present"])
        self.assertEqual(c["count"], 1)
        self.assertEqual(c["corrections"][0]["id"], "X1")

    def test_absent_corrections_are_stated_as_absent(self):
        self.session({"r000": self.good()}, {"r000": "ok"},
                     {"r000": {"completion_tokens": 64}}, expected=1)
        rc, out, _ = run(STREAM_TOOL, ["--session", str(self.d)])
        self.assertFalse(out["evidence_corrections"]["present"])

    def test_a_corrupt_corrections_file_refuses(self):
        self.session({"r000": self.good()}, {"r000": "ok"},
                     {"r000": {"completion_tokens": 64}}, expected=1)
        (self.d / "evidence_corrections.json").write_text("{nope")
        rc, out, _ = run(STREAM_TOOL, ["--session", str(self.d)])
        self.assertEqual(rc, 95)
        self.assertEqual(out["verdict"], "REFUSED")
        self.assertIn("corrections are not valid JSON", out["reason"])

    def test_the_rule_document_digest_is_the_one_shipped(self):
        import hashlib
        doc = ROOT.parent / "protocols/STREAMING_SUCCESS_RULE.md"
        if not doc.is_file():
            doc = ROOT / "protocols/STREAMING_SUCCESS_RULE.md"
        self.assertTrue(doc.is_file(), doc)
        got = hashlib.sha256(doc.read_bytes()).hexdigest()
        self.session({"r000": self.good()}, {"r000": "ok"},
                     {"r000": {"completion_tokens": 64}}, expected=1)
        _, out, _ = run(STREAM_TOOL, ["--session", str(self.d)])
        self.assertEqual(out["rule"]["sha256"], got,
                         "the tool quotes a digest that is not the shipped "
                         "rule document")
        from zerodiag import streaming as ST
        self.assertEqual(ST.SUCCESS_RULE["sha256"], got)


if __name__ == "__main__":
    unittest.main(verbosity=2)
