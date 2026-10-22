"""The shipped numbers must equal what the shipped tools recompute.

This is the anti-drift guard for the published claims: it recomputes the
reference verdict and the streaming metrics from the evidence in THIS
repository, using the published commands, and compares them against the
numbers recorded in the ledger, the comparison files and the documents.

Standard library only. No GPU, no model cache, no third-party package: the
tools are invoked with `-I -S`, so site-packages is not on the path.
"""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
import unittest

REPO = pathlib.Path(__file__).resolve().parent.parent
EV = REPO / "results/diagnostics/closing"
REF_A = EV / "reference/TITAN_RTX_1"
REF_B = EV / "reference/GTX_1080Ti_1"
SESSION = REPO / "results/diagnostics/streaming_session"
OLD_SESSIONS = [EV / "streaming/attempt1", EV / "streaming/attempt2"]


def run(tool, args):
    p = subprocess.run([sys.executable, "-I", "-S",
                        str(REPO / "tools" / tool)] + args,
                       capture_output=True, text=True, timeout=900)
    return p.returncode, (json.loads(p.stdout) if p.stdout.strip().startswith("{")
                          else None), p.stderr


class ShippedEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for p in (REF_A, REF_B, SESSION):
            if not p.is_dir():
                raise AssertionError(f"shipped evidence is missing: {p}")
        cls.rc_ref, cls.ref, cls.err_ref = run(
            "recompute_reference_verdict.py",
            ["--a", str(REF_A), "--b", str(REF_B)])
        cls.rc_st, cls.st, cls.err_st = run(
            "recompute_streaming_metrics.py",
            ["--session", str(SESSION), "--expected", "100"])
        cls.ledger = json.loads(
            (REPO / "results/diagnostics/context_ledger.json").read_text())
        cls.recorded = json.loads(
            (EV / "reference_comparison.json").read_text())

    # -- the published command works on the published evidence ---------
    def test_the_reference_command_recomputes_a_verdict(self):
        self.assertIn(self.rc_ref, (0, 2), self.err_ref)
        self.assertIn(self.ref["logits"]["verdict"], ("PASS", "FAIL"))
        self.assertTrue(self.ref["inputs"]["matched"],
                        "the inputs must match or the comparison is not a "
                        "controlled one")

    def test_the_recomputed_verdict_matches_the_recorded_one(self):
        a, b = self.ref["logits"], self.recorded["logits"]
        for k in ("verdict", "u", "magnitude_scale_M", "atol", "rtol_term",
                  "tolerance", "max_abs_error", "failing_elements",
                  "total_elements", "digests_identical"):
            self.assertEqual(a[k], b[k], f"{k} drifted")
        self.assertEqual(a["worst_witness"]["flat_index"],
                         b["worst_witness"]["flat_index"])

    def test_the_ledger_matches_the_recomputation(self):
        led = self.ledger["reference_comparison"]
        a = self.ref["logits"]
        self.assertTrue(led["evaluated"])
        for k in ("verdict", "tolerance", "max_abs_error", "failing_elements",
                  "total_elements", "u", "magnitude_scale_M"):
            self.assertEqual(led[k], a[k], f"ledger {k} drifted")
        self.assertFalse(led["threshold_widened"])

    def test_the_readme_quotes_the_recomputed_numbers(self):
        txt = (REPO / "README.md").read_text()
        a = self.ref["logits"]
        self.assertIn(f"{a['tolerance']:.6e}", txt)
        self.assertIn(f"{a['max_abs_error']:.6e}", txt)
        self.assertIn(f"**{a['verdict']}**", txt)

    def test_the_criterion_document_is_shipped_and_matches(self):
        doc = REPO / ("studies/titan-1080/protocols/"
                      "REFERENCE_CRITERION_CLARIFICATION.md")
        self.assertTrue(doc.is_file(), doc)
        txt = doc.read_text()
        self.assertIn("64u", txt)
        self.assertIn("finfo(dtype).eps / 2", txt)
        self.assertIn("maximum", txt.lower())
        self.assertEqual(self.ref["logits"]["criterion"],
                         "titan_reference_logit_criterion_v1")

    # -- integrity of the shipped arrays -------------------------------
    def test_every_shipped_array_checksum_verifies(self):
        import hashlib
        n = 0
        for base in (REF_A, REF_B):
            for sc in sorted((base / "arrays").glob("*.json")):
                rec = json.loads(sc.read_text())
                payload = base / "arrays" / rec["file"]
                raw = payload.read_bytes()
                self.assertEqual(len(raw), rec["n_bytes"], payload)
                self.assertEqual(hashlib.sha256(raw).hexdigest(), rec["sha256"],
                                 payload)
                n += 1
        self.assertGreaterEqual(n, 8, "both families' four arrays must ship")

    def test_the_shipped_arrays_are_not_pickles(self):
        for base in (REF_A, REF_B):
            for p in sorted((base / "arrays").glob("*.bin")):
                head = p.read_bytes()[:2]
                self.assertNotEqual(head[:1], b"\x80", p)
                self.assertNotIn(b"numpy", p.read_bytes()[:64], p)

    # -- the streaming session, recomputed from the shipped events ------
    def test_the_streaming_command_recomputes_the_session(self):
        self.assertEqual(self.rc_st, 0, self.err_st)
        self.assertEqual(self.st["verdict"], "SESSION_OBSERVED")
        acc = self.st["accounting"]
        self.assertTrue(acc["accounting_complete"])
        self.assertEqual(acc["requests_recorded"], acc["requests_expected"])
        self.assertEqual(acc["classes_sum"], acc["requests_recorded"])
        self.assertGreater(acc["counts"]["successful"], 0)

    def test_the_recomputed_counts_match_the_ledger(self):
        acc = self.st["accounting"]
        led = self.ledger["streaming_outcome"]
        self.assertTrue(led["closed"])
        self.assertEqual(led["accounting"]["counts"], acc["counts"])
        self.assertEqual(led["requests_recorded"], acc["requests_recorded"])
        self.assertEqual(led["requests_expected"], acc["requests_expected"])

    def test_the_readme_quotes_the_recomputed_request_counts(self):
        txt = (REPO / "README.md").read_text()
        acc = self.st["accounting"]
        n = acc["counts"]["successful"]
        self.assertIn(f"{n} of {acc['requests_recorded']}", txt)

    def test_the_success_rule_is_shipped_and_its_digest_matches(self):
        import hashlib
        doc = REPO / ("studies/titan-1080/protocols/"
                      "STREAMING_SUCCESS_RULE.md")
        self.assertTrue(doc.is_file(), doc)
        self.assertEqual(self.st["rule"]["sha256"],
                         hashlib.sha256(doc.read_bytes()).hexdigest())
        self.assertTrue(self.st["rule"]["http_200_alone_is_not_success"])

    def test_the_recorded_corrections_are_shipped_and_surfaced(self):
        c = self.st["evidence_corrections"]
        self.assertTrue(c["present"], "the corrections must travel with the "
                                      "evidence they correct")
        ids = {x["id"] for x in c["corrections"]}
        self.assertIn("CORR_STREAM_001", ids)
        self.assertTrue((SESSION / "evidence_corrections.json").is_file())

    def test_no_token_level_timing_is_published(self):
        self.assertFalse(self.st["token_level"]["reported"])
        for name in ("README.md", "results/diagnostics/numerical_results.md",
                     "results/diagnostics/streaming_session.md"):
            f = REPO / name
            if not f.is_file():
                continue
            txt = f.read_text()
            for pat in (r"inter-token latency of", r"time to first token of",
                        r"per-token latency"):
                self.assertEqual(re.findall(pat, txt), [], f"{name}: {pat}")

    def test_the_earlier_failed_sessions_are_still_shipped_and_still_refuse(self):
        for old in OLD_SESSIONS:
            if not (old / "raw_events.json").is_file():
                continue
            rc, out, _ = run("recompute_streaming_metrics.py",
                             ["--session", str(old), "--expected", "100"])
            self.assertIn(rc, (2, 95), f"{old} unexpectedly passed")
            if out and "accounting" in out:
                self.assertEqual(out["accounting"]["counts"]["successful"], 0)

    # -- refusal behaviour on damaged evidence -------------------------
    def test_a_missing_array_refuses_rather_than_reporting_a_number(self):
        import shutil
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            a = pathlib.Path(tmp) / "a"
            shutil.copytree(REF_A, a)
            (a / "arrays/final_logits.bin").unlink()
            rc, out, err = run("recompute_reference_verdict.py",
                               ["--a", str(a), "--b", str(REF_B)])
            self.assertEqual(rc, 95)
            self.assertEqual(out["logits"]["verdict"], "REFUSED")
            self.assertNotIn("max_abs_error", out["logits"])

    def test_a_corrupted_array_refuses(self):
        import shutil
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            a = pathlib.Path(tmp) / "a"
            shutil.copytree(REF_A, a)
            p = a / "arrays/final_logits.bin"
            raw = bytearray(p.read_bytes()); raw[0] ^= 0xFF
            p.write_bytes(bytes(raw))
            rc, out, _ = run("recompute_reference_verdict.py",
                             ["--a", str(a), "--b", str(REF_B)])
            self.assertEqual(rc, 95)
            self.assertIn("checksum mismatch", out["logits"]["reason"])

    def test_missing_streaming_evidence_refuses(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            rc, out, _ = run("recompute_streaming_metrics.py",
                             ["--session", tmp])
            self.assertEqual(rc, 95)
            self.assertEqual(out["verdict"], "REFUSED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
