"""CPU tests for the streaming path: request, events, errors, evidence gate.

No GPU, no vLLM: a stdlib loopback server stands in for the HTTP/SSE contract,
so the client's request construction, event parsing, error handling and the
completion gate are all exercised for real.
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = pathlib.Path(__file__).resolve().parent.parent
ROOT = REPO / "studies" / "titan-1080"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from zerodiag import stream_runner as SR          # noqa: E402
from zerodiag import streaming as ST              # noqa: E402

MODE = {"kind": "ok", "bodies": []}


class Stub(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):                     # keep the test output clean
        pass

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200); self.send_header("Content-Length", "0")
            self.end_headers()
        else:
            self.send_response(404); self.send_header("Content-Length", "0")
            self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n)
        MODE["bodies"].append(json.loads(raw))
        if MODE["kind"] == "http_error":
            msg = b'{"error":"deliberate"}'
            self.send_response(500)
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers(); self.wfile.write(msg)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def sse(obj):
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
            self.wfile.flush()

        for i in range(3):
            sse({"choices": [{"text": f"c{i}", "finish_reason": None}]})
            time.sleep(0.01)
        if MODE["kind"] == "truncated":            # no finish_reason, no [DONE]
            self.wfile.write(b"")
            return
        if MODE["kind"] == "unparsed":
            self.wfile.write(b"data: {not json\n\n"); self.wfile.flush()
        sse({"choices": [{"text": "", "finish_reason": "length"}],
             "usage": {"completion_tokens": 64, "prompt_tokens": 256}})
        self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()


class StreamClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
        cls.port = cls.srv.server_address[1]
        cls.th = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.th.start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown(); cls.srv.server_close()

    def setUp(self):
        MODE["kind"] = "ok"; MODE["bodies"] = []
        self.events, self.status, self.usage = {}, {}, {}

    def run_one(self, rid="r000", ids=None, max_tokens=64):
        SR._stream_one(self.port, "stub-model", ids or list(range(1, 257)),
                       max_tokens, rid, self.events, self.status, self.usage,
                       time.monotonic())

    # -- request construction ------------------------------------------
    def test_request_is_token_ids_and_greedy(self):
        ids = list(range(1000, 1256))
        self.run_one(ids=ids)
        self.assertEqual(len(MODE["bodies"]), 1)
        body = MODE["bodies"][0]
        self.assertEqual(body["prompt"], ids,
                         "the prompt must be sent as token ids, not text")
        self.assertEqual(len(body["prompt"]), 256)
        self.assertEqual(body["max_tokens"], 64)
        self.assertEqual(body["temperature"], 0.0)
        self.assertTrue(body["stream"])
        self.assertTrue(body["stream_options"]["include_usage"])

    # -- event parsing --------------------------------------------------
    def test_events_are_parsed_and_timestamped_monotonically(self):
        self.run_one()
        evs = self.events["r000"]
        self.assertEqual(self.status["r000"], "ok")
        ts = [e["t"] for e in evs if e.get("t") is not None]
        self.assertEqual(ts, sorted(ts))
        self.assertEqual([e.get("text") for e in evs][:3], ["c0", "c1", "c2"])
        self.assertTrue(any(e.get("done") for e in evs))
        self.assertEqual([e.get("finish_reason") for e in evs
                          if e.get("finish_reason")], ["length"])
        self.assertEqual(self.usage["r000"]["completion_tokens"], 64)
        self.assertFalse(any("token_ids" in e for e in evs),
                         "token ids must not be invented when absent")

    def test_unparsable_payload_is_preserved_not_dropped(self):
        MODE["kind"] = "unparsed"
        self.run_one()
        self.assertTrue(any("unparsed" in e for e in self.events["r000"]))
        self.assertEqual(self.status["r000"], "ok")

    def test_chunks_are_not_counted_as_tokens(self):
        self.run_one()
        attr = ST.token_attribution(self.events["r000"])
        self.assertFalse(attr["per_token_timing_reportable"])
        s = ST.summarize_session(self.events, completion_tokens={"r000": 64},
                                 status=self.status)
        self.assertFalse(s["per_token_timing_reported"])
        self.assertIsNone(s["per_request"]["r000"]["per_token_timing"])
        self.assertNotEqual(s["per_request"]["r000"]["n_chunks"], 64)

    # -- error responses ------------------------------------------------
    def test_http_error_is_recorded_and_not_a_pass(self):
        MODE["kind"] = "http_error"
        self.run_one()
        self.assertEqual(self.status["r000"], "http_500")
        self.assertTrue(any("error" in e for e in self.events["r000"]))
        ev = self.build_ev()
        gate = SR.session_evidence_complete(ev)
        self.assertFalse(gate["ok"])
        self.assertTrue(any("frozen success rule" in m
                            for m in gate["missing"]), gate["missing"])

    # -- missing completion --------------------------------------------
    def test_missing_completion_marker_refuses(self):
        MODE["kind"] = "truncated"
        self.run_one()
        ev = self.build_ev()
        self.assertFalse(ev["completion_markers"]["any"])
        gate = SR.session_evidence_complete(ev)
        self.assertFalse(gate["ok"])
        self.assertTrue(any("completion marker" in m for m in gate["missing"]))

    def test_complete_session_passes_the_gate(self):
        self.run_one()
        ev = self.build_ev()
        gate = SR.session_evidence_complete(ev)
        self.assertTrue(gate["ok"], gate)
        self.assertEqual(gate["detail"]["counts"]["successful"], 1)
        self.assertEqual(gate["detail"]["requests_recorded"], 1)
        self.assertIn("one session", gate["scope"])
        cm = ev["completion_markers"]["per_request"]["r000"]
        self.assertEqual(cm["finish_reason"], "length")
        self.assertEqual(cm["http_status"], "ok")
        self.assertEqual(cm["server_usage"]["completion_tokens"], 64)
        self.assertFalse(cm["token_identities_supplied"])

    def test_zero_requests_cannot_pass(self):
        ev = self.build_ev(expected=1)
        gate = SR.session_evidence_complete(ev)
        self.assertFalse(gate["ok"])
        self.assertTrue(any("accounting is incomplete" in m
                            for m in gate["missing"]), gate["missing"])
        self.assertTrue(any("frozen success rule" in m
                            for m in gate["missing"]))

    def test_unpersisted_events_cannot_pass(self):
        self.run_one()
        ev = self.build_ev(); ev.pop("raw_events_file")
        self.assertIn("raw events were not persisted",
                      SR.session_evidence_complete(ev)["missing"])

    def build_ev(self, expected=None) -> dict:
        """Evidence in the shape the CURRENT gate reads: frozen accounting."""
        completion = {rid: (self.usage.get(rid) or {}).get("completion_tokens")
                      for rid in self.events}
        return {"summary": ST.summarize_session(self.events,
                                                completion_tokens=completion,
                                                status=self.status),
                "accounting": ST.account_session(
                    self.events, self.status, self.usage,
                    requests_expected=(expected
                                       if expected is not None
                                       else len(self.events)),
                    output_tokens=64),
                "raw_events_file": "raw_events.json",
                "frozen_workload": {"issued_as_frozen": True,
                                    "input_length_verified": True},
                "server": {}, "server_shutdown": {"confirmed_absent": True},
                "completion_markers": SR.completion_markers(
                    self.events, self.status, self.usage)}

    # -- health probe ---------------------------------------------------
    def test_health_probe_reports_a_live_server(self):
        class P:
            def poll(self): return None
        self.assertTrue(SR._wait_health(self.port, timeout_s=10, proc=P()))

    def test_health_probe_fails_fast_on_a_dead_process(self):
        class Dead:
            def poll(self): return 1
        t = time.monotonic()
        self.assertFalse(SR._wait_health(self.port, timeout_s=30, proc=Dead()))
        self.assertLess(time.monotonic() - t, 5.0)


class MetricsDir(unittest.TestCase):
    """The repaired Prometheus multiprocess directory contract."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self.tmp.name) / "prometheus_multiproc"

    def tearDown(self):
        self.tmp.cleanup()

    def test_fresh_private_directory_exists_and_is_empty(self):
        rec = SR.prepare_metrics_dir(self.base)
        self.assertTrue(self.base.is_dir())
        self.assertTrue(rec["created_fresh"])
        self.assertTrue(rec["is_empty_at_start"])
        self.assertTrue(rec["exists_before_server_start"])
        self.assertEqual(rec["env_var"], "PROMETHEUS_MULTIPROC_DIR")
        self.assertEqual(rec["pre_existing_entries_removed"], [])
        self.assertFalse(rec["shared_metrics_files_deleted"])

    def test_a_previous_attempts_db_files_are_not_carried_over(self):
        self.base.mkdir(parents=True)
        (self.base / "gauge_mostrecent_999.db").write_bytes(b"stale")
        rec = SR.prepare_metrics_dir(self.base)
        self.assertEqual(rec["pre_existing_entries_removed"],
                         ["gauge_mostrecent_999.db"])
        self.assertEqual(list(self.base.iterdir()), [])

    def test_directory_is_private_to_the_attempt(self):
        a = SR.prepare_metrics_dir(self.base)
        b = SR.prepare_metrics_dir(pathlib.Path(self.tmp.name) / "attempt2")
        self.assertNotEqual(a["dir"], b["dir"])

    def test_outcome_reports_whether_the_stack_used_it(self):
        rec = SR.prepare_metrics_dir(self.base)
        out = SR.metrics_dir_outcome(rec)
        self.assertFalse(out["used_by_server"])
        self.assertEqual(out["n_db_files"], 0)
        (self.base / "gauge_mostrecent_1.db").write_bytes(b"x")
        out = SR.metrics_dir_outcome(rec)
        self.assertTrue(out["used_by_server"])
        self.assertEqual(out["db_files_written"], ["gauge_mostrecent_1.db"])

    def test_installed_basis_is_recorded_from_the_installed_sources(self):
        rec = SR.prepare_metrics_dir(self.base)
        joined = " ".join(rec["installed_basis"])
        self.assertIn("api_server.py:149", joined)
        self.assertIn("values.py:95", joined)


if __name__ == "__main__":
    unittest.main(verbosity=2)
