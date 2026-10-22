"""Failure controls for the streaming path, against a real HTTP/SSE server.

Every control here is a way the previous attempts failed, or could fail, and
each one is checked against the FROZEN success rule
(`protocols/STREAMING_SUCCESS_RULE.md`) through the same
`zerodiag.streaming.classify_request` the runner and the published offline tool
use. None of them may come out as a successful streaming observation.

The client under test is the production `_stream_one`, and the request body is
the production `build_request_payload`. Nothing is re-implemented here.
"""
from __future__ import annotations

import json
import os
import pathlib
import socket
import subprocess
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = pathlib.Path(__file__).resolve().parent.parent
ROOT = REPO / "studies" / "titan-1080"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _zd_isolation as _isolation                  # noqa: E402
from zerodiag import stream_runner as SR            # noqa: E402
from zerodiag import streaming as ST                # noqa: E402

MODE = {"kind": "ok", "bodies": [], "usage_tokens": 64}
OUT = ST.PROTOCOL["output_tokens"]


class Stub(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        code = 200 if self.path == "/health" else 404
        self.send_response(code)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _sse(self, obj):
        self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
        self.wfile.flush()

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        MODE["bodies"].append(json.loads(self.rfile.read(n)))
        kind = MODE["kind"]
        if kind == "http_500":
            msg = b'{"error":"deliberate"}'
            self.send_response(500)
            self.send_header("Content-Length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return
        if kind == "empty_body_200":
            # exactly the previous failure: 200, no body at all
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        # No Content-Length and `Connection: close`, so the client reads the
        # event stream until EOF - the same shape a real SSE response has.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        if kind == "no_sse_events":
            return                               # 200, event-stream, no events
        for i in range(3):
            self._sse({"choices": [{"text": f"c{i}", "finish_reason": None}]})
            time.sleep(0.01)
        if kind == "error_after_headers":
            # an exception after the response started: the stream just stops
            self.close_connection = True
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            return
        if kind == "missing_completion":
            return                               # content, but no marker
        if kind == "malformed_events":
            self.wfile.write(b"data: {not json\n\n")
            self.wfile.flush()
        self._sse({"choices": [{"text": "", "finish_reason": "length"}],
                   "usage": {"completion_tokens": MODE["usage_tokens"],
                             "prompt_tokens": ST.PROTOCOL["prompt_len_tokens"]}})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
        cls.port = cls.srv.server_address[1]
        cls.th = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.th.start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        MODE.update(kind="ok", bodies=[], usage_tokens=OUT)
        self.events, self.status, self.usage = {}, {}, {}

    def issue(self, kind, rid="r000", n_ids=None):
        MODE["kind"] = kind
        ids = list(range(1000, 1000 + (n_ids or ST.PROTOCOL["prompt_len_tokens"])))
        SR._stream_one(self.port, ST.PROTOCOL["model"], ids, OUT, rid,
                       self.events, self.status, self.usage, time.monotonic())
        return ST.classify_request(self.events[rid], self.status.get(rid),
                                   self.usage.get(rid), output_tokens=OUT)

    def account(self, expected=1):
        return ST.account_session(self.events, self.status, self.usage,
                                  requests_expected=expected,
                                  output_tokens=OUT)

    # ---------------- the seven required failure controls --------------
    def test_control_http_200_with_an_empty_body_is_not_success(self):
        c = self.issue("empty_body_200")
        self.assertEqual(c["class"], "incomplete")
        self.assertEqual(c["http_status"], "ok", "the transport did return 200")
        self.assertEqual(c["n_events"], 0)
        self.assertFalse(c["checks"]["has_content_event"])
        self.assertEqual(self.account()["counts"]["successful"], 0)

    def test_control_no_sse_events_is_not_success(self):
        c = self.issue("no_sse_events")
        self.assertEqual(c["class"], "incomplete")
        self.assertEqual(c["n_content_events"], 0)
        self.assertEqual(self.account()["counts"]["successful"], 0)

    def test_control_error_after_headers_is_not_success(self):
        c = self.issue("error_after_headers")
        self.assertIn(c["class"], ("failed", "incomplete"))
        self.assertFalse(c["checks"]["has_completion_marker"])
        self.assertEqual(self.account()["counts"]["successful"], 0)

    def test_control_missing_completion_is_not_success(self):
        c = self.issue("missing_completion")
        self.assertEqual(c["class"], "incomplete")
        self.assertTrue(c["checks"]["has_content_event"],
                        "content arrived; only the completion marker is absent")
        self.assertFalse(c["checks"]["has_completion_marker"])
        self.assertEqual(self.account()["counts"]["successful"], 0)

    def test_control_malformed_events_are_preserved_not_silently_dropped(self):
        c = self.issue("malformed_events")
        self.assertEqual(c["class"], "successful",
                         "a malformed event beside valid content and a "
                         "completion marker does not void the observation")
        raw = self.events["r000"]
        self.assertTrue(any("unparsed" in e for e in raw),
                        "the unparsable payload must be retained verbatim")

    def test_control_only_malformed_events_is_not_success(self):
        rid = "rmal"
        self.events[rid] = [{"t": 0.1, "unparsed": "{not json"}]
        self.status[rid] = "ok"
        c = ST.classify_request(self.events[rid], "ok", None, output_tokens=OUT)
        self.assertEqual(c["class"], "incomplete")

    @unittest.skipUnless(_isolation.available(), "requires verified device-denying sandbox")
    def test_control_incorrect_actual_input_length_is_refused(self):
        """The runner refuses a wrong consumed length before issuing."""
        want = ST.PROTOCOL["prompt_len_tokens"]
        short = {"prompts": [list(range(want - 1))], "prompt_len": want - 1}
        with _tmpdir() as d:
            corpus = d / "corpus.json"
            corpus.write_text(json.dumps(short))
            rc, ev = _run_runner_cpu(d, corpus)
            self.assertEqual(rc, 90, ev)
            self.assertEqual(ev["stage"], "refused")
            self.assertIn("token", ev["reason"])

    @unittest.skipUnless(sys.platform == "linux", "requires Linux process groups; never spawn first on Windows")
    def test_control_a_server_that_leaves_a_descendant_is_confirmed_gone(self):
        code = ("import subprocess,sys,time;"
                "subprocess.Popen([sys.executable,'-c','import time;"
                "time.sleep(120)']);time.sleep(120)")
        p = subprocess.Popen([sys.executable, "-c", code],
                             start_new_session=True)
        pgid = os.getpgid(p.pid)
        # Wait for the descendant rather than guessing how long two Python
        # interpreters take to start under a loaded machine.
        deadline = time.monotonic() + 60.0
        before = SR.process_group_members(pgid)
        while len(before) < 2 and time.monotonic() < deadline:
            time.sleep(0.2)
            before = SR.process_group_members(pgid)
        self.assertGreaterEqual(len(before), 2,
                                "the control needs a real descendant")
        rec = SR.shutdown_server(p, grace_s=25.0)
        self.assertTrue(rec["descendants_before"])
        self.assertEqual(rec["members_after"], [])
        self.assertTrue(rec["confirmed_absent"])

    def test_control_an_unconfirmed_shutdown_blocks_the_gate(self):
        ev = _complete_ev()
        ev["server_shutdown"] = {"confirmed_absent": False,
                                 "members_after": [{"pid": 1, "state": "S",
                                                    "cmdline": "engine"}]}
        g = SR.session_evidence_complete(ev)
        self.assertFalse(g["ok"])
        self.assertTrue(any("not confirmed absent" in m for m in g["missing"]))

    def test_control_a_server_that_exited_early_blocks_the_gate(self):
        ev = _complete_ev()
        ev["server"]["exited_before_session"] = True
        g = SR.session_evidence_complete(ev)
        self.assertFalse(g["ok"])
        self.assertTrue(any("exited before the session" in m
                            for m in g["missing"]))

    # ---------------- accounting is complete, nothing discarded --------
    def test_every_issued_request_appears_in_exactly_one_class(self):
        self.issue("ok", "r000")
        self.issue("empty_body_200", "r001")
        self.issue("http_500", "r002")
        MODE["usage_tokens"] = 12
        self.issue("ok", "r003")
        acc = self.account(expected=4)
        self.assertEqual(acc["requests_recorded"], 4)
        self.assertEqual(acc["classes_sum"], 4)
        self.assertTrue(acc["accounting_complete"])
        self.assertEqual(acc["counts"], {"successful": 1, "length_mismatch": 1,
                                         "incomplete": 1, "failed": 1})
        self.assertEqual(acc["per_request"]["r003"]["class"], "length_mismatch")
        self.assertEqual(
            acc["per_request"]["r003"]["reported_completion_tokens"], 12)

    def test_a_missing_request_makes_the_accounting_incomplete(self):
        self.issue("ok", "r000")
        acc = self.account(expected=100)
        self.assertFalse(acc["accounting_complete"])
        ev = _complete_ev()
        ev["accounting"] = acc
        g = SR.session_evidence_complete(ev)
        self.assertFalse(g["ok"])
        self.assertTrue(any("accounting is incomplete" in m
                            for m in g["missing"]))

    def test_a_reduced_workload_blocks_the_gate(self):
        ev = _complete_ev()
        ev["frozen_workload"]["issued_as_frozen"] = False
        g = SR.session_evidence_complete(ev)
        self.assertFalse(g["ok"])
        self.assertTrue(any("does not match the frozen" in m
                            for m in g["missing"]))

    # ---------------- the valid session passes -------------------------
    def test_a_valid_session_satisfies_the_frozen_rule(self):
        c = self.issue("ok")
        self.assertEqual(c["class"], "successful")
        self.assertTrue(all(c["checks"].values()))
        self.assertEqual(c["finish_reason"], "length")
        self.assertTrue(c["done_sentinel"])
        self.assertEqual(c["reported_completion_tokens"], OUT)
        self.assertTrue(c["length_matches_protocol"])
        ev = _complete_ev()
        ev["accounting"] = self.account()
        self.assertTrue(SR.session_evidence_complete(ev)["ok"])

    # ---------------- request construction is the production one -------
    def test_the_issued_body_is_the_production_payload(self):
        ids = list(range(1000, 1000 + ST.PROTOCOL["prompt_len_tokens"]))
        self.issue("ok", n_ids=len(ids))
        body = MODE["bodies"][-1]
        self.assertEqual(body, SR.build_request_payload(
            ST.PROTOCOL["model"], ids, OUT))
        self.assertEqual(len(body["prompt"]), ST.PROTOCOL["prompt_len_tokens"])
        self.assertEqual(body["max_tokens"], OUT)
        self.assertEqual(body["temperature"], 0.0)
        for f in SR.CONSTRAINT_FIELDS:
            self.assertNotIn(f, body)

    # ---------------- chunks are not tokens ----------------------------
    def test_no_token_timing_is_invented_from_chunks(self):
        self.issue("ok")
        s = ST.summarize_session(self.events,
                                 completion_tokens={"r000": OUT},
                                 status=self.status)
        self.assertFalse(s["per_token_timing_reported"])
        self.assertIsNone(s["per_request"]["r000"]["per_token_timing"])
        self.assertNotEqual(s["per_request"]["r000"]["n_chunks"], OUT)
        c = ST.classify_request(self.events["r000"], "ok",
                                self.usage.get("r000"), output_tokens=OUT)
        self.assertFalse(c["token_identity_in_payload"])


def _complete_ev() -> dict:
    """A minimal evidence dict that the gate accepts, for negative controls."""
    rid = "r000"
    events = {rid: [{"t": 0.1, "text": "a"},
                    {"t": 0.2, "text": "", "finish_reason": "length"},
                    {"t": 0.21, "done": True}]}
    return {"accounting": ST.account_session(events, {rid: "ok"},
                                             {rid: {"completion_tokens": OUT}},
                                             requests_expected=1,
                                             output_tokens=OUT),
            "raw_events_file": "raw_events.json",
            "frozen_workload": {"issued_as_frozen": True,
                                "input_length_verified": True},
            "server": {}, "server_shutdown": {"confirmed_absent": True}}


class _tmpdir:
    def __enter__(self):
        import tempfile
        _isolation.SANDBOX_TMP.mkdir(parents=True, exist_ok=True)
        self._t = tempfile.TemporaryDirectory(dir=str(_isolation.SANDBOX_TMP))
        return pathlib.Path(self._t.name)

    def __exit__(self, *a):
        self._t.cleanup()


def _run_runner_cpu(d: pathlib.Path, corpus: pathlib.Path):
    """Run the REAL runner entry point inside the device-denying sandbox.

    No CUDA device exists there, so the runner must refuse - and it must still
    persist its evidence file. That exercises production argument parsing, the
    corpus checks and evidence persistence on the real path.
    """
    out = d / "run"
    argv = _isolation.wrap(
        [sys.executable, "-I", str(ROOT / "zerodiag/stream_runner.py"),
         "--output", str(out), "--receipt", str(d / "receipt.json"),
         "--case-id", "S_test", "--rank", "0", "--world-size", "1",
         "--corpus", str(corpus), "--device-index", "0",
         "--max-cpus", "2", "--memory-budget-gib", "2"],
        writable=[str(d), str(_isolation.SANDBOX_TMP)],
        extra_env={"PYTHONPATH": str(ROOT)})
    r = subprocess.run(argv, capture_output=True, text=True, timeout=900)
    ep = out / "streaming_evidence.json"
    ev = json.loads(ep.read_text()) if ep.is_file() else {"stderr": r.stderr[-400:]}
    return r.returncode, ev


class ProductionPathOnCpu(unittest.TestCase):
    """Real parsing, real dispatch, real persistence - inside the sandbox."""

    @classmethod
    def setUpClass(cls):
        if not _isolation.available():
            raise AssertionError("device-denying isolation is unavailable: "
                                 + str(_isolation.probe().get("reason")))

    def test_the_runner_refuses_without_a_device_and_still_persists_evidence(self):
        want = ST.PROTOCOL["prompt_len_tokens"]
        good = {"prompts": [list(range(want))], "prompt_len": want}
        with _tmpdir() as d:
            corpus = d / "corpus.json"
            corpus.write_text(json.dumps(good))
            rc, ev = _run_runner_cpu(d, corpus)
            self.assertEqual(rc, 90, ev)
            self.assertEqual(ev["stage"], "refused")
            self.assertIn("CUDA", ev["reason"])
            self.assertIn("protocol", ev)
            self.assertEqual(ev["protocol"]["prompt_len_tokens"], want)
            self.assertIsNotNone(ev.get("memory_monitor"))

    def test_the_runner_refuses_a_non_singleton_world(self):
        with _tmpdir() as d:
            argv = _isolation.wrap(
                [sys.executable, "-I", str(ROOT / "zerodiag/stream_runner.py"),
                 "--output", str(d / "o"), "--receipt", str(d / "r.json"),
                 "--case-id", "S", "--rank", "0", "--world-size", "2",
                 "--corpus", str(d / "c.json")],
                writable=[str(d), str(_isolation.SANDBOX_TMP)],
                extra_env={"PYTHONPATH": str(ROOT)})
            r = subprocess.run(argv, capture_output=True, text=True,
                               timeout=600)
            self.assertEqual(r.returncode, 95)
            self.assertIn("world size one", r.stderr)

    def test_the_launcher_dispatches_streaming_to_this_runner(self):
        src = (ROOT / "zerodiag/cli.py").read_text()
        self.assertIn('"streaming": "zerodiag/stream_runner.py"', src)
        self.assertIn('"--device-index", str(rec["device_indices"][r])', src)

    def test_the_metrics_directory_is_fresh_and_private_per_attempt(self):
        with _tmpdir() as d:
            a = SR.prepare_metrics_dir(d / "attempt1/prometheus_multiproc")
            (pathlib.Path(a["dir"]) / "gauge_mostrecent_1.db").write_bytes(b"x")
            b = SR.prepare_metrics_dir(d / "attempt2/prometheus_multiproc")
            self.assertNotEqual(a["dir"], b["dir"])
            self.assertTrue(b["is_empty_at_start"])
            again = SR.prepare_metrics_dir(a["dir"])
            self.assertEqual(again["pre_existing_entries_removed"],
                             ["gauge_mostrecent_1.db"])
            self.assertFalse(again["shared_metrics_files_deleted"])


class BindAddressAndAttribution(unittest.TestCase):
    """What the runner measures, and what it deliberately does not claim."""

    def test_the_bind_address_is_measured_and_discriminates(self):
        import socket
        a = socket.socket()
        a.bind(("127.0.0.1", 0))
        a.listen(1)
        b = socket.socket()
        b.bind(("", 0))
        b.listen(1)
        try:
            lo = SR.listening_bind_addresses(a.getsockname()[1])
            allif = SR.listening_bind_addresses(b.getsockname()[1])
            self.assertEqual(lo["addresses"], ["127.0.0.1"])
            self.assertTrue(lo["loopback_only"])
            self.assertIn("0.0.0.0", allif["addresses"])
            self.assertFalse(allif["loopback_only"])
            self.assertIn("NOT restricted to loopback", allif["note"])
            self.assertIn("api_server.py:544", allif["installed_basis"])
        finally:
            a.close()
            b.close()

    def test_a_closed_port_reports_no_listener_rather_than_claiming_one(self):
        import socket
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        rec = SR.listening_bind_addresses(port)
        self.assertEqual(rec["addresses"], [])
        self.assertIsNotNone(rec.get("note"))
        self.assertFalse(rec["loopback_only"])

    def test_the_runner_does_not_assert_loopback_without_measuring(self):
        src = (ROOT / "zerodiag/stream_runner.py").read_text()
        self.assertNotIn('"loopback_only": True', src)
        self.assertIn('ev["server"]["bind"] = listening_bind_addresses(port)',
                      src)

    def test_the_server_is_in_its_own_group_and_the_monitor_is_pgid_scoped(self):
        """The cleanup guarantee and the RSS attribution limit are linked."""
        src = (ROOT / "zerodiag/stream_runner.py").read_text()
        self.assertIn("start_new_session=True", src)
        mon = (ROOT / "zerodiag/containment.py").read_text()
        self.assertIn("pgid", mon,
                      "the monitor attributes RSS by process group, so a "
                      "server in its own group is outside it; that limitation "
                      "is reported rather than papered over")

    def test_the_corrections_file_states_the_rss_attribution_limit(self):
        import json as _json
        p = next((c for c in (
            ROOT.parent / "out/batch/final_attempts/attempt1/run/rank0"
            / "evidence_corrections.json",
            ROOT.parent / "results/diagnostics/streaming_session"
            / "evidence_corrections.json",
            ROOT.parent.parent / "results/diagnostics/streaming_session"
            / "evidence_corrections.json") if c.is_file()),
            ROOT / "missing")
        if not p.is_file():
            self.skipTest("no executed session in this tree")
        ids = {c["id"] for c in _json.loads(p.read_text())["corrections"]}
        self.assertIn("CORR_STREAM_001", ids)
        self.assertIn("CORR_STREAM_003", ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
