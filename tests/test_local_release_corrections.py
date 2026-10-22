"""Local release regressions; no Torch, CUDA, server or model is imported."""
import copy
import importlib.util
import json
import pathlib
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "studies/titan-1080"))
from zerodiag import containment, stream_server, streaming, ownership

spec = importlib.util.spec_from_file_location("stream_recompute", ROOT / "tools/recompute_streaming_metrics.py")
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)
SESSION = ROOT / "results/diagnostics/streaming_session"


class Evidence(unittest.TestCase):
    def setUp(self):
        self.raw, self.ev, _ = tool.load(str(SESSION))

    def refuse(self, change):
        change(self.raw, self.ev)
        with self.assertRaises(tool.Refusal):
            tool.recompute(self.raw, self.ev, 100)

    def test_retained_session_is_observed_without_inventing_latency(self):
        r = tool.recompute(self.raw, self.ev, 100)
        self.assertEqual(r["accounting"]["counts"]["successful"], 100)
        c = r["chunk_measurements"]
        self.assertIsNone(c["first_text_latency_s"])
        self.assertFalse(c["request_latency_available"])
        self.assertEqual(c["session_relative_first_event_s"]["median"], 6.753719)
        self.assertEqual(c["inter_event_gap_s"]["n"], 6500)

    def test_nonfinite_clocks_refuse(self):
        for value in (float("nan"), float("inf"), -float("inf"), -0.1, True, "0"):
            with self.subTest(value=value):
                raw = copy.deepcopy(self.raw)
                raw["events"]["r000"][0]["t"] = value
                with self.assertRaises(tool.Refusal):
                    tool.recompute(raw, self.ev, 100)

    def test_backwards_clock_refuses_instead_of_sorting(self):
        def change(raw, ev):
            a = raw["events"]["r000"]
            a[0], a[1] = a[1], a[0]
        self.refuse(change)

    def test_wrong_prompt_length_refuses(self):
        self.refuse(lambda raw, ev: raw["usage"]["r000"].update(prompt_tokens=255))

    def test_malformed_token_counts_refuse(self):
        for value in (True, -1, 64.0, "64", None):
            raw = copy.deepcopy(self.raw)
            raw["usage"]["r000"]["completion_tokens"] = value
            with self.assertRaises(tool.Refusal):
                tool.recompute(raw, self.ev, 100)

    def test_unknown_or_missing_ids_refuse(self):
        for target in ("status", "usage"):
            raw = copy.deepcopy(self.raw)
            raw[target]["alien"] = "ok" if target == "status" else {}
            with self.assertRaises(tool.Refusal):
                tool.recompute(raw, self.ev, 100)
        self.refuse(lambda raw, ev: raw["status"].pop("r000"))

    def test_nondictionary_event_is_not_discarded(self):
        self.refuse(lambda raw, ev: raw["events"]["r000"].append("bad"))

    def test_missing_protocol_or_false_workload_refuses(self):
        self.refuse(lambda raw, ev: ev.pop("protocol"))
        self.setUp()
        self.refuse(lambda raw, ev: ev["frozen_workload"].update(input_length_verified=False))

    def test_expected_override_cannot_change_the_protocol(self):
        with self.assertRaises(tool.Refusal):
            tool.recompute(self.raw, self.ev, 1)

    def test_true_latency_requires_an_observed_request_start(self):
        self.raw["request_timing"] = {
            rid: {"request_start_s": evs[0]["t"] - 0.01}
            for rid, evs in self.raw["events"].items()}
        r = tool.recompute(self.raw, self.ev, 100)
        self.assertAlmostEqual(r["chunk_measurements"]["first_text_latency_s"]["median"], .01)
        self.raw["request_timing"]["r000"]["request_start_s"] = 100
        with self.assertRaises(tool.Refusal):
            tool.recompute(self.raw, self.ev, 100)

    def test_cli_refusals_are_clean_and_json_is_finite(self):
        for change in ("nan", "metadata", "accounting"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                raw, ev = copy.deepcopy(self.raw), copy.deepcopy(self.ev)
                if change == "nan":
                    raw["events"]["r000"][0]["t"] = float("nan")
                if change == "accounting":
                    ev["accounting"]["counts"]["successful"] = 99
                d = pathlib.Path(tmp)
                (d / "raw_events.json").write_text(json.dumps(raw))
                if change != "metadata":
                    (d / "streaming_evidence.json").write_text(json.dumps(ev))
                p = subprocess.run([sys.executable, "-I", "-S", str(ROOT / "tools/recompute_streaming_metrics.py"),
                                    "--session", tmp], capture_output=True, text=True, timeout=30)
                self.assertEqual(p.returncode, 95, p.stdout)
                self.assertEqual(p.stderr, "")
                self.assertEqual(json.loads(p.stdout)["verdict"], "REFUSED")
                # A quoted diagnostic may name NaN; an unquoted numeric JSON
                # NaN is forbidden. Check the actual JSON grammar, not spelling.
                json.loads(p.stdout, parse_constant=lambda s: self.fail("non-finite numeric JSON"))

    def test_even_sample_median_is_average_of_middle_pair(self):
        self.assertEqual(tool._stats([1, 3])["median"], 2)

    def test_library_does_not_label_session_origin_as_latency(self):
        r = streaming.summarize_session({"r": [{"t": 20, "text": "x"}]},
                                         completion_tokens={}, status={"r": "ok"})
        self.assertIsNone(r["per_request"]["r"]["first_text_latency_s"])
        self.assertEqual(r["per_request"]["r"]["session_relative_first_event_s"], 20)


class FutureGuards(unittest.TestCase):
    def test_real_socket_wildcard_is_restricted_before_bind(self):
        original = socket.socket
        with original() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        with stream_server.loopback_api_socket(port):
            with socket.socket() as s:
                s.bind(("", port))
                self.assertEqual(s.getsockname()[0], "127.0.0.1")
        self.assertIs(socket.socket, original)

    def test_nonloopback_target_is_refused(self):
        with stream_server.loopback_api_socket(19999), socket.socket() as s:
            with self.assertRaises(ValueError):
                s.bind(("192.0.2.1", 19999))

    def test_socket_restored_after_exception(self):
        original = socket.socket
        with self.assertRaises(RuntimeError):
            with stream_server.loopback_api_socket(19999):
                raise RuntimeError("fixture")
        self.assertIs(socket.socket, original)

    def test_aggregate_sample_includes_separate_server_group_once(self):
        b = containment.budget_arithmetic()
        self.assertTrue(b["planning_model_only"])
        self.assertTrue(b["not_an_admission_criterion"])
        self.assertEqual(b["host_resident_bytes"]["update_oracle_workspace_chunked"], 9 * 4194304 * 4)
        self.assertEqual(b["host_resident_bytes"]["state_hashing_workspace"], 16 * 1048576)
        def rss(pids):
            self.assertEqual(pids, [11, 22, 33])
            return {"total_bytes": 123, "unreadable": []}
        with mock.patch.object(containment, "owned_pids", return_value=[11]), \
             mock.patch.object(containment, "owned_rss_bytes", side_effect=rss):
            m = containment.MemoryMonitor(11, budget_bytes=100, additional_pids=lambda: [11, 22, 33])
            self.assertEqual(m.sample()["bytes"], 123)
            self.assertTrue(m.report()["breached"])
            self.assertIn("across sessions", m.report()["attribution"])

    def test_unreadable_owned_memory_refuses(self):
        with mock.patch.object(containment, "owned_pids", return_value=[11]), \
             mock.patch.object(containment, "owned_rss_bytes", return_value={"total_bytes": 0, "unreadable": [11]}):
            m = containment.MemoryMonitor(11, budget_bytes=100)
            with self.assertRaises(containment.ContainmentRefusal):
                m.sample()

    def test_server_ledger_cannot_register_client_group(self):
        with mock.patch.object(ownership.os, "getuid", return_value=1000, create=True), \
             mock.patch.object(ownership.os, "getpgid", return_value=123, create=True):
            own = ownership.JobOwnership("fixture-token", excluded_pgids=(123,))
            with self.assertRaises(ValueError):
                own.register(0, 456)

    def test_future_runner_uses_bind_gate_and_outer_job_token(self):
        from zerodiag import stream_runner
        argv = stream_runner.build_server_argv("python", model="m", revision="r", host="127.0.0.1", port=1234)
        self.assertTrue(argv[2].endswith("stream_server.py"))
        text = pathlib.Path(stream_runner.__file__).read_text()
        self.assertIn('if ev["server"]["loopback_only"] is not True:', text)
        self.assertIn("additional_pids=server_ownership.owned_pids", text)
        self.assertIn("os.environ.get(JOB_TOKEN_ENV)", text)


if __name__ == "__main__":
    unittest.main()
