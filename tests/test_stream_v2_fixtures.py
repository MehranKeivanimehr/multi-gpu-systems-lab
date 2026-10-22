"""Fixtures through the ACTUAL prospective driver and the downstream certifier.

No server, no GPU, no device query. Covers missing, empty and multi-token
chunks, failed and truncated responses, and verifies the certifier downstream
reaches the same narrowed conclusion.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


def row(*, n_chunks=64, completion=64, status=200, done=True,
        finish="stop", drop=(), start=0.08, gap=0.006):
    r = {"prompt_token_ids_sha256": "a" * 64,
         "http_status": status,
         "finish_reason": finish,
         "usage_prompt_tokens": 256,
         "usage_completion_tokens": completion,
         "raw_sse_frames": [{"t": start + i * gap, "data": "{...}"}
                            for i in range(n_chunks)],
         "done_sentinel_seen": done,
         "chunk_arrivals_s": [start + i * gap for i in range(n_chunks)]}
    for k in drop:
        r.pop(k, None)
    return r


class DriverVerdictFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = load("sv2", "studies/titan-1080/bench/infer_vllm_stream_v2.py")

    def test_the_adopted_protocol_is_the_dedicated_256_64(self):
        self.assertEqual(self.d.PROTOCOL["prompt_len"], 256)
        self.assertEqual(self.d.PROTOCOL["output_len"], 64)
        self.assertIn("run_plan.json", self.d.PROTOCOL["source"])
        self.assertIn("128", self.d.PROTOCOL["shared_config_conflict"])

    def test_a_complete_one_token_per_chunk_request_supports_per_token(self):
        v = self.d.verdict_for_request(row(n_chunks=64, completion=64))
        self.assertIn("per_token_latency", v["usable_for"])
        self.assertEqual(v["tokens_per_chunk"], 1)

    def test_multi_token_chunks_refuse_per_token_but_keep_chunk_arrival(self):
        v = self.d.verdict_for_request(row(n_chunks=16, completion=64))
        self.assertIn("chunk_arrival", v["usable_for"])
        self.assertIn("per_token_latency", v["refused_for"])
        self.assertEqual(v["tokens_per_chunk"], 4.0)
        self.assertIn("shared an arrival", v["reason"])

    def test_empty_chunk_list_refuses_everything(self):
        v = self.d.verdict_for_request(row(n_chunks=0, completion=0))
        self.assertIn("per_token_latency", v["refused_for"])

    def test_each_missing_evidence_field_refuses(self):
        for k in self.d.REQUIRED_PER_REQUEST:
            with self.subTest(field=k):
                v = self.d.verdict_for_request(row(drop=(k,)))
                self.assertIn(k, v["missing_evidence"])
                self.assertIn("chunk_arrival", v["refused_for"])

    def test_a_failed_response_refuses(self):
        v = self.d.verdict_for_request(row(status=503))
        self.assertIn("chunk_arrival", v["refused_for"])
        self.assertIn("503", v["reason"])

    def test_a_truncated_stream_without_done_is_partial_only(self):
        v = self.d.verdict_for_request(row(done=False))
        self.assertIn("chunk_arrival_partial", v["usable_for"])
        self.assertIn("completion", v["refused_for"])
        self.assertIn("truncation not excluded", v["reason"])

    def test_a_length_finish_reason_is_accepted_but_an_error_one_is_not(self):
        self.assertIn("completion",
                      self.d.verdict_for_request(row(finish="length"))["usable_for"])
        v = self.d.verdict_for_request(row(finish="error"))
        self.assertIn("completion", v["refused_for"])

    def test_arm_summary_names_gaps_as_chunk_gaps(self):
        s = self.d.summarise_arm([row() for _ in range(4)])
        self.assertIn("chunk_gap_median_s", s)
        self.assertNotIn("inter_token_latency_s", s)
        self.assertIn("not per-token latencies", s["naming"])

    def test_one_bad_request_blocks_the_arm_level_per_token_claim(self):
        rows = [row() for _ in range(5)] + [row(n_chunks=16, completion=64)]
        s = self.d.summarise_arm(rows)
        self.assertFalse(s["per_token_latency_claimable"])
        self.assertIsNotNone(s["per_token_reason"])

    def test_a_clean_arm_may_claim_per_token(self):
        s = self.d.summarise_arm([row() for _ in range(6)])
        self.assertTrue(s["per_token_latency_claimable"])

    def test_tail_percentiles_are_never_reported(self):
        s = self.d.summarise_arm([row() for _ in range(100)])
        self.assertFalse(s["tail_percentiles_reported"])

    def test_prompt_digest_is_deterministic_and_id_sensitive(self):
        a = self.d.prompt_digest([1, 2, 3]); b = self.d.prompt_digest([1, 2, 3])
        c = self.d.prompt_digest([1, 2, 4])
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertEqual(len(a), 64)


class DownstreamCertifierFixtures(unittest.TestCase):
    """The same fixtures must narrow identically in the certifier."""

    @classmethod
    def setUpClass(cls):
        cls.tc = load("tc_sv2", "tools/titan_certify.py")
        cls.dir = pathlib.Path(tempfile.mkdtemp())
        cls._n = 0

    def certify_streaming(self, rows, **cfg_over):
        type(self)._n += 1
        tel_src = load("tcs", "tools/titan_certify.py")
        n = len(rows)
        wall = 11.6
        rec = {"status": "ok",
               "config": {"engine": "vllm_loopback_streaming", "requests": n,
                          "concurrency": 4, "prompt_len_requested": 256,
                          "prompt_len_tokens_actual": 256, "output_len": 64,
                          "engine_pid": 1234, "enable_prefix_caching": False,
                          "guided_decoding_backend_configured": "lm-format-enforcer",
                          "guided_backend_exercised_in_this_arm": False},
               "meta": {"phys_devices": [0]},
               "startup_s_outside_measurement": 30.0,
               "perf": {"wall_time_s": wall, "requests_per_s": n / wall,
                        "true_ttft_measured": True,
                        "ttft_median_s": 0.08, "itl_median_s": 0.006,
                        "itl_n": 0, "tail_percentiles_reported": False},
               "energy": {"energy_total_j": 1200.0},
               "telemetry": {}, "rows": rows}
        rec["config"].update(cfg_over)
        p = self.dir / f"s{self._n}.json"
        p.write_text(json.dumps(rec, default=str))
        return self.tc.certify(p)

    def test_no_streaming_record_is_an_unqualified_pass(self):
        r = self.certify_streaming([{"ttft_s": 0.08, "n_chunks": 63,
                                     "inter_token_latencies_s": [0.006] * 63}])
        self.assertNotEqual(r["verdict"], self.tc.PASS)

    def test_removed_rows_fail_in_the_certifier_too(self):
        r = self.certify_streaming([])
        self.assertEqual(r["verdict"], self.tc.FAIL)
        self.assertTrue(any(k == "raw_request_rows_present" for k, _ in r["failures"]))

    def test_the_certifier_records_cleanup_as_unproven_by_a_pid(self):
        r = self.certify_streaming([{"ttft_s": 0.08, "n_chunks": 63,
                                     "inter_token_latencies_s": [0.006] * 63}])
        names = [k for k, _ in r["limitations"]] + [k for k, _ in r["failures"]]
        self.assertIn("cleanup_is_not_proven_by_a_pid_integer", names)


if __name__ == "__main__":
    unittest.main()
