"""Certification must be strict and non-vacuous.

Each test below is a counterexample that the DELIVERED titan_audit.py accepted.
Reproductions are preserved under out/repro/*_FIRST_FAILING.*; these tests pin
the corrected behaviour. All device access is mocked or absent: no test touches
a GPU, and the certifier imports nothing that queries one.
"""
from __future__ import annotations

import copy
import importlib.util
import statistics as _stats
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tc = load("tc_certify", "tools/titan_certify.py")
        cls.dir = pathlib.Path(tempfile.mkdtemp())
        cls._n = 0

    def verdict(self, obj):
        type(self)._n += 1
        p = self.dir / f"r{self._n}.json"
        p.write_text(json.dumps(obj, default=str))
        return self.tc.certify(p)


class UnknownAndEmptyTests(_Base):
    def test_empty_object_is_not_a_pass(self):
        r = self.verdict({})
        self.assertEqual(r["verdict"], self.tc.UNSUPPORTED)
        self.assertEqual(r["n_checks"], 0)

    def test_unknown_schema_is_unsupported_not_pass(self):
        r = self.verdict({"totally": "unknown", "schema": [1, 2, 3]})
        self.assertEqual(r["verdict"], self.tc.UNSUPPORTED)

    def test_unreadable_file_is_invalid(self):
        p = self.dir / "broken.json"
        p.write_text("{not json")
        self.assertEqual(self.tc.certify(p)["verdict"], self.tc.INVALID)

    def test_no_verdict_may_be_pass_with_zero_checks(self):
        for obj in ({}, {"x": 1}, {"config": {}}, [] if False else {"rows": []}):
            r = self.verdict(obj)
            if r["n_checks"] == 0:
                self.assertNotEqual(r["verdict"], self.tc.PASS)


def _telemetry(devices=("0",), dur=10.0, interval=0.2, power=120.0):
    n = int(dur / interval) + 3
    raw = []
    for k in range(n):
        t = 1000.0 - interval + k * interval
        raw.append({"time_s": t, "query_start_s": t - 0.01, "query_stop_s": t + 0.01,
                    "gpus": [{"index": int(d), "power.draw": power} for d in devices]})
    t0, t1 = 1000.0, 1000.0 + dur
    per = {}
    for d in devices:
        ts = [s["time_s"] for s in raw]
        ps = [power] * len(ts)
        e, _, _, _ = None, None, None, None
        per[d] = power * dur
    return {"window_start_s": t0, "window_stop_s": t1, "duration_s": dur,
            "raw_samples": raw, "sample_interval_s": interval,
            "sample_interval_matches_frozen": interval == 0.2,
            "max_permitted_gap_s": 0.6, "sampling_errors": [],
            "energy_total_j": power * dur * len(devices),
            "energy_per_gpu_j": per,
            "power_mean_w": {d: power for d in devices},
            "power_max_w": {d: power for d in devices},
            "util_mean_pct": {d: 95.0 for d in devices},
            "util_max_pct": {d: 98.0 for d in devices},
            "mem_max_mib": {d: 5000.0 for d in devices},
            "temp_max_c": {d: 60.0 for d in devices},
            "integration": {d: {"status": "ok", "energy_j": power * dur,
                                "max_support_gap_s": 0.2} for d in devices}}


def _vllm_record(n_rows=4):
    rows = []
    for i, conc in enumerate((1, 4, 16, 64)[:n_rows]):
        tel = _telemetry()
        rows.append({"concurrency": conc, "status": "ok", "elapsed_s": 10.0,
                     "output_tokens": 1000 * conc,
                     "output_tokens_per_s": 100.0 * conc,
                     "energy_total_j": tel["energy_total_j"],
                     "energy_per_1k_output_tokens_j": 12.0,
                     "telemetry": tel})
    return {"config": {"engine": "vllm", "enable_prefix_caching": False,
                       "guided_decoding_backend_configured": "lm-format-enforcer",
                       "model": "m", "model_revision": "rev"},
            "device": {"name": "NVIDIA TITAN RTX"}, "model_load_s": 1.0,
            "meta": {"phys_devices": [0]},
            "guided_decoding": {"configured_backend": "lm-format-enforcer",
                                "exercised": True, "conformed": True},
            "rows": rows}


class VllmRowCollisionTests(_Base):
    """The delivered audit keyed findings on batch_size, which vLLM rows lack,
    so all rows collapsed onto one key and only the LAST row was audited."""

    def test_clean_vllm_record_passes(self):
        self.assertEqual(self.verdict(_vllm_record())["verdict"], self.tc.PASS)

    def test_corrupting_the_FIRST_row_is_detected(self):
        bad = _vllm_record()
        bad["rows"][0]["energy_total_j"] = 1
        bad["rows"][0]["telemetry"]["energy_total_j"] = 1
        bad["rows"][0]["telemetry"]["sample_interval_s"] = 99
        r = self.verdict(bad)
        self.assertEqual(r["verdict"], self.tc.FAIL)
        self.assertTrue(any("i=0" in k for k, _ in r["failures"]),
                        f"failure must name row 0: {[k for k,_ in r['failures']]}")

    def test_corrupting_any_single_row_is_detected(self):
        for i in range(4):
            bad = _vllm_record()
            bad["rows"][i]["telemetry"]["sample_interval_s"] = 99
            r = self.verdict(bad)
            self.assertEqual(r["verdict"], self.tc.FAIL, f"row {i} undetected")
            self.assertTrue(any(f"i={i}" in k for k, _ in r["failures"]))

    def test_every_row_gets_a_unique_finding_key(self):
        r = self.verdict(_vllm_record())
        keys = [k for k in r["checks"] if k.startswith("vllm[")]
        ids = {k.split("]")[0] + "]" for k in keys}
        self.assertEqual(len(ids), 4, f"expected 4 distinct row ids, got {ids}")


def _streaming_record(n_rows=100):
    rows = [{"ttft_s": 0.08 + i * 1e-5, "total_s": 0.5, "n_chunks": 63,
             "chars": 200, "inter_token_latencies_s": [0.006] * 63,
             "median_itl_s": 0.006} for i in range(n_rows)]
    wall = 11.6
    return {"status": "ok",
            "config": {"engine": "vllm_loopback_streaming", "requests": n_rows,
                       "concurrency": 4, "prompt_len_requested": 256,
                       "prompt_len_tokens_actual": 256, "output_len": 64,
                       "engine_pid": 12345, "controller_pid": 12344,
                       "enable_prefix_caching": False,
                       "guided_decoding_backend_configured": "lm-format-enforcer",
                       "guided_backend_exercised_in_this_arm": False},
            "meta": {"phys_devices": [0]},
            "startup_s_outside_measurement": 30.0,
            "perf": {"wall_time_s": wall, "requests_per_s": n_rows / wall,
                     "true_ttft_measured": True,
                     # statistics.median averages the two middle values at even n,
                     # which is what the certifier recomputes.
                     "ttft_median_s": _stats.median(r["ttft_s"] for r in rows),
                     "itl_median_s": 0.006,
                     "itl_n": sum(len(r["inter_token_latencies_s"]) for r in rows),
                     "tail_percentiles_reported": False},
            "energy": {"energy_total_j": 1200.0},
            "telemetry": _telemetry(),
            "rows": rows}


class StreamingEvidenceTests(_Base):
    """Removing all raw request rows and setting a nonsense PID passed before."""

    def test_a_complete_streaming_record_is_qualified_not_pass(self):
        r = self.verdict(_streaming_record())
        self.assertEqual(r["verdict"], self.tc.QUALIFIED,
                         "raw SSE/status/cleanup evidence is absent, so the arm "
                         "cannot be an unqualified pass")
        names = [k for k, _ in r["limitations"]]
        for expected in ("raw_sse_evidence_absent", "http_status_evidence_absent",
                         "cleanup_is_not_proven_by_a_pid_integer"):
            self.assertIn(expected, names)

    def test_removing_every_request_row_is_detected(self):
        bad = _streaming_record()
        bad["rows"] = []
        bad["config"]["engine_pid"] = -123
        r = self.verdict(bad)
        self.assertEqual(r["verdict"], self.tc.FAIL)
        self.assertTrue(any(k == "raw_request_rows_present" for k, _ in r["failures"]))

    def test_row_count_must_match_the_requested_count(self):
        bad = _streaming_record(n_rows=100)
        bad["rows"] = bad["rows"][:3]
        r = self.verdict(bad)
        self.assertEqual(r["verdict"], self.tc.FAIL)
        self.assertTrue(any(k == "row_count_matches_requested" for k, _ in r["failures"]))

    def test_a_negative_pid_is_never_cleanup_proof(self):
        """Integer presence is not evidence; the limitation stands either way."""
        for pid in (-123, 0, 12345):
            rec = _streaming_record()
            rec["config"]["engine_pid"] = pid
            r = self.verdict(rec)
            self.assertIn("cleanup_is_not_proven_by_a_pid_integer",
                          [k for k, _ in r["limitations"]])

    def test_a_per_token_latency_claim_is_refused(self):
        bad = _streaming_record()
        bad["perf"]["per_token_latency_s"] = 0.006
        r = self.verdict(bad)
        self.assertEqual(r["verdict"], self.tc.FAIL)
        self.assertTrue(any(k == "per_token_latency_is_NOT_claimed" for k, _ in r["failures"]))

    def test_undeclared_prompt_length_mismatch_is_a_limitation(self):
        rec = _streaming_record()
        rec["config"]["prompt_len_tokens_actual"] = 278
        r = self.verdict(rec)
        self.assertEqual(r["verdict"], self.tc.QUALIFIED)
        self.assertIn("prompt_length_requested_equals_realised",
                      [k for k, _ in r["limitations"]])

    def test_first_arrival_median_is_recomputed_from_rows(self):
        bad = _streaming_record()
        bad["perf"]["ttft_median_s"] = 0.00001
        r = self.verdict(bad)
        self.assertEqual(r["verdict"], self.tc.FAIL)
        self.assertTrue(any(k == "first_arrival_median_matches_rows" for k, _ in r["failures"]))

    def test_chunk_gap_count_is_recomputed_from_rows(self):
        bad = _streaming_record()
        bad["perf"]["itl_n"] = 999999
        r = self.verdict(bad)
        self.assertEqual(r["verdict"], self.tc.FAIL)
        self.assertTrue(any(k == "chunk_gap_count_matches_rows" for k, _ in r["failures"]))


class MandatoryCheckTests(_Base):
    def test_every_supported_schema_declares_mandatory_checks(self):
        for schema in self.tc.AUDITORS:
            self.assertIn(schema, self.tc.MANDATORY)
            self.assertTrue(self.tc.MANDATORY[schema],
                            f"{schema} declares no mandatory checks")

    def test_a_record_missing_a_mandatory_check_is_invalid(self):
        """Tests the mechanism directly: a mandatory name the auditor never
        emits must make the record INVALID, not PASS. Deleting a field instead
        would change schema detection and test something else."""
        from unittest import mock
        extended = dict(self.tc.MANDATORY)
        extended["vllm_offline"] = extended["vllm_offline"] + ("a_check_that_is_never_emitted",)
        with mock.patch.object(self.tc, "MANDATORY", extended):
            r = self.verdict(_vllm_record())
        self.assertEqual(r["verdict"], self.tc.INVALID)
        self.assertIn("a_check_that_is_never_emitted", r["missing_mandatory"])

    def test_adding_a_mandatory_check_invalidates_older_records(self):
        """The point of the mandatory list: tightening it must not silently
        accept records produced before the tightening."""
        from unittest import mock
        extended = dict(self.tc.MANDATORY)
        for schema in extended:
            extended[schema] = extended[schema] + ("future_requirement",)
        with mock.patch.object(self.tc, "MANDATORY", extended):
            r = self.verdict(_vllm_record())
            r2 = self.verdict(_streaming_record())
        self.assertEqual(r["verdict"], self.tc.INVALID)
        self.assertEqual(r2["verdict"], self.tc.INVALID)

    def test_strict_bool_refuses_strings(self):
        self.assertIsNone(self.tc.strict_bool("false"))
        self.assertIsNone(self.tc.strict_bool("true"))
        self.assertIsNone(self.tc.strict_bool(1))
        self.assertIs(self.tc.strict_bool(False), False)
        self.assertIs(self.tc.strict_bool(True), True)

    def test_is_finite_number_refuses_inf_and_nan(self):
        self.assertFalse(self.tc.is_finite_number(float("inf")))
        self.assertFalse(self.tc.is_finite_number(float("-inf")))
        self.assertFalse(self.tc.is_finite_number(float("nan")))
        self.assertFalse(self.tc.is_finite_number(True))
        self.assertTrue(self.tc.is_finite_number(1.5))


if __name__ == "__main__":
    unittest.main()
