"""CPU fixtures against the ACTUALLY EXECUTED streaming driver.

These exercise `infer_vllm_stream.one_request` -- the function that produced the
delivered records -- against synthetic SSE byte streams. No server is started,
no model is loaded, and no device is queried: aiohttp's session and response are
replaced by fakes, and the driver's module-level import of `common` (which
touches telemetry helpers) is stubbed before import.

What the fixtures establish about the executed driver, as facts rather than
assertions of adequacy:
  * it timestamps only NON-EMPTY text chunks, so an empty-text token is invisible
  * a chunk carrying several tokens is counted once, so counts are chunk counts
  * it ignores HTTP status entirely
  * a malformed or error payload is silently skipped
  * `[DONE]` ends the stream, and early completion is indistinguishable from
    truncation because no finish_reason or usage is read
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import pathlib
import sys
import types
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
DRIVER = ROOT / "studies/titan-1080/bench/infer_vllm_stream.py"


def _stub_aiohttp():
    """aiohttp lives in the vLLM prefix, not the CPU test interpreter. The
    driver only uses ClientSession/ClientTimeout, so a minimal stub lets the
    REAL parsing code run unchanged under CPU-only tests."""
    if "aiohttp" in sys.modules:
        return
    m = types.ModuleType("aiohttp")

    class ClientTimeout:
        def __init__(self, total=None): self.total = total

    class ClientSession:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    m.ClientTimeout = ClientTimeout
    m.ClientSession = ClientSession
    sys.modules["aiohttp"] = m


def _load_driver():
    """Import the executed driver with its device-touching deps stubbed."""
    _stub_aiohttp()
    fake_common = types.ModuleType("common")
    fake_common.SAMPLE_INTERVAL_S = 0.2
    fake_common.GpuSampler = object
    fake_common.phys_devices = lambda: []
    fake_common.stats_to_dict = lambda s: {}
    fake_common.write_result = lambda rec, name: pathlib.Path("/dev/null")
    sys.modules["common"] = fake_common
    spec = importlib.util.spec_from_file_location("stream_driver", DRIVER)
    m = importlib.util.module_from_spec(spec)
    sys.modules["stream_driver"] = m
    spec.loader.exec_module(m)
    return m


class _FakeContent:
    def __init__(self, lines): self._lines = list(lines)
    def __aiter__(self):
        async def gen():
            for l in self._lines:
                yield l
        return gen()


class _FakeResp:
    def __init__(self, lines, status=200):
        self.content = _FakeContent(lines); self.status = status
    async def __aenter__(self): return self
    async def __aexit__(self, *a): return False


class _FakeSession:
    def __init__(self, lines, status=200): self._lines, self._status = lines, status
    def post(self, *a, **k): return _FakeResp(self._lines, self._status)


def sse(*texts, done=True):
    out = []
    for t in texts:
        out.append(b"data: " + json.dumps({"choices": [{"text": t}]}).encode() + b"\n")
    if done:
        out.append(b"data: [DONE]\n")
    return out


class ExecutedDriverFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = _load_driver()

    def run_one(self, lines, status=200):
        sess = _FakeSession(lines, status)
        return asyncio.run(self.d.one_request(sess, "http://127.0.0.1:1",
                                              "m", "prompt", 8))

    # -- the central misnomer ------------------------------------------------
    def test_an_empty_text_token_is_never_timestamped(self):
        """A real tokenizer can emit a token whose decoded text is empty. The
        driver skips it, so such a token contributes no arrival at all."""
        r = self.run_one(sse("a", "", "b"))
        self.assertEqual(r["n_chunks"], 2, "the empty-text token was counted")
        self.assertEqual(len(r["inter_token_latencies_s"]), 1)

    def test_a_multi_token_chunk_is_counted_once(self):
        """vLLM may coalesce tokens into one SSE chunk. The driver counts
        chunks, so n_chunks is a LOWER bound on tokens."""
        r = self.run_one(sse("one two three four"))
        self.assertEqual(r["n_chunks"], 1)
        self.assertEqual(r["inter_token_latencies_s"], [])
        self.assertGreater(r["chars"], 10)

    def test_chunk_count_is_not_a_token_count(self):
        many = self.run_one(sse(*["x"] * 10))
        few = self.run_one(sse("x" * 10))
        self.assertEqual(many["n_chunks"], 10)
        self.assertEqual(few["n_chunks"], 1)
        self.assertEqual(many["chars"], few["chars"],
                         "identical text, different chunk counts: the metric "
                         "tracks transport framing, not tokens")

    # -- error and malformed handling ---------------------------------------
    def test_a_malformed_json_payload_is_silently_skipped(self):
        lines = [b"data: {not json\n"] + sse("a")
        r = self.run_one(lines)
        self.assertEqual(r["n_chunks"], 1, "malformed payload must not crash, "
                                           "but note it is skipped silently")

    def test_an_error_payload_is_skipped_rather_than_raised(self):
        lines = [b"data: " + json.dumps({"error": {"message": "boom"}}).encode() + b"\n"] + sse("a")
        r = self.run_one(lines)
        self.assertEqual(r["n_chunks"], 1)
        self.assertIsNotNone(r["ttft_s"], "an error frame did not surface")

    def test_a_non_200_status_is_ignored_by_the_driver(self):
        r = self.run_one(sse("a"), status=503)
        self.assertEqual(r["n_chunks"], 1,
                         "the driver never inspects response.status, so a 503 "
                         "that still streams text is recorded as success")

    def test_a_stream_with_no_text_at_all_yields_no_first_arrival(self):
        r = self.run_one(sse(done=True))
        self.assertIsNone(r["ttft_s"])
        self.assertEqual(r["n_chunks"], 0)
        self.assertIsNone(r["median_itl_s"])

    # -- completion accounting ---------------------------------------------
    def test_early_completion_is_indistinguishable_from_truncation(self):
        """Two tokens then DONE, versus two tokens then the socket ending. The
        driver produces the same record, because it reads no finish_reason and
        no usage block."""
        with_done = self.run_one(sse("a", "b", done=True))
        without = self.run_one(sse("a", "b", done=False))
        self.assertEqual(with_done["n_chunks"], without["n_chunks"])
        self.assertEqual(len(with_done["inter_token_latencies_s"]),
                         len(without["inter_token_latencies_s"]))

    def test_done_terminates_the_stream(self):
        lines = sse("a", done=True) + sse("b", done=False)
        r = self.run_one(lines)
        self.assertEqual(r["n_chunks"], 1, "frames after [DONE] were consumed")

    def test_the_record_carries_no_token_usage_or_finish_reason(self):
        r = self.run_one(sse("a", "b"))
        for absent in ("usage", "finish_reason", "completion_tokens",
                       "http_status", "raw_sse"):
            self.assertNotIn(absent, r,
                             f"{absent} is absent from the executed driver's record")

    def test_first_arrival_is_monotonic_and_positive(self):
        r = self.run_one(sse("a", "b", "c"))
        self.assertGreater(r["ttft_s"], 0.0)
        self.assertGreaterEqual(r["total_s"], r["ttft_s"])
        self.assertTrue(all(x >= 0 for x in r["inter_token_latencies_s"]))


class RequiredFieldsForAFutureDriver(unittest.TestCase):
    """What a validated per-token driver would have to retain. Stated as a
    checkable contract so a future driver can be held to it; nothing here
    authorises running one."""

    REQUIRED = ("prompt_token_ids", "prompt_corpus_sha256", "http_status",
                "finish_reason", "usage_prompt_tokens", "usage_completion_tokens",
                "raw_sse_frames_with_timestamps", "done_sentinel_seen",
                "owned_group_exit_confirmed")

    def test_the_executed_driver_satisfies_none_of_the_token_requirements(self):
        src = DRIVER.read_text()
        present = [f for f in self.REQUIRED if f in src]
        self.assertEqual(present, [],
                         f"unexpectedly present: {present}; update the contract")

    def test_the_contract_names_every_missing_evidence_class(self):
        for cls in ("prompt", "status", "finish", "usage", "sse", "exit"):
            self.assertTrue(any(cls in f for f in self.REQUIRED),
                            f"no requirement covers {cls}")


if __name__ == "__main__":
    unittest.main()
