import io
import unittest

from validation.stream_latency import read_events, summarize


class StreamingTests(unittest.TestCase):
    def test_ignore_empty_output_and_usage_events(self):
        events = [
            {"received_s": 11, "payload": {"choices": [{"text": "", "logprobs": None}]}},
            {"received_s": 12, "payload": {"choices": [{"text": "hello"}]}},
            {"received_s": 14, "payload": {"choices": [{"text": "world"}]}},
            {"received_s": 15, "payload": {"choices": [], "usage": {"completion_tokens": 2}}}]
        result = summarize(events, 10, 16, 2)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["client_ttft_s"], 2)
        self.assertEqual(result["client_end_to_end_s"], 6)
        self.assertEqual(result["stream_content_chunk_gaps_s"], [2])
        self.assertFalse(result["internal_gpu_ttft_measured"])

    def test_token_evidence_can_be_present_before_nonempty_text(self):
        result = summarize([{"received_s": 2, "payload": {"choices": [{"text": "", "logprobs": {"tokens": ["x"]}}],
                             "usage": {"completion_tokens": 1}}}], 1, 3, 1)
        self.assertEqual(result["client_ttft_s"], 1)

    def test_missing_or_wrong_token_count_is_not_success(self):
        for usage in ({}, {"completion_tokens": 1}):
            result = summarize([{"received_s": 2, "payload": {"choices": [{"text": "x"}], "usage": usage}}], 1, 3, 2)
            self.assertNotEqual(result["status"], "ok")

    def test_sse_payloads_and_completion_are_retained(self):
        response = io.BytesIO(b'data: {"choices": [{"text": "x"}]}\n\ndata: [DONE]\n')
        events = read_events(response, clock=lambda: 5)
        self.assertEqual(events[0]["received_s"], 5)
        self.assertEqual(events[0]["payload"]["choices"][0]["text"], "x")

    def test_truncated_or_error_stream_is_rejected(self):
        for data in (b'data: {"choices": []}\n', b'data: {"error": "failed"}\n'):
            with self.assertRaises(ValueError):
                read_events(io.BytesIO(data), clock=lambda: 5)


if __name__ == "__main__":
    unittest.main()
