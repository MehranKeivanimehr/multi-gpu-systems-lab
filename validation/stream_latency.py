#!/usr/bin/env python3
"""Loopback-only, client-observed streaming latency with raw event retention.

Separate serving experiment, NOT a replacement for offline-batch throughput.
No claim of internal GPU token timestamps or one-token-per-stream-event ITL.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit


def read_events(response, clock=time.perf_counter, events=None):
    """Read vLLM's data-only SSE lines, retaining payload and receive time."""
    events = [] if events is None else events
    completed = False
    for raw in response:
        received = clock()
        line = raw.decode("utf-8").strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            raise ValueError("Unexpected non-SSE response line")
        data = line[5:].strip()
        if data == "[DONE]":
            completed = True
            break
        payload = json.loads(data, parse_constant=lambda value: value)
        if "error" in payload:
            raise ValueError("Server stream error: " + str(payload["error"]))
        events.append({"received_s": received, "payload": payload, "raw_data": data})
    if not completed:
        raise ValueError("Stream ended without [DONE]")
    return events


def summarize(events, sent_s, done_s, expected_tokens):
    content_times, usage = [], None
    for event in events:
        payload = event["payload"]
        if payload.get("usage"):
            usage = payload["usage"]
        for choice in payload.get("choices", []):
            logprobs = choice.get("logprobs") or {}
            if choice.get("text") or logprobs.get("tokens"):
                content_times.append(event["received_s"])
                break
    actual = usage.get("completion_tokens") if usage else None
    valid = bool(content_times) and actual == expected_tokens
    return {"status": "ok" if valid else "invalid_or_missing_output_evidence",
            "client_ttft_s": content_times[0] - sent_s if content_times else None,
            "client_end_to_end_s": done_s - sent_s,
            "stream_content_chunk_gaps_s": [b - a for a, b in zip(content_times, content_times[1:])],
            "output_tokens": actual, "expected_output_tokens": expected_tokens,
            "token_usage": usage,
            "metric_definition": "request dispatch to first received output-bearing SSE event; includes client/network/server queueing",
            "itl_status": "not claimed: SSE chunks may contain multiple tokens",
            "internal_gpu_ttft_measured": False}


def request_one(index, ids, args, queued_s):
    payload = {"model": args.model, "prompt": ids, "max_tokens": args.output_len,
               "min_tokens": args.output_len, "ignore_eos": True,
               "temperature": 0, "stream": True, "echo": False, "logprobs": 1,
               "stream_options": {"include_usage": True}}
    encoded = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(args.base_url.rstrip("/") + "/v1/completions", data=encoded,
                                     headers={"Content-Type": "application/json"}, method="POST")
    sent = time.perf_counter()
    result = {"request_index": index, "queued_s": queued_s, "sent_s": sent,
              "prompt_ids": ids, "request_payload": payload,
              "request_payload_sha256": hashlib.sha256(encoded).hexdigest()}
    events = []
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            read_events(response, events=events)
        done = time.perf_counter()
        result.update(summarize(events, sent, done, args.output_len))
        result.update(done_s=done, events=events)
    except Exception as exc:
        result.update(status="failed", error=repr(exc), done_s=time.perf_counter(), events=events)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True, help="exact served model alias")
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--prompt-len", type=int, default=256)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--vocab-upper", type=int, required=True, help="exclusive verified valid token-ID upper bound")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    parsed = urlsplit(args.base_url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        parser.error("Only a local loopback HTTP server is allowed; do not expose the service publicly")
    if min(args.requests, args.concurrency, args.prompt_len, args.output_len) < 1 or args.vocab_upper <= 1000:
        parser.error("positive sizes and a verified vocabulary upper bound >1000 are required")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the file before sending any requests; never overwrite a session.
    failures = 0
    with args.out.open("x", encoding="utf-8") as stream:
        rng = random.Random(args.seed)
        prompts = [[rng.randrange(1000, args.vocab_upper) for _ in range(args.prompt_len)] for _ in range(args.requests)]
        queued = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(request_one, i, ids, args, queued) for i, ids in enumerate(prompts)]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                failures += result["status"] != "ok"
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
    print("Saved client-observed raw streaming events:", args.out)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
