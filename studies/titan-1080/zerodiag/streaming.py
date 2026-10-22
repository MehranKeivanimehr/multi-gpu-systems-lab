"""The prospective dedicated streaming case: 256 input / 64 output.

This protocol is frozen before any new streaming result exists.

Naming discipline, which is the whole point of this module: a server emits
*text chunks*, not tokens. A chunk boundary is not a token boundary, so a
chunk arrival time is not a token time. Per-token timing is reported ONLY when
token identity and timestamp attribution are both genuinely established by the
response payload; otherwise the metrics are reported as first-text and chunk
metrics, under those names.

Preserved history, not rewritten here: the historical runs used 278-token
inputs, and the corpus records `exact_output_tokens: 128` alongside
`streaming_output_tokens: 64`. That 64/128 difference is a real protocol
contradiction in the historical material and is preserved as such. This case
uses 64 output tokens, as the frozen streaming defaults specify.
"""
from __future__ import annotations

import hashlib
import json
import math

PROTOCOL = {
    "schema": "titan_streaming_protocol_v1",
    "server": "one fresh loopback-only server, bound to localhost",
    "requests": 100,
    "concurrency": 4,
    "prompt_len_tokens": 256,
    "output_tokens": 64,
    "greedy": True,
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "model_revision": "fe8a4ea1ffedaf415f4da2f062534de366a451e6",
    "corpus": "protocols/prompt_corpus.json",
    "corpus_sha256": ("f2d11005cbce3e5419be77a3c3b05eff7a0b3e56b791e0d3c13"
                      "327d2ac611350"),
    "sessions": 1,
    "sessions_note": ("ONE new session. One session is not six independent "
                      "repeat sessions, carries no session-variation estimate, "
                      "and is never reported as repeats."),
    "reporting_allowed": ["per-request descriptive summaries",
                          "first-text latency", "chunk counts",
                          "inter-chunk gaps", "wall duration",
                          "server-reported completion token counts"],
    "reporting_forbidden": ["tail percentiles from one session",
                            "internal GPU TTFT",
                            "chunk gaps presented as inter-token latency",
                            "per-token timing without established attribution"],
    "historical_untouched": ("the historical 278-token inputs and the "
                             "preserved 64/128 protocol contradiction are not "
                             "rewritten by this case"),
}


def protocol_digest() -> str:
    return hashlib.sha256(
        json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()


def verify_prompt_lengths(tokenized_lengths: list[int]) -> dict:
    """Actual tokenized lengths must be the protocol length, not assumed."""
    want = PROTOCOL["prompt_len_tokens"]
    bad = [(i, n) for i, n in enumerate(tokenized_lengths) if n != want]
    return {"ok": not bad and bool(tokenized_lengths),
            "expected_tokens": want, "n_prompts": len(tokenized_lengths),
            "offending": bad[:10], "n_offending": len(bad),
            "basis": "length of the actual tokenizer output, not a text guess",
            "reason": (None if (not bad and tokenized_lengths) else
                       (f"{len(bad)} prompt(s) are not exactly {want} tokens"
                        if bad else "no tokenized lengths were supplied"))}


def verify_revision(resolved_revision: str | None,
                    tokenizer_revision: str | None) -> dict:
    """Both model and tokenizer must resolve to the pinned revision."""
    want = PROTOCOL["model_revision"]
    ok = resolved_revision == want and tokenizer_revision == want
    return {"ok": ok, "expected": want, "model_revision": resolved_revision,
            "tokenizer_revision": tokenizer_revision,
            "reason": (None if ok else
                       "model and/or tokenizer revision is not the pinned one")}


def token_attribution(events: list[dict]) -> dict:
    """Decide, from the actual events, whether per-token timing is reportable.

    Per-token timing requires BOTH:
      * identity: each event names the token(s) it carries (e.g. explicit token
        ids), not just a text fragment; and
      * attribution: each event carries its own timestamp, and the number of
        timestamped events equals the number of tokens.

    If either fails, per-token timing is not reportable and the caller must use
    the chunk metrics under their own names.
    """
    n = len(events)
    timestamped = sum(1 for e in events if e.get("t") is not None)
    with_ids = sum(1 for e in events
                   if isinstance(e.get("token_ids"), list) and e["token_ids"])
    total_ids = sum(len(e.get("token_ids") or []) for e in events)
    identity = n > 0 and with_ids == n
    one_to_one = identity and total_ids == n and timestamped == n
    return {
        "n_events": n, "n_timestamped": timestamped,
        "n_events_with_token_ids": with_ids, "total_token_ids": total_ids,
        "identity_established": identity,
        "timestamp_attribution_established": timestamped == n and n > 0,
        "per_token_timing_reportable": bool(one_to_one),
        "reason": (None if one_to_one else
                   ("events do not establish a one-to-one timestamped token "
                    "mapping; chunk arrival times are NOT token times")),
    }


SUCCESS_RULE = {
    "schema": "titan_streaming_success_rule_v1",
    "document": "protocols/STREAMING_SUCCESS_RULE.md",
    "sha256": "860f6954ba4d5514143fa8083c54287334303247ad761f82952fb10fda0dc849",
    "conditions": ["transport status is ok (HTTP 200, body read to its end)",
                   "at least one data event carries non-empty text",
                   "a completion marker: finish_reason or the [DONE] sentinel",
                   "no error event recorded for the request"],
    "http_200_alone_is_not_success": True,
    "length_check": ("where usage.completion_tokens is present it must equal "
                     "the frozen output length; otherwise the request is "
                     "length_mismatch and is not counted as successful"),
    "classes": ["successful", "length_mismatch", "incomplete", "failed"],
}


def classify_request(events: list, http_status, usage, *,
                     output_tokens: int) -> dict:
    """The frozen per-request rule. One definition; nothing else decides this.

    Used by the runner's gate, the session summary and the published offline
    tool, so a request cannot be judged by one rule on the node and another in
    the repository.
    """
    if not isinstance(events, list) or any(not isinstance(e, dict) for e in events):
        raise ValueError("events must be a list of objects; nothing may be silently discarded")
    evs = events
    previous = -1
    for e in evs:
        t = e.get("t")
        if type(t) not in (int, float) or not math.isfinite(t) or t < previous or t < 0:
            raise ValueError("event timestamps must be finite, nonnegative and in retained order")
        previous = t
    usage = {} if usage is None else usage
    if not isinstance(usage, dict):
        raise ValueError("usage must be an object")
    for k in ("completion_tokens", "prompt_tokens"):
        if k in usage and (type(usage[k]) is not int or usage[k] < 0):
            raise ValueError("token counts must be nonnegative integers")
    if "prompt_tokens" in usage and usage["prompt_tokens"] != PROTOCOL["prompt_len_tokens"]:
        raise ValueError("server-reported prompt length does not match the protocol")
    errors = [e for e in evs if e.get("error")]
    content = [e for e in evs if isinstance(e.get("text"), str) and e["text"]]
    fins = [e.get("finish_reason") for e in evs if e.get("finish_reason")]
    done = any(e.get("done") is True for e in evs)
    transport_ok = (http_status == "ok")
    reported = (usage or {}).get("completion_tokens")
    checks = {"transport_ok": transport_ok,
              "has_content_event": bool(content),
              "has_completion_marker": bool(fins or done),
              "no_error_event": not errors}
    if not transport_ok or errors:
        cls = "failed"
    elif not (content and (fins or done)):
        cls = "incomplete"
    elif reported is not None and int(reported) != int(output_tokens):
        cls = "length_mismatch"
    else:
        cls = "successful"
    return {"class": cls, "checks": checks,
            "http_status": http_status,
            "n_events": len(evs), "n_content_events": len(content),
            "output_chars": sum(len(e["text"]) for e in content),
            "finish_reason": (fins[-1] if fins else None),
            "done_sentinel": done,
            "reported_completion_tokens": reported,
            "length_evidence_present": reported is not None,
            "length_matches_protocol": (None if reported is None
                                        else int(reported) == int(output_tokens)),
            "error_events": [e.get("error") for e in errors][:3],
            "token_identity_in_payload": any(e.get("token_ids") is not None
                                             for e in evs)}


def account_session(events_by_request: dict, status: dict, usage: dict, *,
                    requests_expected: int, output_tokens: int) -> dict:
    """Session accounting under the frozen rule. Nothing is dropped."""
    if set(events_by_request) != set(status) or not set(usage).issubset(events_by_request):
        raise ValueError("request identity maps disagree")
    per = {rid: classify_request(events_by_request.get(rid), status.get(rid),
                                 usage.get(rid), output_tokens=output_tokens)
           for rid in sorted(events_by_request)}
    counts = {c: sum(1 for d in per.values() if d["class"] == c)
              for c in ("successful", "length_mismatch", "incomplete", "failed")}
    no_len = sum(1 for d in per.values()
                 if d["class"] == "successful" and not d["length_evidence_present"])
    recorded = len(per)
    return {"rule": SUCCESS_RULE,
            "requests_expected": int(requests_expected),
            "requests_recorded": recorded,
            "counts": counts,
            "successful_without_length_evidence": no_len,
            "classes_sum": sum(counts.values()),
            "accounting_complete": (recorded == int(requests_expected)
                                    and sum(counts.values()) == recorded),
            "per_request": per}


def summarize_session(events_by_request: dict, *, completion_tokens: dict,
                      status: dict, request_timing: dict | None = None) -> dict:
    """Descriptive, correctly named metrics for ONE session.

    `events_by_request[rid]` is the raw, preserved event list for that request.
    Nothing here converts a chunk into a token.
    """
    per_request, attribution_ok = {}, True
    for rid, events in sorted(events_by_request.items()):
        attr = token_attribution(events)
        attribution_ok &= attr["per_token_timing_reportable"]
        ts = [e["t"] for e in events if e.get("t") is not None]
        if any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in ts):
            raise ValueError("non-finite or invalid event timestamp")
        if any(b < a for a, b in zip(ts, ts[1:])):
            raise ValueError("event timestamps are out of retained order")
        first_content = next((e["t"] for e in events if e.get("text")), None)
        start = (request_timing or {}).get(rid, {}).get("request_start_s")
        if start is not None and (type(start) not in (int, float)
                or not math.isfinite(start) or start < 0 or (ts and start > ts[0])):
            raise ValueError("invalid per-request start timestamp")
        latency = None if start is None or first_content is None else first_content - start
        gaps = [round(b - a, 6) for a, b in zip(ts, ts[1:])]
        per_request[rid] = {
            "status": status.get(rid),
            "n_chunks": len(events),
            "first_text_latency_s": latency,
            "session_relative_first_event_s": ts[0] if ts else None,
            "request_start_s": start,
            "wall_duration_s": (max(ts) - min(ts)) if len(ts) > 1 else None,
            "inter_chunk_gaps_s": gaps,
            "server_reported_completion_tokens": completion_tokens.get(rid),
            "token_attribution": attr,
            "per_token_timing": (None if not attr["per_token_timing_reportable"]
                                 else "reportable"),
        }
    n_ok = sum(1 for d in per_request.values() if d["status"] == "ok")
    return {
        "sessions": 1,
        "sessions_note": PROTOCOL["sessions_note"],
        "requests_recorded": len(per_request),
        "requests_ok": n_ok,
        "requests_failed": len(per_request) - n_ok,
        "per_request": per_request,
        "metric_names": ("session_relative_first_event_s is an arrival offset, not latency. "
                         "first_text_latency_s requires a retained per-request start. "
                         "Inter-event gaps include metadata events and are not inter-token latency."),
        "per_token_timing_reported": bool(per_request) and attribution_ok,
        "per_token_timing_withheld_reason": (
            None if (per_request and attribution_ok) else
            "token identity and timestamp attribution were not established for "
            "every request, so per-token timing is withheld"),
        "no_tail_percentiles": ("withheld: one session does not support tail "
                                "percentile estimates"),
    }
