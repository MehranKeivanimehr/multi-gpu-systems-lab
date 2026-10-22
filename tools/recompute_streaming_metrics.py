#!/usr/bin/env python
"""Recompute the streaming session's accounting and timings from shipped events.

Standard library only. No GPU, no running server, no model cache, no
third-party package: it reads the session's `raw_events.json` and
`streaming_evidence.json` and recomputes everything from the retained raw
events and their recorded monotonic timestamps.

It implements the frozen rule in `protocols/STREAMING_SUCCESS_RULE.md`
independently of the repository library, and the repository's
`zd_test_offline_recompute` suite asserts that the two implementations agree on
a matrix of fixtures, so this is a second opinion rather than a copy.

Naming discipline, enforced here: a server emits *text chunks*, not tokens. A
chunk boundary is not a token boundary, so a chunk arrival time is not a token
time. Per-token timing is reported only when the payload itself carried
per-token identity and per-token timestamps for every event. Token times are
never derived by dividing chunk intervals and text is never retokenized to
invent emission times.

Exit codes:
  0   the session is accounted for and has at least one successful request
  2   the session is accounted for and has NO successful request
  95  refused: missing, corrupt or unreadable evidence

Usage:
  python tools/recompute_streaming_metrics.py --session <dir> [--expected 100]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

RULE = "protocols/STREAMING_SUCCESS_RULE.md"
RULE_SHA = "860f6954ba4d5514143fa8083c54287334303247ad761f82952fb10fda0dc849"


class Refusal(Exception):
    pass


def _constant(value):
    raise Refusal(f"non-finite JSON constant: {value}")


def _pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise Refusal(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _int(value, label, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise Refusal(f"{label} must be a {'positive' if positive else 'nonnegative'} integer")
    return value


def _time(value, label):
    try:
        valid = type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise Refusal(f"{label} must be finite and nonnegative")
    return value


def _object(value, label):
    if not isinstance(value, dict):
        raise Refusal(f"{label} must be an object")
    return value


def load(session: str):
    rp = os.path.join(session, "raw_events.json")
    if not os.path.isfile(rp):
        raise Refusal(f"missing raw events: {rp}")
    try:
        with open(rp) as f:
            raw = json.load(f, parse_constant=_constant, object_pairs_hook=_pairs)
    except ValueError as exc:
        raise Refusal(f"raw events are not valid JSON: {exc}")
    if not isinstance(raw, dict) or not isinstance(raw.get("events"), dict):
        raise Refusal("raw events file has no 'events' mapping")
    corr = None
    cp = os.path.join(session, "evidence_corrections.json")
    if os.path.isfile(cp):
        try:
            with open(cp) as f:
                corr = json.load(f, parse_constant=_constant, object_pairs_hook=_pairs)
        except ValueError as exc:
            raise Refusal(f"evidence corrections are not valid JSON: {exc}")
    ep = os.path.join(session, "streaming_evidence.json")
    if not os.path.isfile(ep):
        raise Refusal("missing streaming_evidence.json: protocol metadata is required")
    try:
        with open(ep) as f:
            ev = json.load(f, parse_constant=_constant, object_pairs_hook=_pairs)
    except ValueError as exc:
        raise Refusal(f"streaming evidence is not valid JSON: {exc}")
    _object(ev, "streaming evidence")
    if corr is not None:
        _object(corr, "evidence corrections")
    return raw, ev, corr


def classify(events, http_status, usage, output_tokens):
    """The frozen per-request rule, reimplemented from the document."""
    if not isinstance(events, list):
        raise Refusal("each request's events must be a list")
    evs = events
    previous = -1
    for e in evs:
        _object(e, "event")
        t = _time(e.get("t"), "event timestamp")
        if t < previous:
            raise Refusal("event timestamps must be nondecreasing in retained order")
        previous = t
        if e.get("text") is not None and not isinstance(e["text"], str):
            raise Refusal("event text must be a string")
        if e.get("finish_reason") is not None and not isinstance(e["finish_reason"], str):
            raise Refusal("finish_reason must be a string or null")
        if "done" in e and type(e["done"]) is not bool:
            raise Refusal("done must be boolean")
    if not isinstance(http_status, str):
        raise Refusal("every request requires a recorded transport status")
    _object(usage, "request usage")
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if k in usage:
            _int(usage[k], f"usage.{k}")
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
    ts = [e["t"] for e in evs]
    has_tok = [e for e in evs if e.get("token_ids") is not None]
    return {
        "class": cls, "checks": checks, "http_status": http_status,
        "n_events": len(evs), "n_content_events": len(content),
        "n_timestamped_events": len(ts),
        "n_unparsed_events": sum(1 for e in evs if "unparsed" in e),
        "output_chars": sum(len(e["text"]) for e in content),
        "finish_reason": (fins[-1] if fins else None),
        "done_sentinel": done,
        "reported_completion_tokens": reported,
        "length_evidence_present": reported is not None,
        "length_matches_protocol": (None if reported is None
                                    else int(reported) == int(output_tokens)),
        "reported_prompt_tokens": (usage or {}).get("prompt_tokens"),
        "error_events": [e.get("error") for e in errors][:3],
        "session_relative_first_event_s": (ts[0] if ts else None),
        "session_relative_first_content_s": next((e["t"] for e in content), None),
        "first_text_latency_s": None,
        "last_event_s": (ts[-1] if ts else None),
        "wall_duration_s": ((ts[-1] - ts[0]) if len(ts) > 1 else None),
        "inter_event_gaps_s": [round(b - a, 6) for a, b in zip(ts, ts[1:])],
        "token_identity_in_payload": bool(has_tok),
        "per_token_timing_supported": False,
        "per_token_timing": None,
        "per_token_timing_withheld_reason":
            ("the payload did not carry per-token identity and per-token "
             "timestamps for every event, and a chunk arrival time is not a "
             "token time"),
    }


def _stats(values):
    if not values:
        return None
    v = sorted(values)
    return {"n": len(v), "min": v[0], "max": v[-1],
            "mean": statistics.fmean(v), "median": statistics.median(v)}


def recompute(raw: dict, shipped: dict, expected) -> dict:
    _object(raw, "raw events")
    _object(shipped, "streaming evidence")
    events = _object(raw.get("events"), "events")
    status = _object(raw.get("status"), "status")
    usage = _object(raw.get("usage"), "usage")
    if set(events) != set(status) or not set(usage).issubset(events):
        raise Refusal("event/status request IDs must agree; usage may not name unrecorded IDs")
    if any(not isinstance(r, str) or not r for r in events):
        raise Refusal("request IDs must be nonempty strings")
    proto = _object(shipped.get("protocol"), "protocol")
    for key in ("schema", "model", "model_revision"):
        if not isinstance(proto.get(key), str) or not proto[key]:
            raise Refusal(f"protocol.{key} is required")
    _int(proto.get("requests"), "protocol.requests", True)
    _int(proto.get("concurrency"), "protocol.concurrency", True)
    if type(proto.get("greedy")) is not bool:
        raise Refusal("protocol.greedy must be boolean")
    out_tokens = _int(proto.get("output_tokens"), "protocol.output_tokens", True)
    prompt_tokens = _int(proto.get("prompt_len_tokens"), "protocol.prompt_len_tokens", True)
    wl = _object(shipped.get("frozen_workload"), "frozen_workload")
    declared_expected = _int(wl.get("requests_expected"), "requests_expected", True)
    if expected is None:
        expected = declared_expected
    if _int(expected, "expected", True) != declared_expected:
        raise Refusal("requested count disagrees with frozen workload")
    for k in ("requests_issued",):
        if k in wl and _int(wl[k], k) != len(events):
            raise Refusal(f"{k} disagrees with retained requests")
    if "request_ids" in wl and (not isinstance(wl["request_ids"], list)
            or len(wl["request_ids"]) != len(set(wl["request_ids"]))
            or set(wl["request_ids"]) != set(events)):
        raise Refusal("frozen request IDs disagree with retained requests")
    for k, value in (("prompt_len_tokens", prompt_tokens), ("output_tokens", out_tokens)):
        if k in wl and (type(wl[k]) is not int or wl[k] != value):
            raise Refusal(f"workload {k} disagrees with protocol")
    if "concurrency" in wl and (_int(wl["concurrency"], "workload.concurrency", True) != proto["concurrency"]):
        raise Refusal("workload concurrency disagrees with protocol")
    for u in usage.values():
        _object(u, "request usage")
        if "prompt_tokens" in u and (_int(u["prompt_tokens"], "prompt_tokens") != prompt_tokens):
            raise Refusal("server-reported prompt length disagrees with protocol")
    per = {rid: classify(events.get(rid), status.get(rid), usage.get(rid, {}),
                         out_tokens) for rid in sorted(events)}
    counts = {c: sum(1 for d in per.values() if d["class"] == c)
              for c in ("successful", "length_mismatch", "incomplete",
                        "failed")}
    ok_ids = [r for r, d in per.items() if d["class"] == "successful"]
    no_len = [r for r in ok_ids if not per[r]["length_evidence_present"]]
    if ok_ids and (wl.get("input_length_verified") is not True
                   or wl.get("issued_as_frozen") is not True
                   or wl.get("consumed_prompt_lengths") != [prompt_tokens]):
        raise Refusal("successful-session workload identity/length metadata is missing or inconsistent")
    timing = raw.get("request_timing")
    if timing is not None:
        _object(timing, "request_timing")
        if set(timing) != set(events):
            raise Refusal("request timing ID set disagrees with events")
        for rid, td in timing.items():
            _object(td, "request timing entry")
            start = _time(td.get("request_start_s"), "request_start_s")
            if events[rid] and start > events[rid][0]["t"]:
                raise Refusal("request starts after its first event")
            content_t = per[rid]["session_relative_first_content_s"]
            per[rid]["first_text_latency_s"] = None if content_t is None else content_t - start
    first = [per[r]["session_relative_first_event_s"] for r in ok_ids
             if per[r]["session_relative_first_event_s"] is not None]
    latencies = [per[r]["first_text_latency_s"] for r in ok_ids
                 if per[r]["first_text_latency_s"] is not None]
    gaps = [g for r in ok_ids for g in per[r]["inter_event_gaps_s"]]
    chunks = [per[r]["n_content_events"] for r in ok_ids]
    lens = sorted({per[r]["reported_completion_tokens"] for r in ok_ids
                   if per[r]["length_evidence_present"]})
    plens = sorted({per[r]["reported_prompt_tokens"] for r in per
                    if per[r]["reported_prompt_tokens"] is not None})
    tok_ok = all(per[r]["per_token_timing_supported"] for r in ok_ids) and ok_ids
    return {
        "rule": {"document": RULE, "sha256": RULE_SHA,
                 "http_200_alone_is_not_success": True},
        "implementation": "standard library only, independent of the library",
        "accounting": {
            "requests_expected": expected,
            "requests_recorded": len(per),
            "counts": counts,
            "classes_sum": sum(counts.values()),
            "successful_without_length_evidence": len(no_len),
            "accounting_complete": (len(per) == expected
                                    and sum(counts.values()) == len(per)),
            "nothing_discarded": ("every recorded request is in exactly one "
                                  "class"),
        },
        "lengths_supported_by_evidence": {
            "server_reported_completion_tokens": lens,
            "protocol_output_tokens": out_tokens,
            "all_successful_match_protocol": (lens in ([], [out_tokens])),
            "successful_requests_with_length_evidence": len(
                [r for r in ok_ids if per[r]["length_evidence_present"]]),
            "length_mismatch_reported_tokens": sorted(
                {per[r]["reported_completion_tokens"] for r in per
                 if per[r]["class"] == "length_mismatch"}),
            "server_reported_prompt_tokens": plens,
            "protocol_prompt_len_tokens": proto.get("prompt_len_tokens"),
            "consumed_prompt_lengths_recorded_by_runner":
                wl.get("consumed_prompt_lengths"),
            "input_length_verified_by_runner": wl.get("input_length_verified"),
        },
        "chunk_measurements": {
            "note": ("CHUNK measurements, not time to first token. Session-relative "
                     "arrivals are not request latency. Missing request starts cannot be recovered; "
                     "inter_event_gaps_s are gaps between network events, not "
                     "inter-token latencies."),
            "clock": (shipped.get("clock_definitions") or {}).get("event_t"),
            "session_relative_first_event_s": _stats(first),
            "first_text_latency_s": _stats(latencies),
            "request_latency_available": timing is not None,
            "request_latency_limitation": (None if timing is not None else
                "No per-request start timestamps were retained. True first-text latency is unavailable."),
            "gap_population": "all adjacent timestamped events, including metadata and completion events",
            "inter_event_gap_s": _stats(gaps),
            "content_events_per_successful_request": _stats(chunks),
            "computed_over": "successful requests only",
        },
        "token_level": {
            "reported": bool(tok_ok),
            "reason": (None if tok_ok else
                       "no request carried per-token identity and per-token "
                       "timestamps, so no token-level timing is computed; "
                       "chunk intervals were not divided and text was not "
                       "retokenized"),
        },
        "session_scope": {
            "sessions": 1,
            "note": ("one prospective session. Its concurrent requests are "
                     "not independent repeated sessions; no cross-session "
                     "distribution and no tail percentile is reported."),
            "no_tail_percentiles": True,
        },
        "completion": {
            "requests_with_finish_reason":
                sum(1 for d in per.values() if d["finish_reason"]),
            "requests_with_done_sentinel":
                sum(1 for d in per.values() if d["done_sentinel"]),
            "finish_reasons": sorted({d["finish_reason"] for d in per.values()
                                      if d["finish_reason"]}),
        },
        "runner_reported": {
            "stage": shipped.get("stage"),
            "reason": shipped.get("reason"),
            "streaming_case_complete": shipped.get("streaming_case_complete"),
            "server_shutdown_confirmed":
                (shipped.get("server_shutdown") or {}).get("confirmed_absent"),
            "effective_guided_backend":
                (shipped.get("guided_decoding_backend") or {}).get("selected"),
        },
        "per_request": per,
    }


def cross_check(mine: dict, shipped: dict) -> dict:
    """Compare against the accounting the runner itself recorded."""
    acc = shipped.get("accounting") or {}
    if not acc:
        return {"compared": False,
                "reason": "the session evidence carries no runner accounting"}
    fields = {}
    for k in ("requests_expected", "requests_recorded", "accounting_complete"):
        fields[k] = {"recomputed": mine["accounting"].get(k),
                     "recorded": acc.get(k),
                     "agree": mine["accounting"].get(k) == acc.get(k)}
    rc = acc.get("counts") or {}
    fields["counts"] = {"recomputed": mine["accounting"]["counts"],
                        "recorded": rc,
                        "agree": mine["accounting"]["counts"] == rc}
    per = acc.get("per_request") or {}
    cls = [(mine["per_request"][r]["class"], (per.get(r) or {}).get("class"))
           for r in mine["per_request"] if r in per]
    return {"compared": True, "fields": fields,
            "per_request_classes_agree": all(a == b for a, b in cls),
            "per_request_compared": len(cls)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--session", required=True,
                    help="directory holding raw_events.json")
    ap.add_argument("--expected", type=int, default=None,
                    help="expected request count; default: from the evidence")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    try:
        raw, shipped, corrections = load(a.session)
        res = recompute(raw, shipped, a.expected)
        # A correction recorded beside the evidence is surfaced by the
        # published command itself, so a reader of the numbers cannot miss it.
        res["evidence_corrections"] = (
            {"present": False,
             "note": "no corrections were recorded for this session"}
            if corrections is None else
            {"present": True,
             "applies_to": corrections.get("applies_to"),
             "count": len(corrections.get("corrections") or []),
             "corrections": corrections.get("corrections")})
        res["session"] = a.session
        res["cross_check"] = cross_check(res, shipped)
        cc = res["cross_check"]
        if cc["compared"] and (not all(f["agree"] for f in cc["fields"].values())
                or not cc["per_request_classes_agree"]
                or cc["per_request_compared"] != len(res["per_request"])):
            raise Refusal("runner accounting disagrees with retained evidence")
    except (Refusal, OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
        out = {"verdict": "REFUSED", "reason": str(exc),
               "note": ("missing or corrupt streaming evidence is a refusal; "
                        "no metric and no session result is reported for it")}
        print(json.dumps(out, indent=2, sort_keys=True, allow_nan=False))
        if a.json:
            with open(a.json, "w") as f:
                json.dump(out, f, indent=2, sort_keys=True)
        return 95
    acc = res["accounting"]
    res["verdict"] = ("SESSION_OBSERVED"
                      if acc["accounting_complete"] and acc["counts"]["successful"]
                      else "NO_SUCCESSFUL_REQUEST"
                      if acc["accounting_complete"]
                      else "ACCOUNTING_INCOMPLETE")
    print(json.dumps(res, indent=2, sort_keys=True, allow_nan=False))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=2, sort_keys=True)
    return 0 if res["verdict"] == "SESSION_OBSERVED" else 2


if __name__ == "__main__":
    sys.exit(main())
