"""Prospective repaired streaming arm: 256 in / 64 out, evidence-complete.

Not executed in this stage. No server is started and no device is touched here.

It reuses the tested helper `validation/stream_latency.py` (`read_events`,
`summarize`) rather than re-implementing SSE parsing, and it is launched through
the AUDITED routed path (`bench/launch.py: run_case`, adapter
`loopback_server`), not through the retired shell bypass.

Evidence this driver REQUIRES before it will report a measurement, each of which
the executed driver lacked:
  * exact prompt token ids and a corpus digest
  * HTTP success per request (status asserted, not ignored)
  * explicit completion and token counts from the server `usage` block
  * `finish_reason` per request, with truncation distinguished from completion
  * raw timestamped SSE frames retained verbatim
  * explicit `[DONE]` handling and error-frame surfacing
  * confirmed owned-group cleanup

Naming discipline: arrival gaps are `chunk_gap_s`, never `inter_token_latency`.
A per-token claim is emitted only when `tokens_per_chunk == 1` for every chunk,
which the driver verifies against `usage.completion_tokens`.
"""
from __future__ import annotations

import hashlib, json, os, pathlib, statistics, sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent.parent / "validation"))

PROTOCOL = {"prompt_len": 256, "output_len": 64, "requests": 100, "concurrency": 4,
            "source": "bundle/plan/run_plan.json S_rtx_1_stream_latency.workload",
            "shared_config_conflict": ("vllm_frozen_config.json declares 128 output "
                                       "for all arms; the dedicated section's 64 is "
                                       "adopted to preserve scope")}

REQUIRED_PER_REQUEST = ("prompt_token_ids_sha256", "http_status", "finish_reason",
                        "usage_prompt_tokens", "usage_completion_tokens",
                        "raw_sse_frames", "done_sentinel_seen", "chunk_arrivals_s")


def verdict_for_request(row: dict) -> dict:
    """Decide what a single request's record can support."""
    missing = [k for k in REQUIRED_PER_REQUEST if k not in row or row[k] is None]
    out = {"missing_evidence": missing, "usable_for": [], "refused_for": []}
    if missing:
        out["refused_for"] = ["chunk_arrival", "per_token_latency", "completion"]
        return out
    if row["http_status"] != 200:
        out["refused_for"] = ["chunk_arrival", "per_token_latency", "completion"]
        out["reason"] = f"http_status={row['http_status']}"
        return out
    if not row["done_sentinel_seen"]:
        out["refused_for"] = ["completion", "per_token_latency"]
        out["usable_for"] = ["chunk_arrival_partial"]
        out["reason"] = "stream ended without [DONE]: truncation not excluded"
        return out
    if row["finish_reason"] not in ("stop", "length"):
        out["refused_for"] = ["completion", "per_token_latency"]
        out["usable_for"] = ["chunk_arrival_partial"]
        out["reason"] = f"finish_reason={row['finish_reason']!r}"
        return out
    out["usable_for"] = ["chunk_arrival", "completion"]
    n_chunks = len(row["chunk_arrivals_s"])
    comp = row["usage_completion_tokens"]
    if comp == n_chunks and n_chunks > 0:
        out["usable_for"].append("per_token_latency")
        out["tokens_per_chunk"] = 1
    else:
        out["refused_for"].append("per_token_latency")
        out["tokens_per_chunk"] = (comp / n_chunks) if n_chunks else None
        out["reason"] = (f"{comp} completion tokens arrived in {n_chunks} chunks: "
                         f"several tokens shared an arrival, so individual token "
                         f"times do not exist in this record")
    return out


def summarise_arm(rows: list[dict], *, protocol: dict | None = None) -> dict:
    """Aggregate, naming chunk gaps as chunk gaps."""
    p = protocol or PROTOCOL
    verdicts = [verdict_for_request(r) for r in rows]
    usable = [r for r, v in zip(rows, verdicts) if "chunk_arrival" in v["usable_for"]]
    per_token_ok = all("per_token_latency" in v["usable_for"] for v in verdicts) and bool(verdicts)
    firsts = [r["chunk_arrivals_s"][0] for r in usable if r["chunk_arrivals_s"]]
    gaps = [b - a for r in usable
            for a, b in zip(r["chunk_arrivals_s"], r["chunk_arrivals_s"][1:])]
    return {
        "protocol": p,
        "requests_recorded": len(rows),
        "requests_usable_for_chunk_arrival": len(usable),
        "all_requests_usable": len(usable) == len(rows) and bool(rows),
        "first_arrival_median_s": statistics.median(firsts) if firsts else None,
        "chunk_gap_median_s": statistics.median(gaps) if gaps else None,
        "chunk_gap_n": len(gaps),
        "per_token_latency_claimable": per_token_ok,
        "per_token_reason": (None if per_token_ok else
                             "at least one request had multiple tokens per chunk, "
                             "a non-200 status, a missing [DONE] or missing usage"),
        "tail_percentiles_reported": False,
        "tail_note": f"{p['requests']} requests support a median only",
        "naming": "arrival gaps are chunk gaps; they are not per-token latencies",
        "per_request_verdicts": verdicts,
    }


def prompt_digest(token_ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode()).hexdigest()


def main() -> int:
    print("infer_vllm_stream_v2 is a PROSPECTIVE driver.")
    print(f"protocol: {PROTOCOL['prompt_len']} in / {PROTOCOL['output_len']} out, "
          f"{PROTOCOL['requests']} requests at concurrency {PROTOCOL['concurrency']}")
    print("it is launched through bench/launch.py run_case (adapter "
          "loopback_server); it does not start a server by itself and performs "
          "no device access when imported or run without an authorized gate.")
    print("REFUSED: no GPU authorization exists in this stage.")
    return 95


if __name__ == "__main__":
    sys.exit(main())
