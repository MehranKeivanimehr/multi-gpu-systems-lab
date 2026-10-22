# Frozen streaming success rule

Frozen before the session that it judges. It decides, per request, what counts
as a streaming observation, and what the session as a whole establishes.

## Why it is written down separately

The previous attempt returned **HTTP 200 for all 100 requests with an empty
body in every one of them**. Any rule that treats a 200 as success would have
reported a complete session from zero observations. This rule exists so that
cannot happen, and so that the bar is fixed before the numbers exist.

## Per-request classification

A request is classified into exactly one of these:

| class | condition |
|---|---|
| `successful` | the success rule below holds AND the length check below is not violated |
| `length_mismatch` | the success rule holds, but the server's own `usage.completion_tokens` is present and is not the frozen output length |
| `incomplete` | the transport succeeded, but there is no content event, or no completion marker |
| `failed` | a non-200 HTTP status, a transport error, or an error event was recorded |

### The success rule

All four must hold, from the retained raw events for that request:

1. **transport** — the recorded HTTP status for the request is `ok` (200 and
   the body was read to its end without a transport exception);
2. **content** — at least one parsed `data:` event carries non-empty `text`;
3. **completion** — that request has a completion marker: a non-null
   `finish_reason`, or the `[DONE]` sentinel;
4. **no error** — no `error` event was recorded for the request.

HTTP 200 alone is explicitly **not** success. Conditions 2 and 3 are the ones
the previous attempt failed.

### The length check

Where the server supplies `usage.completion_tokens` for the request, it must
equal the frozen output length (64). A request whose reported completion length
differs is `length_mismatch` and is **not** counted as successful: the frozen
output policy was not met for it.

Where `usage` is absent, the request can still be `successful`, and it is
additionally counted under `successful_without_length_evidence`. That count is
always reported. It is a limitation of the observation, not a silent pass.

## Session-level accounting

Every issued request appears in exactly one class; the classes sum to the
number of requests issued, which must equal the frozen request count. A request
that produced nothing is reported, never dropped.

Reported for the session:

- `requests_expected`, `requests_issued`, `requests_recorded`;
- the four class counts, plus `successful_without_length_evidence`;
- `frozen_workload_issued` — the issued request count, concurrency, input
  length and output policy all equal the frozen protocol;
- `input_length_verified` — every request's consumed prompt is exactly the
  frozen input length in token ids;
- `accounting_complete` — recorded equals issued equals expected, and the
  classes sum to it.

## What a session may and may not establish

- The execution gap is closed only for the observations the session actually
  supports: it needs `accounting_complete`, `frozen_workload_issued` and at
  least one `successful` request.
- `first_text_latency_s` and inter-event gaps are **chunk** measurements. A
  chunk boundary is not a token boundary. Per-token timing is reported only if
  the payload itself carried per-token identity and per-token timestamps for
  every event; otherwise it is withheld with that reason stated. Token
  timestamps are never derived by dividing chunk intervals, and text is never
  retokenized to invent emission times.
- One session is one session. Its 100 requests are concurrent requests within a
  single prospective session, not independent repeated sessions, and no
  cross-session latency distribution or tail percentile is reported.
- Nothing about training quality, gradient correctness, throughput scaling,
  energy or the historical performance cohort follows from any streaming
  observation.
