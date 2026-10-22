# The streaming session — what was run and what it supports

> Local correction: streaming is **qualified**, not a fully protocol-conforming timing measurement. True request latency is unavailable; arrival offsets use one session origin. Current census: 17 supported / 2 qualified. See [corrections](../../docs/LOCAL_RELEASE_CORRECTIONS.md).

Authorization `FINAL-20261020T141800Z`, issued 2026-10-20T14:18:00Z,
expiry 2026-10-20T15:48:00Z, nonrenewable, 5400 s, streaming only,
physical GPU 0 (TITAN RTX) only. Used **178.1 s** on
**1 of 3** permitted attempts. No safety
stop. Cleanup verified: `True`.

## The rule, frozen before the session

`studies/titan-1080/protocols/STREAMING_SUCCESS_RULE.md`
(sha256 `860f6954ba4d5514143fa8083c54287334303247ad761f82952fb10fda0dc849`). A request is `successful`
only with a 200 **and** a content event **and** a completion marker **and** no
error event; a server-reported completion length other than
64 makes it `length_mismatch`. HTTP 200 alone is
not success. The rule has one implementation in the repository library and a
second, independent one in the published tool; the test suite asserts they
agree on a matrix that exercises every class.

## Accounting

| class | count |
|---|---|
| successful | **100** |
| length mismatch | 0 |
| incomplete | 0 |
| failed | 0 |
| **expected / recorded** | **100 / 100** |
| accounting complete | True |
| successful without length evidence | 0 |

Every issued request appears in exactly one class. Nothing was discarded, and
the workload was not reduced: 100
requests at concurrency 4,
256 input token ids verified **before**
the first request, and a 64-token output
policy.

## What the server itself reported

| quantity | value |
|---|---|
| prompt tokens | [256] |
| completion tokens | [64] |
| finish reasons | ['length'] |
| requests with finish_reason | 100 |
| requests with `[DONE]` | 100 |
| model / revision | TinyLlama/TinyLlama-1.1B-Chat-v1.0 @ fe8a4ea1ffedaf415f4da2f062534de366a451e6 |
| effective guided-decoding backend | lm-format-enforcer |

## Chunk measurements, named as chunk measurements

| quantity | min | median | max | n |
|---|---|---|---|---|
| session-relative first-event arrival (s; NOT latency) | 1.223 | 6.754 | 12.227 | 100 |
| inter-event gap (s) | 0.000078 | 0.006000 | 0.011106 | 6500 |
| content events per request | 64 | 64 | 64 | 100 |

Clock: `seconds from t_origin, from time.monotonic(); monotonic, same process, not wall-clock and not comparable across machines`.

- First-event values are **session-relative arrival offsets**, including pre-dispatch client scheduling. They are not request latency; request starts were not retained.
- For every request the content-event count equalled the server's own reported
  completion-token count (64). That is an observation about this
  server configuration. It does **not** make an inter-event gap an inter-token
  latency.
- **Per-token timing is withheld**: no request carried per-token identity and per-token timestamps, so no token-level timing is computed; chunk intervals were not divided and text was not retokenized

## Corrections recorded beside the evidence

The measured evidence file is never rewritten. These are recorded in
`streaming_session/evidence_corrections.json` and printed by the recompute
command:

- **CORR_STREAM_001** — `server.loopback_only`: WRONG as recorded: it was an assertion, not a measurement. the listener was bound to 0.0.0.0 (all interfaces)
- **CORR_STREAM_002** — `cuda_peak_alloc_mib`: correct but easy to misread. torch.cuda.max_memory_allocated() inside the RUNNER (client) process, which never allocates on the device: generation happens in the separate server process. It is 0.0 and it is NOT the server's device memory, which this session did not measure
- **CORR_STREAM_003** — `memory_monitor.peak_owned_rss_gib`: correct for what it measures, narrower than the window's limit. the monitor attributes RSS by the runner's own process group (pgid). The server is deliberately started with start_new_session=True so its whole group can be signalled and confirmed gone, which puts it in a DIFFERENT process group, outside this monitor's attribution. The figure is the client/runner group only

## What this session establishes, and what it does not

Establishes: that the frozen streaming workload ran to completion on this node
and this device under the frozen success rule, with the accounting above and
the chunk measurements above, once.

Does not establish: any token-level timing; any repeated-session latency
distribution or tail percentile; anything about training quality, gradient
correctness, throughput scaling, energy, or the historical performance cohort.
The vLLM environment's `datasets`/`huggingface-hub` conflict is still present
and was not repaired.

## Preserved history

This context failed under three earlier authorizations before it succeeded, and
those failures remain published:

- `BATCH-20261020T095841Z` — invalid_diagnostic_or_software_failure: original attempt; preserved unchanged
- `FOLLOWUP-20261020T113750Z` — invalid_diagnostic_or_software_failure: two attempts; the loopback server never became healthy
- `CLOSED-20261020T125221Z` — invalid_diagnostic_or_software_failure: two attempts; the server became healthy and 100 requests were answered with HTTP 200 and empty bodies, caused by the outlines/datasets import chain
- `FINAL-20261020T141800Z` — completed_with_supported_checks: one attempt; the frozen success rule was met
