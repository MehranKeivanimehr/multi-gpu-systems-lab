| context | ws | stage | precision | boundaries | base calls | updates | skips | max dev | tolerance | discriminating | chunks | elements/rank | whole-model rank set | module/master | cross-rank coverage |

> Local correction: streaming is **qualified**, not a fully protocol-conforming timing measurement. True request latency is unavailable; arrival offsets use one session origin. Current census: 17 supported / 2 qualified. See [corrections](../../docs/LOCAL_RELEASE_CORRECTIONS.md).
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `D_p1080_2_zero1_fp32` | 2 | 1 | fp32 | 2 | 2 | 2 | 0 | 5.960e-08 | 7.630e-06 | True | 15 | 62,219,904 | True | established (fp32_exact_equality) | 124,439,808 of 124,439,808 exactly once |
| `D_p1080_4_zero1_fp32` | 4 | 1 | fp32 | 4 | 4 | 4 | 0 | 1.490e-08 | 4.255e-06 | True | 8 | 31,109,952 | True | established (fp32_exact_equality) | 124,439,808 of 124,439,808 exactly once |
| `D_p1080_6_zero1_fp32` | 6 | 1 | fp32 | 6 | 6 | 6 | 0 | 1.490e-08 | 4.255e-06 | True | 5 | 20,739,968 | True | established (fp32_exact_equality) | 124,439,808 of 124,439,808 exactly once |
| `D_p1080_1_zero2_fp32` | 1 | 2 | fp32 | 1 | 1 | 1 | 0 | 5.960e-08 | 7.630e-06 | True | 30 | 124,439,808 | True | original criterion, see Gate-0 note | n/a at world size 1 |
| `D_p1080_2_zero2_fp32` | 2 | 2 | fp32 | 2 | 2 | 2 | 0 | 5.960e-08 | 7.630e-06 | True | 15 | 62,219,904 | True | established (fp32_exact_equality) | 124,439,808 of 124,439,808 exactly once |
| `D_p1080_4_zero2_fp32` | 4 | 2 | fp32 | 4 | 4 | 4 | 0 | 1.490e-08 | 4.255e-06 | True | 8 | 31,109,952 | True | established (fp32_exact_equality) | 124,439,808 of 124,439,808 exactly once |
| `D_p1080_6_zero2_fp32` | 6 | 2 | fp32 | 6 | 6 | 6 | 0 | 1.490e-08 | 4.255e-06 | True | 5 | 20,739,968 | True | established (fp32_exact_equality) | 124,439,808 of 124,439,808 exactly once |
| `D_rtx_1_zero1_fp32` | 1 | 1 | fp32 | 1 | 1 | 1 | 0 | 5.960e-08 | 7.630e-06 | True | 30 | 124,439,808 | True | original criterion, see Gate-0 note | n/a at world size 1 |
| `D_rtx_2_zero1_fp32` | 2 | 1 | fp32 | 2 | 2 | 2 | 0 | 1.490e-08 | 7.630e-06 | True | 15 | 62,219,904 | True | established (fp32_exact_equality) | 124,439,808 of 124,439,808 exactly once |
| `D_rtx_1_zero2_fp32` | 1 | 2 | fp32 | 1 | 1 | 1 | 0 | 5.960e-08 | 7.630e-06 | True | 30 | 124,439,808 | True | original criterion, see Gate-0 note | n/a at world size 1 |
| `D_rtx_2_zero2_fp32` | 2 | 2 | fp32 | 2 | 2 | 2 | 0 | 1.490e-08 | 7.630e-06 | True | 15 | 62,219,904 | True | established (fp32_exact_equality) | 124,439,808 of 124,439,808 exactly once |
| `D_rtx_1_zero1_amp` | 1 | 1 | amp_fp16 | 1 | 1 | 1 | 0 | 1.490e-08 | 7.630e-06 | True | 30 | 124,439,808 | True | original criterion, see Gate-0 note | n/a at world size 1 |
| `D_rtx_2_zero1_amp` | 2 | 1 | amp_fp16 | 2 | 2 | 2 | 0 | 1.490e-08 | 7.630e-06 | True | 15 | 62,219,904 | True | established (mixed_precision_expected_cast) | 124,439,808 of 124,439,808 exactly once |
| `D_rtx_1_zero2_amp` | 1 | 2 | amp_fp16 | 1 | 1 | 1 | 0 | 1.192e-07 | 7.630e-06 | True | 30 | 124,439,808 | True | original criterion, see Gate-0 note | n/a at world size 1 |
| `D_rtx_2_zero2_amp` | 2 | 2 | amp_fp16 | 2 | 2 | 2 | 0 | 1.192e-07 | 7.630e-06 | True | 15 | 62,219,904 | True | established (mixed_precision_expected_cast) | 124,439,808 of 124,439,808 exactly once |

## Cross-family reference comparison (closing window)

Recomputed offline from the shipped arrays by
`tools/recompute_reference_verdict.py`.

| quantity | value |
|---|---|
| criterion | `titan_reference_logit_criterion_v1` |
| tolerance | 1.226736e-04 |
| max absolute deviation | 2.765656e-05 (22.54% of tolerance) |
| failing elements | 0 of 256,000 |
| arrays bitwise identical | False |
| greedy output tokens | IDENTICAL (0/512) |
| verdict | **PASS** |

## Streaming metrics (closing window)

None. The session recorded 100 requests
and 0 stream events, so no latency metric is
reported. `tools/recompute_streaming_metrics.py` exits 95 on this evidence.

## Streaming session (streaming-only window)

Recomputed offline from the shipped raw events by
`tools/recompute_streaming_metrics.py`, under the frozen success rule.

| quantity | value |
|---|---|
| requests expected / recorded | 100 / 100 |
| successful / length mismatch / incomplete / failed | 100 / 0 / 0 / 0 |
| server-reported completion tokens | [64] |
| session-relative first-event arrival (s; NOT latency), median | 6.754 (min 1.223, max 12.227) |
| inter-event gap (s), median | 0.006000 (n=6500) |
| content events per request | 64–64 |
| per-token timing | withheld |
| verdict | **SESSION_OBSERVED** |

Session-relative arrivals include pre-dispatch client scheduling and are not request latency; request starts were not retained. Content events are chunks, not tokens.
