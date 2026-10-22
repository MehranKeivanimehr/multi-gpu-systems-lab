# Closing window — what was executed and what it established

Authorization `CLOSED-20261020T125221Z`, issued 2026-10-20T12:52:21Z,
expiry 2026-10-20T14:52:21Z, nonrenewable, 7200 s.
Used **266.1 s**; **4** of
6 permitted attempts; no safety
stop; cleanup verified after every attempt
(`cleanup_verified: True`).

Approved devices, one job and one GPU at a time:

| case | device | runtime | disposition |
|---|---|---|---|
| `R_p1080_1_tinyllama_fp32` | GTX_1080Ti_1 (physical index 1) | 33.9 s | completed with supported checks |
| `R_rtx_1_tinyllama_fp32` | TITAN_RTX_1 (physical index 0) | 33.7 s | completed with supported checks |
| `S_rtx_1_stream_256_64` | TITAN_RTX_1 (physical index 0) | 41.7 s, 2 attempts | invalid diagnostic / software failure |

Device identity was checked before every launch by **full** UUID against the
frozen group, never by prefix, together with a foreign-process and occupancy
check. No substitute device, multi-GPU job or second concurrent job occurred.

## 1. The reference comparison was evaluated

The runners retained the complete final-step logit array for each family, in a
non-pickle raw little-endian binary format with an explicit JSON schema and a
SHA-256 over the exact payload bytes, together with the input ids, the
attention mask and the greedy output ids. The frozen criterion
(`protocols/REFERENCE_CRITERION_CLARIFICATION.md`, frozen before these values
existed) was then recomputed **offline** from those arrays.

| quantity | value |
|---|---|
| dtype | float32 |
| u = finfo(dtype).eps / 2 | 5.9604644775390625e-08 |
| M = max(max abs A, max abs B) | 16.079071044921875 |
| atol = 64u·max(1, M) | 6.133679e-05 |
| rtol term = 64u·max(M, 1e-300) | 6.133679e-05 |
| tolerance | 1.226736e-04 |
| max absolute deviation | 2.765656e-05 |
| deviation as a fraction of tolerance | 22.54% |
| failing elements | 0 of 256,000 |
| worst witness | index 250964, position [7, 26964], -11.310945511 vs -11.310917854 |
| aggregation | maximum absolute deviation over all elements |
| threshold widened | False |
| arrays bitwise identical | False |
| **verdict** | **PASS** |

Greedy output tokens are reported separately: **IDENTICAL**,
0 of 512 differ. Token agreement does not
substitute for the logit comparison.

The inputs were verified to match before the comparison was trusted: realized
input digest, attention-mask digest, initial-state digest, input shape, model
dtype, pinned revision and eval mode all equal, on two **different** device
families (NVIDIA TITAN RTX, NVIDIA GeForce GTX 1080 Ti). Matching is established from
recorded digests, not from seeds.

Two independent implementations agree on every quantity: the repository library
(numpy) and the published tool (standard library only, a different Python
version). Agreement recorded:
`True`.

## 2. The streaming session was not obtained

The Prometheus defect is closed and the server became healthy, the private
per-attempt metrics directory was used by the installed client, and
100 requests were issued and each returned
an HTTP status. **0** stream events were
recorded, and **0**
requests produced a completion marker, so the session evidence is incomplete:

- no raw stream events were recorded
- no completion marker (DONE sentinel or finish_reason)

Root cause, from the installed sources rather than from current online
examples: every completion request builds a guided-decoding processor
(`GuidedDecodingParams.from_optional`, `sampling_params.py:63`, never returns
`None`), so `engine/multiprocessing/client.py:545` always calls the builder and
`guided_decoding/__init__.py:12` imports `outlines` for the default backend;
`outlines` → `datasets 5.1.0` → `huggingface_hub 0.36.2` no longer resolves.
The exception was raised inside the already-started SSE response, which is why
the status was 200 and the body empty.

No streaming metric is reported, no latency distribution is implied, and the
single session would in any case have established only that session's
observations. The gap is **unresolved**, not a negative result.

## 3. What is deliberately not claimed

- No throughput, scaling or energy conclusion follows from any closing-window
  case.
- The reference control is inference agreement, not gradient correctness,
  convergence or retrospective validation of the historical performance
  cohort.
- Independent external review has not occurred.
