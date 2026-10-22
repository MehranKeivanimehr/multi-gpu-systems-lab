# Streaming protocol: a preserved contradiction and the prospective policy

## The contradiction (both sources immutable, neither edited)

| source | input | output |
|---|---|---|
| `bundle/plan/run_plan.json` -> `S_rtx_1_stream_latency.workload` (dedicated) | `prompt_len: 256` | **`output_len: 64`** |
| `bundle/plan/vllm_frozen_config.json` (shared) | `prompt_len: 256` | **`max_new_tokens: 128`**, declared to apply *identically* to `vllm_offline_tp1`, `vllm_offline_tp2` and `vllm_loopback_streaming` |

The two frozen documents disagree about the streaming output length. The
executed arm used **64**, which **matches the dedicated streaming section** and
contradicts only the shared config's identical-across-arms claim.

### Correction to the previous stage
The previous correction stage recorded this as the streaming arm "not running
the declared workload". That was wrong in attribution: the arm ran the output
length its own dedicated plan section declared. What is actually defective is
the shared config's claim that one workload applied identically to all three
arms. The earlier log is preserved; this is the appended correction.

## Prospective policy: 256 / 64, minimal and scope-preserving
A future repaired streaming arm uses **input 256, output 64**, because that is
the dedicated section's declaration and changing it would widen scope. The
shared config's 128 is **not** adopted for streaming, and the offline arms keep
their own 128 unchanged. No frozen document is edited; this policy is recorded
here and is subject to owner amendment.

**An amendment the owner may prefer instead**, stated once rather than assumed:
make the shared config authoritative at 128 for all three arms, which would buy
cross-arm workload comparability at the cost of changing the streaming scope and
making the historical 64 non-comparable with the repaired arm. This stage does
not choose it.

## What remains qualified or invalid regardless of the above
* The realised input was **278 tokens**, not 256. Unchanged disclosure.
* Arrivals are **non-empty text chunks**, not identified tokens. A chunk is not
  a token; several tokens sharing one arrival do not yield individual arrival
  times. Chunk gaps remain chunk gaps unless token identity and attribution are
  proven.
* No raw SSE, HTTP status, `usage`, `finish_reason` or confirmed cleanup exists
  in the delivered records.
