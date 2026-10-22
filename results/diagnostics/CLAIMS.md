# Claim ledger

> Local correction: streaming is **qualified**, not a fully protocol-conforming timing measurement. True request latency is unavailable; arrival offsets use one session origin. Current census: 17 supported / 2 qualified. See [corrections](../../docs/LOCAL_RELEASE_CORRECTIONS.md).

Each claim states its evidence class. Classes are never merged.

## A. Historical performance measurements (earlier cohort)

Measured on this node in the Stage-1 campaign. Retained unchanged; the new
diagnostics do **not** retroactively certify them.

- 1080 Ti pairs/quads scale better than the TITAN RTX pair (83.7% vs 66.5% DDP
  efficiency at 4 vs 2 GPUs). **P2P capability is an association, not a
  demonstrated causal explanation.**
- Throughput, scaling efficiency and energy rank the groups in three different
  orders; no single-number verdict is complete.
- Energy figures are **gross GPU-board energy** from sampled board power, not
  wall or PSU energy.
- ResNet-50 "achieved TFLOPS" is **analytical** work over time, not hardware
  occupancy.
- Memory peaks are **post-warmup** peaks, not whole-process peaks.
- 90 historical endpoint groups with their actual repeat counts are preserved,
  as is the distinction between the **original frozen endpoint criterion**
  (77/90) and the separately labelled **post-hoc stricter span criterion**
  (74/90). The post-hoc criterion is an informed follow-up, not blind
  preregistration.
- Interval statements assume IID continuous observations (or conditional
  independence with a common median); exchangeability alone is insufficient.

## B. Current numerical diagnostics — qualified closeout

19 terminal contexts: **17 supported / 2 qualified / 0 numerical failures / 0 software failures**. Gate-0 and streaming are qualified. Earlier failure/qualification states remain in the versioned context ledger; they are not the current census.

- All 16 training contexts support one-boundary conditional optimizer-update agreement, not independent gradient correctness. Recorded parameter deviations 1.49e-08–1.19e-07 lie within their declared tolerances. Discrimination is assessed per partition, not individually for every element.
- Four-/six-GPU GTX 1080 Ti contexts did execute. Exact rank/partition reconciliation covers 124,439,808 unique elements once. Later working/master checks pass under the corrected identity/dtype contract. These are not throughput, scaling or energy measurements.
- Full boundary tensors were not retained; the complete training oracle cannot be independently replayed from this package. Gate-0's original non-discriminating module/master criterion remains qualified; its observed numerical difference was zero, but no bit-pattern comparison ran then.
- Independent gradient recomputation, convergence, training quality, extended AMP/skip behavior, retroactive historical performance certification, and A6000 certification are not established.

## C. Cross-family reference

| Claim | Disposition | Evidence and limit |
|---|---|---|
| Final logits agree within the frozen 64u criterion | Supported | 2.765656e-05 maximum deviation vs 1.226736e-04 tolerance; zero failing elements of 256,000. Complete arrays retained and independently recomputed. One pinned TinyLlama model/corpus, single-device inference only. |
| Greedy generated IDs agree | Supported | All 512 identical; token agreement does not imply logit equality. |
| Logits are bitwise equal | Not supported | Arrays differ. A difference alone is not proof of sensitivity to every meaningful alternative; explicit out-of-tolerance CPU controls test comparator refusal/failure behavior. |

## D. Streaming observations — qualified

| Claim | Disposition | Evidence and limit |
|---|---|---|
| Successful streaming responses were observed | Supported within qualified context | 100 of 100 successful requests, complete accounting, 256/64 reported prompt/completion lengths, markers retained. One session, not independent repeated sessions. |
| True per-request first-text latency is available | Withdrawn | No request starts retained. The median 6.753719 s is a session-relative first-event arrival offset, including client work-queue scheduling. |
| Adjacent all-event gaps can be described | Supported | Median 0.006 s over 6,500 gaps, including metadata and completion events. Not the 6,300 content-only gaps or token emission times. |
| The original server listener was loopback-only | Withdrawn | It listened on all interfaces. Accepted connections in the retained log were loopback. Local future code does not rewrite this fact. |
| Aggregate server/client host RSS was measured | Not established | Original monitor covered client group only. No retrospective aggregate figure is available. |
| The dependency conflict is repaired | Not established | lm-format-enforcer avoids the broken outlines import path; the datasets/huggingface-hub conflict remains. |
| New operational code is hardware-validated | Not established | CPU socket, schema and monitor regressions only. Linux process and installed-vLLM/CUDA behavior are separately identified in acceptance receipts. |

## E. Review status

The local review independently checked archive integrity, numerical reference arrays and record reconciliation. It is not a new CUDA experiment or a complete independent replay of missing training tensors. No external peer review or GitHub CI run is claimed. A6000 is unfinished, separate and unchanged.
