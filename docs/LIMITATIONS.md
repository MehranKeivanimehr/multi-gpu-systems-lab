# Limitations and data availability

Mehran Keivanimehr · October 2026

This is the detailed scope boundary for the release. Qualifications needed to interpret a particular result also remain beside that result. [Current status](SCIENTIFIC_STATUS.md), [A6000 claims](../results/a6000_validation/evidence/70_final_review/CLAIM_LEDGER_v2.csv), [Titan contexts](../results/diagnostics/context_ledger.json).

## Scientific scope

- Synthetic inputs and short diagnostics do not establish convergence, training quality or downstream task accuracy.
- Historical performance and later diagnostics are different cohorts. Later success does not retrospectively validate a run that lacked update or gradient observations.
- There is no matched cross-platform ranking: model lengths, batches, precision, optimizers, serving settings and repeat counts differ.
- A6000 ResNet-50 agreement with FP32, inference output/reference agreement and complete two-GPU validation across every framework remain unestablished.
- Four-GPU/8B training, ZeRO-3 subgroup validation and four-GPU A6000 inference remain deferred. The 8B serving result is a qualified historical observation.
- The matched BF16 contrast concerns one model, device and accumulation boundary. It does not isolate all precision components or certify DeepSpeed generally.
- Titan oracle agreement is conditional on the observed consumed gradient. It is not independent gradient correctness. One boundary with zero observed skips does not establish extended AMP behavior.

## Measurement and statistics

- Historical A6000 training has three records per retained group; the serving contrast has two per arm. The observed spread is not a universal confidence bound or automatic significance threshold.
- The selected historical Titan scaling plot shows point records, not invented repeated-run uncertainty. Fixed per-GPU batch makes it weak scaling.
- Allocator allocated/reserved peaks are not total physical board usage or proof that a model fits a smaller GPU. Approximate MFU is not hardware occupancy.
- Energy headlines and causal bottleneck explanations are not promoted. P2P/NCCL failures are host-specific observations with cause unresolved, not a diagnosed physical fault.
- Streaming is one qualified session. Historical event times share a session origin; true request latency is unavailable. Content events are not demonstrated one-to-one emitted token timestamps. The original listener and aggregate memory conditions were not fully established.

## What can be replayed

| Evidence | From this public release | Boundary |
|---|---|---|
| A6000 original fixed-slice control C | Yes: four FP32 arrays, raw-sample loss records and CPU verifier | 787,968 elements per array; not full-model equality |
| Titan cross-family final logits | Yes: complete retained arrays and verifier | One pinned model/corpus, single-device inference |
| Historical rate/dispersion arithmetic | Yes: source JSON/CSV and derivation tools | Recomputing records is not repeating the experiment |
| A6000 full-vector control and BF16 attribution | Aggregate/index/hash records only | Full payloads remain outside the public package |
| S1a original full-vector recipe comparison | Per-tensor statistics can be reaggregated | Recipe tensors were not retained; the original elementwise comparison cannot be replayed from saved statistics |
| Titan complete training-boundary oracle | Recorded checks and coverage can be reconciled | Full boundary tensors were not retained |

Private full-size A6000 payloads and original collection archives should be backed up and hash-verified before working directories are removed. Metadata anonymization and publication checksums are described in [provenance](../provenance/README.md).

## Operational and release boundaries

The original A6000 outage cause remains unknown. Later clean execution is not a PSU diagnosis or a guarantee of safety. New code changes are not retroactively hardware-tested. Publication source hashes are not old execution authorizations; any future cluster work needs a fresh allocation, identity verification and source-bound plan. Historical unauthorized Titan execution is excluded from results and retained in private incident records; foreign-job impact was unknown.

Portable local tests, optional CPU numerical tests, Linux lifecycle tests and actual GitHub Actions are different contracts. A configured workflow is not a passed remote run. No institutional, vendor or peer-review endorsement is claimed.
