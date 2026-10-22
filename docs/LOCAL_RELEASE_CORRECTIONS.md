# Local release corrections — Titan/1080

Author: Mehran Keivanimehr. 2026-10-20. This is a derived, CPU-only correction of the final on-node archive; no new GPU observations were collected.

Historical note: the A6000-pending statements below describe the 2026-10-20 state. The later [A6000 closeout](A6000.md) and [current status](SCIENTIFIC_STATUS.md) supersede those statements; the Titan qualifications remain in force.

## Supported scope

The GTX 1080 Ti and TITAN RTX training records reconcile one-boundary **conditional optimizer-update agreement**, module/master comparisons and exact partition/rank coverage. The complete training-boundary tensors were not retained: archived scalar verdicts can be reconciled but the full oracle cannot be replayed independently. No independent gradient correctness, convergence, training quality, historical performance certification or new throughput/energy scaling result follows from them. The retained TinyLlama arrays independently support the two-family reference PASS.

The current context census is **17 supported / 2 qualified / 0 numerical failures / 0 software failures**, across 19 terminal contexts. Gate-0 retains its original qualification. Streaming is qualified because its arrival offsets were mislabelled as latency and its original listener/memory containment conditions were not fully met. The earlier 18/1 census is superseded, not erased.

## Corrections and evidence

| Finding | Local correction | What remains unavailable |
|---|---|---|
| F01, timestamp origin | Relabel session-relative arrivals; conventional even-sample median 6.753719 s. Old first-text latency fields are null in derived output. Future collector records request start separately. | True historical per-request latency: request starts were never retained. |
| F02, verifier | Refuse non-finite/backwards clocks, invalid event types or token counts, mismatched IDs, missing metadata and contradictory accounting. No sorting to hide clock reversals. | Missing raw observations cannot be invented. Frozen missing-usage limitation remains explicit. |
| F03, CI | Ship explicit portable, Torch-CPU and Linux POSIX profiles; exclude installed-vLLM contracts from ordinary CI. Root commands now match package layout. | GitHub CI has not run. |
| F04, status | Generate the current phase summary from the ledger; preserve the old finding registry as historical. Acceptance uses --check and verifies an unchanged tree. | A6000 is unfinished and uncertified. |
| F05, privacy | Replace one missed historical measurement UUID with a stable label; original and mapping remain private; update measurement provenance. | Privacy scan is not a guarantee about every possible indirect identifier. |
| F06, future execution | Project-local API socket wrapper restricts the API bind before listen; refuse an unconfirmed loopback listener. Monitor union of client and token/ledger-bound server descendants across sessions; preserve outer ownership token. | New operational code is locally CPU-tested, not installed-vLLM/CUDA validated. No retrospective aggregate server RSS is available. |

Raw events, measured streaming_evidence.json, all retained reference arrays and training observations are unchanged. Corrections are sidecars and revised derived summaries. Historical failures and the unauthorized-execution incident remain preserved privately and excluded from scientific results.

## Streaming interpretation

100/100 successful requests, 256 reported prompt tokens and 64 reported completion tokens, 64 content events each. There are **6,500 adjacent all-event gaps**, including metadata/completion events; content-only gaps would number 6,300. Median all-event gap 0.006 s. Neither is a per-token emission time. The session-relative first arrival range is 1.223144–12.226531 s; its median is 6.753719 s, **not request latency**. One session does not provide independent session repeats or a tail-latency study.

The original API listener was all-interface; accepted connections recorded in the shipped access log were loopback. The original client monitor did not include the server. The datasets/huggingface-hub conflict remains unresolved; the selected lm-format-enforcer backend avoids the broken path but does not repair the environment.

The live legacy planning helper also retained an obsolete five-buffer oracle estimate and a 1 MiB digest-workspace charge. Local code now charges nine oracle buffers, comparison/transfer allowances and the 16 MiB digest workspace, and labels all totals as planning estimates rather than measurements or admission criteria. The measured Gate-0 RSS is unchanged; the planning model cannot prove fit or allocator/transient headroom.

## Verification and next work

Run `python -B tools/release_checks.py --profile portable` from the repository root. This checks without regenerating documents. The separate `torch` profile requires CPU-only Torch; `posix` requires Linux. Local receipts state exactly what ran. New Linux operational tests are deferred on Windows, not counted as a pass. No claim that an unrun workflow or installed-vLLM path was validated is made.

Titan data collection can remain closed for this qualified scope. No broad training/reference rerun is required. Move to the separate A6000 study; it is not validated by these results. The final review ZIP includes private records: publish only the selected public_repo tree after reviewing the rights/privacy choices, never the entire review archive.
