# Verification record and evidence boundary

Author: Mehran Keivanimehr · Publication revision 2026-10-22

## Current public release

This release integrates the reviewed A6000 closeout with the corrected Titan/1080 base. The standard-library profile verifies public measurement checksums, source syntax, local links, generated outputs, retained numerical evidence, deterministic SVG figures and refusal regressions. [Commands and environments](../TESTING.md), [profile implementation](../tools/release_checks.py).

The A6000 verifier recomputes all six original fixed-slice control endpoints from the shipped FP32 arrays and validates the matched-cap serving cohorts against raw results. The Titan reference verifier recomputes its frozen final-logit criterion from complete retained arrays. Streaming recomputation reports the qualified session without inventing historical request latency. [A6000 verifier](../tools/verify_a6000.py), [reference verifier](../tools/recompute_reference_verdict.py), [streaming verifier](../tools/recompute_streaming_metrics.py).

CPU numerical fixtures and Linux lifecycle checks are separate profiles. They are not established by a Windows skip or by the presence of a workflow file. The historical 269-test A6000 receipt and other on-node receipts identify their original sources/environments; they are not reruns of every edited publication byte. [Test boundaries](../TESTING.md), [scientific status](SCIENTIFIC_STATUS.md).

## Source and data identity

The private collection archives remain unchanged. The public tree anonymizes operator metadata and uses its own manifest. Tensor payloads and numerical observations are not replaced with synthetic values. Collection-time hashes and authorizations do not authorize this edited tree for GPU execution. [Publication provenance](../provenance/README.md), [input hashes](../provenance/publication.json), [public evidence manifest](../provenance/measurements.sha256.json).

## Remote CI

GitHub Actions is configured for portable Linux/Windows checks, optional CPU numerical fixtures and Linux process tests. No actual remote workflow run or peer-review endorsement is claimed. The owner should confirm the first run after publication.

## Limitations

Passing these checks establishes only their stated properties. It does not certify convergence, historical gradient correctness, every model/GPU configuration or the root cause of a host outage. Read [the full limitations and replay boundaries](LIMITATIONS.md).
