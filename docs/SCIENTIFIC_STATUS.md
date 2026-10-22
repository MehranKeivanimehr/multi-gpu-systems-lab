# Scientific scope and evidence status

Author: Mehran Keivanimehr · Updated 2026-10-22

Current scope is a qualified release of two reviewed cluster studies plus a historical A100 reference. A supported check means the stated check has evidence, not that a framework or an entire training recipe is universally correct.

## Current study ledgers

- Titan/1080: **17 supported / 2 qualified** over 19 contexts. The streaming observations do not establish request latency; Gate-0 retains its original criterion qualification. [Context ledger](../results/diagnostics/context_ledger.json).
- A6000: replayable five-update fixed-slice control; matched-policy first-boundary contrasts; corrected historical serving and repeat analysis. The 28 claims remain separately classified. [Claim ledger](../results/a6000_validation/evidence/70_final_review/CLAIM_LEDGER_v2.csv).
- A100: historical records only; not certified by either cluster closeout. [Interpretation](../studies/a100/CURRENT_REPORT.md).

## A6000 claim dispositions

| Disposition | Claims |
|---|---:|
| deferred | 2 |
| historical_qualified | 1 |
| not_established | 3 |
| supported | 8 |
| supported_qualified | 10 |
| withdrawn | 4 |

The [62-row evidence table](../results/a6000_validation/evidence/70_final_review/CLAIM_EVIDENCE_TABLE.csv) separates calls, gradient coverage, sampled update checks and reference agreement. Old Stage-1 rows are not retrospectively validated. The original finding list remains a [historical correction plan](../validation/findings.json), not the current completion census.

## Recompute

```sh
python -I -S tools/verify_a6000.py
python -I -S tools/recompute_reference_verdict.py --a results/diagnostics/closing/reference/TITAN_RTX_1 --b results/diagnostics/closing/reference/GTX_1080Ti_1
python -B tools/release_checks.py --profile portable
```

The public tree is a metadata-anonymized derivative. Private archive hashes, unchanged tensor payloads and publication checksums have distinct roles; see [provenance](../provenance/README.md). GitHub Actions has not been run as part of this local release.

## Limitations

Short diagnostics are not training-quality tests. Missing historical gradients cannot be reconstructed from a verdict or a seed. Full scope exclusions, data-availability boundaries and operational qualifications are in [LIMITATIONS.md](LIMITATIONS.md).
