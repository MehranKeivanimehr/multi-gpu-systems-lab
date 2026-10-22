# RTX A6000: current interpretation

Author: Mehran Keivanimehr · qualified closeout, 2026-10-22

The A6000 controls and follow-up are complete within their declared scope. This supersedes the older pending-status narrative, not the original measurements. The current reader-facing report is [the A6000 study](../../docs/A6000.md), with [results](../../reports/REPORT.md) and [claim-by-claim evidence](../../results/a6000_validation/evidence/70_final_review/CLAIM_LEDGER_v2.csv).

- Equal-cap serving: DP2 exceeds TP2 by 33.23% for Qwen1.5B and 14.99% for historical Llama8B, two repeats per arm at aggregate cap 512.
- The five-update FP32 one/two-GPU fixed-slice calibration replays from included arrays; all six endpoints pass.
- The matched first-boundary native-BF16/DeepSpeed-BF16 contrast records identical hashes. It is not a general framework correctness certificate.
- Historical performance, observed calls, complete gradient accounting, sampled updates and reference agreement remain separate evidence categories.

[Source evidence (2026/10), [current ledger](../../results/a6000_validation/evidence/70_final_review/CLAIM_LEDGER_v2.csv), [replay](../../tools/verify_a6000.py)].

## Limitations

ResNet numerical agreement, A6000 inference correctness and complete all-framework two-GPU validation are not established. Four-GPU/8B training and ZeRO-3 validation are deferred. Energy, latency and causal explanations are not promoted. [Full scope](../../docs/LIMITATIONS.md).
