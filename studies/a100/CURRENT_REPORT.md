# A100 reference: corrected scientific interpretation

Author: Mehran Keivanimehr. Status: historical observations; no scientific-v2 A100 rerun.

This interpretation supersedes the claims in the preserved [historical report](REPORT.md) and [summary](summary.md). The 40 configuration records remain unchanged: 36 completed, four GPT-2 OOMs. [A100 records (2026/10), [JSONL/CSV catalog](all_results.csv)]

## What the records establish

The recorded GPT-2 BF16 batch-16/sequence-1024 loop yielded 66,284 tokens/s. Qwen vLLM at 512 requests yielded 25,384.5 output tokens/s for the recorded model, prompt/output lengths and engine settings. These are scoped point observations; repeated calls in one loaded engine are not independent fresh sessions. [Recorded rates (2026/10), [GPT-2](train_gpt2.jsonl), [vLLM](infer_vllm.jsonl), [implementation](../../provenance/a100-original.ipynb)]

Qwen `bf16-pure` used BF16 parameters and optimizer-state tensors; `bf16` used FP32 parameters with BF16 autocast. They are different numerical recipes, not equivalent-memory implementations of one verified training protocol. Successful update counts, step-level gradients and task-quality comparisons were not recorded. [Precision evidence (2026/10), [Qwen measurements](train_qwen2.5-1.5b.jsonl), [original notebook](../../provenance/a100-original.ipynb)]

## Withdrawn or restricted interpretations

- **Physical capacity:** no assertion that the pure-BF16 workload fits a 16 GiB GPU. At batch 8/sequence 512, the recorded allocator peak was 15.98 GiB allocated, 20.12 GiB reserved and sampled device memory 21.2 GiB. These do not establish portable 16 GiB fit. No subtraction of the allocated peak from board capacity is presented as usable headroom. [Memory evidence (2026/10), [Qwen records](train_qwen2.5-1.5b.jsonl)]
- **Latency:** the fields historically named `ttft_ms` time completed one-token batches; they are not per-request streaming TTFT. Derived decode rates remain proxy estimates based on separate calls. [Timing evidence (2026/10), [notebook](../../provenance/a100-original.ipynb), [vLLM script](vllm_bench.py)]
- **Training work:** FP16 rates are attempted-loop throughput; missing update counters do not prove that skips occurred, nor that every iteration updated parameters. [Evidence boundary (2026/10), [original training code](../../provenance/a100-original.ipynb)]
- **Explanations and extrapolations:** CPU/launch-bound, memory-bound or compute-bound labels are hypotheses without supporting profiler/control evidence. Untested batch-512 training fit, universal model-size ceilings, cost-free TF32 and unchanged precision quality are not established. [Audit (2026/10), [findings A08/A09/A17](../../audit/findings.csv)]
- **Energy/MFU:** rate divided by sampled average board power is not calibrated whole-node efficiency. Analytical MFU omits some work and is not a measured hardware utilization counter or a common cross-campaign denominator. [Definitions (2026/10), [historical implementation](../../provenance/a100-original.ipynb), [methodology](../../docs/METHODOLOGY.md)]

## Remaining evidence

An A100 is not presently available for validation. Retain these data as explicitly qualified historical reference observations. Restoring stronger claims requires the exact target configuration, successful-update/numerical checks, independent repeats, aligned raw power traces and actual streaming timestamps as applicable. No A6000 or Titan rerun can substitute for an A100-specific measurement. [Validation scope (2026/10), [finding register](../../validation/findings.json)]

## Sources

[A100 records, 2026/10] Mehran Keivanimehr. A100 campaign source data and original notebook, linked above.

[Audit, 2026/10] Mehran Keivanimehr. Scientific audit finding registry. [Registry](../../audit/findings.csv).
