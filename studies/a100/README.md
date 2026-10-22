# A100 40 GB reference campaign

Author: Mehran Keivanimehr.

This folder contains the recorded single-GPU Colab A100 campaign: 40 configurations, including four GPT-2 OOM results. Its primary evidence is the five JSONL files, with matching per-benchmark CSVs and `all_results.csv`. [A100 dataset (2026/10), [combined table](all_results.csv)]

- [Corrected scientific interpretation](CURRENT_REPORT.md) is the current report.
- [Detailed historical report](REPORT.md) and [historical generated summary](summary.md) are preserved provenance, not accepted conclusions.
- [Recorded environment](env.json).
- [Clean reproduction notebook](reproduce.ipynb).
- [Exported vLLM benchmark](vllm_bench.py).
- [Original notebook](../../provenance/a100-original.ipynb) for source provenance.

## Reproduction notes

Use an exclusive Colab A100 runtime. The clean notebook is derived from the original training cells and the successful vLLM invocation. Its results go to `/content/gpu_bench/a100_reproduction`, leaving the reference records in this repository intact. Follow its cells in order.

The recorded main environment is PyTorch 2.11.0+cu130, Transformers 5.18.0, Accelerate 1.15.0, Python 3.13.15; the separate vLLM environment records vLLM 0.31.0 and PyTorch 2.13.0+cu132. A full freeze and pinned model revisions were not supplied. Setting those top-level versions cannot guarantee an identical future Colab image. [Recorded versions (2026/10), [environment](env.json) and [vLLM records](infer_vllm.jsonl)]

The historical notebook's Qwen block initially followed the configuration's parameter dtype and then audited/relabelled BF16-only runs before rerunning with FP32 parameters. The clean notebook retains both cells so the two recipes remain distinguishable. [Qwen audit (2026/10), [source notebook](../../provenance/a100-original.ipynb)]

The old summary's cross-GPU table contains unfilled placeholders. The repository-level [report](../../reports/REPORT.md) and [methodology](../../docs/METHODOLOGY.md) describe the available comparisons without filling those cells from incompatible protocols.
