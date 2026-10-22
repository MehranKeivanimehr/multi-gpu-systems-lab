# Titan / GTX 1080 Ti benchmark campaign

Author: **Mehran Keivanimehr**.

Recorded on a PCIe-only node with 2 TITAN RTX, 7 GTX 1080 Ti, and 1 TITAN Xp. The supplied run set covers GPT-2 training, forward-only ResNet-50 inference, TinyLlama inference, dtype GEMM probes, and NCCL communication. [Titan campaign (2026/10), [preflight](results/raw/00_preflight.json) and [results](results/)]

- [Campaign report](reports/README.md).
- [Scientific correction status](../../docs/SCIENTIFIC_STATUS.md) and [verification / future execution](../../validation/README.md).
- [Generated tables](reports/SUMMARY.md).
- [Project-wide reproduction instructions](../../docs/REPRODUCING.md#titan--gtx-1080-ti-gpu-execution).
- [Methodology and comparison matrix](../../docs/METHODOLOGY.md).
- [Historical README](../../provenance/titan-README.original.md).

## Contents

```text
bench/       11 Python benchmark/analysis modules
scripts/     10 numbered drivers and one shared helper
configs/     DeepSpeed ZeRO-1 / ZeRO-2 settings
results/     7 CSV tables and 64 raw JSON artifacts
reports/     corrected narrative and generated tables
```

Environment directories, installed libraries, and downloaded model caches are excluded. Two separate execution environments are required. Recorded package snapshots are `requirements-train.txt` and `requirements-vllm.txt`; curated install files end in `-minimal.txt`. The `environment.yml` file is a training-environment starting point, not a combined vLLM/training setup. See the reproduction guide for the complete install sequence.

## Campaign-local findings

Four GTX 1080 Ti GPUs reached 83.73% GPT-2 FP32 DDP efficiency, while the TITAN RTX pair reached 66.48%. These are weak-scaling results at batch 4/GPU, sequence 512, relative to each family's matching single-GPU baseline. [Titan scaling (2026/10), [CSV](results/scaling_efficiency.csv)]

The communication probe found P2P unavailable for the TITAN RTX pair on this host. It measured 3.86 GB/s versus 11.31 GB/s for the four-card GTX 1080 Ti group, using a 497 MiB buffer. These observations characterize the recorded machine. [Titan communication (2026/10), [RTX raw record](results/raw/C_comm_rtx_2.json), [table](results/comm_bandwidth.csv), [source](bench/comm_bench.py)]

## Regenerate tables

Use an environment with pandas and tabulate:

```bash
python bench/make_report.py
```

Do not regenerate or overwrite the frozen historical CSVs with modified code. New GPU records default to `results/validation-v2`; set `BENCH_RESULTS_DIR` to a fresh session directory. Inference requires an exact `MODEL_REVISION`. The [execution guidance](../../validation/README.md) governs future work; do not launch the full historical matrix automatically.
