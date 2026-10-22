# Titan / GTX 1080 Ti campaign report

**Author: Mehran Keivanimehr**

[Generated measurement tables](SUMMARY.md) provide the numerical detail. [Original historical narrative](../../../provenance/titan-reports-README.original.md) is retained separately; this current report corrects its comparison scope and measurement labels.

## Hardware and execution

The recorded node contains 2 TITAN RTX (24 GB), 7 GTX 1080 Ti (11 GB), and 1 TITAN Xp (12 GB), with no recorded NVLink links. The groups used homogeneous devices, up to six GTX 1080 Ti or two TITAN RTX. [Titan inventory (2026/10), [preflight](../results/raw/00_preflight.json)]

The training stack records PyTorch 2.4.1+cu121; the separate vLLM snapshot records PyTorch 2.4.0, vLLM 0.6.3.post1, and Transformers 4.45.2. [Titan software (2026/10), [training record](../results/raw/A_rtx_1_bs4_fp32.json) and [vLLM snapshot](../requirements-vllm.txt)]

## Training

GPT-2 124M is randomly initialized from its architecture configuration and trained on synthetic tokens at sequence length 512 with AdamW. The successful source run set contains 43 training configurations; a 44th DDP configuration with batch 8/GPU on two GTX 1080 Ti recorded OOM. [Titan training (2026/10), [trainer](../bench/train_llm.py) and [table](../results/llm_runs.csv)]

At FP32 batch 4/GPU, DDP scaling efficiency on 2/4/6 GTX 1080 Ti GPUs is 88.23%/83.73%/80.83%; two TITAN RTX GPUs reach 66.48%. The baselines are matching single-GPU runs from each family. [Titan scaling (2026/10), [CSV](../results/scaling_efficiency.csv)]

FSDP was run only in FP32, using the older FSDP implementation. Native DDP and DeepSpeed both have TITAN RTX FP16 arms. The recorded `amp_fp16` name for DeepSpeed enables its FP16 configuration and should not be read as proof that it uses identical parameter/optimizer precision to native autocast. ZeRO-3 was not measured in this campaign. [Titan methods (2026/10), [trainer](../bench/train_llm.py), [DeepSpeed driver](../scripts/40_deepspeed.sh), [training rows](../results/llm_runs.csv)]

## Interconnect

The NCCL probe records P2P unavailable for the TITAN RTX pair and available for the measured GTX 1080 Ti groups. At the largest probe setting, bus bandwidth is 3.86 GB/s for two TITAN RTX, 11.04 GB/s for two GTX 1080 Ti, 11.31 GB/s for four, and 9.52 GB/s for six. These are host-specific observations. [Titan NCCL (2026/10), [raw records](../results/raw/) and [headline table](../results/comm_bandwidth.csv)]

The probe allocates **497 MiB**, although the historical field is named `size_mb`. GPT-2's recorded parameter count gives approximately 474.70 MiB of FP32 gradients, so the probe is a nearby synthetic payload rather than an exact match. Bandwidth is decimal GB/s and uses the ring factor `2*(N-1)/N`. [Titan units (2026/10), [implementation](../bench/comm_bench.py) and [parameter count](../results/raw/A_rtx_1_bs4_fp32.json)]

## Inference and precision

The dtype probe measures BF16 GEMM slower than FP32 on three representative devices, not all ten GPUs individually. The PyTorch 2.4.1 flag includes emulation by default; it is not a native-acceleration promise. The campaign uses FP32 on the Pascal training arms and adds FP16 on TITAN RTX, without generalizing GEMM behavior to every workload. [Titan dtype probe (2026/10), [CSV](../results/dtype_gemm.csv); PyTorch (v2.4.1), [flag definition](https://raw.githubusercontent.com/pytorch/pytorch/v2.4.1/torch/cuda/__init__.py)]

TinyLlama-1.1B inference uses a 256-token prompt and 128 generated tokens. HF `generate` records both FP32 and FP16 on all three GPU families; vLLM records FP16 on TITAN RTX only. At 16 requests, HF FP32 is faster than FP16 on the two Pascal families in these records. The vLLM arm reaches 3,979.28 output tokens/s at TP1/64 requests and 2,531.73 at TP2/64 requests. [Titan inference (2026/10), [CSV](../results/inference.csv)]

ResNet-50 is a forward-only inference benchmark. Its best recorded throughput is 2,278.25 images/s for TITAN RTX AMP-FP16 at batch 128, 1,062.14 images/s for TITAN RTX FP32 at batch 128, 765.75 for TITAN Xp FP32 at batch 64, and 696.28 for GTX 1080 Ti FP32 at batch 64. The generated table includes precision. [Titan CNN (2026/10), [CSV](../results/cnn_resnet50.csv)]

## Comparison scope

The supplied A6000 suite records ResNet training and Qwen/Llama language-model workloads; its protocols do not directly match this campaign's GPT-2 training, ResNet inference, or TinyLlama inference. The old narrative referred to a separate A6000 CNN study whose source records were not supplied. Its numerical claims are excluded from this current comparison. [Comparison evidence (2026/10), [historical report](../../../provenance/titan-reports-README.original.md), [A6000 config](../../a6000/configs/experiments/train_scaling.yaml), [comparison matrix](../../../docs/METHODOLOGY.md#comparison-matrix)]

## Limitations and reproduction

The dataset supplies one run record per configuration and no full console logs. Historical power integration omits the workload boundaries, so those energy values are partial-window estimates, not validated total-workload energy. Net energy is diagnostic only; a P-state explanation is unverified. [Titan energy audit (2026/10), [findings A02/A14](../../../audit/findings.csv)]

Use the [corrected CNN table](../../../results/corrections/titan_cnn.csv): 4.1 approximate GMAC/image corresponds to 8.2 GFLOP/image with FMA=2, and AMP has no FP32-peak percentage. The values are analytical model-work estimates, not occupancy. The [corrected inference table](../../../results/corrections/titan_inference.csv) labels standalone forward durations as prefill proxies, excludes one token per request from the decode proxy, and does not claim actual TTFT. Batch completion duration is not request latency. [Correction proof (2026/10), [calculations](../../../results/corrections/proof.json), [tests](../../../tests/test_scientific_metrics.py)]

Historical training lacks successful-update counts and identical-initial-state proof. New scientific-v2 code uses identical model seeds and full initial-state hashes across ranks, counts actual optimizer calls and retains step losses. Optional gradient diagnostics are separate from performance runs. Host execution and reference-equivalence validation remain pending. [Training correction (2026/10), [implementation](../bench/train_llm.py), [status](../../../docs/SCIENTIFIC_STATUS.md)]

Read the [reproduction guide](../../../docs/REPRODUCING.md#titan--gtx-1080-ti-gpu-execution) for the separate environment installs and ordered launchers. The historical full vLLM package snapshot contains an unused dependency conflict; the curated environment is a new setup recipe that still needs target-host validation. Repository preparation verified data and source consistency offline and did not rerun GPU workloads.
