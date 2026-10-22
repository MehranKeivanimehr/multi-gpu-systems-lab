# NVIDIA A100-SXM4-40GB — Single-GPU Benchmark Report

Single-GPU reference measurements for the **Multi-GPU Performance Benchmark** project, collected on a Google Colab A100 in October 2026. The goal is a clean, reproducible baseline for training and inference on a modern data-center GPU: throughput, memory, power and energy efficiency across precisions.

## At a glance

| Workload | Best configuration | Result |
|:--|:--|:--|
| GPT-2 124M training | BF16 AMP, batch 16, seq 1024 | **66.3k tokens/s**, 220 tokens/J |
| ResNet-50 training | FP16 AMP, batch 256, 224² | **2,278 images/s**, 6.9 images/J |
| Qwen2.5-1.5B full training | BF16 AMP + FP32 master weights, 8×512 tokens, grad checkpointing | **10.0k tokens/s**, 26.7 GiB peak |
| LLM inference, single request | vLLM, Qwen2.5-1.5B-Instruct | **239 tokens/s**, 13 ms TTFT |
| LLM inference, max throughput | vLLM, 512 concurrent requests | **25.4k tokens/s**, 72 tokens/J |

**Main takeaways**

- Mixed precision is the single biggest lever: **3.5× (GPT-2) and 4.8× (ResNet-50)** over strict FP32, with similar gains in energy per sample. Enabling TF32 alone gives 2.3–2.7×.
- Use **BF16 for transformers, FP16 for CNNs**: they tie on GPT-2, but FP16 is 17% faster on ResNet-50.
- 40 GB comfortably holds **full fine-tuning of a 1.5B model** with the standard AdamW recipe.
- For LLM serving, **the inference engine matters more than the GPU**: vLLM is ~7× faster than Hugging Face `generate` at every batch size measured.

## Hardware and software

| | |
|:--|:--|
| GPU | NVIDIA A100-SXM4-40GB, 39.5 GiB usable, compute capability 8.0, 108 SMs, 400 W power limit, 1410 MHz max SM clock |
| Driver | 580.82.07 |
| Training + HF inference | PyTorch 2.11.0+cu130 (CUDA 13.0, cuDNN 9.27), transformers 5.18.0, Python 3.13 |
| vLLM inference | vLLM 0.31.0, PyTorch 2.13.0+cu132, Python 3.12 (isolated virtual environment) |
| Platform | Google Colab (shared host) |

## Methodology

- **Synthetic data** generated directly on the GPU with fixed seeds. Numbers measure GPU compute only; no dataloader or disk I/O.
- **Training:** 10 warmup + 40 timed steps per configuration (Qwen: 5 + 20). Every timed step includes forward, backward and the full optimizer step.
- **Precision modes:**
  - `FP32`: strict IEEE FP32, TF32 disabled
  - `TF32`: FP32 code with TF32 tensor-core math for matmuls and convolutions
  - `FP16` / `BF16`: autocast mixed precision (FP16 uses a gradient scaler)
- **Telemetry:** NVML sampled every 100 ms during the timed region for GPU utilization, power and memory. Peak memory is `torch.cuda.max_memory_allocated`, reported in **GiB**.
- **Approximate MFU** (model FLOPs utilization) = 6 × parameters × tokens/s, compared against A100 dense peaks: FP32 19.5, TF32 156, FP16/BF16 312 TFLOPS. Attention FLOPs and checkpoint recompute are excluded, so MFU is a lower bound.
- **Inference:** 128 input tokens, exactly 256 output tokens per request, greedy decoding, BF16. Time to first token (TTFT) is the median time to generate one token for the whole batch.
- Out-of-memory configurations are recorded as `OOM`, not dropped.

## Results

### 1. GPT-2 124M training

Sequence length 1024, fused AdamW, PyTorch SDPA attention.

| Precision | tok/s (bs 8) | tok/s (bs 16) | Speedup vs FP32 | Peak mem (GiB, bs 16) | Power (W) | tokens/J | ~MFU |
|:--|--:|--:|--:|--:|--:|--:|--:|
| FP32 | 18,479 | 18,984 | 1.00× | 27.9 | 361 | 52.6 | 72.7% |
| TF32 | 48,612 | 50,451 | 2.66× | 27.9 | 369 | 136.6 | 24.1% |
| FP16 | 62,553 | 65,505 | 3.45× | 23.6 | 314 | 208.7 | 15.7% |
| BF16 | 63,687 | 66,284 | **3.49×** | 23.6 | 302 | **219.6** | 15.9% |

- **BF16 is the best choice:** marginally faster than FP16, lower power, and no loss scaling needed. It delivers **4.2× more tokens per joule** than strict FP32.
- **Batch size 16 is already saturated:** going from 8 to 16 adds only 3–5%. Batch 32 runs out of memory at every precision; the 50k-vocabulary logits and their gradients dominate activation memory.
- **Low MFU under mixed precision is a software effect, not a hardware limit.** Tensor cores make matmuls so fast that non-matmul work dominates: LayerNorm, GELU, dropout, the FP32 loss over 50k classes, and per-kernel launch overhead in eager mode. Strict FP32 shows high MFU only because its peak is low.

### 2. ResNet-50 training

224×224 images, `channels_last`, SGD with momentum.

| Precision | img/s (bs 64) | img/s (bs 128) | img/s (bs 256) | Speedup vs FP32 | Peak mem (GiB, bs 256) | Power (W) | images/J |
|:--|--:|--:|--:|--:|--:|--:|--:|
| FP32 | 444 | 465 | 480 | 1.00× | 21.5 | 332 | 1.45 |
| TF32 | 1,040 | 1,076 | 1,103 | 2.30× | 20.9 | 328 | 3.37 |
| FP16 | 1,870 | 2,189 | **2,278** | **4.75×** | 11.0 | 331 | **6.89** |
| BF16 | 1,747 | 1,869 | 1,944 | 4.05× | 11.0 | 312 | 6.23 |

- **FP16 beats BF16 by 17% for CNNs** (7% at batch 64), likely because cuDNN's FP16 convolution and BatchNorm kernels are more mature. For transformers the two tie, so the best precision on A100 depends on the model family.
- **Mixed precision needs larger batches:** going from batch 64 to 256 adds 22% for FP16 but only 8% for FP32, because small batches leave the tensor cores underused.
- **Mixed precision halves memory:** batch 256 in FP16 uses 11 GiB, so batch 512 would fit, but throughput is already leveling off.

### 3. Qwen2.5-1.5B full-parameter training

Randomly initialized from the official config (1.54B parameters), fused AdamW, SDPA attention. Two recipes were measured:

- **BF16 AMP + FP32 master weights:** the standard mixed-precision recipe, using 16 bytes per parameter for weights, gradients and Adam states.
- **Pure BF16:** weights and optimizer states stored in BF16, using 8 bytes per parameter. It saves memory but loses update precision, so it is not a drop-in replacement.

| Recipe | Batch × seq | Grad ckpt | tok/s | Step (ms) | Peak mem (GiB) | Power (W) | tokens/J | ~MFU |
|:--|:--|:--:|--:|--:|--:|--:|--:|--:|
| BF16 AMP + FP32 master | 4 × 512 | yes | 7,926 | 258 | 23.95 | 306 | 25.9 | 23.5% |
| BF16 AMP + FP32 master | **8 × 512** | yes | **9,995** | 410 | 26.73 | 353 | 28.3 | **29.7%** |
| BF16 AMP + FP32 master | 4 × 1024 | yes | 9,870 | 415 | 26.73 | 361 | 27.3 | 29.3% |
| BF16 AMP + FP32 master | 2 × 512 | no | 6,790 | 151 | 25.11 | 286 | 23.8 | 20.2% |
| Pure BF16 | 4 × 512 | yes | 9,839 | 208 | 12.39 | 322 | 30.6 | 29.2% |
| Pure BF16 | 8 × 512 | yes | 12,008 | 341 | 15.98 | 374 | 32.1 | 35.6% |
| Pure BF16 | 4 × 1024 | yes | 11,793 | 347 | 15.98 | 382 | 30.9 | 35.0% |
| Pure BF16 | 2 × 512 | no | 8,739 | 117 | 13.19 | 281 | 31.1 | 25.9% |

- **The standard recipe fits with ~13 GiB to spare:** 10.0k tokens/s at 26.7 GiB peak. Weights, gradients and Adam states alone take ~23 GiB, and activations with checkpointing add only 1–4 GiB.
- **FP32 master weights cost 17–20% throughput** (9,995 vs 12,008 tok/s at 8×512) **and ~11 GiB of memory**, from the larger optimizer step and the per-step weight casts.
- **Tokens per step matter, not their shape:** 8×512 and 4×1024 run at the same speed.
- **A bigger model uses the GPU better:** ~30% MFU here vs ~16% for GPT-2, because the matmuls are larger.
- **Gradient checkpointing is probably unnecessary at this scale.** Without it, batch 2 used only ~2 GiB of activations, so batch 8 should fit in 40 GiB. This was not measured, so the run without checkpointing at batch 8 remains open.

### 4. LLM inference — Qwen2.5-1.5B-Instruct (BF16)

#### Throughput and latency

| Concurrent requests | HF `generate` (out tok/s) | vLLM (out tok/s) | vLLM speedup | HF TTFT (ms) | vLLM TTFT (ms) |
|--:|--:|--:|--:|--:|--:|
| 1 | 33.4 | 238.7 | **7.1×** | 34.3 | 13.1 |
| 8 | 258.3 | 1,908.6 | 7.4× | 33.6 | 37.9 |
| 32 | 1,012.3 | 6,708.6 | 6.6× | 78.5 | 78.9 |
| 128 | — | 19,151 | — | — | 248.5 |
| 512 | — | **25,385** | — | — | 883.7 |

#### GPU usage and energy

| Concurrent requests | HF util / power / tokens/J | vLLM util / power / tokens/J |
|--:|:--|:--|
| 1 | 31% / 65 W / 0.52 | 98% / 201 W / 1.19 |
| 8 | 33% / 68 W / 3.78 | 91% / 208 W / 9.19 |
| 32 | 36% / 80 W / 12.72 | 93% / 234 W / 28.72 |
| 128 | — | 97% / 308 W / 62.12 |
| 512 | — | 99% / 351 W / **72.24** |

- **HF `generate` is bound by the host CPU, not the GPU.** Each sequence decodes at ~33 tok/s whether the batch is 1 or 32. The ~30 ms per step is Python and kernel-launch overhead, while the GPU idles at 65–80 W.
- **vLLM uses the hardware:** batch-1 decoding reaches 241 tok/s per sequence. Each step must read ~3.1 GB of weights, which at 1,555 GB/s caps speed at ~500 tok/s, so vLLM reaches about **48% of the memory-bandwidth limit**.
- **Throughput levels off around 25k tok/s:** going from 128 to 512 concurrent requests adds only 1.33× as decoding turns compute-bound. At that point the A100 produces 72 tokens per joule, 5.7× better than the best HF configuration.
- **Prefill latency is similar for both engines beyond batch 1:** TTFT is ~38 ms vs ~34 ms at batch 8 and ~79 ms for both at batch 32.

## Assessment: what the A100-40GB is good at

**Strengths**

- **Mixed-precision training.** BF16/FP16 tensor cores give a 3.5–4.8× speedup over FP32 and roughly the same gain in energy per sample. BF16 needs no loss scaling, so it is the low-effort default for transformers.
- **High-throughput LLM serving.** HBM2e bandwidth and tensor cores support 25k+ tok/s for a 1.5B model with a batching engine like vLLM.
- **Full fine-tuning of small LLMs.** About 2B parameters is a practical ceiling for plain AdamW with FP32 master weights in 40 GB.

**Limitations**

- **Legacy FP32 code gains little** unless TF32 is enabled: strict FP32 peaks at 19.5 TFLOPS.
- **40 GB constrains larger models.** Models beyond ~2B parameters need LoRA, 8-bit optimizers, FSDP/ZeRO sharding, or the 80 GB variant.
- **No FP8 tensor cores** (introduced with Hopper), so FP8 training and inference recipes don't apply.
- **Small models are overhead-bound in eager mode.** Reaching high utilization needs torch.compile, fused kernels, or larger batches.

## Practical recommendations

1. At minimum, enable TF32: `torch.set_float32_matmul_precision("high")`. It is free 2.3–2.7×.
2. Use **BF16 AMP for transformers** and **FP16 AMP for CNNs** on this GPU.
3. Batch sizes: GPT-2-class models saturate at 8–16 × 1024 tokens. ResNet-50 with AMP needs a batch of 128 or more.
4. For LLM serving, use vLLM (or a similar continuous-batching engine) instead of HF `generate`.
5. For ~1.5B full fine-tuning, the standard BF16 + FP32-master recipe fits easily. Use pure BF16 only when memory forces it.

## Caveats

- **One run per configuration;** expect a few percent run-to-run variance on a shared Colab host.
- **Synthetic data:** real input pipelines, especially ImageNet JPEG decoding, will reduce training throughput.
- **NVML utilization** reports the fraction of time any kernel is running, not SM occupancy; power draw is the more informative signal.
- **HF inference numbers** depend heavily on host CPU speed and library versions; compare them across machines only with matching software.
- **vLLM setup:** `gpu_memory_utilization=0.6`, prefix caching disabled, FlashInfer sampler disabled (`VLLM_USE_FLASHINFER_SAMPLER=0`). The sampler is not used by greedy decoding, and disabling it avoids a JIT build that needs `ninja`. Engine startup (~68 s) is excluded. Per-sequence decode speed at 512 requests reflects queueing and is not meaningful.
- **Different PyTorch versions:** vLLM ran with PyTorch 2.13 in a separate environment, while training and HF inference ran with PyTorch 2.11.
- **Qwen training used random weights.** Throughput is identical to pretrained weights for the same architecture.

## Files

| File | Contents |
|:--|:--|
| `env.json` | Hardware and software environment |
| `train_gpt2.csv` / `.jsonl` | GPT-2 124M training, 4 precisions × 3 batch sizes |
| `train_resnet50.csv` / `.jsonl` | ResNet-50 training, 4 precisions × 3 batch sizes |
| `train_qwen2.5-1.5b.csv` / `.jsonl` | Qwen2.5-1.5B training, both recipes |
| `infer_hf.csv` / `.jsonl` | HF `generate` inference |
| `infer_vllm.csv` / `.jsonl` | vLLM inference |
| `all_results.csv` | All rows combined, with a `benchmark` column |
| `summary.md` | Auto-generated tables (includes an unfilled cross-GPU comparison template) |
| `vllm_bench.py` | vLLM benchmark script |
| `vllm_log.txt`, `vllm_log_fail_ninja.txt` | vLLM run logs (successful run and the initial `ninja` failure) |
