# Single-GPU reference benchmarks — NVIDIA A100-SXM4-40GB (Google Colab)

## Environment

- GPU: NVIDIA A100-SXM4-40GB, 39.5 GiB, compute capability 8.0, 108 SMs
- nvidia-smi: `NVIDIA A100-SXM4-40GB, 40960 MiB, 580.82.07, 400.00 W, 1410 MHz`
- Training / HF inference: torch 2.11.0+cu130 (CUDA 13.0, cuDNN 92700), transformers 5.18.0, Python 3.13.15
- vLLM inference (separate venv): vLLM 0.31.0, torch 2.13.0+cu132, Python 3.12
- Run date: 2026-10-14

## Method

- Synthetic inputs generated on the GPU (no dataloader / disk I/O). Fixed seeds.
- Training: 10 warmup + 40 timed steps (Qwen: 5 + 20). Full optimizer step included.
- `fp32` = strict IEEE FP32 (TF32 off); `tf32` = FP32 with TF32 matmul/conv; `fp16`/`bf16` = autocast AMP (FP16 uses GradScaler).
- GPU util / power / memory sampled via NVML every 100 ms during the timed region. `peak_alloc_gb` = `torch.cuda.max_memory_allocated` (GiB).
- `approx_mfu_pct` = 6·N·tokens/s (ResNet: 3×fwd FLOPs) vs A100 dense peak (FP32 19.5, TF32 156, FP16/BF16 312 TFLOPS); excludes attention FLOPs and recompute.
- Inference: 128 input tokens, exactly 256 output tokens, greedy decoding. TTFT = median time of a 1-token generate for the whole batch.

## Training

### GPT-2 124M (seq 1024, AdamW fused, SDPA attention)

| precision   |   batch | status   |   tokens_per_s |   step_ms |   peak_alloc_gb |   power_avg_w |   tokens_per_joule |   approx_mfu_pct |
|:------------|--------:|:---------|---------------:|----------:|----------------:|--------------:|-------------------:|-----------------:|
| fp32        |       8 | ok       |          18479 |     443.3 |           14.65 |         346.6 |               53.3 |             70.8 |
| fp32        |      16 | ok       |          18984 |     863.1 |           27.9  |         360.9 |               52.6 |             72.7 |
| fp32        |      32 | OOM      |            nan |     nan   |          nan    |         nan   |              nan   |            nan   |
| tf32        |       8 | ok       |          48612 |     168.5 |           14.65 |         356.5 |              136.4 |             23.3 |
| tf32        |      16 | ok       |          50451 |     324.8 |           27.9  |         369.4 |              136.6 |             24.1 |
| tf32        |      32 | OOM      |            nan |     nan   |          nan    |         nan   |              nan   |            nan   |
| fp16        |       8 | ok       |          62553 |     131   |           12.63 |         306.1 |              204.4 |             15   |
| fp16        |      16 | ok       |          65505 |     250.1 |           23.6  |         313.9 |              208.7 |             15.7 |
| fp16        |      32 | OOM      |            nan |     nan   |          nan    |         nan   |              nan   |            nan   |
| bf16        |       8 | ok       |          63687 |     128.6 |           12.63 |         294.6 |              216.2 |             15.2 |
| bf16        |      16 | ok       |          66284 |     247.2 |           23.6  |         301.8 |              219.6 |             15.9 |
| bf16        |      32 | OOM      |            nan |     nan   |          nan    |         nan   |              nan   |            nan   |

### ResNet-50 (224×224, channels_last, SGD momentum)

| precision   |   batch | status   |   images_per_s |   step_ms |   peak_alloc_gb |   power_avg_w |   images_per_joule |   approx_mfu_pct |
|:------------|--------:|:---------|---------------:|----------:|----------------:|--------------:|-------------------:|-----------------:|
| fp32        |      64 | ok       |          444.2 |     144.1 |            5.42 |         310.4 |               1.43 |             56   |
| fp32        |     128 | ok       |          464.6 |     275.5 |           10.58 |         324   |               1.43 |             58.6 |
| fp32        |     256 | ok       |          479.5 |     533.9 |           21.52 |         331.5 |               1.45 |             60.5 |
| tf32        |      64 | ok       |         1040.4 |      61.5 |            5.41 |         309.1 |               3.37 |             16.4 |
| tf32        |     128 | ok       |         1075.8 |     119   |           10.57 |         323.4 |               3.33 |             17   |
| tf32        |     256 | ok       |         1103.4 |     232   |           20.91 |         327.6 |               3.37 |             17.4 |
| fp16        |      64 | ok       |         1870.3 |      34.2 |            2.95 |         272.3 |               6.87 |             14.7 |
| fp16        |     128 | ok       |         2188.6 |      58.5 |            5.63 |         316   |               6.93 |             17.3 |
| fp16        |     256 | ok       |         2277.9 |     112.4 |           10.96 |         330.7 |               6.89 |             18   |
| bf16        |      64 | ok       |         1747.2 |      36.6 |            2.95 |         282.4 |               6.19 |             13.8 |
| bf16        |     128 | ok       |         1868.7 |      68.5 |            5.63 |         299.3 |               6.24 |             14.7 |
| bf16        |     256 | ok       |         1943.7 |     131.7 |           10.96 |         311.9 |               6.23 |             15.3 |

### Qwen2.5-1.5B (random init from config, AdamW fused, SDPA)

`bf16-pure` = BF16 weights and BF16 optimizer states (memory-lean, not the standard recipe). `bf16` + `fp32` master = standard mixed-precision training.

| precision   | master_weights                          |   batch |   seq_len | grad_ckpt   | status   |   tokens_per_s |   step_ms |   peak_alloc_gb |   power_avg_w |   tokens_per_joule |   approx_mfu_pct |
|:------------|:----------------------------------------|--------:|----------:|:------------|:---------|---------------:|----------:|----------------:|--------------:|-------------------:|-----------------:|
| bf16-pure   | none (bf16 weights + bf16 AdamW states) |       4 |       512 | True        | ok       |           9839 |     208.2 |           12.39 |         321.8 |               30.6 |             29.2 |
| bf16-pure   | none (bf16 weights + bf16 AdamW states) |       8 |       512 | True        | ok       |          12008 |     341.1 |           15.98 |         373.9 |               32.1 |             35.6 |
| bf16-pure   | none (bf16 weights + bf16 AdamW states) |       4 |      1024 | True        | ok       |          11793 |     347.3 |           15.98 |         381.6 |               30.9 |             35   |
| bf16-pure   | none (bf16 weights + bf16 AdamW states) |       2 |       512 | False       | ok       |           8739 |     117.2 |           13.19 |         280.7 |               31.1 |             25.9 |
| bf16        | fp32                                    |       4 |       512 | True        | ok       |           7926 |     258.4 |           23.95 |         305.6 |               25.9 |             23.5 |
| bf16        | fp32                                    |       8 |       512 | True        | ok       |           9995 |     409.8 |           26.73 |         352.9 |               28.3 |             29.7 |
| bf16        | fp32                                    |       4 |      1024 | True        | ok       |           9870 |     415   |           26.73 |         361.3 |               27.3 |             29.3 |
| bf16        | fp32                                    |       2 |       512 | False       | ok       |           6790 |     150.8 |           25.11 |         285.6 |               23.8 |             20.2 |

## Inference — Qwen2.5-1.5B-Instruct, BF16

| engine          |   batch |   ttft_ms |   e2e_s |   output_tokens_per_s |   per_seq_decode_tok_s |   gpu_util_avg |   power_avg_w |   tokens_per_joule |
|:----------------|--------:|----------:|--------:|----------------------:|-----------------------:|---------------:|--------------:|-------------------:|
| hf-transformers |       1 |      34.3 |    7.67 |                  33.4 |                   33.4 |           31.2 |          64.8 |               0.52 |
| hf-transformers |       8 |      33.6 |    7.93 |                 258.3 |                   32.3 |           32.5 |          68.3 |               3.78 |
| hf-transformers |      32 |      78.5 |    8.09 |                1012.3 |                   31.8 |           36   |          79.6 |              12.72 |
| vllm            |       1 |      13.1 |    1.07 |                 238.7 |                  240.8 |           97.6 |         200.9 |               1.19 |
| vllm            |       8 |      37.9 |    1.07 |                1908.6 |                  246.3 |           90.9 |         207.6 |               9.19 |
| vllm            |      32 |      78.9 |    1.22 |                6708.6 |                  223.3 |           93.4 |         233.6 |              28.72 |
| vllm            |     128 |     248.5 |    1.71 |               19151.3 |                  174.4 |           96.7 |         308.3 |              62.12 |
| vllm            |     512 |     883.7 |    5.16 |               25384.5 |                   59.6 |           99.2 |         351.4 |              72.24 |

## Key findings

1. **Transformers train fastest in BF16.** GPT-2: 66.3k tok/s (BF16) vs 19.0k (strict FP32); 4.2× more tokens per joule. TF32 alone gives a free 2.7× over strict FP32.
2. **CNNs prefer FP16 AMP over BF16 on A100** (ResNet-50: 2,278 vs 1,944 img/s at bs256, +17%), while for GPT-2 the two are tied. FP16 AMP is 4.8× more energy-efficient than FP32.
3. **40 GB is comfortable for full fine-tuning of ~1.5B models.** Qwen2.5-1.5B with FP32 master weights: 10.0k tok/s at 26.7 GiB peak (~30% MFU). Pure-BF16 is 20% faster and fits in 16 GiB.
4. **Small-model eager training is overhead-bound.** GPT-2 BF16 reaches only ~16% MFU (LayerNorm/GELU/dropout/50k-vocab loss/launch overhead); the larger Qwen reaches ~30%.
5. **Serving engine matters more than the GPU for decode.** HF `generate` is CPU/launch-bound (~33 tok/s per sequence regardless of batch, GPU util ~33%, 65–80 W). vLLM is 7.1× faster at bs1 (239 tok/s, ~48% of the memory-bandwidth roofline) and peaks at 25.4k out tok/s with 512 concurrent requests (72 tok/J).
6. **Batch saturation points:** GPT-2 saturates at bs8–16 (+3% from 8→16), ResNet-50 AMP at ~bs128–256, vLLM at ~128–512 concurrent requests.

## Caveats

- Single run per config; expect a few % run-to-run variance. Colab A100 is a shared host.
- Synthetic data: numbers exclude input pipeline; real dataloaders (esp. ImageNet JPEG decode) will be lower.
- NVML `gpu_util` only measures the fraction of time any kernel runs — not SM occupancy. Power is more informative.
- HF inference numbers depend strongly on host CPU and library versions; compare across machines with care.
- vLLM: `gpu_memory_utilization=0.6`, prefix caching off, FlashInfer sampler disabled (`VLLM_USE_FLASHINFER_SAMPLER=0`; irrelevant for greedy decoding). Engine startup (67.7 s) excluded. Per-seq decode at n=512 is not meaningful (queueing).
- vLLM ran with a newer torch (separate venv) than the training/HF benchmarks.
- GPT-2 bs32 @ seq1024 OOMs at all precisions (50k-vocab logits + activations); throughput had already saturated at bs16.

## Cross-GPU comparison (fill in from previous runs)

Only compare rows whose config matches exactly (model, precision, batch, seq/img size, synthetic vs real data). Pascal GPUs (1080 Ti, TITAN Xp, CC 6.1) have no tensor cores or BF16 and are not supported by vLLM; TITAN RTX (CC 7.5) supports FP16 tensor cores but not BF16.

| Benchmark                                                             |   A100-SXM4-40GB | RTX A6000   | TITAN RTX   | GTX 1080 Ti   | TITAN Xp   |
|:----------------------------------------------------------------------|-----------------:|:------------|:------------|:--------------|:-----------|
| GPT-2 124M train, strict FP32, bs16, seq1024 (tok/s)                  |          18984   | TBD         | TBD         | TBD           | TBD        |
| GPT-2 124M train, BF16 AMP, bs16, seq1024 (tok/s)                     |          66284   | TBD         | TBD         | TBD           | TBD        |
| GPT-2 124M train, FP16 AMP, bs8, seq1024 (tok/s)                      |          62553   | TBD         | TBD         | TBD           | TBD        |
| ResNet-50 train, strict FP32, bs64 (img/s)                            |            444.2 | TBD         | TBD         | TBD           | TBD        |
| ResNet-50 train, FP16 AMP, bs64 (img/s)                               |           1870.3 | TBD         | TBD         | TBD           | TBD        |
| ResNet-50 train, FP16 AMP, bs256 (img/s)                              |           2277.9 | TBD         | TBD         | TBD           | TBD        |
| Qwen2.5-1.5B train, BF16 AMP + FP32 master, bs8, seq512, ckpt (tok/s) |           9995   | TBD         | TBD         | TBD           | TBD        |
| HF generate, Qwen2.5-1.5B-Instruct, bs1 (out tok/s)                   |             33.4 | TBD         | TBD         | TBD           | TBD        |
| vLLM, Qwen2.5-1.5B-Instruct, bs1 (out tok/s)                          |            238.7 | TBD         | TBD         | TBD           | TBD        |
| vLLM, Qwen2.5-1.5B-Instruct, 512 concurrent (out tok/s)               |          25384.5 | TBD         | TBD         | TBD           | TBD        |
