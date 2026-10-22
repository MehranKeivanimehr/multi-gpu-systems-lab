# Stage-1 Benchmark Results -- Heterogeneous 10-GPU Cluster

Auto-generated from `results/*.csv` by `bench/make_report.py`. Narrative, methodology and caveats live in `reports/README.md`.

## 1. Dtype capability (4096^3 GEMM)

Measured rather than assumed: `torch.cuda.is_bf16_supported()` returns True on every GPU here, yet bf16 is slower than fp32 on all of them.

| gpu         |   capability |   fp32_tflops |   fp16_tflops |   bf16_tflops |   fp16_speedup |   bf16_speedup | is_bf16_supported_flag   |
|:------------|-------------:|--------------:|--------------:|--------------:|---------------:|---------------:|:-------------------------|
| GTX 1080 Ti |          6.1 |         9.952 |         8.413 |         5.786 |          0.845 |          0.581 | True                     |
| TITAN RTX   |          7.5 |        13.036 |       104.907 |         8.268 |          8.047 |          0.634 | True                     |
| TITAN Xp    |          6.1 |        10.07  |        11.609 |         5.966 |          1.153 |          0.592 | True                     |

**Policy derived from this:** FP32 on Pascal (6.1), FP32 + AMP-fp16 on Turing (7.5), bf16 never.

## 2. Single-GPU training (GPT-2 124M, seq 512)

Best configuration per GPU and precision:

| group   | precision   |   micro_batch_per_gpu |   tokens_per_s |   peak_alloc_mib |   util_mean_pct |   power_mean_w_per_gpu |   energy_per_1m_tokens_j |
|:--------|:------------|----------------------:|---------------:|-----------------:|----------------:|-----------------------:|-------------------------:|
| rtx_1   | amp_fp16    |                    16 |        27365.2 |          12655.3 |           98.33 |                 263.23 |                   9599.2 |
| rtx_1   | fp32        |                    12 |        11914.7 |          12361.2 |           99.95 |                 266.57 |                  22158.9 |
| xp_1    | fp32        |                     8 |         7514.7 |           8740   |           99.71 |                 227.98 |                  29919.8 |
| p1080_1 | fp32        |                     8 |         7073.9 |           8740   |           99.74 |                 219.2  |                  30630.3 |

Full batch sweep in `results/llm_runs.csv`.

## 3. DDP weak scaling (bs=4/GPU, FP32)

| family      | group   |   n_gpus |   tokens_per_s |   tokens_per_s_per_gpu |   speedup_vs_1gpu |   scaling_efficiency_pct |   energy_per_1m_tokens_j |
|:------------|:--------|---------:|---------------:|-----------------------:|------------------:|-------------------------:|-------------------------:|
| GTX 1080 Ti | p1080_1 |        1 |         6719.8 |                 6719.8 |             1     |                   100    |                  31637.2 |
| GTX 1080 Ti | p1080_2 |        2 |        11857.9 |                 5929   |             1.765 |                    88.23 |                  36024.5 |
| GTX 1080 Ti | p1080_4 |        4 |        22507.2 |                 5626.8 |             3.349 |                    83.73 |                  30557.9 |
| GTX 1080 Ti | p1080_6 |        6 |        32589   |                 5431.5 |             4.85  |                    80.83 |                  22320.8 |
| TITAN RTX   | rtx_1   |        1 |        10753.1 |                10753.1 |             1     |                   100    |                  24710.8 |
| TITAN RTX   | rtx_2   |        2 |        14296.7 |                 7148.3 |             1.33  |                    66.48 |                  32644.2 |
| TITAN Xp    | xp_1    |        1 |         7077.3 |                 7077.3 |             1     |                   100    |                  31556.5 |

### Interconnect (NCCL all-reduce at the 497 MB gradient payload)

| group   |   n_gpus | p2p_all_enabled   |   time_ms |   algbw_gb_s |   busbw_gb_s |
|:--------|---------:|:------------------|----------:|-------------:|-------------:|
| p1080_2 |        2 | True              |    47.212 |        11.04 |        11.04 |
| rtx_2   |        2 | False             |   134.86  |         3.86 |         3.86 |
| p1080_4 |        4 | True              |    69.142 |         7.54 |        11.31 |
| p1080_6 |        6 | True              |    91.198 |         5.71 |         9.52 |

The 497 MB figure is GPT-2 124M's FP32 gradient volume (124M params x 4 B), i.e. exactly what DDP all-reduces each step.

## 4. Parallelism method comparison (bs=4/GPU)

| group   | precision   |   n_gpus | method   |   tokens_per_s |   pct_of_ddp |   peak_alloc_mib |   energy_per_1m_tokens_j |
|:--------|:------------|---------:|:---------|---------------:|-------------:|-----------------:|-------------------------:|
| p1080_2 | fp32        |        2 | ddp      |        11857.9 |        100   |           5571.3 |                  36024.5 |
| p1080_2 | fp32        |        2 | fsdp     |        10981   |         92.6 |           4564.8 |                  37176.7 |
| p1080_2 | fp32        |        2 | zero-2   |        10811.2 |         91.2 |           5242.8 |                  35521.9 |
| p1080_2 | fp32        |        2 | zero-1   |        10772.1 |         90.8 |           5242.8 |                  35724   |
| p1080_4 | fp32        |        4 | ddp      |        22507.2 |        100   |           5571.3 |                  30557.9 |
| p1080_4 | fp32        |        4 | fsdp     |        20669.7 |         91.8 |           4212.3 |                  37081.6 |
| p1080_4 | fp32        |        4 | zero-1   |        20598   |         91.5 |           4886.9 |                  36184.6 |
| p1080_4 | fp32        |        4 | zero-2   |        20561.5 |         91.4 |           4886.9 |                  37693.7 |
| p1080_6 | fp32        |        6 | ddp      |        32589   |        100   |           5571.3 |                  22320.8 |
| p1080_6 | fp32        |        6 | zero-2   |        28423.8 |         87.2 |           4768.2 |                  31235.7 |
| p1080_6 | fp32        |        6 | zero-1   |        28398.1 |         87.1 |           4768.2 |                  31705   |
| p1080_6 | fp32        |        6 | fsdp     |        26302.3 |         80.7 |           4093.1 |                  35710   |
| rtx_2   | amp_fp16    |        2 | zero-1   |        20237.1 |        107.2 |           3911.5 |                  17078.5 |
| rtx_2   | amp_fp16    |        2 | zero-2   |        20111.6 |        106.6 |           3911.5 |                  17238   |
| rtx_2   | amp_fp16    |        2 | ddp      |        18874.1 |        100   |           4930.7 |                  19445.6 |
| rtx_2   | fp32        |        2 | ddp      |        14296.7 |        100   |           5571.3 |                  32644.2 |
| rtx_2   | fp32        |        2 | zero-2   |        10701.1 |         74.9 |           5242.8 |                  35811.3 |
| rtx_2   | fp32        |        2 | zero-1   |        10689.6 |         74.8 |           5242.8 |                  38427.7 |
| rtx_2   | fp32        |        2 | fsdp     |         9823.5 |         68.7 |           4564.8 |                  41768.6 |

DDP wins on throughput in FP32; FSDP wins on memory everywhere. The penalty for sharded methods scales with how poor the fabric is.

## 5. ResNet-50 inference -- the A6000-comparable axis

| gpu         |   capability |   batch_size |   throughput |   achieved_tflops |   theoretical_fp32_tflops |   compute_util_pct |   energy_per_image |
|:------------|-------------:|-------------:|-------------:|------------------:|--------------------------:|-------------------:|-------------------:|
| TITAN RTX   |          7.5 |          128 |      2278.25 |             9.341 |                     16.31 |              57.27 |           0.115744 |
| TITAN Xp    |          6.1 |           64 |       765.75 |             3.14  |                     12.15 |              25.84 |           0.319826 |
| GTX 1080 Ti |          6.1 |           64 |       696.28 |             2.855 |                     11.34 |              25.18 |           0.329468 |

`energy_per_image` is gross joules per image from sampled power. See the README for why the idle-subtracted variant is unreliable at bs=1 and why absolute energy is not comparable to the prior A6000 study.

## 6. Inference (TinyLlama-1.1B, 256-token prompt, 128 new tokens)

| engine                   | gpu         | dtype   |   tensor_parallel |   requests |   output_tokens_per_s |   ttft_ms |   peak_alloc_mib |   energy_per_1k_output_tokens_j |
|:-------------------------|:------------|:--------|------------------:|-----------:|----------------------:|----------:|-----------------:|--------------------------------:|
| hf_transformers_generate | GTX 1080 Ti | float16 |                 1 |          1 |                 27.3  |    111.33 |           2162.8 |                        3775.31  |
| hf_transformers_generate | GTX 1080 Ti | float16 |                 1 |          4 |                 38.65 |    370.26 |           2322.9 |                        2954.3   |
| hf_transformers_generate | GTX 1080 Ti | float16 |                 1 |         16 |                 55.69 |   1501.83 |           2962.5 |                        1957.8   |
| hf_transformers_generate | GTX 1080 Ti | float32 |                 1 |          1 |                 37.94 |     88.08 |           4249.5 |                        3173.36  |
| hf_transformers_generate | GTX 1080 Ti | float32 |                 1 |          4 |                152    |    248.54 |           4382.5 |                        1131.02  |
| hf_transformers_generate | GTX 1080 Ti | float32 |                 1 |         16 |                344.89 |    965.78 |           4912.7 |                         617.534 |
| hf_transformers_generate | TITAN RTX   | float16 |                 1 |          1 |                 35.35 |     29.27 |           2162.8 |                        2009.06  |
| hf_transformers_generate | TITAN RTX   | float16 |                 1 |          4 |                138.05 |     44.15 |           2322.9 |                         519.199 |
| hf_transformers_generate | TITAN RTX   | float16 |                 1 |         16 |                533.55 |    145.2  |           2962.5 |                         291.797 |
| hf_transformers_generate | TITAN RTX   | float32 |                 1 |          1 |                 37.98 |     51.47 |           4249.5 |                        3062.19  |
| hf_transformers_generate | TITAN RTX   | float32 |                 1 |          4 |                144.34 |    168.41 |           4382.5 |                        1246.37  |
| hf_transformers_generate | TITAN RTX   | float32 |                 1 |         16 |                400.06 |    674.15 |           4912.7 |                         545.127 |
| hf_transformers_generate | TITAN Xp    | float16 |                 1 |          1 |                 26.48 |    105.21 |           2162.8 |                        4294.38  |
| hf_transformers_generate | TITAN Xp    | float16 |                 1 |          4 |                 38.17 |    347.12 |           2322.9 |                        3277.38  |
| hf_transformers_generate | TITAN Xp    | float16 |                 1 |         16 |                 60.16 |   1453.44 |           2962.5 |                        2145.11  |
| hf_transformers_generate | TITAN Xp    | float32 |                 1 |          1 |                 38.23 |     77.04 |           4249.5 |                        3585.31  |
| hf_transformers_generate | TITAN Xp    | float32 |                 1 |          4 |                152.21 |    238.45 |           4382.5 |                        1235.76  |
| hf_transformers_generate | TITAN Xp    | float32 |                 1 |         16 |                387.28 |    902.04 |           4912.7 |                         633.848 |
| vllm                     | TITAN RTX   | float16 |                 1 |          1 |                157.21 |    nan    |            nan   |                         951.953 |
| vllm                     | TITAN RTX   | float16 |                 1 |          4 |                598.2  |    nan    |            nan   |                         273.652 |
| vllm                     | TITAN RTX   | float16 |                 1 |         16 |               1871.92 |    nan    |            nan   |                          93.486 |
| vllm                     | TITAN RTX   | float16 |                 1 |         64 |               3979.28 |    nan    |            nan   |                          51.416 |
| vllm                     | TITAN RTX   | float16 |                 2 |          1 |                145.4  |    nan    |            nan   |                        1241.17  |
| vllm                     | TITAN RTX   | float16 |                 2 |          4 |                532.13 |    nan    |            nan   |                         451.562 |
| vllm                     | TITAN RTX   | float16 |                 2 |         16 |               1383.37 |    nan    |            nan   |                         191.084 |
| vllm                     | TITAN RTX   | float16 |                 2 |         64 |               2531.73 |    nan    |            nan   |                         111.523 |

vLLM requires compute capability >= 7.0, so it runs only on the two TITAN RTX; the 8 Pascal GPUs use HF `generate`. These are different serving stacks and are not interchangeable measurements.
