# Audit re-runs (2026-10-13, 18:36-19:31)

Two targeted re-runs requested by the results audit. Nothing else was re-run or changed; the original results in
`results/raw/train_scaling/` and `results/raw/vllm_scaling/` are untouched.

| | Re-run 1: Llama-3.1-8B memory headroom | Re-run 2: vLLM TP at matched concurrency |
|---|---|---|
| Config | `configs/experiments/llama8b_memory_rerun.yaml` | `configs/experiments/vllm_tp_concurrency_rerun.yaml` |
| Raw results | `results/raw/llama8b_memory_rerun/` (8 runs) | `results/raw/vllm_tp_concurrency_rerun/` (8 runs) |
| Command | `.venv/bin/python scripts/run_matrix.py configs/experiments/llama8b_memory_rerun.yaml` | `.venv/bin/python scripts/run_matrix.py configs/experiments/vllm_tp_concurrency_rerun.yaml` |
| GPUs | 4-7 | 4-5 (TP2), 4-7 (TP4) |
| Outcome | 8/8 OK, no OOM, no leftover processes | 8/8 OK, no OOM, no leftover processes |

Every run folder has `command.txt`, `spec.json`, `run.log`, `result.json`, `status.json` and `nvidia_smi_{before,after}.txt`.
Tables: `results/rerun/tables.md`; per-run data: `llama8b_memory_comparison.csv`, `vllm_tp_concurrency_comparison.csv`
(regenerate with `.venv/bin/python scripts/compare_reruns.py`).

Code changes (minimal): per-run `mem_fraction` override (hard ceiling 0.95) and allocator counters
(`num_alloc_retries`, `num_ooms`, cudaMalloc/cudaFree counts) in training `result.json`; per-layout `max_num_seqs` for vLLM;
optional per-workload `repeats`. Unit tests: 14/14 pass.

## Re-run 1: Llama-3.1-8B training on 4 GPUs

Same workload as the original (2 x 1024 tokens per GPU, activation checkpointing, fp32 gradient reduction) unless noted.

| Setting | Framework | Runs | Tokens/s | vs original | Step (ms) | Peak alloc / reserved (GiB) | Allocator retries in timed window |
|---|---|---:|---:|---:|---:|---:|---:|
| original (cap 0.92) | FSDP2 | 3 | 994 | 1.000x | 8,241 | 37.4 / 43.3 | not recorded |
| original (cap 0.92) | ZeRO-3 | 3 | 973 | 1.000x | 8,429 | 39.3 / 43.6 | not recorded (35 DeepSpeed flushes) |
| **cap 0.95** | FSDP2 | 2 | 982 | 0.988x | 8,340 | 37.4 / 44.3 | **0** |
| **cap 0.95** | ZeRO-3 | 2 | 951 | 0.978x | 8,615 | 39.3 / 45.0 | **30** (35 flushes) |
| cap 0.95 + `expandable_segments` (fallback) | FSDP2 | 1 | 987 | 0.993x | 8,300 | 37.4 / 41.6 | 0 |
| cap 0.95 + `expandable_segments` (fallback) | ZeRO-3 | 1 | 872 | 0.896x | 9,431 | 39.3 / 45.0 | 30 |
| micro-batch 1 (cap 0.92) | FSDP2 | 1 | 492 | not comparable | 8,333 | 37.4 / 42.8 | 0 |
| micro-batch 1 (cap 0.92) | ZeRO-3 | 1 | 478 | not comparable | 8,552 | 39.3 / 43.6 | 30 |

Findings:
1. **The original 8B numbers hold.** With more headroom the throughput is 1-2 % *lower*, not higher
   (FSDP2 982-987 vs 994; ZeRO-3 951 vs 973). SM clocks (1882-1896 MHz), power and temperature match the original runs, so
   the small drop is session-to-session variation, about 1-2 % for these communication-bound runs vs. 0.3 % within a session.
2. **FSDP2 never thrashed.** It made 0 allocator retries and 0 cudaMalloc/cudaFree calls during the timed window, so the
   original cap did not hurt it.
3. **ZeRO-3's allocator churn is intrinsic, not caused by the memory cap.** It shows 30 retries (~2 per step) and ~500
   cudaMalloc/cudaFree pairs per timed window at cap 0.92 and 0.95, at micro-batch 2 and 1, and with expandable segments.
   The latter made it 10 % slower. The allocator simply grows to whatever cap it is given and fragments there.
   This is DeepSpeed ZeRO-3 with its default buffer sizes on a 48 GB GPU. Tuning `reduce_bucket_size` /
   `stage3_max_live_parameters` might remove it, but that changes the framework configuration and was out of scope.
   The original ZeRO-3 numbers are therefore representative of default ZeRO-3, and the 2-3 % FSDP2 lead includes that churn.
4. **The 8B step is communication-bound.** Halving the micro-batch leaves the step time unchanged
   (8,333 vs 8,344 ms for FSDP2), so tokens/s halves. Micro-batch 1 also gives no memory relief. Peak allocation is identical
   (37.4 / 39.3 GiB) because it is set by model state plus gathered parameters, not by activations
   (activation checkpointing is on).

## Re-run 2: vLLM tensor parallel with concurrency matched to the GPU count

Same workload as the original: 1024 requests, 512 random prompt tokens, exactly 256 greedy output tokens, bf16,
`gpu_memory_utilization` 0.85, TP4 with `NCCL_P2P_LEVEL=NVL`. Only `max_num_seqs` changed (256 -> 512 for TP2, 1024 for
TP4); the engine logs confirm the new values. Output tokens/s, medians of 2 runs (spread <= 0.11 %).

| Model | GPUs | TP old (256) | TP new (matched) | Change | DP original (256 per GPU) | DP / TP old | DP / TP new | TP new scaling eff. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | 2 | 11,934 | 11,811 (512) | -1.0 % | 15,736 | 1.32x | 1.33x | 76 % |
| Qwen2.5-1.5B | 4 | 4,214 | 4,993 (1024) | +18.5 % | 31,596 | 7.50x | 6.33x | 16 % |
| Llama-3.1-8B | 2 | 2,984 | 3,101 (512) | +3.9 % | 3,565 | 1.19x | 1.15x | 88 % |
| Llama-3.1-8B | 4 | 1,848 | 1,830 (1024) | -1.0 % | 7,175 | 3.88x | 3.92x | 26 % |

Findings:
1. **The earlier tensor-vs-data-parallel gap was only slightly overstated.** Matching concurrency changed TP throughput by
   -1 % to +4 % in three of four cases and by +18.5 % for Qwen on 4 GPUs. Data parallel beats tensor parallel even at
   matched concurrency (1.15–1.33× on NVLink, 3.9–6.3× across PCIe).
2. **The KV cache never limited the new runs.** It had room for 634 (Llama TP2) to 3,494 (Qwen TP4) requests of 832 tokens,
   all at or above the requested concurrency, and every run produced exactly 262,144 output tokens.
3. **Remaining caveats (identical across old and new runs):**
   - vLLM captures CUDA graphs only up to batch 512 by default, so TP4 decode steps above 512 sequences run without full graphs.
   - The default 8192-token prefill budget per step applies to all layouts.
   - Single-GPU Llama engines (TP1 / DP) were KV-limited to about 240 concurrent requests, which slightly *favours* TP.
   - Latency was not recorded (engine stats off, by decision, to keep throughput exactly comparable).

## Decisions applied to the README (2026-10-13)

The README results tables and findings now use these numbers; no other number was changed.

- **Llama-3.1-8B training:** the original numbers are kept (FSDP2 994 tok/s, ZeRO-3 973 tok/s); they come from the same
  session as the rest of the table and were reproduced within 1-2 % here. Note added under the training table:
  "ZeRO-3 shows ~2 allocator retries per step with default DeepSpeed buffers; FSDP2 shows none."
- **vLLM tensor parallel:** replaced with the matched-concurrency results (Qwen2.5-1.5B TP2 11,811 / TP4 4,993;
  Llama-3.1-8B TP2 3,101 / TP4 1,830). The scaling efficiencies follow from the unchanged TP1 baselines
  (Qwen 76 % / 16 %, Llama 88 % / 26 %). The unmatched-concurrency caveat is gone; the note that TP4 uses
  `NCCL_P2P_LEVEL=NVL` on 4 GPUs is kept. TP1 and data-parallel numbers are unchanged.
- **Added statement:** "Data parallel beats tensor parallel even at matched concurrency (1.15–1.33× on NVLink, 3.9–6.3×
  across PCIe)."

`reports/REPORT.md` is the generated record of the original campaign and still lists the TP numbers at the original
concurrency (256 per engine); its vLLM section points here.
