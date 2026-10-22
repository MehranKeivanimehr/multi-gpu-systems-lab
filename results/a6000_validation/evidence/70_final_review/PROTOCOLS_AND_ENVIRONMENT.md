# A6000: protocols, environment and known limitations (for publication)

Author of the repository: Mehran Keivanimehr. These values are taken from the frozen candidates and the retained
records.

## Hardware and software
| Item | Value |
|---|---|
| GPUs | NVIDIA RTX A6000, 48 GB class; NVML total 49,140 MiB; Torch-usable 50,897,289,216 B (47.40 GiB); enforced power limit 300 W |
| Devices used | physical GPUs 4 and 5 (one NVLink pair). GPUs 0–3 were never used; 6–7 were not used after the 2026-10-14 outage |
| Software | driver 580.178.04, CUDA 13.0 · Ubuntu 24.04.5 LTS, kernel 7.0.0-38 · Python 3.12.3 |
| Libraries | torch 2.11.0 · DeepSpeed 0.19.7 · transformers 5.19.0 · triton 3.6.0 · vLLM 0.26.0 · NCCL 2.28.9 · numpy 2.3.5 · nvidia-ml-py 13.615.71 · PyYAML 6.0.3 |

The full freeze is in `pilot_evidence/environment/pip_freeze.txt`.

## Models (pinned revisions)
| Model | Repository | Revision |
|---|---|---|
| Qwen2.5-0.5B | Qwen/Qwen2.5-0.5B-Instruct | 7ae557604adf67be50417f59c2c2f167def9a775 |
| Qwen2.5-1.5B | Qwen/Qwen2.5-1.5B-Instruct | 989aa7980e4cf806f80c7fef2b1adb7bc71aa306 |
| Llama-3.1-8B | unsloth/Llama-3.1-8B-Instruct | 4699cc75b550f9c6f3173fb80f4703b62d946aa5 |
| ResNet-50 | torchvision definition | — |

The numerical diagnostics build models from their configurations (canonical init), not from released weights.

## Numerical protocols
**Pre-registration.** `pilot_evidence/analysis/numerical_reference_preregistration.md`, sha256 cd367fc5…, hashed with
0 reference runs present.

**Initial state.** One canonical CPU-built FP32 model:
- model seed 1234;
- a bf16-representable start for the Qwen-1.5B controls;
- the tied embedding held once.

Identity is checked by full-state fingerprint after wrapping; a seed alone is not accepted.

**Inputs.**
- One rank-independent generator, seed 1234000.
- 4 global rows per update; for Qwen-1.5B R5/C5, 20 global microbatches with frozen digests
  (`36_c16_design/global_input_map_r5_c5.json`).
- Rank r, micro-step i takes row i·world + r.
- Loss is the mean of the 4 samples' mean token loss.

**Optimizer.** AdamW: lr 1e-4, betas (0.9, 0.999), eps 1e-8, weight decay 0.01; no clipping; warmup 0.

**Precision.**
| Run | Settings |
|---|---|
| FP32 reference and controls | TF32 off, matmul precision "highest", deterministic cuDNN, no autocast |
| Control C and first-boundary runs | additionally math-only SDPA |
| Recipe runs | production settings (TF32 on, cuDNN benchmark), per addendum 1 |
| First-boundary attribution | matched policy: TF32 off, math SDPA, reduced-precision reductions off |

**Recipes:**
- DDP: FP32 parameters with bf16 autocast;
- FSDP2: FP32 shards with bf16 compute;
- DeepSpeed ZeRO-1/2: bf16 parameters with an FP32 master;
- G3: DDP with native bf16 parameters.

**Endpoints.**
- **Slices:** the first 262,144 elements of `layers.0.self_attn.q_proj.weight`, `layers.0.mlp.down_proj.weight` and
  `layers.<last>.mlp.up_proj.weight`, plus all of `model.norm.weight`.
- **G1:** relative L2 and cosine of the first-update gradient.
- **L:** relative per-step loss difference.
- **U:** relative L2 and cosine of Δθ after 5 updates.
- **Control C limits:** 1e-4 / 0.99999 / 1e-6 (k0) / 1e-5 (max) / 0.05 / 0.999.
- **Recipe limits** are in the pre-registration table.
- **Verdict rule:** between a limit and 3× is "marginal (investigate)"; beyond 3× is "fail"; a cosine below its limit
  is "not_pass"; interpretation is allowed only if every endpoint passes.
- **Implementation:** `src/mgbench/control_c.py` (frozen since Candidate 16).

## Measurement definitions (performance)
- **Training throughput:** the median of 3 records under fixed per-GPU batches.
- **Dispersion:** sample CV and range/median.
- **MFU:** approximate; omits checkpoint recompute.
- **Memory:** allocator-reserved peaks are not physical capacity; historical NVML peaks include setup.
- **Inference throughput:** output tokens over the worker generation makespan, max(end) − min(start), on a common
  monotonic clock.
- **Historical telemetry:** includes waiting and teardown, so it is not generation-only.
- **Excluded:** energy, latency and TTFT (see the ledger).

## Execution safety (A6000 node)
- Supervised windows with:
  - device identity checks, and same-window enrollment of a second GPU under a frozen policy;
  - bounded systemd scopes: supervisor 1 CPU / 2 GiB, worker 15 CPU / 46 GiB, swap 0;
  - ownership-scoped cleanup with three outcomes;
  - a project-wide host lease.
- Node guards:
  - start only if other GPUs + 300 W per GPU ≤ 1,500 W;
  - abort above 1,700 W for 5 s;
  - at most 2 GPUs.

These guards are empirical, not a PSU rating; the 2026-10-14 outage cause is unknown.

## Known limitations
See `CLAIM_LEDGER_v2.csv`: statuses `not_established`, `withdrawn` and `deferred`, plus the limits column. They also
include `30_historical_accounting/` corrections C1–C36 and the permanent S1a admission defect (C17).
