# RTX A6000: validation closeout (qualified), v2

on-node verification, not an external review.

**What v2 replaces.** It supersedes `FINAL_REPORT.md` (r2). It applies the owner's completion checklist and
corrections C34–C36:
- the DP-vs-TP comparison at equal concurrency caps;
- the true dispersion medians;
- consistent scope labels.

**Closeout type.** Qualified, as for Titan/1080. It establishes what was compared and what the evidence supports. It
does not validate every historical row or certify any framework in general.

**Where the detail is.**
- `CLAIM_LEDGER_v2.csv`: 28 claims, each with its evidence, limits and the experiment needed before any stronger claim.
- `CLAIM_EVIDENCE_TABLE.csv`: per model × recipe × GPU count, with observed optimizer calls, gradient coverage, sampled
  update checks and reference agreement in separate columns.
- `recompute_final_tables_v2.py`: re-derives every number below.

## 1. Hardware evidence: 7 supervised windows at 1–2 GPUs, every cleanup confirmed
| Window | What ran | Outcome |
|---|---|---|
| Smoke 1 and 2 (2026-10-16) | — | ZeRO-2's first-smoke failure was a launcher defect, fixed |
| S1a (2026-10-16) | FP32 control + 4 recipes, world 1 | valid |
| w18 (2026-10-17) | enrollment, R5, C5 | R5 valid; C5 refused by the power start gate (other users' load) |
| w18b (2026-10-20) | R5 + C5 | control C passed |
| w19fb (2026-10-21) | first-boundary attribution | G4 rejected by a harness ledger defect (C32) |
| w20fb (2026-10-21) | attribution rerun on Candidate 20 | 5/5 valid |

The power guard never aborted (peaks 966–1,495 W). GPUs 0–3 were never used.

## 2. Training
**Calibration.**
- **Setup:** pre-registered control C, Qwen2.5-1.5B FP32 DDP with five updates; world 1 (GA 4) vs world 2 (GA 2); same
  start and the same 20 global microbatches.
- **Result:** G1 5.23e-8, loss ≤ 1.95e-8, Δθ 1.01e-5. Every endpoint passes.
- **Replayable on CPU** from the 12.6 MB of exported slices.
- **Reproducibility:** the FP32 gradient is bitwise reproducible across days. The S1a FP32 reference is bitwise
  identical to the calibrated R5.

**Recipes vs FP32 (Qwen2.5-1.5B).**
| Recipe | Pilot, pre-registered (world 2, 5 updates) | S1a (world 1, 3 updates) | Status |
|---|---|---|---|
| DDP + bf16 autocast | 0.0482 | 0.0399 | within 0.05 |
| FSDP2 bf16 | 0.0574 (marginal) | 0.0496 | at the limit, qualified |
| ZeRO-1/2 bf16 | 0.230 | 0.218 | marginal on L2; cosine 0.974–0.976 below 0.99; losses pass; norms within 0.3 % |

S1a differs from the pre-registered design:
- world 1 instead of world 2, and 3 updates instead of 5;
- graded on full vectors;
- production precision settings;
- a tolerance file reused with the pilot result visible (C18);
- the permanent a4 admission defect (C17).

**Attribution (specific tested result).**
- **Setup:** GPU 4, one accumulation boundary of Qwen2.5-1.5B, matched precision policy (TF32 off, math SDPA).
- **Result:** DeepSpeed ZeRO-2 bf16 and plain DDP with bf16 parameters produced bitwise-identical first-boundary
  gradients (338/338 tensors). Both deviate 0.2175 from FP32. Autocast alone gives 0.0362; FP32 accumulation order gives
  5.4e-8.
- **Conclusion:** in this configuration the ZeRO deviation comes from the bf16-parameter policy.
- **Limits:** this is not a general DeepSpeed correctness certificate. ZeRO-1, later updates, multi-GPU and other models
  were not tested.

**Other training claims.**
- **Qwen2.5-0.5B:** all recipes pass (mechanics model).
- **ResNet-50:** agreement with FP32 is **not established**. All frameworks fail identically while the control passes.
  It needs a decisive precision/control experiment.
- **Update accounting:**
  - observed for DDP, FSDP2 and ZeRO-1/2 at 1–2 GPUs: optimizer calls and finite gradients;
  - per-unit coverage and sampled AdamW checks: only at 1 GPU (S1a) and for FP32 DDP at 2 GPUs (C5);
  - complete two-GPU validation across every framework is not claimed.
- **Comparison type:** rows are complete-recipe comparisons.
- **Initialization:** established for Qwen/ResNet diagnostics, not for the 8B meta path.

## 3. Inference
- **Qwen2.5-1.5B throughput:** measured 1–2 GPU layouts, 2 repeats; qualified.
- **At equal total cap 512, DP2 beats TP2:**
  - **+33.23 %** for Qwen2.5-1.5B;
  - **+14.99 %** for Llama-3.1-8B, a qualified historical 8B observation, not revalidated.

  Repeats agree within 0.4 %.
- **KV capacity:** converted at the actual request length; capacity only.
- **Withdrawn:** historical generation-only power/utilization, latency/TTFT headlines and energy.
- **Not established:** inference correctness, which needs matched inputs, reference outputs/logits and a frozen
  acceptance rule.
- **Deferred:** 4-GPU inference.

## 4. Performance
**Training throughput** is the median of 3 records per configuration:
| Groups | Median CV | Max CV | Median range/median | Max range/median |
|---|---|---|---|---|
| All 28 | 0.15 % | 0.66 % | 0.29 % | 1.22 % |
| Current-scope (1–2 GPU, non-8B), 16 | 0.12 % | 0.47 % | 0.23 % | 0.85 % |
| Historical 4-GPU/8B, 12 | 0.19 % | 0.66 % | 0.36 % | 1.22 % |

- Differences below the observed range are not rankings.
- Memory and MFU are definitions only.
- Causal bottleneck explanations are withdrawn.
- **Deferred:** four-GPU training, all 8B training and ZeRO-3, pending separately scoped and approved validation.

## 5. Corrections during closeout (append-only)
| Correction | What |
|---|---|
| C27–C31 | execution-contract corrections |
| C32 | DeepSpeed first-boundary ledger defect, fixed in Candidate 20 |
| C33 | an observer sample holding other users' process names: redacted in the package; owner decision on the original |
| C34–C36 | this revision's corrections |

Earlier limitations stay.

## 6. Source and verification
- **Final source:** Candidate 20 (e12df77a…e33b).
- **Repository checks:** 6/6 (269 tests).
- **Self-tests:** first-boundary 19/19, two-GPU 49/49, lifecycle 11/11.
- **Reproducibility:** what is and isn't replayable is in `REPRODUCIBILITY.md`; protocols and environment are in
  `PROTOCOLS_AND_ENVIRONMENT.md`.
- **Repository integration:** remaining local steps are in `INTEGRATION_NOTES.md`.
