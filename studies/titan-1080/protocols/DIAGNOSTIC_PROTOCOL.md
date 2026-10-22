# ZeRO numerical diagnostic — frozen protocol

Frozen before any new diagnostic result. These are **diagnostics, not
performance measurements**, and they deliberately differ from the retained
performance recipe. Every deviation is declared here.

## Declared deviations from the performance recipe

| aspect | performance recipe | diagnostic | why |
|---|---|---|---|
| warmup | 5 untimed attempts | **0** | the audited boundary must be the first optimizer step, so the optimizer state at entry is known |
| measured attempts | 30 | **1 audited boundary** | one boundary is the unit of the numerical check |
| instrumentation | sampled telemetry only | two full CPU snapshots per boundary plus a chunked oracle | the snapshots and oracle cost host memory and wall time, so no timing from these runs is comparable to the performance campaign |
| timing | reported | **not reported** | instrumentation overhead makes it meaningless here |

No throughput, scaling or energy number is derived from a diagnostic run.

## What one training context checks

1. Rank, world size, physical UUID and child-visible identity.
2. Model inventory by distinct storage (tied `lm_head`/`wte` counted once).
3. Realized initialization and input digests, recorded per rank.
4. Expected partition identities, offsets, padding and coverage derived from
   the installed engine's own `partition_size` and `groups_padding`.
5. Base-call multiplicity derived from `len(single_partition_of_fp32_groups)`,
   one `_optimizer_step(i)` per fp32 master group per rank. Gate-0's single
   call reflected a single-group layout and is **not** generalised.
6. The gradient **consumed by the base optimizer**, observed at the entry of
   the base optimizer's own `step()`. In DeepSpeed 0.15.1 ZeRO-1 the master
   partition's `.grad` exists only inside `DeepSpeedZeroOptimizer.step()`
   (`stage_1_and_2.py`: assigned :1899, clipped :1905, consumed :1911, freed
   :1913), so this is the only point where it is live. It is observed, not
   reconstructed from `averaged_gradients`.
7. Attempts, accumulation boundaries, base calls, updates and skips counted
   separately. A skip is never inferred from an absent hook.
8. Working/master consistency **after** the engine has finished propagating
   (after `engine.step()` returns). FP32 requires exact elementwise equality
   with no tolerance; a bit-pattern comparison is reported separately under its
   own name. Mixed precision is compared against the resolved expected cast.
9. Chunked CPU snapshots and chunked oracle arithmetic, default 4,194,304
   elements, with the final short chunk included and complete coverage.
10. Exact rank-set reconciliation. A missing, duplicate or failed rank cannot
    become a smaller successful job.

## Tolerance policy (unchanged)

From the dtype unit roundoff `u`: `rtol = 64u`, `atol = 64u·max(1,|p|)`.
Thresholds are never adjusted after seeing a result. A check that cannot
distinguish the intended update from a no-op is reported as
**non-discriminating** and is never promoted to verified.

## Separate conclusions, never merged

* conditional optimizer-update agreement given the observed gradient;
* independent gradient correctness — **not** established by this diagnostic
  (the gradient is observed, not recomputed, and dropout p=0.1 is active);
* training quality or convergence — **not** addressed;
* historical throughput and energy observations — a separate cohort that these
  diagnostics do not retroactively certify.

## AMP

Resolved from the retained benchmark's own configuration
(`bench/train_llm.py:144`): `fp16 enabled, loss_scale 0,
initial_scale_power 12`, i.e. DeepSpeed's **dynamic** loss scaling. The same
source constructs `torch.amp.GradScaler` only when mode is not `deepspeed`
(:188), so native GradScaler semantics are **not** assumed. Overflow skips are
reported as observed. No fault injection and no manual scale tuning.

## Reference and streaming

See `zerodiag/reference.py` and `zerodiag/streaming.py`, whose module
docstrings are the frozen protocols. In summary: two singleton FP32
cross-family reference cases matched on realized inputs and initial state
(seeds alone are not accepted as proof, and generation runs in eval mode so no
dropout-mask claim is made); and one streaming session at 256 input / 64 output
tokens, where a network chunk is never treated as a token and per-token timing
is reported only when token identity and timestamp attribution are genuinely
established.
