# Numerical reference test — pre-registration (A6000, session 20261014a)

Written and hashed **before** any reference-comparison run was executed. Author of the repository: Mehran

## Question

For one short training run from an identical logical initial state and identical global inputs, how far does each
retained A6000 training recipe (DDP+autocast, FSDP2 bf16-compute, DeepSpeed ZeRO-1/ZeRO-2 bf16, DeepSpeed autocast for
ResNet) deviate from a declared fp32 reference? This tests that every recipe computes the intended update (correct
loss normalisation over accumulation and data-parallel ranks, correct reduction, no silent skips). It does **not**
establish equal model quality, and different recipes are not expected to be bitwise equal.

## Controls held identical

- **Initial state**: one canonical model built on the CPU in fp32 from model seed 1234 (CPU RNG). Every framework loads
  it (bf16-weight recipes cast it to bf16). Identity is checked by fingerprinting the framework's full logical state
  gathered **after** wrapping (DDP module state, FSDP2 `full_tensor()`, DeepSpeed module state + fp32 master via
  `safe_get_full_fp32_param`) against the canonical fp32 or bf16-cast fingerprint. A seed alone is not accepted.
- **Inputs**: rank-independent global batches (one CPU generator, seed 1234000); rank r, micro-step i takes global rows
  `(i*world + r)*micro_batch …`. Global batch per optimizer step G is identical across the reference and the recipes.
- **Accumulation / normalisation**: loss = mean over the G samples' per-sample mean token loss (equal lengths, no
  padding); gradients averaged over ranks. **Clipping**: none in any recipe. **Optimizer**: AdamW, lr 1e-4, default
  betas/eps/weight decay. **Warmup**: 0 (measured step 0 is the first update from the canonical state). 5 updates.
- **Model-initialisation RNG** (seed 1234, CPU) is separate from the **input RNG** (seed 1234000, CPU); no dropout in
  these models.

## Reference and positive control

- **Reference R**: DDP, world 1, fp32 parameters and fp32 compute (autocast disabled), grad_accum = G / micro_batch.
- **Positive control C**: the same fp32 recipe at world 2 with grad_accum halved. It differs from R only by
  data-parallel reduction order, so it calibrates the harness: if C fails its tight tolerance, the harness (not the
  recipes) is at fault and recipe outcomes are not interpreted.

## Metrics (computed on fixed slices: the first 262,144 elements of 4 representative tensors)

LM tensors: `model.layers.0.self_attn.q_proj.weight`, `model.layers.0.mlp.down_proj.weight`,
`model.layers.<last>.mlp.up_proj.weight`, `model.norm.weight`. ResNet: `conv1.weight`, `layer1.0.conv1.weight`,
`layer4.2.conv3.weight`, `fc.weight`.

1. **G1 (primary)** — first-update gradient, relative L2 error over the concatenated slices:
   `||g − g_R|| / ||g_R||`, and cosine similarity. For DeepSpeed the reduced fp32 gradient is read with
   `safe_get_full_grad` after the boundary backward and before the optimizer step.
2. **L (primary)** — per-step global loss, `|L_k − L_R,k| / L_R,k` for k = 0 (identical state, forward-only error)
   and max over k = 0..4 (includes trajectory drift).
3. **U (secondary, descriptive)** — parameter change after 5 updates, relative L2 error of `Δθ = θ_5 − θ_0` vs R and
   cosine. For bf16-weight DeepSpeed recipes the fp32 master copy is used, and θ_0 is the bf16-rounded canonical
   state.

## Tolerances and their justification

bf16 has an 8-bit significand: unit roundoff u = 2^-8 ≈ 3.9e-3. fp32: u = 2^-24 ≈ 6e-8.

| Comparison | G1 rel. L2 | G1 cosine | L at k=0 | L max k≤4 | U |
|---|---|---|---|---|---|
| Positive control C (fp32 vs fp32, reduction order only) | ≤ 1e-4 | ≥ 0.99999 | ≤ 1e-6 | ≤ 1e-5 | rel ≤ 0.05 (Adam sign flips of near-zero gradients), cosine ≥ 0.999 |
| bf16-compute recipes (DDP autocast, FSDP2, DS autocast) | ≤ 0.05 | ≥ 0.995 | ≤ 5e-3 | ≤ 1e-2 | descriptive only |
| bf16-weight recipes (DeepSpeed ZeRO-1/2 bf16) | ≤ 0.08 | ≥ 0.99 | ≤ 1e-2 | ≤ 2e-2 | descriptive only |

Justification. Positive control: fp32 accumulations of ~10^6–10^7 terms in different order give relative differences
of order 10^-6–10^-5 in reduced gradients; 1e-4 allows an order of magnitude margin. bf16-compute: each matmul output
carries O(u) relative error; through ~24–28 transformer layers (or ResNet-50) gradient errors of 1–3% are expected, so
0.05 (≈13 u) is a pass limit that still catches normalisation mistakes (a missing 1/GA or 1/world factor gives errors
of 0.5–1.0, a wrong sign or skipped reduction gives ≥ 1). Loss at k = 0 is a forward-only quantity averaged over
~2–4k tokens (LM), so 5e-3 ≈ u. bf16 weights add one more rounding of every parameter (relative 2^-9 ≈ 2e-3) before
the forward, hence the looser 0.08 / 1e-2 limits.

Outcomes between the pass limit and 3× the limit are reported as **marginal (investigate)**; beyond 3× as **fail**.
U is reported but not graded for bf16 recipes: Adam updates of size ~lr are close to the bf16 spacing of weights of
magnitude ~0.02, so update-vector agreement is not dtype-bounded in a simple way.

## Runs

Experiments `validation_v2_20261014a_numref_<model>_ref` (world 1) and `validation_v2_20261014a_numref_<model>_recipes`
(world 2), for qwen2.5-0.5b (mechanics), qwen2.5-1.5b (retained recipe, seq 1024) and resnet50 (retained recipe).
Llama-3.1-8B: the meta-initialised FSDP2 path has **no canonical-state control** in this pilot. 8B results remain
complete-recipe comparisons; only update/finite/dtype diagnostics are run for it.

Analysis script: `evidence/analysis/compare_numerical_reference.py`. It reads `reference_tensors.pt` and
`result.json` of each run and applies exactly the limits above.

## Addendum 1 (before any reference run)

The trainer enables TF32 matmuls/convolutions and `cudnn.benchmark` globally. For the declared reference R and the
positive control C (`fp32_reference: true`) these are now disabled (`allow_tf32=False`, `cudnn.benchmark=False`,
`cudnn.deterministic=True`, matmul precision "highest") so that "fp32 compute" is literal and algorithm selection does
not vary between R and C. Recipe runs keep their production settings (TF32 allowed, benchmark on); that is part of
each complete recipe. Metrics and tolerances are unchanged.
