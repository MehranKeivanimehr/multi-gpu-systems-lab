# Scientific-v2 acceptance criteria

Author: Mehran Keivanimehr. These are proposed requirements, not completed experiments.

## Evidence and claim gate

For every finding, report separately: code/interpretation corrected, local regression checked, target-host pilot checked, repeated experiment checked, and remaining limitations. Do not replace an old measurement with a predicted or rescaled value and call it a rerun. The baseline audit and new status are separate records. [Scope (2026/10), [baseline registry](../audit/findings.csv), [current registry](findings.json)]

## Training gate — A08, A10, A16

- Run a short diagnostic for **each retained model/framework/precision recipe**, not only the small smoke model. Check every measured rank and step for finite loss and, in diagnostic runs, finite nonempty gradients. Record actual optimizer-call counts, AMP/DeepSpeed overflow/skip counters and representative parameter changes.
- Successful logical updates must equal the prescribed update count on every rank. Retain underlying optimizer-call counts separately: ZeRO-3 can update parameter subgroups using multiple base-optimizer calls per logical update. Cross-check DeepSpeed boundary/skip counters, base calls and parameter changes. A mismatch is evidence to investigate, not a reason to erase the run. Verify a known skipped-update test in an isolated diagnostic; do not inject invalid gradients into performance measurements.
- Record live parameter, gradient, optimizer-state and communication/compute dtypes after lazy state initialization. A common label such as BF16 is not numerical equivalence. Existing instrumentation is partial and must be extended on the host where needed.
- Before claiming a matched framework control, verify identical logical initial parameters/buffers, intended rank-specific input IDs, optimizer hyperparameters, clipping, accumulation and loss normalization. A seed alone is insufficient. A6000's large-model meta path currently remains explicitly unverified.
- Compare short reference updates using the same global inputs and declared numerical recipe. Choose and document error metrics/tolerances **before** examining comparative outcomes; justify them for the dtype/workload. Bitwise equality across different recipes is not required or claimed. If recipes cannot be matched, restrict the comparison to complete configurations and do not claim equal quality.
- Run numerical diagnostics separately from performance timings. Log update counters during clean performance runs, but do not include expensive hash/gradient profiling work in reported throughput without declaring it.

The need to count updates follows the loss scaler's documented conditional execution of `optimizer.step()`. [PyTorch (v2.4.1), [GradScaler implementation](https://raw.githubusercontent.com/pytorch/pytorch/v2.4.1/torch/amp/grad_scaler.py)]

DeepSpeed increments its update-boundary and skipped-update counters separately; these are cross-checks, not a substitute for the target-host parameter-change test. [DeepSpeed (v0.15.1), [engine counters](https://raw.githubusercontent.com/deepspeedai/DeepSpeed/v0.15.1/deepspeed/runtime/engine.py); DeepSpeed (v0.19.7), [engine counters](https://raw.githubusercontent.com/deepspeedai/DeepSpeed/v0.19.7/deepspeed/runtime/engine.py)]

## Energy and telemetry gate — A02, A03, A14, A18

- Preserve raw power readings, per-query start/stop timestamps, GPU identity, query failures, workload boundaries, P-states, clocks and temperatures. Use one host-local monotonic timebase for samples and generation workers.
- Independently recompute the integral over the exact throughput window. Require valid readings bracketing both boundaries for every GPU. Reject incomplete integration; do not zero-fill, silently shorten the window or extrapolate without a separately justified method.
- Use a pilot to choose long steady-state windows; start by testing approximately 30- and 60-second windows for the same workload. Fix the eventual protocol before the repeat campaign. Set and justify maximum sampling/query gaps and a stability/uncertainty target; these are not universal constants.
- Distinguish GPU-board sensor estimates from wall-socket/whole-node energy. Record device-specific sensor semantics. Sampling faster does not necessarily create independent new sensor measurements or calibrate physical accuracy.
- Keep idle-subtracted differences diagnostic unless before/after baselines, thermal/clock/P-state controls and uncertainty support promotion. Retain negative differences instead of clamping them to zero.
- A6000 historical inference telemetry is not generation-only. New means must reproduce from worker-window-filtered samples. Report timed-window and whole-run memory peaks separately.

Sensor semantics differ by device generation; do not apply the Ampere one-second averaging statement to GA100. [NVIDIA (accessed 2026/10), [NVML power query documentation](https://docs.nvidia.com/deploy/nvml-api/latest/api/group__nvmlDeviceQueries.html)]

## Latency gate — A05, A06, A07

- Keep offline batch duration, standalone forward proxies and completed one-token-batch proxies separate from request-level streaming latency.
- For a new serving experiment, preserve submission/dispatch, first output-bearing event and completion timestamps for each request, plus output-token counts and failures. Record concurrency, arrival policy, queueing boundary, model revision, prefix caching and all engine defaults that affect the comparison.
- The supplied [stream client](stream_latency.py) measures client-observed first-output latency on loopback, not internal GPU prefill. It retains stream chunks; do not label chunk gaps as per-token ITL when chunks contain multiple tokens. Validate the server/client interface in the pilot.
- Require exact expected output-token counts for the fixed-length test; report incomplete/failed requests separately. Report latency distributions from enough requests to support the chosen quantiles; do not report a precise tail percentile from a tiny sample.
- Do not retrofit streaming latency into the historical offline throughput experiment or compare different prompt/arrival/caching protocols as one engine-only effect.

[vLLM (accessed 2026/10), [latency boundary definitions](https://docs.vllm.ai/en/stable/benchmarking/cli/#understanding-the-latency-metrics); vLLM (v0.6.3.post1), [streaming request schema](https://raw.githubusercontent.com/vllm-project/vllm/v0.6.3.post1/vllm/entrypoints/openai/protocol.py)]

## Repetition and interpretation gate — A09, A12, A13, A15, A17

Use five fresh process/engine sessions per retained headline configuration as the initial proposal, not a statistical guarantee. Balance or randomize case order within repeat blocks, preserve the order, and predefine any extension rule and resource limit. Report raw points, median, sample CV and full range/median. Distinguish independent sessions from repeated timed calls within one session. Estimate comparisons with paired blocks where justified; do not assert a stable ranking if variability does not resolve the effect.

Do not restore physical-fit, model-size ceiling, universal hardware cause, identical quality, perfect scaling or KV-scheduler explanations without the corresponding direct evidence. A profile can motivate a hypothesis; a causal explanation also needs an appropriate control. Current withdrawn claims may remain withdrawn instead of consuming cluster time. [Claim scope (2026/10), [A6000 current report](../studies/a6000/CURRENT_REPORT.md), [A100 current report](../studies/a100/CURRENT_REPORT.md), [register](findings.json)]

## Sources

[PyTorch, v2.4.1] PyTorch contributors. Gradient scaling implementation. Source linked above.

[DeepSpeed, v0.15.1 and v0.19.7] DeepSpeed contributors. Training engine update-boundary and skip counters. Sources linked above.

[NVIDIA, accessed 2026/10] NVIDIA. NVML Device Queries. Source linked above.

[vLLM, accessed 2026/10] vLLM contributors. Benchmark CLI and latency definitions. Source linked above.

[vLLM, v0.6.3.post1] vLLM contributors. Serving request protocol. Source linked above.

[Claim scope, 2026/10] Mehran Keivanimehr. Audit and correction register, linked above.
