# Methodology and comparison rules

Current scientific corrections and outstanding experimental evidence are tracked in [SCIENTIFIC_STATUS.md](SCIENTIFIC_STATUS.md). Source code now contains unexecuted scientific-v2 changes; historical records retain their original protocols. [Version boundary (2026/10), [finding register](../validation/findings.json)]

## Experimental scope

The campaigns measure synthetic-workload execution on single nodes. Their measurements characterize the recorded workload and software stack; they do not establish model accuracy, training convergence, application quality, or performance with a production input pipeline. [Campaign methods (2026/10), [A6000](../studies/a6000/README.md), [Titan](../studies/titan-1080/bench/train_llm.py), [A100](../studies/a100/summary.md)]

## Comparison matrix

| Axis | A6000 | A100 | Titan/1080 | Interpretation |
|---|---|---|---|---|
| Small-LM training | Qwen2.5-1.5B, seq 1024, microbatch 1, accumulation 4 | GPT-2 seq 1024; Qwen seq 512/1024, several batches/checkpoint settings | GPT-2 seq 512, several batches, accumulation 1 | Models or token/step recipes differ; no direct GPU speed ratio |
| Large-LM training | Llama-3.1-8B, checkpointed, FSDP2 / ZeRO-3 | Not recorded | Not recorded | A6000 campaign only |
| ResNet-50 | Training, BF16, AdamW, batch 128/GPU | Training, FP32/TF32/AMP, SGD momentum, batch 64/128/256 | Inference, FP32/AMP, batch sweep | Forward-only and optimizer-step throughput are different tasks |
| LLM inference | Qwen2.5-1.5B / Llama-3.1-8B; prompt 512, output 256; vLLM TP/DP | Qwen2.5-1.5B; prompt 128, output 256; HF/vLLM | TinyLlama-1.1B; prompt 256, output 128; HF/vLLM | Different models, lengths, engine versions and concurrency |
| Communication | NCCL 1–1024 MiB; NVLink pairs / PCIe | Not recorded | NCCL 4–497 MiB; PCIe | Compare payload, group size and topology explicitly |

[A6000 protocols (2026/10), [training](../studies/a6000/configs/experiments/train_scaling.yaml) and [inference](../studies/a6000/configs/experiments/vllm_scaling.yaml)]; [A100 protocol (2026/10), [summary](../studies/a100/summary.md)]; [Titan protocols (2026/10), [training](../studies/titan-1080/bench/train_llm.py), [CNN](../studies/titan-1080/bench/bench_cnn.py), [inference](../studies/titan-1080/results/inference.csv)].

## Scaling

For weak scaling with a fixed per-GPU workload, efficiency is `throughput(N) / (N * throughput(1))`. A6000 normalizes against the same framework's baseline; Titan normalizes every method against its family's matching single-GPU baseline. These different baseline conventions must be kept visible. [Scaling implementations (2026/10), [A6000](../studies/a6000/scripts/aggregate.py) and [Titan](../studies/titan-1080/bench/aggregate.py)]

A6000 vLLM main runs and matched-concurrency audit runs are distinct datasets. The combined report uses audited TP rows and original DP rows when explicitly identified; it does not silently overwrite original TP results. [A6000 inference audit (2026/10), [analysis](../studies/a6000/results/rerun/RERUN_ANALYSIS.md)]

## Precision and memory

A100 `bf16-pure` Qwen rows use BF16 parameters and optimizer states, whereas `bf16` rows force FP32 parameters with BF16 autocast. Titan DeepSpeed's `amp_fp16` label enables DeepSpeed's FP16 path; it is not identical to the native AMP implementation in DDP. FSDP in Titan is the older API, while A6000 uses FSDP2. [Precision implementations (2026/10), [A100 notebook](../provenance/a100-original.ipynb), [Titan trainer](../studies/titan-1080/bench/train_llm.py), [A6000 frameworks](../studies/a6000/src/mgbench/train/frameworks.py)]

The frozen Titan CNN table retains its original incorrect MAC/FLOP convention and AMP/FP32-peak percentage as provenance. Current derived tables correct the dominant conv/linear work to approximate 8.2 GFLOP/image at FMA=2 and omit a peak percentage for AMP. Neither quantity is hardware occupancy. [Titan correction (2026/10), [corrected table](../results/corrections/titan_cnn.csv), [proof](../results/corrections/proof.json), [original table](../studies/titan-1080/results/cnn_resnet50.csv)]

## Timing, energy and utilization

The historical A100 HF/vLLM TTFT fields time completed one-token batches; Titan HF uses a timed standalone forward/prefill pass. These are not request-level streamed TTFT. Titan vLLM supplies batch makespan, not per-request latency. Current code and tables relabel those quantities, with a separate streaming client proposed for actual client-observed first-output latency. [Timing corrections (2026/10), [A100 notebook](../provenance/a100-original.ipynb), [corrected inference table](../results/corrections/titan_inference.csv), [stream client](../validation/stream_latency.py)]

Historical Titan energy integrates between sampled timestamps and omits workload boundaries. A6000 inference mean telemetry includes waiting/teardown outside generation. A100 rate-over-average-power values are not aligned integrals. These do not support validated energy rankings even within a campaign without additional evidence. New Titan code brackets the exact monotonic workload window and retains raw readings; new A6000 inference code selects worker generation makespan. Host validation and sensor uncertainty remain pending. [Energy scope (2026/10), [findings A02/A03/A14/A18](../validation/findings.json), [acceptance criteria](../validation/ACCEPTANCE.md)]

MFU is an analytical model-work estimate, not a measured utilization counter. A6000 includes an attention term while the A100 formula differs; checkpoint recompute and other operations are not consistently counted. Do not rank hardware using these different MFU formulas. Allocated, reserved and sampled device memory have distinct scopes. Historical A6000 device peaks include setup; corrected code additionally labels timed-window peaks. [Metric scope (2026/10), [A6000 metrics](../studies/a6000/src/mgbench/metrics.py), [A100 notebook](../provenance/a100-original.ipynb), [A6000 monitor](../studies/a6000/src/mgbench/gpu_monitor.py)]

## Communication units

Titan's historical `size_mb=497` allocates `497 * 1024 * 1024` bytes: **497 MiB**, or 521,142,272 bytes. The recorded GPT-2 parameter count is 124,439,808, whose FP32 parameter-sized gradient payload is 497,759,232 bytes, approximately 474.70 MiB. The probe buffer is roughly 4.7% larger and must not be described as the exact DDP payload. `busbw = algbw * 2 * (N-1) / N`; both bandwidths are decimal GB/s. The raw measurements remain unchanged. [Titan buffer (2026/10), [communication source](../studies/titan-1080/bench/comm_bench.py) and [parameter count](../studies/titan-1080/results/raw/A_rtx_1_bs4_fp32.json)]

## Evidence boundaries

P2P availability is observed on the recorded hosts. These results do not identify a universal hardware/driver policy or prove a platform-independent root cause. Similarly, approximate FLOP utilization and near-100% NVML utilization are not direct SM occupancy measurements. [Recorded observations (2026/10), [Titan preflight](../studies/titan-1080/results/raw/00_preflight.json) and [A6000 interconnect discussion](../studies/a6000/reports/REPORT.md)]

Sources are linked at the relevant claims; the campaign source files define the measured protocols.
