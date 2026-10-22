# Primary-source reading guide

Mehran Keivanimehr · Sources checked 2026-10-22

These references explain mechanisms and measurement definitions. None endorses these measurements or supplies missing experimental evidence. Project-specific claims are supported by the linked run records, not by citing a paper that studied a different system.

## NVIDIA documentation

**Floating-point comparisons.** Operation ordering and rounding can change numerical results even when arithmetic conforms to IEEE 754. This supports explicit precision settings and declared comparisons; it is not a guarantee that this project's error thresholds are correct. [NVIDIA floating point (accessed 2026/10), [§2.2](https://docs.nvidia.com/cuda/floating-point/index.html#operations-and-accuracy)].

**Mixed precision.** NVIDIA's FP16 training procedure keeps an FP32 primary weight copy and describes loss scaling and skipped updates on overflow. It provides context for observing actual update behavior rather than equating loop iterations with successful steps. It does not establish BF16 error bounds for this project. [NVIDIA mixed precision (accessed 2026/10), [training procedure](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)].

**Collective bandwidth.** NCCL tests distinguish algorithm bandwidth from bus bandwidth. For AllReduce the documented conversion is `busbw = algbw × 2(N−1)/N`. Buffer bytes, collective and rank count must accompany a comparison; the conversion is not evidence of a particular physical bottleneck. [NVIDIA NCCL tests (accessed 2026/10), [AllReduce](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce)].

**Telemetry.** NVIDIA documents the semantics of its GPU utilization, power and memory reports. This repository keeps sensor observations separate from allocator peaks and from workload-boundary energy integration. [NVIDIA SMI (accessed 2026/10), [documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html)].

## Papers

**Distributed data parallelism.** The DDP paper describes replicated models, gradient communication and implementation techniques such as bucketing and overlap. It is background for the DDP recipe, not evidence that these small-node runs scale like the paper's system. [Li et al. (2020/06), [abstract](https://arxiv.org/abs/2006.15704)].

**Fully sharded training.** The FSDP paper describes sharded training and integration with PyTorch's tensor and allocator machinery. The actual FSDP2 behavior here is defined by the recorded implementation and precision settings, not inferred from the older paper alone. [Zhao et al. (2023/09), [abstract](https://arxiv.org/abs/2304.11277)].

**ZeRO.** ZeRO's staged optimizations partition optimizer states, then gradients, then parameters. That model motivates expected partition inventories; it does not prove a particular runtime layout or a successful update. [Rajbhandari et al. (2020/05), [§5](https://arxiv.org/html/1910.02054v3)].

**Mixed-precision training.** The paper presents reduced-precision training with techniques for retaining accuracy. Its results do not turn a short synthetic diagnostic into a model-quality evaluation. [Micikevicius et al. (2017/10), [paper](https://arxiv.org/abs/1710.03740)].

**AdamW.** Decoupled weight decay differs from adding an L2 penalty inside an adaptive optimizer. The oracle therefore records which update rule it checks. [Loshchilov and Hutter (2019/01), [abstract](https://arxiv.org/abs/1711.05101)].

**PagedAttention.** The vLLM paper studies KV-cache memory management and serving throughput. It provides serving-system context, not an explanation of this project's DP/TP outcome without additional controls. [Kwon et al. (2023/09), [abstract](https://arxiv.org/abs/2309.06180)].

## Source list

[NVIDIA floating point, accessed 2026/10] NVIDIA. “Floating Point and IEEE 754 Compliance for NVIDIA GPUs.” NVIDIA documentation. https://docs.nvidia.com/cuda/floating-point/index.html

[NVIDIA mixed precision, accessed 2026/10] NVIDIA. “Train With Mixed Precision.” NVIDIA Deep Learning Performance. https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html

[NVIDIA NCCL tests, accessed 2026/10] NVIDIA. “Performance.” nccl-tests documentation. https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md

[NVIDIA SMI, accessed 2026/10] NVIDIA. “NVIDIA System Management Interface.” NVIDIA documentation. https://docs.nvidia.com/deploy/nvidia-smi/index.html

[Li et al., 2020/06] Shen Li et al. “PyTorch Distributed: Experiences on Accelerating Data Parallel Training.” arXiv:2006.15704. https://arxiv.org/abs/2006.15704

[Zhao et al., 2023/09] Yanli Zhao et al. “PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel.” arXiv:2304.11277v2. https://arxiv.org/abs/2304.11277

[Rajbhandari et al., 2020/05] Samyam Rajbhandari et al. “ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.” arXiv:1910.02054v3. https://arxiv.org/abs/1910.02054

[Micikevicius et al., 2017/10] Paulius Micikevicius et al. “Mixed Precision Training.” arXiv:1710.03740. https://arxiv.org/abs/1710.03740

[Loshchilov and Hutter, 2019/01] Ilya Loshchilov and Frank Hutter. “Decoupled Weight Decay Regularization.” arXiv:1711.05101v3. https://arxiv.org/abs/1711.05101

[Kwon et al., 2023/09] Woosuk Kwon et al. “Efficient Memory Management for Large Language Model Serving with PagedAttention.” SOSP 2023 / arXiv:2309.06180. https://arxiv.org/abs/2309.06180

## Limitations

Documentation versions may differ from the recorded cluster stack. Historical package versions and source hashes remain part of the evidence. The literature supplies definitions and context, not independent replication of the project.
