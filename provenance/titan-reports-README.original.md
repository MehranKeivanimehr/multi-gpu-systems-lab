# Stage 1 — Multi-GPU Performance Benchmark Suite
## Heterogeneous 10-GPU Cluster (`ten.ece.ufl.edu`)

A reproducible benchmark of single- and multi-GPU training and inference across
three GPU generations sharing one PCIe-only node. All results were produced on a
shared machine under strict non-interference rules: no `sudo`, no host changes,
no foreign process ever signalled, and every job capped so an out-of-memory
condition lands inside the benchmark process rather than starving the node.

Auto-generated result tables: [`SUMMARY.md`](SUMMARY.md).
Raw per-run JSON: [`../results/raw/`](../results/raw/). Tidy CSVs:
[`../results/`](../results/).

---

## 1. The headline finding

**On this cluster, the four "obsolete" GTX 1080 Ti scale better than the two
TITAN RTX, and the reason is not the GPUs — it is that Turing consumer cards
cannot do PCIe peer-to-peer.**

| Group | Link class | CUDA P2P | all-reduce busbw | DDP efficiency |
|---|---|---|---|---|
| 2× TITAN RTX | PHB | **disabled** | **3.86 GB/s** | 66.5% |
| 2× GTX 1080 Ti | PIX | enabled | 11.04 GB/s | **88.2%** |
| 4× GTX 1080 Ti | PIX | enabled | **11.31 GB/s** | **83.7%** |
| 6× GTX 1080 Ti | PHB | enabled | 9.52 GB/s | 80.8% |

The measured P2P matrix across all 10 GPUs shows P2P blocked for **exactly the
two TITAN RTX, against every other GPU**, and enabled between **all 8 Pascal
GPUs including across the PCIe host bridge**:

```
      0   1   2   3   4   5   6   7   8   9      (Y = P2P, . = blocked)
  0   -   .   .   .   .   .   .   .   .   .      <- TITAN RTX  (sm_75)
  1   .   -   .   Y   Y   Y   Y   Y   Y   Y
  2   .   .   -   .   .   .   .   .   .   .      <- TITAN RTX  (sm_75)
  3   .   Y   .   -   Y   Y   Y   Y   Y   Y
  4   .   Y   .   Y   -   Y   Y   Y   Y   Y
  5   .   Y   .   Y   Y   -   Y   Y   Y   Y
  6   .   Y   .   Y   Y   Y   -   Y   Y   Y
  7   .   Y   .   Y   Y   Y   Y   -   Y   Y      <- TITAN Xp   (sm_61)
  8   .   Y   .   Y   Y   Y   Y   Y   -   Y
  9   .   Y   .   Y   Y   Y   Y   Y   Y   -

P2P ALLOWED by link class: {PHB: 32, PIX: 24}
P2P BLOCKED by link class: {PHB: 26, PIX: 8}
```

Link class does not predict P2P: there are 32 *allowed* PHB pairs and 8
*blocked* PIX pairs. GPU generation predicts it exactly. NVIDIA dropped PCIe
peer-to-peer on Turing GeForce/TITAN parts (it requires NVLink there, and
`nvidia-smi topo -m` confirms no NVLink bridge is installed), while Pascal
GeForce retains it. Consequently the TITAN RTX pair must stage every gradient
all-reduce through host memory at 3.86 GB/s.

**Practical consequence:** for distributed training on this node, the 1080 Ti
pool is the better fabric. The TITAN RTX remain the right choice for
single-GPU work and for inference, where they win decisively.

### This corrected two earlier hypotheses
Both are recorded here because the suite was built to test them, not to confirm them.

1. *"PHB forces host staging."* **Wrong.** Only one PHB group was measured at
   first, and it happened to also be the P2P-disabled one, conflating two
   variables. The 6× run (PHB **with** P2P) separated them: crossing the host
   bridge costs ~16% of bandwidth, not 3×.
2. *"The 6× run will show an interconnect cliff."* **Wrong.** Bandwidth fell
   11.31 → 9.52 GB/s and efficiency 83.7% → 80.8%. A gentle slope, not a cliff.

---

## 2. Hardware and its limitations

| idx | GPU | VRAM | Capability | Arch | Role in this suite |
|---|---|---|---|---|---|
| 0, 2 | NVIDIA TITAN RTX | 24 GB | 7.5 | Turing | primary group; only GPUs that can run vLLM |
| 1,3,4,5,6,8,9 | NVIDIA GeForce GTX 1080 Ti | 11 GB | 6.1 | Pascal | scaling group (2× / 4× / 6×) |
| 7 | NVIDIA TITAN Xp | 12 GB | 6.1 | Pascal | single-GPU only |

Driver 535.230.02 (CUDA 12.2), 56 cores, 503 GB RAM. No NVLink: all GPU-to-GPU
traffic is PCIe. Topology forms two switch domains — `{1,2,3,4,5}` and
`{6,7,8,9}` are PIX internally — with GPU 0 attached directly to the root
complex and therefore PHB to everything.

### Constraints that shaped every design decision

**bf16 does not exist here.** It needs sm_80+. Critically,
`torch.cuda.is_bf16_supported()` returns **`True` on all 10 GPUs**, which is
misleading. Measured 4096³ GEMM:

| GPU | fp32 | fp16 | bf16 | fp16 gain | bf16 gain |
|---|---|---|---|---|---|
| TITAN RTX | 13.04 | **104.91** | 8.27 | **8.05×** | 0.63× |
| GTX 1080 Ti | 9.95 | 8.41 | 5.79 | 0.85× | 0.58× |
| TITAN Xp | 10.07 | 11.61 | 5.97 | 1.15× | 0.59× |

bf16 runs but is *slower than fp32 everywhere*. The suite therefore uses FP32
on Pascal and FP32 + AMP-fp16 on Turing, and never bf16. **Do not trust the
capability flag on pre-Ampere hardware — measure.**

**Pascal fp16 is workload-dependent, and the GEMM probe does not generalise.**
The table above shows large square fp16 GEMM at ~0.85–1.15× of fp32 on Pascal,
because cuBLAS computes those in fp32. But real fp16 *transformer inference* on
Pascal is catastrophically slower, because the model executes many genuine fp16
elementwise/attention ops on GP102's native 1:64 fp16 path and skinny decode
GEMMs that miss the optimised route:

| GPU, 16 concurrent | fp16 | fp32 | **fp32 advantage** |
|---|---|---|---|
| GTX 1080 Ti | 55.7 tok/s | 344.9 tok/s | **6.2×** |
| TITAN Xp | 60.2 tok/s | 387.3 tok/s | **6.4×** |

**Serve in FP32 on Pascal, never fp16**, despite the 2× memory cost (4913 MiB
vs 2963 MiB — both fit comfortably in 11 GB).

The TITAN RTX goes the other way, as measured (not assumed): fp16 400.1 → 533.5
tok/s, so **fp16 is correct on Turing and fp32 on Pascal**. Best-dtype-per-GPU
at 16 concurrent requests:

| GPU | best dtype | out tok/s | vs its own worse dtype |
|---|---|---|---|
| TITAN RTX | fp16 | 533.5 | 1.33× over fp32 |
| TITAN Xp | fp32 | 387.3 | 6.4× over fp16 |
| GTX 1080 Ti | fp32 | 344.9 | 6.2× over fp16 |

This cuts the apparent architecture gap dramatically. Compared at each GPU's
own best dtype the TITAN RTX leads the TITAN Xp by **1.38×** and the 1080 Ti by
**1.55×** — not the 9.6× an fp16-everywhere comparison would suggest — and both
are consistent with the 1.63× FP32 training gap.

**vLLM cannot run on 8 of 10 GPUs.** It requires capability ≥ 7.0; Pascal is
6.1. vLLM is therefore TITAN RTX only, and the Pascal GPUs are covered by an
HF `generate` baseline instead. On Turing, vLLM also cannot use FlashAttention
(needs sm_80+) and selects the XFormers backend.

**The host CUDA toolkit is unusable for JIT.** `nvcc` is 10.1 against a 12.2
driver, so no component may JIT-compile. DeepSpeed runs with `DS_BUILD_OPS=0`,
a plain `torch.optim.AdamW` instead of `FusedAdam`, and **no CPU/NVMe offload**
(`cpu_adam` would need a compiler). ZeRO stages 1–2 in that configuration are
pure Python + NCCL and work fine.

**`sm_61` is absent from the torch wheel — and that is fine.** torch 2.4.1+cu121
ships `sm_50/60/70/75/80/86/90`. CUDA guarantees binary compatibility across
*minor* revisions within a major one, so the `sm_60` cubin executes on the 6.1
Pascal cards. The preflight gate applies that rule per device and then *proves
it* by running a real matmul on all 10 GPUs, which is the authoritative check.
A naive "is the literal string `sm_61` present" test would abort the suite
incorrectly.

**11 GB is the binding constraint on batch size, and DDP tightens it.** With
the 0.85 allocator cap (9574 MiB usable of 11264), single-GPU GPT-2 124M fits
bs=8 (8740 MiB), but DDP's gradient buckets plus NCCL buffers push bs=8 into
OOM. Measured DDP ceiling is **bs=6** (7393 MiB). The cross-architecture
scaling config is **bs=4/GPU**, which has baselines on every GPU type and
leaves headroom for ZeRO and FSDP.

---

## 3. Methodology

**Workload.** GPT-2 124M built from `GPT2Config()` defaults (12 layers, d=768,
12 heads, vocab 50257) with random init, seq 512, synthetic random token IDs,
no dataloader. Random vs pretrained weights execute an identical graph, so this
measures GPU and interconnect throughput rather than storage or preprocessing,
and the suite stays hermetic and reproducible. Peak allocation came out
**byte-identical across all three GPU types** at every batch size
(2511 / 3290 / 5099 / 8740 MiB) — direct evidence that Pascal and Turing ran
the same workload.

**Scaling.** Weak scaling: micro-batch per GPU is held constant, so ideal
throughput is linear in GPU count and `efficiency = measured / (1-GPU × N)`.
Each family is normalised against *its own* 1-GPU baseline at the same batch
size and precision.

**Interconnect.** NCCL all-reduce is benchmarked at 4–497 MB. The 497 MB point
is deliberately GPT-2 124M's exact FP32 gradient volume (124M × 4 B), i.e. what
DDP actually moves per step. `busbw = algbw × 2(n−1)/n` makes ring bandwidth
comparable across group sizes.

**Telemetry.** Power, utilisation, SM clock, temperature and memory are sampled
from `nvidia-smi` on a background thread at 100–200 ms, and energy is the
trapezoid integral of the power series. This is a *sampled estimate*, not a
hardware energy counter — consistent across runs on this box, which is what the
comparisons need, but not a calibrated absolute measurement.

**Device selection is topology-aware.** `bench/topology.py` enumerates
same-model combinations and maximises the worst pairwise link
(lexicographically over sorted link ranks), rather than taking "the first N
GPUs of type X" — which would silently hand different runs different fabrics.
The 6× group is pinned to the operator-specified `1,3,4,5,8,9`; the selector's
own choice of `{1,3,4,5,6,8}` is topologically identical (7 PIX pairs, 8 PHB
pairs, PHB worst link).

### Safety mechanics actually enforced

- `CUDA_DEVICE_ORDER=PCI_BUS_ID` plus explicit `CUDA_VISIBLE_DEVICES` on every launch.
- `torch.cuda.set_per_process_memory_fraction(0.85)`, and vLLM's
  `gpu_memory_utilization=0.85` — the same cap expressed in each stack's terms.
- Pre-flight guard refuses to launch if a target GPU holds a **foreign compute
  PID** or unaccounted VRAM. It never kills and never reconfigures; it declines.
  Utilisation counts against a GPU only when a foreign process is resident —
  see the known-issue note in §6 on why.
- Per-job wall-clock timeout via `timeout --signal=TERM --kill-after=30s`,
  scoped to our own process group only.
- Per-job `MASTER_PORT` derived from a hash of the job key, so the suite cannot
  collide with another user's `torchrun`.
- `nvidia-smi` snapshot stored before and after every run. All 10 GPUs returned
  to 6 MiB / 0% after every track.
- All environments and caches live **inside** the project directory.

---

## 4. Results

Full tables in [`SUMMARY.md`](SUMMARY.md). Highlights:

### Training throughput (GPT-2 124M, seq 512)

| Configuration | tok/s | J per 1M tokens |
|---|---|---|
| 6× 1080 Ti, DDP fp32, bs=4 | **32,589** | 22,321 |
| 2× TITAN RTX, DDP amp-fp16, bs=8 | 31,166 | **13,081** |
| 1× TITAN RTX, amp-fp16, bs=16 | 27,365 | 9,599 |
| 4× 1080 Ti, DDP fp32, bs=4 | 22,507 | 30,558 |
| 1× TITAN RTX, fp32, bs=12 | 11,915 | 22,158 |
| 1× TITAN Xp, fp32, bs=8 | 7,515 | 29,920 |
| 1× GTX 1080 Ti, fp32, bs=8 | 7,074 | 30,630 |

Six 1080 Ti beat two TITAN RTX on absolute throughput, but the RTX pair is
**1.71× more energy-efficient per token**. Throughput, scaling efficiency and
energy rank these groups in *three different orders* — any single-number
"which is faster" claim about this cluster is incomplete.

AMP-fp16 yields **2.30×** end-to-end on the TITAN RTX despite an **8.05×** GEMM
advantage: GPT-2 training is far from pure GEMM (softmax, layernorm, fp32
loss/logits, optimizer), so the Tensor Core win is heavily diluted.

### Parallelism methods (bs=4/GPU)

| Group | Method | tok/s | % of DDP | Peak MiB |
|---|---|---|---|---|
| 2× RTX (no P2P) | DDP fp32 | 14,297 | 100% | 5571 |
| 2× RTX | ZeRO-2 fp32 | 10,701 | 74.9% | 5243 |
| 2× RTX | ZeRO-1 fp32 | 10,690 | 74.8% | 5243 |
| 2× RTX | FSDP fp32 | 9,824 | **68.7%** | **4565** |
| 4× 1080 Ti (P2P) | DDP fp32 | 22,507 | 100% | 5571 |
| 4× 1080 Ti | FSDP fp32 | 20,670 | **91.8%** | **4212** |
| 4× 1080 Ti | ZeRO-1 fp32 | 20,598 | 91.5% | 4887 |
| 4× 1080 Ti | ZeRO-2 fp32 | 20,562 | 91.4% | 4887 |

**The cost of sharding tracks fabric quality.** Identical method, model and
batch: FSDP loses 31% on the no-P2P RTX pair but only 8% on the P2P Pascal
group. Sharded parallelism trades compute for communication, making it the
wrong choice on the TITAN RTX pair and nearly free on the 1080 Ti group.

**And sharded methods degrade faster with group size than DDP.** Scaling
efficiency on the 1080 Ti pool, bs=4/GPU FP32:

| GPUs | DDP | ZeRO-1/2 | FSDP |
|---|---|---|---|
| 2 | **88.2%** | 80.3% | 81.7% |
| 4 | **83.7%** | 76.6% | 76.9% |
| 6 | **80.8%** | 70.5% | **65.2%** |

DDP loses 7.4 points going from 2 to 6 GPUs; FSDP loses 16.5. DDP's
communication is one fixed all-reduce per step, whereas FSDP and ZeRO add
parameter/state traffic that grows with world size. On this cluster DDP is the
right default at every scale tested, and the sharded methods are worth their
cost only when memory — not throughput — is the binding constraint.

**One reversal — with AMP on the RTX pair, ZeRO beats DDP** (20,237 vs 18,874
tok/s, 107.2%). torch AMP keeps master weights *and gradients* in fp32, so
DDP's all-reduce payload stays 497 MB under autocast; DeepSpeed's `fp16` mode
keeps fp16 gradients, halving it to ~248 MB. On a 3.86 GB/s fabric that halving
outweighs ZeRO's extra optimizer-state traffic — the sharded method wins
*because* the fabric is bad.

**ZeRO-1 ≈ ZeRO-2** everywhere (within 0.6%, identical peak memory). At 124M
params, stage 2's extra gradient partitioning saves ~0.25 GB that never becomes
the peak, since the peak is the activation high-water mark in backward, not the
optimizer step. The stage was verified as actually applied
(`zero_optimization_stage ...... 2` in the engine log), not silently defaulted.

### Inference (TinyLlama-1.1B, 256-token prompt, 128 new tokens)

| Engine | GPU | dtype | TP | Concurrency | out tok/s | J/ktok |
|---|---|---|---|---|---|---|
| vLLM | TITAN RTX | fp16 | 1 | 64 | **3,979** | **51.4** |
| vLLM | TITAN RTX | fp16 | **2** | 64 | 2,532 | 111.5 |
| vLLM | TITAN RTX | fp16 | 1 | 16 | 1,872 | 93.5 |
| HF generate | TITAN RTX | fp16 | 1 | 16 | 534 | 291.8 |
| HF generate | TITAN Xp | **fp32** | 1 | 16 | 387 | 633.8 |
| HF generate | GTX 1080 Ti | **fp32** | 1 | 16 | 345 | 617.5 |
| HF generate | TITAN Xp | fp16 | 1 | 16 | 60 | 2,145.1 |
| HF generate | GTX 1080 Ti | fp16 | 1 | 16 | 56 | 1,957.8 |

Two results worth acting on:

1. **vLLM TP=2 is slower than TP=1 at every concurrency** (0.64× at conc=64).
   TinyLlama-1.1B needs 2.05 GB and fits trivially on one 24 GB card, so tensor
   parallelism adds an all-reduce after every attention and MLP block across a
   fabric with no P2P. Do not use TP on this pair unless the model genuinely
   does not fit.
2. **vLLM is 3.5–7.5× faster than HF `generate`** on identical hardware
   (continuous batching, paged KV cache, CUDA graphs), and 5.7× more
   energy-efficient per token at high concurrency.

Quantization was deliberately **not** used: fp16 TinyLlama is 2.05 GB against
24 GB of VRAM, so there is no capacity pressure to relieve, and AWQ/Marlin
kernels would add a confound rather than information.

---

## 5. Comparison with the prior RTX A6000 study

The earlier A6000 work in this tree
(`exp1_calibration_titan/`, `experiments_titan_1080/A6000_COMPLETE_RESULTS.md`)
is a **single-GPU, CNN, energy-focused** study — not multi-GPU scaling. There is
no A6000 multi-GPU baseline to compare against, so the only genuinely
comparable axis is ResNet-50 inference throughput and compute utilisation,
which Track B reproduces using the same column schema and the same
4.1 GFLOP/image constant.

| GPU | Precision | Peak img/s | at bs | Achieved TFLOP/s | FP32 theoretical | **% of FP32 peak** |
|---|---|---|---|---|---|---|
| **TITAN RTX** | **AMP-fp16** | **2,278** | 128 | 9.34 | 16.31 | 57.3%* |
| **RTX A6000** (prior) | FP32 | 1,677 | 256 | 6.88 | 38.71 | **17.8%** |
| TITAN RTX | FP32 | 1,062 | 128 | 4.36 | 16.31 | **26.7%** |
| TITAN Xp | FP32 | 766 | 64 | 3.14 | 12.15 | 25.8% |
| GTX 1080 Ti | FP32 | 696 | 64 | 2.86 | 11.34 | 25.2% |

Comparing FP32 to FP32, the A6000 is 1.58× faster in absolute terms but has
2.37× the theoretical FP32. **All three older GPUs extract a higher fraction of
their peak than the A6000 did** (~25–27% vs 17.8%), which independently
corroborates the memory-bound conclusion of the original study: adding peak
FLOPs to this workload buys less than proportional throughput.

**With Tensor Cores enabled, the TITAN RTX beats the A6000 outright** — 2,278
vs 1,677 img/s, a 1.36× lead, from AMP-fp16 giving 2.14× over its own FP32
result. If the A6000 study's numbers are FP32-only (its 4.1 GFLOP/img and
38.7 TFLOP/s FP32 basis suggest so), then that comparison is FP32-vs-fp16 and
the fair statement is narrower: *a 2019 TITAN RTX using Tensor Cores
out-throughputs an Ampere A6000 that is not.* An AMP arm on the A6000 would
very likely restore its lead and is the right follow-up.

*\* The 57.3% figure is against the **FP32** theoretical peak, which is the
wrong denominator for an fp16 run — it is shown only for continuity with the
A6000 study's basis. Against the measured fp16 Tensor Core ceiling
(104.9 TFLOP/s) the same run is just **8.9%** utilised, i.e. ResNet-50
inference remains heavily memory-bound even with Tensor Cores; AMP raises the
roof far more than it raises the achieved throughput.*

### Why absolute energy is NOT compared
The two prior A6000 artifacts disagree with each other. For ResNet-50 at
batch 1, `A6000_COMPLETE_RESULTS.md` reports **507.63 mJ/image** while
`batch_scaling_results_a6000.csv` reports **1.336 J/image** — a 2.6×
discrepancy. Their energy accounting could not be reconstructed from the
artifacts, so this suite treats **throughput and compute utilisation as the
comparable axes** and reports its own energy with its method stated. Resolving
that discrepancy is a prerequisite for any cross-study energy claim.

---

## 6. Known issues and honest caveats

- **Idle-subtracted energy over-subtracts at low load.** The idle baseline is
  measured in the P8 power state, but even a bs=1 kernel puts the GPU in P2 with
  a higher power floor. This makes `energy_per_image_net` at bs=1 anomalously
  low (TITAN RTX: 38.9 mJ net vs 220.1 mJ at bs=2). **Use the gross figures at
  low load.** Both are reported.
- **Energy is sampled, not counted.** Trapezoid integration of `nvidia-smi`
  power at 100–200 ms. Good for comparing runs on this box; not a calibrated
  absolute measurement.
- **The pre-flight guard cannot distinguish a foreign idle allocation from a
  foreign active job** beyond VRAM and PID. It is deliberately conservative: it
  declines rather than competes. Note it keys on foreign compute PIDs and VRAM,
  *not* raw utilisation — `nvidia-smi` reports a decaying utilisation figure for
  seconds after a process exits, and an earlier version of this guard aborted
  runs by reading its own tail as foreign load.
- **ZeRO-3 was not run.** It is compiler-free and would work, but its per-layer
  parameter all-gather would mostly measure the 3.86 GB/s fabric rather than the
  method. Stage-2 item.
- **FSDP was run in pure FP32 only.** fp16 FSDP needs `ShardedGradScaler`,
  which changes the numerics relative to DDP and ZeRO; holding FP32 keeps all
  four parallelism modes comparable. FSDP and ZeRO therefore have no AMP arm on
  the TITAN RTX pair, where DDP does.
- **The A6000 cross-comparison is FP32-vs-FP32 only.** The prior study appears
  to have no mixed-precision arm, so the TITAN RTX AMP result (2,278 img/s)
  should not be read as beating a like-configured A6000. See §5.
- **No 8×/10× runs.** Mixing GPU types in one job was prohibited by design, and
  the largest same-type pool is 7× 1080 Ti.
- **Single-node only.** No multi-node, no InfiniBand (none present).
- **vLLM env has one unresolved pip conflict**: `datasets 5.1.0` wants
  `huggingface-hub>=1.31` while the pinned vLLM stack uses 0.36.2. `datasets`
  is unused by the benchmark and the core stack is coherent.
- **`transformers` must be pinned in the vLLM env.** Left to pip, vLLM
  0.6.3.post1 resolves `transformers` 5.x, which it cannot use. Install vLLM
  first, then pin `transformers==4.45.2` and `tokenizers<0.21`.

---

## 7. Project layout

```
gpu_bench_stage1/
├── bench/
│   ├── common.py        telemetry, pre-flight guard, memory cap, result I/O
│   ├── topology.py      GPU inventory + topology-aware group selection
│   ├── preflight.py     hard gate: arch coverage + per-GPU kernel execution
│   ├── dtype_probe.py   fp32/fp16/bf16 GEMM throughput per GPU type
│   ├── train_llm.py     GPT-2: single / DDP / FSDP / DeepSpeed (one code path)
│   ├── bench_cnn.py     ResNet-50 inference sweep (A6000-comparable)
│   ├── comm_bench.py    NCCL all-reduce bandwidth + P2P matrix
│   ├── infer_vllm.py    vLLM (TITAN RTX only)
│   ├── infer_hf.py      HF generate baseline (Pascal substitute)
│   ├── aggregate.py     results/raw/*.json -> tidy CSVs
│   └── make_report.py   CSVs -> SUMMARY.md
├── configs/             ds_zero1.json, ds_zero2.json
├── scripts/             00_preflight .. 70_infer_vllm, each runnable alone
├── results/raw/*.json   one record per run, with pre-run GPU snapshot
├── results/*.csv        tidy tables
├── reports/             README.md (this file), SUMMARY.md (generated)
├── envs/                bench-train, bench-vllm (conda --prefix, deletable)
└── requirements-train.txt, requirements-vllm.txt, environment.yml
```

## 8. Reproducing

```bash
# environments (see environment.yml for the exact sequence and the
# transformers pin that the vLLM env requires)
conda create --prefix ./envs/bench-train python=3.10 -y
./envs/bench-train/bin/pip install torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu121
./envs/bench-train/bin/pip install -r requirements-train.txt

bash scripts/00_preflight.sh          # HARD GATE - must pass before anything
bash scripts/01_dtype_probe.sh        # precision policy evidence
bash scripts/10_single_gpu.sh         # Track A: single-GPU baselines
bash scripts/20_cnn_a6000_compare.sh  # Track B: A6000-comparable CNN
bash scripts/31_comm_bw.sh  rtx_2 p1080_2 p1080_4 p1080_6
BS=4 bash scripts/30_ddp.sh rtx_2 p1080_2 p1080_4   # clean groups first
BS=4 bash scripts/30_ddp.sh p1080_6                 # heavy: 6 of 7 1080 Ti
BS=4 bash scripts/40_deepspeed.sh rtx_2 p1080_4
BS=4 bash scripts/50_fsdp.sh rtx_2 p1080_4
bash scripts/60_infer_hf.sh
bash scripts/70_infer_vllm.sh

./envs/bench-train/bin/python bench/aggregate.py
./envs/bench-train/bin/python bench/make_report.py
```

Every script is independently runnable, guards its own devices, and records an
OOM as a result rather than crashing the suite.
