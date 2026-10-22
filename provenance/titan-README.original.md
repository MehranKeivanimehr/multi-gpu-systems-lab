# Stage 1 — Multi-GPU Benchmark Suite (heterogeneous 10-GPU cluster)

**Start here → [`reports/README.md`](reports/README.md)** — hardware limitations,
methodology, full results, A6000 comparison, and honest caveats.
**Generated tables → [`reports/SUMMARY.md`](reports/SUMMARY.md)**.

Benchmarked on `ten.ece.ufl.edu`: 2× TITAN RTX (24 GB, sm_75), 7× GTX 1080 Ti
(11 GB, sm_61), 1× TITAN Xp (12 GB, sm_61). Driver 535.230.02 / CUDA 12.2.
No NVLink — PCIe only.

## Headline result

The four "obsolete" GTX 1080 Ti scale **better** than the two TITAN RTX
(83.7% vs 66.5% DDP efficiency), because CUDA P2P is disabled on exactly the
two Turing cards while all 8 Pascal GPUs have it — including across the PCIe
host bridge. The RTX pair stages every gradient all-reduce through host memory
at 3.86 GB/s versus ~11 GB/s for Pascal.

| Group | P2P | all-reduce busbw | DDP efficiency |
|---|---|---|---|
| 2× TITAN RTX | **disabled** | 3.86 GB/s | 66.5% |
| 2× GTX 1080 Ti | enabled | 11.04 GB/s | **88.2%** |
| 4× GTX 1080 Ti | enabled | **11.31 GB/s** | 83.7% |
| 6× GTX 1080 Ti | enabled | 9.52 GB/s | 80.8% |

Throughput, scaling efficiency and energy rank the groups in **three different
orders**, so no single-number verdict about this cluster is complete.

## Contents

```
bench/      9 benchmark modules (telemetry, topology, preflight, train, infer, aggregate)
scripts/    8 numbered shell drivers, each independently runnable
configs/    DeepSpeed ZeRO stage 1 / 2 configs
results/    7 tidy CSVs + raw/ (64 per-run JSON records)
reports/    README.md (main document), SUMMARY.md (auto-generated)
requirements-train.txt, requirements-vllm.txt, environment.yml
```

## Reproducing

`envs/` and the Hugging Face cache are **not** included in this archive.
Recreate them with the sequence in [`environment.yml`](environment.yml), then:

```bash
bash scripts/00_preflight.sh     # HARD GATE — must pass before anything runs
bash scripts/01_dtype_probe.sh
bash scripts/10_single_gpu.sh
# ... see reports/README.md §8 for the full ordered list
```

Note the `transformers` pin required by the vLLM environment — left to pip,
vLLM 0.6.3 resolves transformers 5.x, which it cannot use.
