# Test environments and exact commands

Run every command at the flattened repository root (the folder containing this file). There is no `source/` subdirectory.

## Portable offline acceptance

```sh
python -B tools/release_checks.py --profile portable
```

Standard library only. Includes the five generated-artifact/integrity checks, portable scientific tests, shipped evidence comparisons and local release regressions. Generated files are checked with --check, never overwritten before testing. The profile verifies that the tree did not change.

The combined release also checks the A6000 calibration payloads and matched-cap raw rates, current report/status generation and deterministic SVG figures. The CPU numerical profile additionally includes the A6000 boundary ledger and Candidate 20 regression fixtures. Neither profile starts a model or queries a GPU.

The optional NumPy cross-implementation test is explicitly skipped when NumPy is absent; the standalone stdlib commands remain required. `python -S -B tools/release_checks.py --profile portable` propagates -S to child interpreters.

The standalone evidence commands need no model, server, GPU or site-packages:

```sh
python -I -S tools/recompute_reference_verdict.py --a results/diagnostics/closing/reference/TITAN_RTX_1 --b results/diagnostics/closing/reference/GTX_1080Ti_1
python -I -S tools/recompute_streaming_metrics.py --session results/diagnostics/streaming_session --expected 100
```

## Torch CPU numerical tests

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.4.1+cpu
python -m pip install numpy==1.26.4
python -B tools/release_checks.py --profile torch
```

Torch is required, not silently skipped. This profile requires a CPU-only build and exercises numerical fixtures, not CUDA or installed DeepSpeed. Python 3.10 and 3.12 are the configured CI combinations.

## Linux POSIX lifecycle tests

```sh
python -B tools/release_checks.py --profile posix
```

Linux /proc, process groups and signals are required. Harmless dummy processes only. This includes the separate-session server/escaped-descendant accounting and cleanup test with an unrelated sentinel. Windows cannot establish this contract by skipping it.

## Shipped workflow and limits

The actual workflow is `.github/workflows/validate.yml`: portable checks on Ubuntu/Windows with Python 3.10/3.12, a separate Ubuntu Torch-CPU job, and a separate Ubuntu POSIX job. Installed-vLLM, model-cache and GPU-capable-worker contracts are deliberately not in ordinary offline CI. No GitHub workflow run is claimed.

Do not use unqualified unittest discovery to claim offline acceptance: the repository also carries on-node tests requiring the exact installed vLLM stack or a verified device-denying bubblewrap sandbox. They must not run on a shared GPU host merely because CUDA_VISIBLE_DEVICES is set. Synthetic UUIDs and test labels are not isolation. Missing sandbox support means an explicit deferral.

New listener/monitoring code has local CPU regression evidence only. A future authorized deployment must separately establish installed-stack behavior; no GPU rerun is required to correct historical claims. Local release receipts are private; historical Linux/CUDA receipts remain historical, not a rerun of modified code.
