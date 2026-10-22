# Reproduce the analysis

Run commands from the repository root. Python 3.10 or 3.12 can run the standard-library profile. These commands do not start a GPU workload. [Verification implementation](../tools/release_checks.py).

## 1. Verify the public evidence

```sh
python tools/validate.py
python -I -S tools/verify_a6000.py
python -I -S tools/recompute_reference_verdict.py --a results/diagnostics/closing/reference/TITAN_RTX_1 --b results/diagnostics/closing/reference/GTX_1080Ti_1
python -I -S tools/recompute_streaming_metrics.py --session results/diagnostics/streaming_session --expected 100
```

The A6000 command validates the exported payloads, input identities and matched-cap raw rates. The Titan commands separately recompute reference agreement and qualified streaming observations. Expected outcomes and boundaries are in [the results](../reports/REPORT.md) and [limitations](LIMITATIONS.md).

## 2. Check or regenerate derived files

```sh
python tools/build_showcase.py --check
python tools/build_report.py --check
python tools/scientific_status.py --check
python tools/scientific_corrections.py --check
python tools/prepare_notebook.py --check
```

To regenerate intentionally, run the same commands without `--check`. The input measurements are not regenerated. For the figures:

```sh
python tools/render_figures.py --svg-only
python tools/render_figures.py --check
```

SVG generation is standard-library-only. PNG/GIF export additionally needs Pillow (`requirements-figures.txt`); the output uses a local system font. PNG font rendering can differ across systems, while the SVG geometry and data values are deterministic.

## 3. Run the explicit test profiles

```sh
python -B tools/release_checks.py --profile portable
```

See [TESTING.md](../TESTING.md) for CPU numerical and Linux-only profiles. Do not use broad test discovery as an unattended cluster command: retained on-node tests have different sandbox and dependency requirements.

## 4. If you want a new hardware experiment

The historical training sources/configs remain under `studies/`. This release is an analysis artifact, not permission to launch them on a shared node. Reconstruct the recorded environment deliberately, verify the model revisions, review resource estimates and obtain an appropriate current allocation. Keep the launch holds until a new, source-bound plan is approved. [Execution controls](../validation/README.md), [A6000 protocol](../results/a6000_validation/evidence/70_final_review/PROTOCOLS_AND_ENVIRONMENT.md).

## Limitations

Offline replay verifies retained numerical evidence, not a new training/inference measurement. Missing full tensors, historical timing proxies and deferred model/GPU combinations stay explicit in [LIMITATIONS.md](LIMITATIONS.md).
