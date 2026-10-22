# Contributing

Repository maintainer: Mehran Keivanimehr. Read [LICENSE](LICENSE) before redistributing changes.

## Add measurements

1. Create a new campaign directory instead of overwriting the recorded datasets.
2. Include source code, complete environment versions, hardware inventory, GPU topology, and the exact run configuration.
3. Record warmup, timing window, repetitions, batch size, sequence length, optimizer, parameter/reduction precision, checkpointing, and serving concurrency.
4. Preserve failure and skip records alongside successful runs.
5. Add a loader and protocol description to `tools/build_report.py` and regression checks to `tests/`.
6. Extend the measurement manifest only when intentionally accepting a new dataset. Never use a new hash to conceal an unexplained change.
7. Run the offline checks and document any GPU execution validation separately.

## Change code or reports

Use small changes with a clear reason. A change to the benchmark implementation creates a new experimental version; it does not retroactively change how existing measurements were obtained. The root report is generated, so change its generator and rebuild it together with `results/`.

```bash
python tools/validate.py
python -B tools/release_checks.py --profile portable
python tools/build_report.py --check
```

Do not commit environment directories, model weights, access tokens, or new run outputs over the reference records. Describe whether a validation was performed offline or on a GPU in the pull request.
