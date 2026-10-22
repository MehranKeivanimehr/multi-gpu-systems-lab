# Verification and future execution

The finished public analysis has CPU-only entry points. Start with [reproduction](../docs/REPRODUCING.md) and [the test profiles](../TESTING.md).

```sh
python -B tools/release_checks.py --profile portable
python -I -S tools/verify_a6000.py
```

The earlier `findings.json` registry is retained as a historical correction plan. Current outcomes are in [scientific status](../docs/SCIENTIFIC_STATUS.md), the [A6000 ledger](../results/a6000_validation/evidence/70_final_review/CLAIM_LEDGER_v2.csv) and the [Titan context ledger](../results/diagnostics/context_ledger.json).

## Do not turn offline verification into a GPU launch

No portable profile launches a model or imports CUDA/vLLM/DeepSpeed. Historical execution scripts and on-node tests are retained for inspection, not included in ordinary offline acceptance. Device-denying isolation is required when a test exercises a GPU-capable production worker; environment labels alone are not isolation.

Future cluster work requires current facility permission, exact device identity, a fresh time/resource window and source-bound authorization. Shared-node safety controls are not a substitute for an allocation. Keep the shipped launch holds active unless those conditions are deliberately met.

## Limitations

CPU checks do not establish hardware behavior. Historical allocations expired; public metadata changes do not renew them. See [the complete scope](../docs/LIMITATIONS.md).
