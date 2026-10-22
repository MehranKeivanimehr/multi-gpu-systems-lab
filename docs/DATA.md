# Data catalog and schemas

## Canonical measurements

| Campaign | Canonical source | Derived campaign tables |
|---|---|---|
| A6000 | `studies/a6000/results/raw/<experiment>/<run>/status.json`, `spec.json`, `result.json`, and logs | `results/runs.csv`, `summary_train.csv`, `summary_vllm.csv`, `nccl.csv`, `rerun/*.csv` within the campaign |
| Titan/1080 | `studies/titan-1080/results/raw/*.json`; some records contain several batch or size rows | `results/*.csv` within the campaign |
| A100 | `studies/a100/*.jsonl`, one measured configuration per line; `env.json` | per-benchmark CSV and `all_results.csv` |

[Campaign sources (2026/10), [A6000](../studies/a6000/results/), [Titan/1080](../studies/titan-1080/results/), [A100](../studies/a100/)]

The measurement manifest records relative paths, file sizes, and SHA-256 hashes for these imported measurements and historical documents. `python tools/validate.py` verifies their bytes. New reproduction runs should be made in a separate working copy; the supplied reference records are the baseline dataset.

## Combined run index

`results/run_index.csv` is generated from canonical sources, including A6000 audit reruns that are absent from its older `runs.csv`. One row means one A6000 run directory, one Titan JSON record, or one A100 JSONL measurement. These units differ: a Titan CNN/inference/communication record can contain multiple observations. Therefore the index is an artifact catalog, not a statistical sample table.

| Column | Meaning |
|---|---|
| `campaign` | `a6000`, `titan-1080`, or `a100` |
| `experiment` | Original experiment or artifact family |
| `run_id` | Original run ID, Titan filename stem, or A100 configuration key |
| `kind` | Training, inference, communication, dtype probe, or preflight |
| `status` | Original status; `rows_ok` for Titan nested rows; `recorded` for unlabelled probe records |
| `model`, `framework`, `gpu` | Recorded model and execution stack; GPU family for the campaign |
| `n_gpus`, `batch_per_gpu`, `seq_len`, `precision` | Configuration when available; empty means unrecorded or not applicable |
| `source` | Relative canonical source path; JSONL references include `#L<number>` |

`results/inventory.json` counts catalog entries, status distributions, canonical observation counts, and supplied aggregate-table rows. It never treats skipped configurations as successful measurements.

## Units and missing values

Memory normalized in explanatory documents uses GiB or MiB. A100 fields named `*_gb` are calculated with `2**30` and therefore represent GiB. Titan `size_mb` is a historical field name for a MiB allocation; its bandwidth is decimal GB/s. The raw names and numerical values are preserved. [Unit implementations (2026/10), [A100 notebook](../provenance/a100-original.ipynb) and [Titan communication code](../studies/titan-1080/bench/comm_bench.py)]

Empty CSV cells, JSON `null`, or missing fields mean unavailable, not zero. A failed/OOM/skipped run has no valid throughput. Repetition counts represent independent recorded runs; timed iterations inside one run are not independent repeats. [Run schemas (2026/10), [A6000 example](../studies/a6000/results/raw/train_scaling/qwen2.5-1.5b__ddp__g4__r0/result.json) and [Titan example](../studies/titan-1080/results/raw/A_rtx_1_bs4_fp32.json)]

## Metadata retained

Reference artifacts retain hostnames, GPU indices/topology, timestamps, environment versions, and command-line paths from the experiments. They support traceability. Model weights, installed environments, download caches, and credential files are excluded. The original A100 notebook may contain historical output and remote upload cells; use the cleaned reproduction notebook for execution.
