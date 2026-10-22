#!/usr/bin/env bash
# Full Stage-1 sequence, smallest and safest first: 1 GPU -> 2 GPUs -> 4 GPUs, training before inference.
# Every step is resumable (finished runs are skipped). Stop cleanly before the next run:  touch .cache/STOP
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

run() {
  "$PY" scripts/run_matrix.py "$@"
  rc=$?
  if [ $rc -ne 0 ]; then echo "run_matrix.py $* exited with $rc -> stopping the sequence"; exit $rc; fi
  if [ -e .cache/STOP ]; then echo "STOP file present -> stopping the sequence"; exit 0; fi
}

run configs/experiments/smoke.yaml
run configs/experiments/vllm_smoke.yaml
run configs/experiments/nccl.yaml
run configs/experiments/train_scaling.yaml --gpus 1
run configs/experiments/train_scaling.yaml --gpus 2
run configs/experiments/train_scaling.yaml --gpus 4
run configs/experiments/vllm_scaling.yaml
"$PY" scripts/aggregate.py && "$PY" scripts/make_report.py
echo "RUN_ALL_DONE"
