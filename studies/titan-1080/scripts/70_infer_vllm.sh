#!/usr/bin/env bash
# Track F -- vLLM on the TITAN RTX pair only (sm_75). Pascal is 6.1 < 7.0.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
MODEL=${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}
: "${MODEL_REVISION:?Set MODEL_REVISION to an exact model/tokenizer commit}"
echo "### Track F: vLLM inference (TITAN RTX only)  $(date -Is)"
echo "### model=$MODEL dtype=fp16 gpu_mem_util=0.85 prompt=256 max_new=128"
echo
for spec in "rtx_1:TITAN_RTX:1" "rtx_2:TITAN_RTX:2"; do
  group="${spec%%:*}"; rest="${spec#*:}"; label="${rest%%:*}"; tp="${rest##*:}"
  devs="$(bench_group_devices "$group")"
  echo "--- vLLM TP=$tp on $label, devices=$devs ---"
  bench_guard "$devs" || { echo "  [guard refused -- skipping]"; continue; }
  CUDA_VISIBLE_DEVICES="$devs" BENCH_PHYS_DEVICES="$devs" \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    timeout --signal=TERM --kill-after=60s 2400s \
    "$PY_VLLM" "$BENCH/infer_vllm.py" --model "$MODEL" --group "$group" \
      --revision "$MODEL_REVISION" --label "$label" --tp "$tp" --concurrency 1,4,16,64 \
    2>&1 | grep -Ev "^\s*$|WARNING|Processed prompts|it/s\]|s/it\]" | tail -25 || echo "  [exit $?]"
  echo
done
echo "### Track F complete."
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | head -3
