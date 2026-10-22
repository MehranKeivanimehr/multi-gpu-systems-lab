#!/usr/bin/env bash
# Track F' -- HF generate inference baseline on Pascal (vLLM cannot run there).
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
MODEL=${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}
: "${MODEL_REVISION:?Set MODEL_REVISION to an exact model/tokenizer commit}"
echo "### Track F': HF generate inference  $(date -Is)"
echo "### model=$MODEL  prompt=256 tok  max_new=128 tok  dtype=float32,float16"
echo
for spec in "rtx_1:TITAN_RTX" "p1080_1:GTX_1080Ti" "xp_1:TITAN_Xp"; do
  group="${spec%%:*}"; label="${spec##*:}"
  devs="$(bench_group_devices "$group")"
  echo "--- $label (physical GPU $devs) ---"
  for dtype in float32 float16; do
    CUDA_VISIBLE_DEVICES="$devs" BENCH_PHYS_DEVICES="$devs" \
      timeout --signal=TERM --kill-after=30s 1800s \
      "$PY" "$BENCH/infer_hf.py" --model "$MODEL" --group "$group" \
        --revision "$MODEL_REVISION" --label "$label" --batches 1,4,16 --dtype "$dtype" \
      2>&1 | grep -Ev "FutureWarning|warnings.warn|^\s*$|Loading checkpoint" || echo "  [exit $?]"
  done
  echo
done
echo "### Track F' complete."
