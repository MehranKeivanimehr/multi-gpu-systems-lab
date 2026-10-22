#!/usr/bin/env bash
# Track B -- ResNet-50 inference batch sweep, the A6000-comparable axis.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
echo "### Track B: ResNet-50 inference sweep (A6000-comparable)  $(date -Is)"
echo
for spec in "rtx_1:TITAN_RTX:1,2,4,8,16,32,64,128" \
            "p1080_1:GTX_1080Ti:1,2,4,8,16,32,64" \
            "xp_1:TITAN_Xp:1,2,4,8,16,32,64"; do
  group="${spec%%:*}"; rest="${spec#*:}"; label="${rest%%:*}"; batches="${rest##*:}"
  devs="$(bench_group_devices "$group")"
  echo "--- $label (physical GPU $devs), fp32 ---"
  CUDA_VISIBLE_DEVICES="$devs" BENCH_PHYS_DEVICES="$devs" \
    timeout --signal=TERM --kill-after=30s 1800s \
    "$PY" "$BENCH/bench_cnn.py" --group "$group" --label "$label" \
      --batches "$batches" || echo "  [exit $?]"
  echo
done
echo "### Track B complete."
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
