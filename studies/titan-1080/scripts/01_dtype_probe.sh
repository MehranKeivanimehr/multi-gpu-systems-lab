#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
echo "### dtype capability probe  $(date -Is)"
for spec in "rtx_1:TITAN_RTX" "p1080_1:GTX_1080Ti" "xp_1:TITAN_Xp"; do
  group="${spec%%:*}"; label="${spec##*:}"
  devs="$(bench_group_devices "$group")"
  echo; echo "--- $label  (physical GPU $devs) ---"
  CUDA_VISIBLE_DEVICES="$devs" BENCH_PHYS_DEVICES="$devs" \
    timeout --signal=TERM --kill-after=30s 600s \
    "$PY" "$BENCH/dtype_probe.py" --label "$label" || echo "  [exit $?]"
done
