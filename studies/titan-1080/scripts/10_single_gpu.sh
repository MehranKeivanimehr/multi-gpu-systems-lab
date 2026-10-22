#!/usr/bin/env bash
# Track A -- single-GPU GPT-2 124M baselines.
#
# Identical config on all three GPU types (seq 512, FP32) so Pascal and Turing
# are directly comparable, plus an AMP-fp16 arm on the TITAN RTX only, where
# the dtype probe measured a real 8x Tensor-Core GEMM advantage.
#
# Batch sweep finds both the throughput-optimal and the largest safe batch per
# GPU. An OOM is an expected, recorded outcome (exit 42), not a failure of the
# run: the allocator is capped at 0.85 so it lands inside our process.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

STEPS=${STEPS:-30}
WARMUP=${WARMUP:-5}
SEQ=${SEQ:-512}

echo "### Track A: single-GPU baselines  $(date -Is)"
echo "### GPT-2 124M, seq=$SEQ, steps=$STEPS (warmup $WARMUP), memory_fraction=0.85"
echo

run_one() {
  local group="$1" bs="$2" prec="$3" devs="$4"
  local name="A_${group}_bs${bs}_${prec}"
  CUDA_VISIBLE_DEVICES="$devs" BENCH_PHYS_DEVICES="$devs" \
    timeout --signal=TERM --kill-after=30s 900s \
    "$PY" "$BENCH/train_llm.py" \
      --mode single --group "$group" --batch-size "$bs" \
      --seq-len "$SEQ" --steps "$STEPS" --warmup "$WARMUP" \
      --precision "$prec" --out-name "$name" 2>&1 | grep -Ev "FutureWarning|warnings.warn"
  local rc=${PIPESTATUS[0]}
  case $rc in
    0)  ;;
    42) echo "  [OOM at bs=$bs $prec -- recorded, stopping sweep for this arm]" ;;
    124|137) echo "  [TIMEOUT at bs=$bs $prec -- killed cleanly]" ;;
    *)  echo "  [exit $rc at bs=$bs $prec]" ;;
  esac
  return $rc
}

for spec in "rtx_1:fp32 amp_fp16" "p1080_1:fp32" "xp_1:fp32"; do
  group="${spec%%:*}"; precs="${spec##*:}"
  devs="$(bench_group_devices "$group")"
  for prec in $precs; do
    echo "--- $group ($prec), physical GPU $devs ---"
    for bs in 1 2 4 8 12 16; do
      # 1080 Ti / Xp have 11-12 GB; stop the common sweep at 8 and let the
      # larger sizes run only where there is headroom.
      if [[ "$group" != "rtx_1" && $bs -gt 8 ]]; then break; fi
      run_one "$group" "$bs" "$prec" "$devs" || break
    done
    echo
  done
done

echo "### Track A complete. Post-run GPU state:"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
