#!/usr/bin/env bash
# Track D -- DeepSpeed ZeRO stages 1 and 2, no offload.
#
# Why light stages only: the host nvcc is 10.1 against a 12.2 driver, so any
# DeepSpeed op that JIT-compiles would fail. DS_BUILD_OPS=0 is exported in
# _common.sh, the optimizer is a plain torch AdamW (not FusedAdam), and no
# CPU/NVMe offload is configured -- cpu_adam would need a compiler. ZeRO 1 and 2
# in that configuration are pure Python + NCCL.
#
# ZeRO-3 is deliberately not run: it is also compiler-free, but its per-layer
# parameter all-gather multiplies communication volume, and on a fabric where
# the TITAN RTX pair already has no PCIe P2P (3.86 GB/s) the result would
# measure the fabric rather than the method. Noted as a Stage-2 item.
#
# Launched via torchrun rather than the `deepspeed` CLI so device selection
# stays under CUDA_VISIBLE_DEVICES and matches every other track exactly.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

BS=${BS:-4}
STEPS=${STEPS:-30}
WARMUP=${WARMUP:-5}
SEQ=${SEQ:-512}
STAGES=${STAGES:-"1 2"}
BENCH_GROUPS=${@:-"rtx_2 p1080_4"}

echo "### Track D: DeepSpeed ZeRO  $(date -Is)"
echo "### GPT-2 124M, seq=$SEQ, micro-batch=$BS/GPU, stages=$STAGES"
echo "### groups: $BENCH_GROUPS"
echo

for group in $BENCH_GROUPS; do
  devs="$(bench_group_devices "$group")"
  n=$(awk -F, '{print NF}' <<<"$devs")
  case "$group" in
    rtx_*) precs="fp32 amp_fp16" ;;
    *)     precs="fp32" ;;
  esac
  for stage in $STAGES; do
    for prec in $precs; do
      port=$(bench_port "ds${stage}_${group}_${prec}_${BS}")
      name="D_zero${stage}_${group}_bs${BS}_${prec}"
      echo "--- ZeRO-$stage $group ($prec): ${n}x GPU, devices=$devs, port=$port ---"
      bench_guard "$devs" || { echo "  [guard refused -- skipping]"; continue; }
      CUDA_VISIBLE_DEVICES="$devs" BENCH_PHYS_DEVICES="$devs" \
        timeout --signal=TERM --kill-after=30s 1500s \
        "$PY" -m torch.distributed.run \
          --nnodes=1 --nproc_per_node="$n" \
          --master_addr=127.0.0.1 --master_port="$port" \
          "$BENCH/train_llm.py" \
            --mode deepspeed --zero-stage "$stage" \
            --group "$group" --batch-size "$BS" --seq-len "$SEQ" \
            --steps "$STEPS" --warmup "$WARMUP" --precision "$prec" \
            --out-name "$name" --skip-guard \
        2>&1 | grep -Ev "FutureWarning|warnings.warn|^\s*$|Setting OMP_NUM_THREADS|^\*\*\*|torch.distributed.run|Utils.hpp|NCCL version"
      rc=${PIPESTATUS[0]}
      case $rc in
        0)  ;;
        42) echo "  [OOM -- recorded]" ;;
        124|137) echo "  [TIMEOUT -- killed cleanly]" ;;
        *)  echo "  [exit $rc]" ;;
      esac
      echo
    done
  done
done

echo "### Track D complete. GPU state:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
