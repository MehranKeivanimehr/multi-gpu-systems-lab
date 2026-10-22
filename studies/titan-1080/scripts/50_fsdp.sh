#!/usr/bin/env bash
# Track E -- FSDP FULL_SHARD, pure FP32.
#
# FP32 and not mixed precision on purpose: bf16 does not exist on sm_61/sm_75
# (measured at 0.58-0.63x of fp32), and fp16 FSDP would require
# ShardedGradScaler, changing the numerics relative to DDP and ZeRO. Holding
# fp32 keeps all four parallelism modes numerically comparable.
#
# Wrapping is per-GPT2Block via transformer_auto_wrap_policy, which is the
# unit FSDP is designed to shard for transformer models.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
BS=${BS:-4}; STEPS=${STEPS:-30}; WARMUP=${WARMUP:-5}; SEQ=${SEQ:-512}
BENCH_GROUPS=${@:-"rtx_2 p1080_4"}
echo "### Track E: FSDP FULL_SHARD fp32  $(date -Is)"
echo "### GPT-2 124M, seq=$SEQ, micro-batch=$BS/GPU"
echo
for group in $BENCH_GROUPS; do
  devs="$(bench_group_devices "$group")"
  n=$(awk -F, '{print NF}' <<<"$devs")
  port=$(bench_port "fsdp_${group}_${BS}")
  name="E_fsdp_${group}_bs${BS}_fp32"
  echo "--- FSDP $group: ${n}x GPU, devices=$devs, port=$port ---"
  bench_guard "$devs" || { echo "  [guard refused -- skipping]"; continue; }
  CUDA_VISIBLE_DEVICES="$devs" BENCH_PHYS_DEVICES="$devs" \
    timeout --signal=TERM --kill-after=30s 1500s \
    "$PY" -m torch.distributed.run --nnodes=1 --nproc_per_node="$n" \
      --master_addr=127.0.0.1 --master_port="$port" \
      "$BENCH/train_llm.py" --mode fsdp --group "$group" \
        --batch-size "$BS" --seq-len "$SEQ" --steps "$STEPS" \
        --warmup "$WARMUP" --precision fp32 --out-name "$name" --skip-guard \
    2>&1 | grep -Ev "FutureWarning|warnings.warn|^\s*$|Setting OMP_NUM_THREADS|^\*\*\*|torch.distributed.run|Utils.hpp|NCCL version"
  rc=${PIPESTATUS[0]}
  case $rc in
    0) ;; 42) echo "  [OOM -- recorded]" ;; 124|137) echo "  [TIMEOUT -- killed cleanly]" ;;
    *) echo "  [exit $rc]" ;;
  esac
  echo
done
echo "### Track E complete."
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
