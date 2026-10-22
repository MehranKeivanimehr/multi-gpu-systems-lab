#!/usr/bin/env bash
# Track C -- PyTorch DDP weak scaling.
#
# Weak scaling: micro-batch per GPU is held at the value Track A established as
# the largest that fits an 11 GB 1080 Ti (bs=8 -> 8740 MiB of 11264 MiB), so
# ideal throughput is linear in GPU count and the shortfall is attributable to
# gradient all-reduce over PCIe.
#
# Groups are passed as arguments so the heavy 6x run can be held back:
#   ./30_ddp.sh rtx_2 p1080_2 p1080_4      # clean groups (PHB pair, PIX groups)
#   ./30_ddp.sh p1080_6                    # the PHB-crossing 6x run
#
# Never mixes GPU types in one job. Each launch gets its own MASTER_PORT so it
# cannot collide with another user's torchrun on this shared node.

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

BS=${BS:-8}
STEPS=${STEPS:-30}
WARMUP=${WARMUP:-5}
SEQ=${SEQ:-512}
# NB: do not name this GROUPS -- that is a bash special variable (user GIDs)
#     and assignment to it is silently ignored.
BENCH_GROUPS=${@:-"rtx_2 p1080_2 p1080_4"}

echo "### Track C: DDP weak scaling  $(date -Is)"
echo "### GPT-2 124M, seq=$SEQ, micro-batch=$BS/GPU, steps=$STEPS, mem_frac=0.85"
echo "### groups: $BENCH_GROUPS"
echo

run_ddp() {
  local group="$1" prec="$2" devs="$3" n="$4"
  local port name rc
  port=$(bench_port "${group}_${prec}_${BS}")
  name="C_ddp_${group}_bs${BS}_${prec}"

  echo "--- DDP $group ($prec): $n GPUs, devices=$devs, port=$port ---"
  bench_guard "$devs" || { echo "  [guard refused -- skipping]"; return 1; }

  CUDA_VISIBLE_DEVICES="$devs" BENCH_PHYS_DEVICES="$devs" \
    timeout --signal=TERM --kill-after=30s 1200s \
    "$PY" -m torch.distributed.run \
      --nnodes=1 --nproc_per_node="$n" \
      --master_addr=127.0.0.1 --master_port="$port" \
      "$BENCH/train_llm.py" \
        --mode ddp --group "$group" --batch-size "$BS" \
        --seq-len "$SEQ" --steps "$STEPS" --warmup "$WARMUP" \
        --precision "$prec" --out-name "$name" --skip-guard \
    2>&1 | grep -Ev "FutureWarning|warnings.warn|^\s*$|master_addr is only used|Setting OMP_NUM_THREADS|^\*\*\*|torch.distributed.run"
  rc=${PIPESTATUS[0]}
  case $rc in
    0)  ;;
    42) echo "  [OOM -- recorded]" ;;
    124|137) echo "  [TIMEOUT -- killed cleanly]" ;;
    *)  echo "  [exit $rc]" ;;
  esac
  echo
  return 0
}

for group in $BENCH_GROUPS; do
  devs="$(bench_group_devices "$group")"
  n=$(awk -F, '{print NF}' <<<"$devs")
  # AMP-fp16 only on the Turing pair, where the dtype probe measured a real
  # 8x GEMM advantage. On Pascal fp16 is 0.85-1.15x of fp32: not worth a run.
  case "$group" in
    rtx_*) precs="fp32 amp_fp16" ;;
    *)     precs="fp32" ;;
  esac
  for prec in $precs; do
    run_ddp "$group" "$prec" "$devs" "$n"
  done
done

echo "### Track C complete. GPU state:"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
