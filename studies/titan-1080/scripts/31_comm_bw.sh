#!/usr/bin/env bash
# NCCL all-reduce bandwidth per group -- explains the DDP scaling numbers.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
BENCH_GROUPS=${@:-"rtx_2 p1080_2 p1080_4"}
echo "### NCCL all-reduce bandwidth  $(date -Is)"
for group in $BENCH_GROUPS; do
  devs="$(bench_group_devices "$group")"
  n=$(awk -F, '{print NF}' <<<"$devs")
  port=$(bench_port "comm_${group}")
  link=$("$PY" -c "
import sys,os; sys.path.insert(0,os.environ['BENCH'])
from topology import resolve_groups; print(resolve_groups()['$group']['worst_link'])")
  echo; echo "--- $group: ${n}x GPU, devices=$devs, worst_link=$link ---"
  bench_guard "$devs" || { echo "  [guard refused]"; continue; }
  CUDA_VISIBLE_DEVICES="$devs" BENCH_PHYS_DEVICES="$devs" \
    timeout --signal=TERM --kill-after=30s 900s \
    "$PY" -m torch.distributed.run --nnodes=1 --nproc_per_node="$n" \
      --master_addr=127.0.0.1 --master_port="$port" \
      "$BENCH/comm_bench.py" --group "$group" \
    2>&1 | grep -Ev "Utils.hpp|NCCL version|Setting OMP|^\s*$|^\*\*\*|torch.distributed.run" || echo "  [exit $?]"
done
