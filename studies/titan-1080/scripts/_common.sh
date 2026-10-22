#!/usr/bin/env bash
# Shared environment for every Stage-1 script. Source this, do not execute it.
#
# Safety posture encoded here:
#   * PCI_BUS_ID ordering so CUDA_VISIBLE_DEVICES indices match nvidia-smi.
#   * All caches inside the project dir, so nothing lands in $HOME or /tmp and
#     cleanup is a single rm -rf of one directory.
#   * NCCL kept to PCIe reality: no InfiniBand on this host, warnings only.
#   * No sudo, no system package manager, no host modification anywhere.

set -euo pipefail

PROJ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJ
export PY="$PROJ/envs/bench-train/bin/python"
# vLLM lives in its own env: it hard-pins torch, and sharing one env risks
# silently breaking the DDP/FSDP/DeepSpeed stack.
export PY_VLLM="$PROJ/envs/bench-vllm/bin/python"
export BENCH="$PROJ/bench"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONPATH="$BENCH:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# keep every cache inside the project
export HF_HOME="$PROJ/cache/hf"   # supersedes TRANSFORMERS_CACHE (deprecated)
export TORCH_HOME="$PROJ/cache/torch"
export TRITON_CACHE_DIR="$PROJ/cache/triton"
export XDG_CACHE_HOME="$PROJ/cache/xdg"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" \
         "$PROJ/results/raw" "$PROJ/logs"

# DeepSpeed: never JIT-compile. Host nvcc is 10.1 against a 12.2 driver.
export DS_BUILD_OPS=0
export DS_SKIP_CUDA_CHECK=1

# NCCL on a PCIe-only box
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=4

# Derive a per-job MASTER_PORT so we cannot collide with another user's
# torchrun on this shared node. Range 29600-30599.
bench_port() {
  local key="$1"
  local h
  h=$(printf '%s' "$key" | cksum | cut -d' ' -f1)
  echo $(( 29600 + (h % 1000) ))
}

# Resolve a named group (rtx_1, rtx_2, p1080_1, p1080_2, p1080_4, p1080_6, xp_1)
# to its live, topology-chosen device list.
bench_group_devices() {
  "$PY" - "$1" <<'EOF'
import json, sys
sys.path.insert(0, __import__("os").environ["BENCH"])
from topology import resolve_groups
g = resolve_groups()[sys.argv[1]]
print(g["cuda_visible_devices"])
EOF
}

# Guard a device list from the launcher, before any rank starts. Ranks then run
# with --skip-guard so N processes do not each re-query nvidia-smi and so no
# rank mistakes its siblings for foreign load.
bench_guard() {
  BENCH_PHYS_DEVICES="$1" "$PY" - "$1" <<'PYEOF'
import os, sys
sys.path.insert(0, os.environ["BENCH"])
from common import guard_devices, wait_for_idle
devs = [int(x) for x in sys.argv[1].split(",")]
guard_devices(devs)
waited = wait_for_idle(devs)
if waited > 1.0:
    print(f"  (waited {waited:.0f}s for GPUs {devs} to settle)")
PYEOF
}
