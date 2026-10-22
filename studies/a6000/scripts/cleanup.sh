#!/usr/bin/env bash
# Remove caches and temporary files created by benchmark runs. Results, logs and the venv are kept.
# Only touches paths inside this project directory.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "before: $(du -sh . --exclude=.venv | cut -f1) (excluding .venv)"
rm -rf .cache/triton .cache/inductor .cache/torch_extensions .cache/vllm .cache/STOP .cache/*.out .cache/runner.lock \
       .cache/llama_download.log .pytest_cache
find . -path ./.venv -prune -o -name __pycache__ -type d -print0 | xargs -0 rm -rf
find results/raw -name 'ready_*' -o -name 'go' -o -name '*.tmp' | xargs -r rm -f
echo "after:  $(du -sh . --exclude=.venv | cut -f1) (excluding .venv); venv: $(du -sh .venv | cut -f1)"
df -h . | tail -1
