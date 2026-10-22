#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
echo "### Stage-1 preflight  $(date -Is)"
echo "### project: $PROJ"
exec "$PY" "$BENCH/preflight.py"
