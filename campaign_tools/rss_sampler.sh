#!/usr/bin/env bash
# Sample total RSS of this user's python/torchrun processes. Peak host RAM for
# the campaign was a PLANNING CEILING in estimates.json, never measured; this
# records what the jobs actually used. Read-only, no privileged calls.
OUT="$1"; INT="${2:-2}"
echo "utc,rss_total_mib,n_procs,top_proc_mib" > "$OUT"
while true; do
  read -r tot n top <<<"$(ps -u "$(id -u)" -o rss=,comm= 2>/dev/null \
    | awk '$2 ~ /python|torchrun|pt_main/ {s+=$1; n++; if($1>m) m=$1} END{printf "%.1f %d %.1f", s/1024, n+0, m/1024}')"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),$tot,$n,$top" >> "$OUT"
  sleep "$INT"
done
