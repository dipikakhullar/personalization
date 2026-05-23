#!/usr/bin/env bash
# Launch one nohup'd process per incomplete (subreddit, kind) so they pull in
# parallel. Each writes to its own log under ../data/fetch_<sub>_<kind>.log.
#
# The arctic-shift API is bottlenecked by ~1s per-request latency on a single
# connection, so single-process pacing changes don't speed things up. Running
# multiple processes is the only way to use more of the rate-limit ceiling
# (50 req/sec global per IP).
#
# Per-process rate is set generously (--rate-seconds 0.05 = up to 20 req/sec
# target) but in practice each process runs at ~1 req/sec due to latency.
#
# Usage:
#   nohup bash parallel_runner.sh > ../data/parallel.log 2>&1 &
#
# Then later:
#   bash auto_users_docs.sh    # regenerate per-sub user-trace md files

set -u
cd "$(dirname "$0")"

OUT_DIR="../data/dumps"
LOG_DIR="../data"
RATE="${RATE:-0.05}"

SUBS=(
  AskStatistics
  AskAcademia
  AskCulinary
  askphilosophy
  AskDocs
  AskEngineers
  AskHistorians
  askscience
)

declare -a PIDS=()

# One process per sub (it does posts then comments internally, resuming each
# from saved state). Sub-level parallelism avoids state.json write races that
# kind-level parallelism would cause.
for sub in "${SUBS[@]}"; do
  state="$OUT_DIR/sub-${sub}/fetch_state.json"
  fully_done="0"
  if [ -f "$state" ]; then
    fully_done=$(python3 -c "
import json
d = json.load(open('$state'))
print('1' if d['rs']['done'] and d['rc']['done'] else '0')
")
  fi
  if [ "$fully_done" = "1" ]; then
    echo "[$(date -u +%FT%TZ)] skip $sub (fully complete)"
    continue
  fi
  log="$LOG_DIR/fetch_${sub}.log"
  echo "[$(date -u +%FT%TZ)] launching $sub  → $log"
  nohup python fetch_subreddit.py \
      --sub "$sub" \
      --out-dir "$OUT_DIR" \
      --rate-seconds "$RATE" \
      >> "$log" 2>&1 &
  PIDS+=($!)
done

echo "[$(date -u +%FT%TZ)] launched ${#PIDS[@]} parallel fetcher(s): ${PIDS[*]}"
echo "[$(date -u +%FT%TZ)] waiting for all to finish..."

wait "${PIDS[@]}" 2>/dev/null
echo "[$(date -u +%FT%TZ)] all parallel fetchers exited"

# After parallel pass, run sequential cleanup for anything that's still not done
# (catches transient failures during the parallel pass).
echo "[$(date -u +%FT%TZ)] running sequential cleanup pass to fill any gaps..."
bash run_curated_8.sh
echo "[$(date -u +%FT%TZ)] cleanup pass done"

echo "[$(date -u +%FT%TZ)] regenerating user-trace docs"
bash auto_users_docs.sh
echo "[$(date -u +%FT%TZ)] all done"
