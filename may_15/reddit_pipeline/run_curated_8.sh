#!/usr/bin/env bash
# Sequentially fetch the 8 curated expert subreddits, smallest-first.
# Designed to run in background; survives login/logout via nohup.
#
# Usage:
#   nohup bash run_curated_8.sh > fetch.log 2>&1 &
#   tail -f fetch.log
#
# Resume: just re-run; each sub's fetch_state.json is checked before re-fetching.

set -u

cd "$(dirname "$0")"

OUT_DIR="${OUT_DIR:-../data/dumps}"
RATE="${RATE:-0.2}"

# Ordered smallest-first by approximate volume so user gets visible output quickly.
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

mkdir -p "$OUT_DIR"

for sub in "${SUBS[@]}"; do
  echo "============================================================"
  echo "[$(date -u +%FT%TZ)] starting $sub"
  echo "============================================================"
  python fetch_subreddit.py --sub "$sub" --out-dir "$OUT_DIR" --rate-seconds "$RATE"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date -u +%FT%TZ)] $sub exited with code $rc — moving on"
  else
    echo "[$(date -u +%FT%TZ)] $sub done"
  fi
done

echo "[$(date -u +%FT%TZ)] all subs complete"
