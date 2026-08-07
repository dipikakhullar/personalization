#!/usr/bin/env bash
# Fetch the 20 most-different ("diversity_v1") subreddits, 3 at a time.
# Resumable: fetch_subreddit.py resumes each sub from its fetch_state.json.
#
#   nohup bash fetch_diverse20.sh > ../data/fetch_diverse20.log 2>&1 &
#   tail -f ../data/fetch_diverse20.log

set -u
cd "$(dirname "$0")"

SUBS=(
  soccer nosleep nfl nba formula1
  hiphopheads CryptoCurrency gaming wallstreetbets
  WritingPrompts StarWars leagueoflegends aww space
  cars todayilearned television movies funny dogs
)
PARALLEL=3
OUT_DIR=/workspace/personalization/may_15/data/dumps
RATE=0.05

ts() { date -u +%FT%TZ; }
echo "[$(ts)] diverse20 fetch start, ${#SUBS[@]} subs, parallelism=$PARALLEL"

for sub in "${SUBS[@]}"; do
  # resume check: skip subs already fully fetched
  state="$OUT_DIR/sub-${sub}/fetch_state.json"
  if [ -f "$state" ]; then
    done_flag=$(python3 -c "
import json
try:
    d=json.load(open('$state')); print('1' if d['rs']['done'] and d['rc']['done'] else '0')
except Exception: print('0')")
    if [ "$done_flag" = "1" ]; then
      echo "[$(ts)] skip $sub (already complete)"; continue
    fi
  fi
  while [ "$(jobs -r | wc -l)" -ge "$PARALLEL" ]; do wait -n; done
  echo "[$(ts)] starting sub=$sub"
  python fetch_subreddit.py --sub "$sub" --out-dir "$OUT_DIR" --rate-seconds "$RATE" \
    >>"/tmp/fetch_${sub}.log" 2>&1 &
done
wait
echo "[$(ts)] diverse20 fetch done, all ${#SUBS[@]} subs finished"
