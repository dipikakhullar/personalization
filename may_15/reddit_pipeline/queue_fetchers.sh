#!/bin/bash
# Process a list of subreddits two-at-a-time via fetch_subreddit.py.
# Each sub's fetcher logs to /tmp/fetch_<sub>.log; this script logs queue
# transitions to its own stdout (typically /tmp/queue_fetchers.log when run
# under nohup).

set -u

SUBS=(
  MachineLearning
  learnmachinelearning
  programming
  learnprogramming
  personalfinance
  povertyfinance
  legaladvice
  Ask_Lawyers
  medicine
  relationships
  relationship_advice
  Cooking
  GradSchool
  statistics
  datascience
)

PARALLEL=2
OUT_DIR=/workspace/personalization/may_15/data/dumps
RATE=0.05

cd /workspace/personalization/may_15/reddit_pipeline

ts() { date -u +%FT%TZ; }

echo "[$(ts)] queue start, ${#SUBS[@]} subs, parallelism=$PARALLEL"

for sub in "${SUBS[@]}"; do
  while [ "$(jobs -r | wc -l)" -ge "$PARALLEL" ]; do
    wait -n
  done
  echo "[$(ts)] starting sub=$sub"
  python fetch_subreddit.py --sub "$sub" --out-dir "$OUT_DIR" --rate-seconds "$RATE" \
    >>"/tmp/fetch_${sub}.log" 2>&1 &
done

wait
echo "[$(ts)] queue done, all ${#SUBS[@]} subs finished"
