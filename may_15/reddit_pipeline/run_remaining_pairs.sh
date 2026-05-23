#!/usr/bin/env bash
# Walk the remaining allowlist subreddits in pairs (BATCH=2 by default),
# launching each pair in parallel via nohup and waiting on the pair before
# starting the next. Resumable: re-running skips subs whose fetch_state.json
# reports both rs.done and rc.done.
#
# Order is roughly smallest-first so progress is visible early; huge subs
# (AskReddit, personalfinance, NoStupidQuestions, ELI5) run last.
#
# Usage:
#   nohup bash run_remaining_pairs.sh > ../data/pairs.log 2>&1 &
#   tail -f ../data/pairs.log
#
# Override batch size:
#   BATCH=1 nohup bash run_remaining_pairs.sh > ../data/pairs.log 2>&1 &

set -u
cd "$(dirname "$0")"

OUT_DIR="${OUT_DIR:-../data/dumps}"
LOG_DIR="${LOG_DIR:-../data}"
RATE="${RATE:-0.05}"
BATCH="${BATCH:-2}"

SUBS=(
  # tiny — language / hobby
  Coffee tea houseplants Sewing LearnJapanese German French Spanish
  languagelearning Shoestring JapanTravel solotravel
  AskBaking woodworking bicycling rust golang learnjavascript
  LanguageTechnology Sewing
  # small-to-medium — hobby / tech
  homeimprovement DIY gardening running bodyweightfitness Fitness
  travel datascience MachineLearning selfhosted homelab buildapc
  techsupport TooAfraidToAsk answers
  careerguidance jobs learnpython learnprogramming
  Cooking
  # big — life advice / general QA last
  relationship_advice relationships legaladvice personalfinance
  explainlikeimfive NoStupidQuestions AskReddit
)

# De-dupe while preserving order, then filter out completed subs.
declare -A seen=()
queue=()
for sub in "${SUBS[@]}"; do
  key=$(echo "$sub" | tr '[:upper:]' '[:lower:]')
  if [ -n "${seen[$key]:-}" ]; then continue; fi
  seen[$key]=1
  state="$OUT_DIR/sub-${sub}/fetch_state.json"
  if [ -f "$state" ]; then
    done_flag=$(python3 -c "
import json
try:
    d = json.load(open('$state'))
    print('1' if d['rs']['done'] and d['rc']['done'] else '0')
except Exception:
    print('0')
")
    if [ "$done_flag" = "1" ]; then
      echo "[$(date -u +%FT%TZ)] skip $sub (already complete)"
      continue
    fi
  fi
  queue+=("$sub")
done

echo "[$(date -u +%FT%TZ)] queue: ${#queue[@]} subs, batch size $BATCH"
echo "[$(date -u +%FT%TZ)] order: ${queue[*]}"

i=0
while [ $i -lt ${#queue[@]} ]; do
  batch=("${queue[@]:$i:$BATCH}")
  pids=()
  for sub in "${batch[@]}"; do
    log="$LOG_DIR/fetch_${sub}.log"
    echo "[$(date -u +%FT%TZ)] launching $sub → $log"
    nohup python fetch_subreddit.py \
        --sub "$sub" \
        --out-dir "$OUT_DIR" \
        --rate-seconds "$RATE" \
        >> "$log" 2>&1 &
    pids+=($!)
  done
  echo "[$(date -u +%FT%TZ)] waiting on batch: ${batch[*]} (pids: ${pids[*]})"
  wait "${pids[@]}" 2>/dev/null
  echo "[$(date -u +%FT%TZ)] batch done: ${batch[*]}"
  i=$((i + BATCH))
done

echo "[$(date -u +%FT%TZ)] all remaining subs complete"
