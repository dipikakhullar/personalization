#!/usr/bin/env bash
# Extract (query, preferred_answer) pairs for the newly-fetched subreddits whose
# dumps exist under data/dumps/sub-<Name>/ but were never turned into pairs.
# Output lands in data/extracted_current/pairs/sub-<Name>.jsonl alongside the
# original 29 subs. Resumable-ish: skips a sub whose pairs file already exists
# and is non-empty.
#
# Usage:
#   nohup bash extract_new_subs.sh > ../data/extract_new.log 2>&1 &
#   tail -f ../data/extract_new.log

set -u
cd "$(dirname "$0")"

DUMPS_DIR=../data/dumps
OUT_DIR=../data/extracted_current
PARALLEL=2

# smallest dumps first so we see output early; the 100G+ subs run last
SUBS=(
  statistics
  learnmachinelearning
  datascience
  GradSchool
  MachineLearning
  learnprogramming
  povertyfinance
  programming
  Cooking
  legaladvice
  personalfinance
  relationships
  relationship_advice
)

ts() { date -u +%FT%TZ; }
echo "[$(ts)] extract queue: ${#SUBS[@]} subs, parallelism=$PARALLEL"

run_one() {
  local sub="$1"
  local batch="sub-${sub}"
  local out="$OUT_DIR/pairs/${batch}.jsonl"
  if [ -s "$out" ]; then
    echo "[$(ts)] skip $sub (pairs already exist: $(wc -l < "$out") rows)"
    return
  fi
  if [ ! -d "$DUMPS_DIR/$batch" ]; then
    echo "[$(ts)] SKIP $sub (no dump dir)"; return
  fi
  echo "[$(ts)] START $sub"
  if python extract.py --month "$batch" --dumps-dir "$DUMPS_DIR" --out-dir "$OUT_DIR" \
        >> "../data/extract_${sub}.log" 2>&1; then
    echo "[$(ts)] DONE  $sub ($( [ -f "$out" ] && wc -l < "$out" || echo 0) pairs)"
  else
    echo "[$(ts)] FAIL  $sub (see data/extract_${sub}.log)"
  fi
}

i=0
while [ $i -lt ${#SUBS[@]} ]; do
  pids=()
  for sub in "${SUBS[@]:$i:$PARALLEL}"; do
    run_one "$sub" &
    pids+=($!)
  done
  wait "${pids[@]}" 2>/dev/null
  i=$((i + PARALLEL))
done

echo "[$(ts)] all extraction done"

# Option A: always follow extraction with the one-per-post merge so the
# canonical dataset (pairs_by_post/) stays current. See PIPELINE.md.
echo "[$(ts)] running by-post merge (first_preferred/other_preferred)"
python3 merge_pairs_by_post.py >> ../data/merge_by_post.log 2>&1
echo "[$(ts)] by-post merge done"
