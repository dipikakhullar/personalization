#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
DUMPS_DIR=../data/dumps; OUT_DIR=../data/extracted_current; PARALLEL=3
SUBS=(soccer nosleep nfl nba formula1 hiphopheads CryptoCurrency gaming wallstreetbets
      WritingPrompts StarWars leagueoflegends aww space cars todayilearned television movies funny dogs)
ts(){ date -u +%FT%TZ; }
echo "[$(ts)] diverse20 extract start, ${#SUBS[@]} subs, parallelism=$PARALLEL"
run_one(){
  local sub="$1" batch="sub-$1" out="$OUT_DIR/pairs/sub-$1.jsonl"
  [ -s "$out" ] && { echo "[$(ts)] skip $sub ($(wc -l <"$out") pairs)"; return; }
  [ -d "$DUMPS_DIR/$batch" ] || { echo "[$(ts)] SKIP $sub (no dump)"; return; }
  echo "[$(ts)] START $sub"
  if python extract.py --month "$batch" --dumps-dir "$DUMPS_DIR" --out-dir "$OUT_DIR" >>"../data/extract_${sub}.log" 2>&1; then
    echo "[$(ts)] DONE $sub ($([ -f "$out" ] && wc -l <"$out" || echo 0) pairs)"
  else echo "[$(ts)] FAIL $sub"; fi
}
i=0
while [ $i -lt ${#SUBS[@]} ]; do
  pids=(); for s in "${SUBS[@]:$i:$PARALLEL}"; do run_one "$s" & pids+=($!); done
  wait "${pids[@]}" 2>/dev/null; i=$((i+PARALLEL))
done
echo "[$(ts)] diverse20 extract done"
