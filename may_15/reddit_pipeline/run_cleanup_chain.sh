#!/usr/bin/env bash
# Wait for the original run_curated_8.sh (PID passed as $1) to exit, then run
# the curated-8 queue again. Re-running is idempotent — completed subs skip
# (state.json has done=True), failed/partial ones resume from saved cursors.
# Loops up to 3 passes so a sustained Cloudflare outage doesn't strand us.

set -u

ORIG_PID="${1:?usage: $0 <pid-to-wait-for>}"
MAX_PASSES="${MAX_PASSES:-3}"

cd "$(dirname "$0")"

echo "============================================================"
echo "[$(date -u +%FT%TZ)] cleanup chain started, waiting for PID $ORIG_PID"
echo "============================================================"
while kill -0 "$ORIG_PID" 2>/dev/null; do
  sleep 30
done
echo "[$(date -u +%FT%TZ)] PID $ORIG_PID exited. beginning cleanup passes."

for pass in $(seq 1 "$MAX_PASSES"); do
  echo ""
  echo "============================================================"
  echo "[$(date -u +%FT%TZ)] CLEANUP PASS $pass / $MAX_PASSES"
  echo "============================================================"
  bash run_curated_8.sh

  # Count subs that still have any not-done kind.
  remaining=$(python3 -c "
import json, glob
n = 0
for f in glob.glob('../data/dumps/sub-*/fetch_state.json'):
    d = json.load(open(f))
    if not d.get('rs', {}).get('done') or not d.get('rc', {}).get('done'):
        n += 1
print(n)
")
  echo "[$(date -u +%FT%TZ)] after pass $pass: $remaining sub(s) still not fully done"
  if [ "$remaining" -eq 0 ]; then
    echo "[$(date -u +%FT%TZ)] all subs complete — cleanup chain exiting"
    break
  fi
  sleep 60
done

echo "[$(date -u +%FT%TZ)] cleanup chain finished"
