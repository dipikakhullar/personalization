#!/usr/bin/env bash
# Scan data/dumps/sub-*/, run extract.py + generate_user_traces.py for every
# sub with substantial data (>=100 submissions AND >=1000 comments). Writes:
#   ../data/extracted_current/pairs/sub-<name>.jsonl   (overwritten each run)
#   ../data/users_<name>.md                            (overwritten each run)
#
# If fetch_state.json reports either kind not yet done, the users file gets a
# "PARTIAL — fetch still in progress" header note. Re-running once the fetch
# completes overwrites with the full-data version.
#
# Usage:
#   bash auto_users_docs.sh
#
# Cheap (seconds per sub for extract; doesn't touch the running fetcher).

set -u

cd "$(dirname "$0")"

OUT_DIR="../data/extracted_current"
MIN_RS=100
MIN_RC=1000

mkdir -p "$OUT_DIR"

for d in ../data/dumps/sub-*/; do
  d="${d%/}"
  sub_name="$(basename "$d" | sed 's/^sub-//')"
  rs_file="$d/RS_${sub_name}.ndjson"
  rc_file="$d/RC_${sub_name}.ndjson"
  state_file="$d/fetch_state.json"

  rs_n=0
  rc_n=0
  [ -f "$rs_file" ] && rs_n=$(wc -l < "$rs_file")
  [ -f "$rc_file" ] && rc_n=$(wc -l < "$rc_file")

  if [ "$rs_n" -lt "$MIN_RS" ] || [ "$rc_n" -lt "$MIN_RC" ]; then
    echo "[skip] sub-$sub_name  (rs=$rs_n  rc=$rc_n  below thresholds)"
    continue
  fi

  is_partial=$(python3 -c "
import json
d = json.load(open('$state_file'))
print('1' if (not d['rs']['done']) or (not d['rc']['done']) else '0')
" 2>/dev/null || echo "1")

  echo ""
  echo "============================================================"
  echo "[extract+gen] sub-$sub_name  (rs=$rs_n  rc=$rc_n  partial=$is_partial)"
  echo "============================================================"

  python extract.py --month "sub-$sub_name" \
      --dumps-dir ../data/dumps --out-dir "$OUT_DIR" || {
    echo "  extract failed for $sub_name — moving on"
    continue
  }

  pair_file="$OUT_DIR/pairs/sub-${sub_name}.jsonl"
  if [ ! -s "$pair_file" ]; then
    echo "  no pairs produced for $sub_name — skipping doc generation"
    continue
  fi

  args=(--pairs "$pair_file" --out "../data/users_${sub_name}.md")
  if [ "$is_partial" = "1" ]; then
    args+=(--note "⚠️ PARTIAL — fetch still in progress; numbers below are a snapshot of the data pulled so far")
  fi

  python generate_user_traces.py "${args[@]}"
done

echo ""
echo "[done] $(date -u +%FT%TZ)"
ls -lh ../data/users_*.md 2>/dev/null
