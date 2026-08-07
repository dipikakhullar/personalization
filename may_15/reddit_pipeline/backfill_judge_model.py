"""Backfill judge_model on legacy is_qa_pair sidecar records.

Records written before the judge_model field existed have an is_qa_pair dict
with no judge_model key. Those were all produced by openai/gpt-4o
(LEGACY_DEFAULT_JUDGE_MODEL). This script writes that value into them on disk.

Only records MISSING judge_model are touched; everything else is byte-identical.
Each sidecar is rewritten atomically (temp file + os.replace).

IMPORTANT: stop the judge workers before running — they append to these files.
"""
import glob
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JUDGE_DIR = (HERE / ".." / "data" / "llm_judge").resolve()
LEGACY_MODEL = "openai/gpt-4o"


def backfill_file(path: Path) -> tuple[int, int]:
    """Returns (changed, total). Rewrites atomically only if something changed."""
    changed = total = 0
    tmp = path.with_suffix(".jsonl.bf_tmp")
    with open(path, "r", encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            total += 1
            try:
                rec = json.loads(s)
            except json.JSONDecodeError:
                fout.write(line if line.endswith("\n") else line + "\n")
                continue
            j = rec.get("is_qa_pair")
            if isinstance(j, dict) and "judge_model" not in j:
                j["judge_model"] = LEGACY_MODEL
                changed += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if changed:
        os.replace(tmp, path)
    else:
        os.remove(tmp)
    return changed, total


def main():
    files = sorted(glob.glob(str(JUDGE_DIR / "sub-*.jsonl")))
    if not files:
        sys.exit(f"no sidecars in {JUDGE_DIR}")
    total_changed = total_rows = 0
    for fp in files:
        changed, total = backfill_file(Path(fp))
        total_changed += changed
        total_rows += total
        if changed:
            print(f"  {os.path.basename(fp):40s} +{changed} backfilled / {total} rows")
    print(f"\nbackfilled {total_changed} legacy records "
          f"(judge_model={LEGACY_MODEL}) across {len(files)} files "
          f"({total_rows} total rows)")


if __name__ == "__main__":
    main()
