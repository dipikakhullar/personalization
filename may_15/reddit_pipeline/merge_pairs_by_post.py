"""Collapse the per-answer pairs into ONE row per post, with no data loss.

Source pairs emit one (query, preferred_answer) row per OP-thanked answer, so a
single post can appear many times (same question, different thanked answers).
This rewrites them to one row per (subreddit, post_id):

  first_preferred  = the answer of the EARLIEST thanks-reply (tiebreak: higher
                     answer_score) -- the user's first positive acknowledgement.
  other_preferred  = the remaining thanked answers (same structure), ordered by
                     thanks-reply time. Nothing is discarded.

is_qa_pair verdicts (judged per answer_comment_id) are attached to each answer.

Output: data/pairs_by_post/sub-<sub>.jsonl
"""
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = (HERE / ".." / "data").resolve()
PAIRS_DIR = DATA / "extracted_current" / "pairs"
JUDGE_DIR = DATA / "llm_judge"
OUT_DIR = DATA / "pairs_by_post"


def load_qa_verdicts():
    v = {}
    for fp in glob.glob(str(JUDGE_DIR / "sub-*.jsonl")):
        for line in open(fp):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = d.get("is_qa_pair") or {}
            if "question_answer_pair" in q:
                v[(d.get("post_id"), d.get("answer_comment_id"))] = q
    return v


def answer_obj(rec, verdict):
    md = rec.get("metadata", {})
    aid = md.get("answer_comment_id")
    return {
        "preferred_answer": rec.get("preferred_answer"),
        "answer_comment_id": aid,
        "answer_score": md.get("answer_score"),
        "answerer_metadata": rec.get("answerer_metadata"),
        "thanks_reply_id": md.get("thanks_reply_id"),
        "thanks_reply_score": md.get("thanks_reply_score"),
        "thanks_reply_text": md.get("thanks_reply_text"),
        "thanks_reply_timestamp": md.get("thanks_reply_timestamp"),
        "is_qa_pair": verdict.get((md.get("post_id"), aid)),
    }


def sort_key(a):
    # earliest thanks first; missing timestamps last; tiebreak higher score
    ts = a.get("thanks_reply_timestamp") or "9999"
    return (ts, -(a.get("answer_score") or 0))


def merge_sub(fp, verdict):
    by_post = defaultdict(list)
    for line in open(fp):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        by_post[r["metadata"]["post_id"]].append(r)

    out_rows = []
    for pid, recs in by_post.items():
        answers = sorted((answer_obj(r, verdict) for r in recs), key=sort_key)
        r0 = recs[0]
        md = r0.get("metadata", {})
        out_rows.append({
            "user_id": r0.get("user_id"),
            "subreddit": r0.get("subreddit"),
            "post_id": pid,
            "timestamp": r0.get("timestamp"),
            "query": r0.get("query"),
            "op_metadata": r0.get("op_metadata"),
            "post_score": md.get("post_score"),
            "top_comment": r0.get("top_comment"),
            "top_comment_id": md.get("top_comment_id"),
            "top_comment_score": md.get("top_comment_score"),
            "top_comment_anon_id": md.get("top_comment_anon_id"),
            "n_preferred": len(answers),
            "first_preferred": answers[0],
            "other_preferred": answers[1:],
        })
    return out_rows, sum(len(v) for v in by_post.values())


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("loading qa verdicts ...", flush=True)
    verdict = load_qa_verdicts()
    print(f"  {len(verdict):,} verdicts", flush=True)

    tot_pairs = tot_posts = 0
    for fp in sorted(glob.glob(str(PAIRS_DIR / "sub-*.jsonl"))):
        sub = os.path.basename(fp)
        rows, n_pairs = merge_sub(fp, verdict)
        with open(OUT_DIR / sub, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tot_pairs += n_pairs
        tot_posts += len(rows)
        multi = sum(1 for r in rows if r["n_preferred"] > 1)
        print(f"  {sub:34s} {n_pairs:7d} pairs -> {len(rows):7d} posts "
              f"({multi} multi-answer)", flush=True)
    print(f"\nTOTAL: {tot_pairs:,} pairs -> {tot_posts:,} posts "
          f"({tot_pairs - tot_posts:,} collapsed into other_preferred, "
          f"{100*(tot_pairs-tot_posts)/tot_pairs:.1f}%)")


if __name__ == "__main__":
    main()
