"""Build the ~50-sample dataset for the june_2 personalization experiment.

For each sample we need a user with enough history:
  - history = the user's 3 chronologically-earliest (query, preferred_answer) pairs
  - target  = a later pair by the SAME user where top_comment != preferred_answer
              (so the "preferred vs top" comparison is meaningful)

Output: june_2/data/samples.jsonl
"""
import argparse
import glob
import json
import random
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAIRS_DIR = (HERE / ".." / "may_15" / "data" / "extracted_current" / "pairs").resolve()
JUDGE_DIR = (HERE / ".." / "may_15" / "data" / "llm_judge").resolve()
OUT_PATH = HERE / "data" / "samples.jsonl"

HISTORY_SIZE = 3
MIN_LEN = 40          # min chars for query / answers (skip trivially short text)
MAX_ANSWER_CHARS = 4000


def load_qa_verdicts():
    """(post_id, answer_comment_id) -> is_qa_pair bool, from the LLM-judge sidecars."""
    verdict = {}
    for fp in glob.glob(str(JUDGE_DIR / "sub-*.jsonl")):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q = d.get("is_qa_pair") or {}
                if "question_answer_pair" in q:
                    verdict[(d.get("post_id"), d.get("answer_comment_id"))] = q[
                        "question_answer_pair"
                    ]
    return verdict


def load_user_pairs():
    """user_id -> list of pair dicts (only fields we need), sorted by timestamp."""
    by_user = defaultdict(list)
    for fp in sorted(glob.glob(str(PAIRS_DIR / "sub-*.jsonl"))):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                md = d.get("metadata", {})
                by_user[d["user_id"]].append(
                    {
                        "timestamp": d.get("timestamp", ""),
                        "subreddit": d.get("subreddit", ""),
                        "query": (d.get("query") or "").strip(),
                        "preferred_answer": (d.get("preferred_answer") or "").strip(),
                        "top_comment": (d.get("top_comment") or "").strip(),
                        "top_equals_preferred": md.get("top_equals_preferred"),
                        "preferred_score": md.get("answer_score"),
                        "top_score": md.get("top_comment_score"),
                        # join keys for the is_qa_pair LLM-judge sidecars
                        "post_id": md.get("post_id"),
                        "answer_comment_id": md.get("answer_comment_id"),
                    }
                )
    for u in by_user:
        by_user[u].sort(key=lambda p: p["timestamp"])
    return by_user


def usable_text(*texts):
    return all(t and len(t) >= MIN_LEN for t in texts)


def build(n_samples: int, seed: int, qa_true_only: bool, out_path: Path):
    rng = random.Random(seed)
    by_user = load_user_pairs()

    verdict = load_qa_verdicts() if qa_true_only else None
    if qa_true_only:
        print(f"loaded {len(verdict):,} qa verdicts "
              f"({sum(verdict.values()):,} qa_true)")

    def target_ok(p):
        if p["top_equals_preferred"] is not False:
            return False
        if not usable_text(p["query"], p["preferred_answer"], p["top_comment"]):
            return False
        if qa_true_only and verdict.get((p["post_id"], p["answer_comment_id"])) is not True:
            return False
        return True

    # Candidate users: history must be HISTORY_SIZE *distinct posts* (distinct
    # questions), and the target must be a *different post* again. Without the
    # post_id dedup, users whose pairs all come from one popular thread (several
    # OP-thanked answers) produce a degenerate history (same question repeated)
    # and target-in-history leakage.
    candidates = []
    for uid, pairs in by_user.items():
        # pairs are timestamp-sorted; walk them collecting distinct posts.
        history, hist_posts = [], set()
        rest = []
        for p in pairs:
            if len(history) < HISTORY_SIZE:
                if p["post_id"] in hist_posts:
                    continue  # same post already represented in history
                if not usable_text(p["query"], p["preferred_answer"]):
                    continue
                history.append(p)
                hist_posts.add(p["post_id"])
            else:
                rest.append(p)
        if len(history) < HISTORY_SIZE:
            continue
        # target must be a post NOT in the history set
        targets = [p for p in rest if p["post_id"] not in hist_posts and target_ok(p)]
        if targets:
            candidates.append((uid, history, targets))

    print(f"eligible users: {len(candidates):,}")
    rng.shuffle(candidates)

    samples = []
    for uid, history, targets in candidates[:n_samples]:
        target = rng.choice(targets)
        samples.append(
            {
                "sample_id": len(samples),
                "user_id": uid,
                "subreddit": target["subreddit"],
                "history": [
                    {"query": h["query"], "preferred_answer": h["preferred_answer"]}
                    for h in history
                ],
                "query": target["query"],
                "preferred_answer": target["preferred_answer"][:MAX_ANSWER_CHARS],
                "top_comment": target["top_comment"][:MAX_ANSWER_CHARS],
                "preferred_score": target["preferred_score"],
                "top_score": target["top_score"],
                "post_id": target["post_id"],
                "answer_comment_id": target["answer_comment_id"],
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"wrote {len(samples)} samples -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="number of samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--qa-true-only", action="store_true",
                    help="only keep targets the LLM-judge marked is_qa_pair=True")
    ap.add_argument("--out", default=str(OUT_PATH), help="output jsonl path")
    args = ap.parse_args()
    build(args.n, args.seed, args.qa_true_only, Path(args.out))
