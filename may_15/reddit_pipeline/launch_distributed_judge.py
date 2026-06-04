"""Partition the is_qa_pair judging across a fleet of models and launch one
judge_qa_pairs.py worker per model, each restricted (--subs) to a disjoint set
of subreddits so workers never judge the same record twice.

Partitioning is greedy by *remaining* work (pairs minus already-judged) so each
model gets a roughly equal share of the outstanding judging.

Usage:
  python3 launch_distributed_judge.py            # compute, launch, print plan
  python3 launch_distributed_judge.py --dry-run  # just print the plan
"""
import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = (HERE / ".." / "data").resolve()
PAIRS_DIR = DATA / "extracted_current" / "pairs"
JUDGE_DIR = DATA / "llm_judge"
LOG_DIR = HERE

# fleet: (model, log-slug). All share OPENROUTER_API_KEY.
FLEET = [
    ("openai/gpt-4o", "gpt4o"),
    ("anthropic/claude-sonnet-4.6", "claude46"),
    ("deepseek/deepseek-v3.2", "deepseek"),
    ("mistralai/mistral-medium-3.1", "mistral"),
    ("google/gemma-3-27b-it", "gemma"),
]


def remaining_by_sub():
    pairs, judged = {}, {}
    for fp in glob.glob(str(PAIRS_DIR / "sub-*.jsonl")):
        sub = os.path.basename(fp)[4:-6]
        pairs[sub] = sum(1 for _ in open(fp))
    for fp in glob.glob(str(JUDGE_DIR / "sub-*.jsonl")):
        sub = os.path.basename(fp)[4:-6]
        judged[sub] = sum(1 for _ in open(fp))
    rem = {s: max(0, n - judged.get(s, 0)) for s, n in pairs.items()}
    return {s: r for s, r in rem.items() if r > 0}


def partition(rem, k):
    """Greedy longest-processing-time: assign biggest sub to lightest bucket."""
    buckets = [{"subs": [], "load": 0} for _ in range(k)]
    for sub, r in sorted(rem.items(), key=lambda kv: -kv[1]):
        b = min(buckets, key=lambda x: x["load"])
        b["subs"].append(sub)
        b["load"] += r
    return buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rem = remaining_by_sub()
    total = sum(rem.values())
    buckets = partition(rem, len(FLEET))

    print(f"remaining work: {total:,} pairs across {len(rem)} subs\n")
    procs = []
    for (model, slug), b in zip(FLEET, buckets):
        subs_csv = ",".join(b["subs"])
        print(f"=== {model}  (~{b['load']:,} pairs, {len(b['subs'])} subs) ===")
        print(f"    {subs_csv}\n")
        if args.dry_run:
            continue
        log = LOG_DIR / f"judge_dist_{slug}.log"
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        with open(log, "w") as lf:
            p = subprocess.Popen(
                ["python", "judge_qa_pairs.py", "--model", model, "--subs", subs_csv],
                cwd=str(HERE), stdout=lf, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True, env=env,
            )
        procs.append((model, p.pid, log.name))
    if procs:
        print("launched workers:")
        for model, pid, log in procs:
            print(f"  PID {pid:>8}  {model}  -> {log}")


if __name__ == "__main__":
    main()
