"""Aggregate per-month pair files into per-user temporally-ordered histories.

Input:   <extracted_dir>/pairs/YYYY-MM.jsonl   (output of extract.py)
Outputs:
  <out_dir>/users.shard-NNN.jsonl
      Each line: one user record:
        {"user_id": "...", "interactions": [pair, pair, ...]}
      Interactions are sorted ascending by timestamp.
  <out_dir>/subreddits.csv
      Columns: subreddit, n_pairs, n_users
  <out_dir>/aggregate_stats.json
      Top-line counts for the run.

Sharding: users are assigned to shards by `hash(user_id) % n_shards`. Default
N=16, tunable. This keeps any one shard small enough to stream into a
training pipeline without exhausting memory.

User filters: --min-pairs-per-user, --min-subreddits-per-user. Users below
either threshold are dropped from the final output but still counted in
subreddits.csv totals (so the inventory reflects all observed activity, not
just the kept users — useful for deciding allowlist changes).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import orjson


def _shard_for(user_id: str, n_shards: int) -> int:
    h = hashlib.sha1(user_id.encode()).digest()
    return int.from_bytes(h[:4], "big") % n_shards


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--n-shards", type=int, default=16)
    ap.add_argument("--min-pairs-per-user", type=int, default=3,
                    help="user must have at least N pairs total across all subs")
    ap.add_argument("--min-subreddits-per-user", type=int, default=1,
                    help="user must span at least N distinct subreddits")
    args = ap.parse_args()

    pairs_dir = args.extracted_dir / "pairs"
    if not pairs_dir.is_dir():
        raise SystemExit(f"no pairs dir at {pairs_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # First pass: load every pair, group by user.
    by_user: dict[str, list[dict]] = defaultdict(list)
    sub_pair_counts: Counter[str] = Counter()
    sub_user_sets: dict[str, set[str]] = defaultdict(set)
    total_pairs = 0

    for jsonl in sorted(pairs_dir.glob("*.jsonl")):
        with open(jsonl, "rb") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = orjson.loads(line)
                by_user[rec["user_id"]].append(rec)
                sub_pair_counts[rec["subreddit"]] += 1
                sub_user_sets[rec["subreddit"]].add(rec["user_id"])
                total_pairs += 1

    # Subreddit inventory — written from the full distribution, before
    # per-user filtering, so the index reflects what's actually in the data.
    inv_path = args.out_dir / "subreddits.csv"
    with open(inv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subreddit", "n_pairs", "n_users"])
        for sub, n in sub_pair_counts.most_common():
            w.writerow([sub, n, len(sub_user_sets[sub])])

    # Filter users, sort interactions, shard out.
    shard_paths = [args.out_dir / f"users.shard-{i:03d}.jsonl" for i in range(args.n_shards)]
    shard_fhs = [open(p, "wb") for p in shard_paths]

    kept_users = 0
    kept_pairs = 0
    try:
        for user_id, pairs in by_user.items():
            if len(pairs) < args.min_pairs_per_user:
                continue
            subs = {p["subreddit"] for p in pairs}
            if len(subs) < args.min_subreddits_per_user:
                continue
            pairs.sort(key=lambda p: p["timestamp"])
            # Drop redundant user_id key from per-interaction dicts so it's
            # only present at the user level.
            interactions = [
                {k: v for k, v in p.items() if k != "user_id"} for p in pairs
            ]
            rec = {"user_id": user_id, "interactions": interactions}
            shard = _shard_for(user_id, args.n_shards)
            shard_fhs[shard].write(orjson.dumps(rec))
            shard_fhs[shard].write(b"\n")
            kept_users += 1
            kept_pairs += len(interactions)
    finally:
        for fh in shard_fhs:
            fh.close()

    stats = {
        "total_pairs": total_pairs,
        "total_users": len(by_user),
        "kept_users": kept_users,
        "kept_pairs": kept_pairs,
        "min_pairs_per_user": args.min_pairs_per_user,
        "min_subreddits_per_user": args.min_subreddits_per_user,
        "n_shards": args.n_shards,
        "n_subreddits": len(sub_pair_counts),
    }
    with open(args.out_dir / "aggregate_stats.json", "wb") as fh:
        fh.write(orjson.dumps(stats, option=orjson.OPT_INDENT_2))
    print(orjson.dumps(stats, option=orjson.OPT_INDENT_2).decode())


if __name__ == "__main__":
    main()
