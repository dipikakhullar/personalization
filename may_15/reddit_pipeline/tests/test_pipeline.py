"""End-to-end pipeline test on a tiny synthetic .zst fixture.

Builds a one-month synthetic dump exercising every filtering branch:

  Post A (Cooking, alice, question):
    - bob answers "add butter"                                 ← KEEP
    - alice replies "Thanks, this worked!" to bob              ← thanks-trigger
    - carl answers something else
    - alice replies "no thanks" to carl                        ← negated, skip

  Post B (AskHistorians, dave, question):
    - eve answers in detail
    - dave replies "perfect, exactly what I needed" to eve     ← KEEP
    - AutoModerator posts a sticky                              ← bot, skip if it were target
    - dave thanks AutoModerator                                 ← parent is bot, skip

  Post C (memes, frank, question):                              ← off-allowlist, skip whole thread
    - grace answers
    - frank thanks grace

  Post D (Cooking, helen, statement not question):              ← non-question, skip
    - ivan replies
    - helen thanks ivan

  Post E (Cooking, [deleted], question):                        ← deleted OP, skip

  Post F (Cooking, jane, question):
    - jane self-answers                                          ← OP=answerer, skip
    - jane thanks self

Expected pairs: 2 (alice→bob, dave→eve).
Expected users in shard output, with default min-pairs=5, min-subs=2: 0
(neither alice nor dave has enough). We assert at the pair-extract level
instead of the aggregate level, and separately run aggregate with
min-pairs=1, min-subs=1 to verify it shards correctly.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import orjson
import zstandard as zstd


HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent


def _write_zst(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cctx = zstd.ZstdCompressor(level=3)
    with open(path, "wb") as fh:
        with cctx.stream_writer(fh) as w:
            for r in records:
                w.write(orjson.dumps(r))
                w.write(b"\n")


def _build_fixture(dumps_dir: Path) -> None:
    month = "2024-06"
    rs = dumps_dir / month / f"RS_{month}.zst"
    rc = dumps_dir / month / f"RC_{month}.zst"

    base_t = 1_700_000_000

    submissions = [
        {"id": "pA", "author": "alice",   "subreddit": "Cooking",
         "title": "How do I make pasta sauce less acidic?", "selftext": "",
         "created_utc": base_t, "score": 12},
        {"id": "pB", "author": "dave",    "subreddit": "AskHistorians",
         "title": "Why did the Library of Alexandria decline?", "selftext": "Context here.",
         "created_utc": base_t + 100, "score": 50},
        {"id": "pC", "author": "frank",   "subreddit": "memes",
         "title": "Anyone else feel this?", "selftext": "",
         "created_utc": base_t + 200, "score": 5},
        {"id": "pD", "author": "helen",   "subreddit": "Cooking",
         "title": "I made a great risotto today.", "selftext": "",
         "created_utc": base_t + 300, "score": 8},
        {"id": "pE", "author": "[deleted]", "subreddit": "Cooking",
         "title": "How do I clean a cast iron pan?", "selftext": "",
         "created_utc": base_t + 400, "score": 3},
        {"id": "pF", "author": "jane",    "subreddit": "Cooking",
         "title": "What's the best knife for vegetables?", "selftext": "",
         "created_utc": base_t + 500, "score": 7},
    ]
    _write_zst(submissions, rs)

    comments = [
        # Post A (Cooking, alice)
        {"id": "cA1", "author": "bob",   "link_id": "t3_pA", "parent_id": "t3_pA",
         "body": "Add a small amount of butter or baking soda to balance acidity.",
         "subreddit": "Cooking", "created_utc": base_t + 5, "score": 20},
        {"id": "cA2", "author": "alice", "link_id": "t3_pA", "parent_id": "t1_cA1",
         "body": "Thanks, this worked!",
         "subreddit": "Cooking", "created_utc": base_t + 10, "score": 3},
        {"id": "cA3", "author": "carl",  "link_id": "t3_pA", "parent_id": "t3_pA",
         "body": "Just throw it out and start over.",
         "subreddit": "Cooking", "created_utc": base_t + 15, "score": 1},
        {"id": "cA4", "author": "alice", "link_id": "t3_pA", "parent_id": "t1_cA3",
         "body": "no thanks, that's wasteful.",
         "subreddit": "Cooking", "created_utc": base_t + 20, "score": 2},

        # Post B (AskHistorians, dave)
        {"id": "cB1", "author": "eve",   "link_id": "t3_pB", "parent_id": "t3_pB",
         "body": "It declined gradually over centuries due to fires, neglect, and political upheaval.",
         "subreddit": "AskHistorians", "created_utc": base_t + 105, "score": 80},
        {"id": "cB2", "author": "dave",  "link_id": "t3_pB", "parent_id": "t1_cB1",
         "body": "Perfect, exactly what I needed.",
         "subreddit": "AskHistorians", "created_utc": base_t + 110, "score": 5},
        {"id": "cB3", "author": "AutoModerator", "link_id": "t3_pB", "parent_id": "t3_pB",
         "body": "I am a bot. Please read the rules. beep boop",
         "subreddit": "AskHistorians", "created_utc": base_t + 106, "score": 1},
        {"id": "cB4", "author": "dave",  "link_id": "t3_pB", "parent_id": "t1_cB3",
         "body": "thanks bot",
         "subreddit": "AskHistorians", "created_utc": base_t + 107, "score": 0},

        # Post C (off-allowlist sub) — should never enter keep_posts
        {"id": "cC1", "author": "grace", "link_id": "t3_pC", "parent_id": "t3_pC",
         "body": "Yes, all the time.",
         "subreddit": "memes", "created_utc": base_t + 205, "score": 4},
        {"id": "cC2", "author": "frank", "link_id": "t3_pC", "parent_id": "t1_cC1",
         "body": "Thanks!",
         "subreddit": "memes", "created_utc": base_t + 210, "score": 1},

        # Post D (non-question) — should never enter keep_posts
        {"id": "cD1", "author": "ivan",  "link_id": "t3_pD", "parent_id": "t3_pD",
         "body": "Nice, share the recipe?",
         "subreddit": "Cooking", "created_utc": base_t + 305, "score": 2},
        {"id": "cD2", "author": "helen", "link_id": "t3_pD", "parent_id": "t1_cD1",
         "body": "Thanks, will do!",
         "subreddit": "Cooking", "created_utc": base_t + 310, "score": 1},

        # Post F (OP=answerer) — should be filtered
        {"id": "cF1", "author": "jane",  "link_id": "t3_pF", "parent_id": "t3_pF",
         "body": "Figured it out — a Santoku works great.",
         "subreddit": "Cooking", "created_utc": base_t + 505, "score": 1},
        {"id": "cF2", "author": "jane",  "link_id": "t3_pF", "parent_id": "t1_cF1",
         "body": "Thanks past-me!",
         "subreddit": "Cooking", "created_utc": base_t + 510, "score": 1},
    ]
    _write_zst(comments, rc)


def _run(cmd: list[str], cwd: Path) -> None:
    print(">", " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        raise AssertionError(f"command failed: {cmd}")


def main() -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        dumps = tmp / "dumps"
        extracted = tmp / "extracted"
        users = tmp / "users"

        _build_fixture(dumps)

        _run(
            [sys.executable, "extract.py", "--month", "2024-06",
             "--dumps-dir", str(dumps), "--out-dir", str(extracted)],
            cwd=PIPELINE,
        )

        pair_file = extracted / "pairs" / "2024-06.jsonl"
        pairs = [orjson.loads(l) for l in pair_file.read_bytes().splitlines() if l.strip()]
        print(f"got {len(pairs)} pairs")
        for p in pairs:
            print(" ", p["subreddit"], "|", p["query"][:60], "→", p["preferred_answer"][:60])

        # Assertions
        assert len(pairs) == 2, f"expected 2 pairs, got {len(pairs)}"

        subs = {p["subreddit"] for p in pairs}
        assert subs == {"Cooking", "AskHistorians"}, f"unexpected subs: {subs}"

        # Find the alice→bob pair
        cooking = next(p for p in pairs if p["subreddit"] == "Cooking")
        assert cooking["query"].startswith("How do I make pasta sauce")
        assert "butter" in cooking["preferred_answer"]
        assert cooking["metadata"]["post_id"] == "pA"
        assert cooking["metadata"]["answer_comment_id"] == "cA1"
        assert cooking["user_id"].startswith("anon_"), "user_id not anonymized"
        assert "alice" not in json.dumps(cooking), "raw username leaked"
        assert "bob" not in json.dumps(cooking), "raw answerer leaked"

        hist = next(p for p in pairs if p["subreddit"] == "AskHistorians")
        assert hist["metadata"]["answer_comment_id"] == "cB1"
        assert "Alexandria" in hist["query"]
        assert "dave" not in json.dumps(hist)
        assert "eve" not in json.dumps(hist)

        # Aggregate with permissive thresholds so we get a shard out
        _run(
            [sys.executable, "aggregate.py",
             "--extracted-dir", str(extracted),
             "--out-dir", str(users),
             "--min-pairs-per-user", "1",
             "--min-subreddits-per-user", "1",
             "--n-shards", "4"],
            cwd=PIPELINE,
        )

        shards = sorted(users.glob("users.shard-*.jsonl"))
        assert shards, "no shard files written"
        all_users = []
        for s in shards:
            for line in s.read_bytes().splitlines():
                if line.strip():
                    all_users.append(orjson.loads(line))
        print(f"got {len(all_users)} users across shards")
        assert len(all_users) == 2, f"expected 2 users, got {len(all_users)}"

        # Interactions are sorted ascending; with one pair each, trivially true
        for u in all_users:
            ts = [i["timestamp"] for i in u["interactions"]]
            assert ts == sorted(ts), f"interactions not time-sorted: {ts}"

        # subreddits.csv inventory
        inv = (users / "subreddits.csv").read_text().splitlines()
        assert inv[0] == "subreddit,n_pairs,n_users"
        rows = [r.split(",") for r in inv[1:]]
        sub_to_count = {r[0]: int(r[1]) for r in rows}
        assert sub_to_count == {"Cooking": 1, "AskHistorians": 1}, sub_to_count

        # aggregate_stats.json
        stats = orjson.loads((users / "aggregate_stats.json").read_bytes())
        assert stats["total_pairs"] == 2
        assert stats["kept_users"] == 2
        assert stats["n_subreddits"] == 2

        # Per-month extract stats
        em_stats = orjson.loads((extracted / "stats" / "2024-06.json").read_bytes())
        assert em_stats["keep_posts"] == 3, f"keep_posts should be 3 (pA, pB, pF): {em_stats}"
        assert em_stats["pairs_emitted"] == 2

        print("\nALL ASSERTIONS PASSED")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
