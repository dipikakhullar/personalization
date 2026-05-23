# Reddit user-level (query, preferred_answer) pipeline

Builds temporally-ordered user histories from Reddit dumps. A "preferred answer"
is a comment that the post's author replied to with thanks-shaped language —
the highest-precision personalization signal Reddit exposes natively.

## Data fetch (manual)

arctic_shift hosts monthly Pushshift-style dumps on Academic Torrents:
- Index: https://github.com/ArthurHeitmann/arctic_shift/blob/master/download_links.md
- Coverage: 2005-06 through 2026-04, monthly
- Format: `.zst` (recompressed) — preferred — or `.zst_blocks` (original)

For each month you want, download two files into `data/dumps/<YYYY-MM>/`:
- `RS_<YYYY-MM>.zst`  — submissions (posts)
- `RC_<YYYY-MM>.zst`  — comments

The pipeline streams these in place; no full decompression to disk is needed.

## Run

```bash
pip install -r requirements.txt
cd reddit_pipeline

# Single month, default allowlist:
python extract.py --month 2023-06 --dumps-dir ../data/dumps --out-dir ../data/extracted

# All months present in dumps-dir:
python extract.py --all --dumps-dir ../data/dumps --out-dir ../data/extracted

# Aggregate per-month pair files into user-history shards + subreddit inventory:
python aggregate.py \
    --extracted-dir ../data/extracted \
    --out-dir ../data/users \
    --min-pairs-per-user 5 \
    --min-subreddits-per-user 2
```

## Output schema

`data/users/users.shard-NNN.jsonl` — one line per user:

```json
{
  "user_id": "anon_<sha256-16>",
  "interactions": [
    {"timestamp": "...", "subreddit": "...", "query": "...", "preferred_answer": "...",
     "metadata": {"post_id": "...", "answer_comment_id": "...", "answer_score": 42,
                  "thanks_reply_id": "...", "thanks_reply_text": "..."}}
  ]
}
```

`data/users/subreddits.csv` — `subreddit,n_pairs,n_users`.

User IDs are SHA-256 of the original Reddit username (first 16 hex chars), so
the output never contains raw usernames.
