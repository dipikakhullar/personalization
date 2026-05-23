# may_22/ — multi-turn OP↔preferred-answerer conversations

Companion dataset to `may_15/reddit_pipeline/`. Same subreddit dumps, same
OP-thanks-reply anchor, but extends each anchor into the contiguous
alternating back-and-forth between **only** the OP and the answerer they
positively acknowledged.

A record is only emitted when there is at least one further turn beyond the
OP's thanks reply — otherwise it would just duplicate the existing pairs
dataset at `may_15/data/extracted_current/pairs/sub-<SUB>.jsonl`.

## Usage

```bash
python may_22/extract_conversations.py \
    --sub gardening \
    --dumps-dir /workspace/personalization/may_15/data/dumps \
    --out-dir   /workspace/personalization/may_22/data/conversations
```

Output is written incrementally with per-record `flush()` + `fsync()`, so you
can inspect it while the job runs:

```bash
tail -f /workspace/personalization/may_22/data/conversations/sub-gardening.jsonl
```

Pass `--resume` to append (skipping `post_id`s already present). Without it
the output file is truncated on start.

Stats summary written to `…/stats/sub-<SUB>.json` at end of run.

## Anonymization

Uses `anon.anon_user_id` from `may_15/reddit_pipeline/` — so `user_id` and
`answerer_user_id` in this dataset cross-link with the existing pairs
dataset, provided the same `ANON_SALT` env var is set.

## Output schema (`sub-<SUB>.jsonl`, one JSON per line)

| field | type | notes |
| --- | --- | --- |
| `user_id` | str | anon id of OP |
| `answerer_user_id` | str | anon id of the preferred answerer |
| `subreddit` | str | raw subreddit name |
| `timestamp` | str | ISO8601, post created_utc |
| `post_id` | str | reddit submission id |
| `question` | str | `title\n\nselftext` |
| `preferred_answer` | str | body of the comment OP thanked |
| `full_conversation` | list[turn] | see below |
| `metadata` | object | counts + anchor ids |

A `turn` is:

```jsonc
{
  "role":       "OP" | "answerer",
  "user_id":    "anon_...",
  "comment_id": "abc",           // post_id for turn 0, comment_id otherwise
  "kind":       "post" | "comment",
  "text":       "...",
  "timestamp":  "2024-...",
  "score":      12
}
```

Turn layout:
- Turn 0: OP's question post.
- Turn 1: answerer's preferred answer (A1).
- Turn 2: OP's thanks reply.
- Turns 3+: alternating answerer / OP, walked depth-first by earliest
  `created_utc` at each branch, stopping when the next expected speaker has
  no reply or the chain breaks (bot/deleted/empty body).

**Contiguity:** turns 1 through N are guaranteed to form a strict parent→child
chain in the reply tree, authored only by OP and the preferred answerer.
**Turn 0 → Turn 1 is not always direct**: in some records the OP thanked a
nested comment rather than a top-level reply, so third-party comments may sit
between the question post and A1 in the original thread. The
`metadata.preferred_answer_is_top_level` boolean lets you filter to records
where the entire chain is contiguous from the question down.

## Algorithm

Two passes over the per-subreddit arctic_shift dump
(`may_15/data/dumps/sub-<SUB>/`):

1. **Submissions (`RS_*.zst`)** — keep questions posted by real users in the
   target subreddit.
2. **Comments (`RC_*.zst`)** — load every comment under those posts into
   memory; build a `parent_id → [child_id]` index sorted by `created_utc`.

Per post:
1. Find OP comments matching `signals.is_thanks_reply(...)`. Dedup by
   `parent_id` so each thanked comment yields at most one record.
2. For each anchor, validate the preferred answer (not deleted/bot, not OP,
   non-empty body).
3. Call `chain.walk_linear` starting at the OP-thanks reply and continuing as
   long as the *expected* next speaker (alternating answerer/OP) has a valid
   reply. Earliest by `created_utc` at any branch.
4. If `walk_linear` returned a non-empty tail, emit a record.

## Files

- `extract_conversations.py` — CLI + 2-pass driver + per-post emit loop.
- `chain.py` — `Comment` dataclass, `build_children_index`, `walk_linear`.
- `tests/test_chain.py` — synthetic tree fixtures for the walker.
