"""Push the may_22 multi-turn conversations dataset to HuggingFace.

Uploads `may_22/data/conversations/sub-*.jsonl` plus `may_22/data/stats/sub-*.json`
to `dipikakhullar/personalization-reddit-multiturn`, with a README that
declares one split per subreddit so the HF data viewer surfaces a subreddit
selector.

Usage:
    python may_22/push_to_hf.py
"""

from __future__ import annotations

import re
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "dipikakhullar/personalization-reddit-multiturn"
REPO_TYPE = "dataset"
PRIVATE = False

HERE = Path(__file__).resolve().parent
DATA_DIR = (HERE / "data").resolve()


# Everything after the YAML frontmatter — the YAML itself is generated.
SCHEMA_DOC = """\

# personalization-reddit-multiturn

Multi-turn `(question, preferred_answer, full_conversation)` records mined
from Reddit. Companion to `dipikakhullar/personalization-reddit`: same
**OP-thanks-reply** heuristic for identifying the preferred answerer, but
this dataset additionally captures any contiguous back-and-forth between
the OP and that single answerer after the thanks.

A record is only emitted when there is at least one further turn beyond
the OP's thanks reply.

## Splits

One split per subreddit; pick from the dropdown in the data viewer or load
programmatically:

```python
from datasets import load_dataset
ds = load_dataset("dipikakhullar/personalization-reddit-multiturn",
                  split="LanguageTechnology")
```

## Source

Raw post + comment dumps from the
[arctic_shift](https://github.com/ArthurHeitmann/arctic_shift) Pushshift
mirror, fetched per-subreddit and extracted with
`may_22/extract_conversations.py` in the `personalization` repo.

Raw NDJSON dumps are kept locally and are not redistributed here.

## Files

```
conversations/sub-<subreddit>.jsonl    # one conversation per line
stats/sub-<subreddit>.json             # funnel counts per subreddit
```

## Record schema (`conversations/sub-*.jsonl`)

| field | type | description |
|---|---|---|
| `user_id` | str | anonymized OP id (joins with `personalization-reddit`) |
| `answerer_user_id` | str | anonymized id of the preferred answerer |
| `subreddit` | str | source subreddit name |
| `timestamp` | str | post creation, ISO 8601 UTC |
| `post_id` | str | Reddit submission id |
| `question` | str | post title, with selftext appended if present |
| `preferred_answer` | str | body of the comment OP thanked |
| `full_conversation` | list[turn] | ordered turns; see below |
| `n_turns` | int | length of `full_conversation` (≥ 4) |
| `n_turns_after_thanks` | int | turns past the OP-thanks reply (≥ 1) |
| `op_metadata` | object | OP user fields captured at post time; see below |
| `answerer_metadata` | object | answerer user fields captured at comment time; see below |
| `metadata` | object | anchor ids + contiguity flag; see below |

A `turn` has:

| field | type | description |
|---|---|---|
| `role` | str | `"OP"` or `"answerer"` |
| `user_id` | str | anonymized author id of this turn |
| `comment_id` | str | reddit comment id (or post id for turn 0) |
| `kind` | str | `"post"` or `"comment"` |
| `text` | str | full body |
| `timestamp` | str | ISO 8601 UTC |
| `score` | int | reddit score at fetch time |

Turn layout:
- Turn 0: OP's question post.
- Turn 1: answerer's preferred answer.
- Turn 2: OP's thanks reply.
- Turn 3+: alternating answerer/OP, walked depth-first by earliest
  `created_utc` at each branch.

### `op_metadata`

Whatever author-level fields appear on the OP's submission record. Most are
flair-related; many records have all fields null.

| field | type | description |
|---|---|---|
| `user_id` | str | anonymized OP id (same as top-level `user_id`) |
| `author_flair_text` | str \\| null | OP's flair text at post time |
| `author_flair_css_class` | str \\| null | flair css class |
| `author_flair_type` | str \\| null | e.g. "text", "richtext" |
| `author_flair_background_color` | str \\| null | hex color |
| `author_flair_text_color` | str \\| null | "light" / "dark" |

### `answerer_metadata`

Author-level fields from the preferred-answer comment record. Comment records
in the dumps carry fewer author fields than submissions.

| field | type | description |
|---|---|---|
| `user_id` | str | anonymized answerer id (same as top-level `answerer_user_id`) |
| `author_flair_text` | str \\| null | answerer's flair text at comment time |
| `author_flair_css_class` | str \\| null | flair css class |

### `metadata`

| field | type | description |
|---|---|---|
| `answer_comment_id` | str | id of the preferred answer comment |
| `thanks_reply_id` | str | id of OP's thanks reply (the anchor signal) |
| `post_score` | int | submission score at fetch time |
| `answer_score` | int | preferred-answer score at fetch time |
| `preferred_answer_is_top_level` | bool | True iff `A1.parent_id == post_id`; see Contiguity below |

## Contiguity

Turns 1 through N always form a strict parent→child chain in the reply tree
authored only by OP and the preferred answerer. **Turn 0 → Turn 1 is not
always direct**: in some records the OP thanked a nested comment rather than
a top-level reply, so third-party comments may sit between the question post
and A1 in the original thread. Filter on
`metadata.preferred_answer_is_top_level == true` to keep only records where
the entire chain is contiguous from the question down.

## Anonymization

Reddit usernames are hashed with the same salted-SHA256 scheme used in
`personalization-reddit`, so `user_id` / `answerer_user_id` join across the
two datasets when generated with the same `ANON_SALT`. Post/comment ids and
text bodies are kept verbatim — content from public Reddit threads can still
be re-identified by searching the post id or quoting the body.

## Heuristic

For each kept question post:
1. Find OP comments matching the same thanks-reply predicate used in
   `personalization-reddit` (`signals.py::is_thanks_reply`).
2. Treat the *parent* of each thanks reply as the preferred answer.
3. From the thanks reply, walk the comment tree: at each step take the
   earliest child by `created_utc` whose author is the expected next speaker
   (alternating answerer / OP). Stop when no such reply exists or the chain
   hits a bot, a deleted account, or an empty/`[deleted]`/`[removed]` body.
4. Emit a record only when this walk yielded at least one further turn.
"""


_SPLIT_NAME_RE = re.compile(r"[^A-Za-z0-9_]")


def _split_name_for(sub: str) -> str:
    """HF split names must match [A-Za-z0-9_.-]+. Sanitize defensively."""
    return _SPLIT_NAME_RE.sub("_", sub)


def _build_readme(conv_files: list[Path]) -> str:
    """Compose YAML frontmatter declaring one split per subreddit."""
    lines = ["---", "configs:", "- config_name: default", "  data_files:"]
    for p in conv_files:
        # filename pattern: sub-<NAME>.jsonl
        name = p.stem  # sub-NAME
        if not name.startswith("sub-"):
            continue
        sub = name[len("sub-"):]
        split = _split_name_for(sub)
        lines.append(f"  - split: {split}")
        lines.append(f"    path: conversations/{p.name}")
    lines.append("---")
    return "\n".join(lines) + SCHEMA_DOC


def main() -> None:
    if not DATA_DIR.exists():
        raise SystemExit(f"missing data dir: {DATA_DIR}")

    conv_files = sorted((DATA_DIR / "conversations").glob("sub-*.jsonl"))
    stats_files = sorted((DATA_DIR / "stats").glob("sub-*.json"))
    if not conv_files:
        raise SystemExit(f"no conversation files under {DATA_DIR / 'conversations'}")

    per_file_counts: list[tuple[str, int]] = []
    total_records = 0
    for p in conv_files:
        with open(p, "rb") as fh:
            n = sum(1 for line in fh if line.strip())
        per_file_counts.append((p.name, n))
        total_records += n
    print(f"will upload {len(conv_files)} conversation file(s), "
          f"{total_records} records, plus {len(stats_files)} stats file(s)")
    for name, n in per_file_counts:
        print(f"  - {name}: {n} records")

    readme = _build_readme(conv_files)

    api = HfApi()
    info = create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        private=PRIVATE,
        exist_ok=True,
    )
    print(f"repo ready: {info.url if hasattr(info, 'url') else info}")

    api.update_repo_settings(repo_id=REPO_ID, repo_type=REPO_TYPE, private=PRIVATE)
    print(f"visibility set to {'private' if PRIVATE else 'public'}")

    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        commit_message=(
            f"docs: README with per-subreddit splits ({len(conv_files)} splits)"
        ),
    )
    print("README.md uploaded")

    api.upload_folder(
        folder_path=str(DATA_DIR),
        path_in_repo=".",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        allow_patterns=["conversations/*.jsonl", "stats/*.json"],
        commit_message=(
            f"data: upload {len(conv_files)} conversation file(s) "
            f"({total_records} records) + {len(stats_files)} stats file(s)"
        ),
    )
    print(f"uploaded → https://huggingface.co/datasets/{REPO_ID}")

    files = api.list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    print(f"\nrepo now contains {len(files)} files:")
    for f in sorted(files):
        print(f"  {f}")


if __name__ == "__main__":
    main()
