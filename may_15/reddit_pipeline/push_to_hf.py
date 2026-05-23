"""Push the extracted (query, preferred_answer) pairs to HuggingFace as a
private dataset repo at `dipikakhullar/personalization-reddit`.

Uploads `may_15/data/extracted_current/` under `extracted/` in the repo
(preserving the `pairs/` + `stats/` split) and writes a README at the repo
root documenting the source, schema, and heuristic.

Raw NDJSON dumps in `may_15/data/dumps/` are NOT uploaded — they stay on the
local box per the run plan.

Usage:
    python push_to_hf.py
"""

from __future__ import annotations

import re
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "dipikakhullar/personalization-reddit"
REPO_TYPE = "dataset"
PRIVATE = False

HERE = Path(__file__).resolve().parent
EXTRACTED_DIR = (HERE / ".." / "data" / "extracted_current").resolve()


# Everything below the YAML frontmatter — the YAML itself is generated.
SCHEMA_DOC = """\

# personalization-reddit

Per-subreddit `(query, preferred_answer)` pairs mined from Reddit using an
**OP-thanks-reply** heuristic: when the original poster (OP) replies to a
comment with thanks/gratitude, that parent comment is treated as their
preferred answer to their own question.

## Source

Raw post + comment dumps from the
[arctic_shift](https://github.com/ArthurHeitmann/arctic_shift) Pushshift
mirror, fetched per-subreddit (entire history through the fetch date) and
extracted with the pipeline in `may_15/reddit_pipeline/` of the
`personalization` repo.

Raw NDJSON dumps are kept locally and are not redistributed here.

## Splits

One split per subreddit; pick from the dropdown in the data viewer or load
programmatically:

```python
from datasets import load_dataset
ds = load_dataset("dipikakhullar/personalization-reddit", split="AskHistorians")
```

## Files

```
extracted/
  pairs/sub-<subreddit>.jsonl   # one (query, preferred_answer) per line
  stats/sub-<subreddit>.json    # funnel counts per subreddit
```

## Record schema (`pairs/sub-*.jsonl`)

| field | type | description |
|---|---|---|
| `user_id` | str | anonymized OP id (HMAC of Reddit username, see `anon.py`) |
| `timestamp` | str | post creation, ISO 8601 UTC |
| `subreddit` | str | source subreddit name |
| `query` | str | post title, with selftext appended if present |
| `preferred_answer` | str | body of the comment OP thanked (via parent of the thanks reply) |
| `top_comment` | str \\| null | body of the highest-scoring non-OP comment (may equal preferred) |
| `op_metadata` | object | OP user fields captured at post time; see below |
| `answerer_metadata` | object | answerer user fields captured at comment time; see below |
| `metadata` | object | see below |

`op_metadata` sub-object (mirrors the multiturn dataset):

| field | type | description |
|---|---|---|
| `user_id` | str | same as top-level `user_id` |
| `author_flair_text` | str \\| null | OP's flair text at post time |
| `author_flair_css_class` | str \\| null | flair css class |
| `author_flair_type` | str \\| null | e.g. "text", "richtext" |
| `author_flair_background_color` | str \\| null | hex color |
| `author_flair_text_color` | str \\| null | "light" / "dark" |

`answerer_metadata` sub-object:

| field | type | description |
|---|---|---|
| `user_id` | str | same as `metadata.answerer_anon_id` |
| `author_flair_text` | str \\| null | answerer's flair text at comment time |
| `author_flair_css_class` | str \\| null | flair css class |

`metadata` sub-object:

| field | type | description |
|---|---|---|
| `post_id` | str | Reddit submission id |
| `post_score` | int | submission score at fetch time |
| `answer_comment_id` | str | comment id of the preferred answer |
| `answer_score` | int | preferred-answer score at fetch time |
| `answerer_anon_id` | str | anonymized author id of the preferred answer |
| `top_comment_id` | str \\| null | comment id of the top-scoring non-OP comment |
| `top_comment_score` | int \\| null | top comment score |
| `top_comment_anon_id` | str \\| null | anonymized top-comment author id |
| `top_equals_preferred` | bool | whether the preferred answer is also the top comment |
| `thanks_reply_id` | str | OP's thanks-reply comment id (the signal that triggered the pair) |
| `thanks_reply_score` | int | score of OP's thanks reply |
| `thanks_reply_text` | str | body of OP's thanks reply |
| `thanks_reply_timestamp` | str | thanks reply creation, ISO 8601 UTC |

## Heuristic — OP thanks-reply

For each post that passes a question filter:
1. Find comments whose author == OP and whose body matches a "thanks"
   pattern (see `signals.py::is_thanks_reply`).
2. The parent of each such reply is recorded as a candidate "preferred answer".
3. Materialize each candidate into a pair, joining post metadata + answer body
   + thanks-reply context.

Bots and deleted/removed authors are filtered out before pair emission
(`signals.py`, `subreddits.py::BOT_AUTHORS`).

## Anonymization

Reddit usernames are hashed via HMAC-SHA256 with a per-run secret salt
(`anon.py::anon_user_id`) before being written. Post/comment ids and text
bodies are kept verbatim — content from public Reddit threads can still be
re-identified by searching the post id or quoting the body.

## Subreddits included in this snapshot

See `extracted/stats/` for the list and per-sub funnel counts
(`rs_records_scanned`, `keep_posts`, `thanks_refs`, `pairs_emitted`, etc.).
"""


_SPLIT_NAME_RE = re.compile(r"[^A-Za-z0-9_]")


def _split_name_for(sub: str) -> str:
    """HF split names must match [A-Za-z0-9_.-]+. Sanitize defensively."""
    return _SPLIT_NAME_RE.sub("_", sub)


def _build_readme(pair_files: list[Path]) -> str:
    """Compose YAML frontmatter declaring one split per subreddit."""
    lines = ["---", "configs:", "- config_name: default", "  data_files:"]
    for p in pair_files:
        name = p.stem  # sub-NAME
        if not name.startswith("sub-"):
            continue
        sub = name[len("sub-"):]
        split = _split_name_for(sub)
        lines.append(f"  - split: {split}")
        lines.append(f"    path: extracted/pairs/{p.name}")
    lines.append("---")
    return "\n".join(lines) + SCHEMA_DOC


def main() -> None:
    if not EXTRACTED_DIR.exists():
        raise SystemExit(f"missing extracted dir: {EXTRACTED_DIR}")

    pair_files = sorted((EXTRACTED_DIR / "pairs").glob("sub-*.jsonl"))
    stats_files = sorted((EXTRACTED_DIR / "stats").glob("sub-*.json"))
    print(f"will upload {len(pair_files)} pair files + {len(stats_files)} stats files")
    print(f"  pairs dir : {EXTRACTED_DIR / 'pairs'}")
    print(f"  stats dir : {EXTRACTED_DIR / 'stats'}")
    for p in pair_files:
        print(f"  - {p.name}")
    readme = _build_readme(pair_files)

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
            f"docs: README with per-subreddit splits ({len(pair_files)} splits)"
        ),
    )
    print("README.md uploaded")

    api.upload_folder(
        folder_path=str(EXTRACTED_DIR),
        path_in_repo="extracted",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        allow_patterns=["pairs/*.jsonl", "stats/*.json"],
        commit_message=(
            f"data: upload {len(pair_files)} sub pair files + {len(stats_files)} stats files"
        ),
    )
    print(f"uploaded folder → https://huggingface.co/datasets/{REPO_ID}")

    files = api.list_repo_files(REPO_ID, repo_type=REPO_TYPE)
    print(f"\nrepo now contains {len(files)} files:")
    for f in sorted(files):
        print(f"  {f}")


if __name__ == "__main__":
    main()
