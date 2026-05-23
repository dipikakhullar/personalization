"""Extract contiguous multi-turn OP↔preferred-answerer conversations.

Companion to may_15/reddit_pipeline/extract.py. Uses the same OP-thanks-reply
signal to identify (OP, preferred_answerer) anchors and then walks the comment
tree to capture any further alternating dialogue between *only* those two
users. Writes one JSON record per anchor (when a tail exists) to
<out-dir>/sub-<SUB>.jsonl with per-record flush+fsync so the file can be
inspected mid-run.

CLI:
  python may_22/extract_conversations.py \\
      --sub gardening \\
      --dumps-dir /workspace/personalization/may_15/data/dumps \\
      --out-dir   /workspace/personalization/may_22/data/conversations \\
      [--resume]

Output schema: see may_22/README.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import orjson

# Pull in the existing pipeline modules — same signals, same anon, same reader.
HERE = Path(__file__).resolve().parent
_PIPE = HERE.parent / "may_15" / "reddit_pipeline"
if str(_PIPE) not in sys.path:
    sys.path.insert(0, str(_PIPE))

from anon import anon_user_id  # noqa: E402
from dump_reader import (  # noqa: E402
    DumpReadStats,
    comment_path,
    discover_months,
    iter_dump,
    submission_path,
)
from signals import is_question_post, is_thanks_reply, looks_like_bot_body  # noqa: E402
from subreddits import is_bot_author, is_deleted_author  # noqa: E402

from chain import Comment, build_children_index, walk_linear  # noqa: E402


# Author-level fields preserved from the raw RS/RC records. These describe
# the user's *state at the time the post or comment was made* (flair etc.),
# not a per-user profile. Listed here so all records have the same key set
# even when most values are null.
_OP_FIELDS = (
    "author_flair_text",
    "author_flair_css_class",
    "author_flair_type",
    "author_flair_background_color",
    "author_flair_text_color",
)
_ANSWERER_FIELDS = (
    "author_flair_text",
    "author_flair_css_class",
)


# ── Local helpers ──────────────────────────────────────────────────────


def _strip_kind(fullname: str | None) -> str | None:
    if not fullname:
        return None
    if "_" in fullname and len(fullname) > 3 and fullname[2] == "_":
        return fullname.split("_", 1)[1]
    return fullname


def _isoformat(epoch: int | str) -> str:
    return dt.datetime.fromtimestamp(int(epoch), tz=dt.timezone.utc).isoformat()


def _ok_text(s: str | None) -> bool:
    if s is None:
        return False
    s = s.strip()
    if not s:
        return False
    return s not in ("[deleted]", "[removed]")


def _valid_chain_author(author: str | None) -> bool:
    """Used by walk_linear: stop the chain when an author is bot/deleted."""
    if not author:
        return False
    if is_deleted_author(author) or is_bot_author(author):
        return False
    return True


@dataclass(slots=True)
class PostMeta:
    post_id: str
    op_author: str
    title: str
    selftext: str
    subreddit: str
    created_utc: int
    score: int
    author_fields: dict  # {field_name: value} for whitelisted author_* fields


# ── Pass 1: submissions ────────────────────────────────────────────────


def pass1_submissions(
    path: Path, target_sub_lc: str, stats: DumpReadStats
) -> dict[str, PostMeta]:
    keep: dict[str, PostMeta] = {}
    for rec in iter_dump(path, stats):
        sub = rec.get("subreddit")
        if not sub or sub.lower() != target_sub_lc:
            continue
        author = rec.get("author")
        if is_deleted_author(author) or is_bot_author(author):
            continue
        title = rec.get("title") or ""
        selftext = rec.get("selftext") or ""
        if not is_question_post(title, selftext):
            continue
        if (rec.get("_meta") or {}).get("removal_type"):
            continue
        pid = rec.get("id")
        if not pid:
            continue
        keep[pid] = PostMeta(
            post_id=pid,
            op_author=author,
            title=title,
            selftext=selftext if _ok_text(selftext) else "",
            subreddit=sub,
            created_utc=int(rec.get("created_utc") or 0),
            score=int(rec.get("score") or 0),
            author_fields={k: rec.get(k) for k in _OP_FIELDS},
        )
    return keep


# ── Pass 2: load all comments under kept posts ─────────────────────────


def pass2_load_comments(
    path: Path, keep_posts: dict[str, PostMeta], stats: DumpReadStats
) -> dict[str, dict[str, Comment]]:
    """Return comments_by_post[post_id][comment_id] = Comment.

    Holding all comments-under-kept-posts in memory is fine for a single sub
    (millions of comments at most), and it lets us tree-walk freely.
    """
    by_post: dict[str, dict[str, Comment]] = {}
    for rec in iter_dump(path, stats):
        link_id = _strip_kind(rec.get("link_id"))
        if not link_id or link_id not in keep_posts:
            continue
        cid = rec.get("id")
        if not cid:
            continue
        parent_id = _strip_kind(rec.get("parent_id")) or link_id
        author = rec.get("author") or ""
        body = rec.get("body") or ""
        by_post.setdefault(link_id, {})[cid] = Comment(
            id=cid,
            parent_id=parent_id,
            author=author,
            body=body,
            created_utc=int(rec.get("created_utc") or 0),
            score=int(rec.get("score") or 0),
            author_fields={k: rec.get(k) for k in _ANSWERER_FIELDS},
        )
    return by_post


# ── Per-post: find anchors and emit conversations ──────────────────────


def _build_record(
    post: PostMeta,
    answerer_author: str,
    answer_cid: str,
    thanks_cid: str,
    tail_cids: list[str],
    comments: dict[str, Comment],
) -> dict:
    op_anon = anon_user_id(post.op_author)
    ans_anon = anon_user_id(answerer_author)

    query = post.title.strip()
    if post.selftext:
        query = f"{query}\n\n{post.selftext.strip()}"

    turns: list[dict] = []
    # Turn 0: the question post itself.
    turns.append({
        "role": "OP",
        "user_id": op_anon,
        "comment_id": post.post_id,
        "kind": "post",
        "text": query,
        "timestamp": _isoformat(post.created_utc),
        "score": post.score,
    })

    def _turn(cid: str, role: str, user_id: str) -> dict:
        c = comments[cid]
        return {
            "role": role,
            "user_id": user_id,
            "comment_id": c.id,
            "kind": "comment",
            "text": c.body.strip(),
            "timestamp": _isoformat(c.created_utc),
            "score": c.score,
        }

    # Turn 1: A1 (the preferred answer).
    turns.append(_turn(answer_cid, "answerer", ans_anon))
    # Turn 2: OP's thanks reply.
    turns.append(_turn(thanks_cid, "OP", op_anon))
    # Tail: alternates answerer, OP, answerer, OP, ...
    role = "answerer"
    for cid in tail_cids:
        turns.append(_turn(cid, role, ans_anon if role == "answerer" else op_anon))
        role = "OP" if role == "answerer" else "answerer"

    ans_comment = comments[answer_cid]

    op_metadata = {"user_id": op_anon}
    op_metadata.update(post.author_fields)

    answerer_metadata = {"user_id": ans_anon}
    answerer_metadata.update(ans_comment.author_fields or {})

    return {
        "user_id": op_anon,
        "answerer_user_id": ans_anon,
        "subreddit": post.subreddit,
        "timestamp": _isoformat(post.created_utc),
        "post_id": post.post_id,
        "question": query,
        "preferred_answer": ans_comment.body.strip(),
        "full_conversation": turns,
        "n_turns": len(turns),
        "n_turns_after_thanks": len(tail_cids),
        "op_metadata": op_metadata,
        "answerer_metadata": answerer_metadata,
        "metadata": {
            "answer_comment_id": answer_cid,
            "thanks_reply_id": thanks_cid,
            "post_score": post.score,
            "answer_score": ans_comment.score,
            # True iff the preferred answer is a direct top-level reply to the
            # post. When False, third-party comments sit between turn 0 (the
            # question post) and turn 1 (the preferred answer) in the reply
            # tree. Turns 1..N are always strictly contiguous OP↔answerer.
            "preferred_answer_is_top_level": ans_comment.parent_id == post.post_id,
        },
    }


def find_and_emit(
    post: PostMeta,
    comments: dict[str, Comment],
    already_emitted: set[str],
    out_fh,
) -> tuple[int, int, int]:
    """For one post: find anchors, walk chains, emit records.

    Returns (n_anchors, n_emitted, n_dropped_no_tail).
    """
    n_anchors = 0
    n_emitted = 0
    n_dropped = 0

    # Find OP thanks-replies; dedup by parent_comment_id (first wins,
    # matching extract.py:303-305 semantics).
    thanks_by_parent: dict[str, str] = {}  # parent_cid -> thanks_reply_cid
    op_author = post.op_author
    for cid, c in comments.items():
        if c.author != op_author:
            continue
        if not is_thanks_reply(c.body):
            continue
        if c.parent_id == post.post_id:
            # OP "thanking" the post itself — not a comment anchor.
            continue
        if c.parent_id in thanks_by_parent:
            # Already have an earlier thanks to this parent; ignore.
            continue
        thanks_by_parent[c.parent_id] = cid

    if not thanks_by_parent:
        return (0, 0, 0)

    children = build_children_index(comments)

    for parent_cid, thanks_cid in thanks_by_parent.items():
        n_anchors += 1
        ans = comments.get(parent_cid)
        if ans is None:
            continue
        if not _valid_chain_author(ans.author):
            continue
        if ans.author == op_author:
            continue
        if not _ok_text(ans.body) or looks_like_bot_body(ans.body):
            continue

        # Walk past the thanks reply, expecting the answerer to speak next.
        tail = walk_linear(
            start_id=thanks_cid,
            next_author=ans.author,
            op_author=op_author,
            answerer_author=ans.author,
            comments=comments,
            children_by_parent=children,
            is_valid_author=_valid_chain_author,
        )
        if not tail:
            n_dropped += 1
            continue

        # Drop a turn if its body would render as empty after stripping.
        # walk_linear already guards against [deleted]/[removed]/empty, but
        # also filter bot-body just in case.
        truncated_tail: list[str] = []
        for cid in tail:
            body = comments[cid].body or ""
            if looks_like_bot_body(body):
                break
            truncated_tail.append(cid)
        if not truncated_tail:
            n_dropped += 1
            continue

        if post.post_id in already_emitted:
            # On --resume we may visit the same post twice; skip duplicate.
            continue

        rec = _build_record(
            post, ans.author, parent_cid, thanks_cid, truncated_tail, comments,
        )
        out_fh.write(orjson.dumps(rec))
        out_fh.write(b"\n")
        out_fh.flush()
        try:
            os.fsync(out_fh.fileno())
        except OSError:
            pass
        already_emitted.add(post.post_id)
        n_emitted += 1

    return (n_anchors, n_emitted, n_dropped)


# ── Driver ─────────────────────────────────────────────────────────────


def _load_already_emitted(path: Path) -> set[str]:
    """For --resume: collect post_ids already written to the output file."""
    out: set[str] = set()
    if not path.exists():
        return out
    with open(path, "rb") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = orjson.loads(line)
            except orjson.JSONDecodeError:
                continue
            pid = rec.get("post_id")
            if pid:
                out.add(pid)
    return out


def run_sub(sub: str, dumps_dir: Path, out_dir: Path, resume: bool) -> dict:
    batch = f"sub-{sub}"
    rs = submission_path(dumps_dir, batch)
    rc = comment_path(dumps_dir, batch)

    out_dir.mkdir(parents=True, exist_ok=True)
    conv_dir = out_dir
    stats_dir = out_dir.parent / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)

    out_path = conv_dir / f"sub-{sub}.jsonl"
    stats_path = stats_dir / f"sub-{sub}.json"

    already: set[str] = set()
    if resume:
        already = _load_already_emitted(out_path)
        open_mode = "ab"
    else:
        open_mode = "wb"

    t0 = time.time()

    rs_stats = DumpReadStats()
    keep_posts = pass1_submissions(rs, sub.lower(), rs_stats)
    t_rs = time.time()

    rc_stats = DumpReadStats()
    comments_by_post = pass2_load_comments(rc, keep_posts, rc_stats)
    t_rc = time.time()

    total_anchors = 0
    total_emitted = 0
    total_dropped = 0
    with open(out_path, open_mode) as fh:
        for post_id, post in keep_posts.items():
            comments = comments_by_post.get(post_id)
            if not comments:
                continue
            a, e, d = find_and_emit(post, comments, already, fh)
            total_anchors += a
            total_emitted += e
            total_dropped += d
    t_emit = time.time()

    stats = {
        "subreddit": sub,
        "batch": batch,
        "rs_records_scanned": rs_stats.records,
        "rs_bad_lines": rs_stats.bad_lines,
        "rc_records_scanned": rc_stats.records,
        "rc_bad_lines": rc_stats.bad_lines,
        "keep_posts": len(keep_posts),
        "anchors_found": total_anchors,
        "conversations_emitted": total_emitted,
        "anchors_dropped_no_tail": total_dropped,
        "resumed_from_existing": len(already) if resume else 0,
        "seconds": {
            "rs": round(t_rs - t0, 1),
            "rc": round(t_rc - t_rs, 1),
            "emit": round(t_emit - t_rc, 1),
            "total": round(t_emit - t0, 1),
        },
    }
    with open(stats_path, "wb") as fh:
        fh.write(orjson.dumps(stats, option=orjson.OPT_INDENT_2))
    return stats


def _print_sub_summary(s: dict) -> None:
    print(
        f"[conversations] sub={s['subreddit']} posts={s['keep_posts']} "
        f"anchors={s['anchors_found']} emitted={s['conversations_emitted']} "
        f"dropped_no_tail={s['anchors_dropped_no_tail']} "
        f"in {s['seconds']['total']}s"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--sub", help="subreddit name (case-insensitive), e.g. gardening")
    grp.add_argument("--all", action="store_true",
                     help="run on every sub-* dir under --dumps-dir that has both RS and RC files")
    ap.add_argument("--dumps-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="conversations output dir; .jsonl written here, stats next to it")
    ap.add_argument("--resume", action="store_true",
                    help="append to existing output, skipping post_ids already present")
    args = ap.parse_args()

    if args.all:
        batches = discover_months(args.dumps_dir)
        subs = [b[len("sub-"):] for b in batches if b.startswith("sub-")]
        if not subs:
            raise SystemExit(f"no sub-* dumps with both RS+RC found under {args.dumps_dir}")
        print(f"running on {len(subs)} subs: {', '.join(subs)}")
        for sub in subs:
            try:
                s = run_sub(sub, args.dumps_dir, args.out_dir, args.resume)
                _print_sub_summary(s)
            except Exception as e:  # one bad sub shouldn't kill the whole run
                print(f"[conversations] sub={sub} FAILED: {type(e).__name__}: {e}")
        return

    s = run_sub(args.sub, args.dumps_dir, args.out_dir, args.resume)
    _print_sub_summary(s)


if __name__ == "__main__":
    main()
