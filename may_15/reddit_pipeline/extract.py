"""Extract (query, preferred_answer) pairs from one or more monthly dumps.

Algorithm, per month:

  Pass 1 — submissions (RS_<YYYY-MM>.zst):
    Keep posts where subreddit ∈ allowlist, author is a real user, the post
    looks like a question. Result: keep_posts[post_id] = PostMeta.

  Pass 2a — comments (RC_<YYYY-MM>.zst), thanks-reply detection:
    For each comment whose link_id is in keep_posts, whose author equals the
    post's author, and whose body matches is_thanks_reply(): record the
    parent comment id as a candidate "preferred answer" pointer.

  Pass 2b — comments (RC_<YYYY-MM>.zst), candidate-answer materialization:
    For each comment whose id is in the candidate-answer set: emit a
    (query, preferred_answer) pair, joining post metadata with answer body.

Two passes over RC avoid holding all comments for popular subs in memory.

Output: <out_dir>/pairs/<YYYY-MM>.jsonl, one pair per line.
Also: <out_dir>/stats/<YYYY-MM>.json with counts for debugging the funnel.
"""

from __future__ import annotations

import argparse
import datetime as dt
import time
from dataclasses import dataclass
from pathlib import Path

import orjson

from anon import anon_user_id
from dump_reader import (
    DumpReadStats,
    comment_path,
    discover_months,
    iter_dump,
    submission_path,
)
from signals import is_question_post, is_thanks_reply, looks_like_bot_body
from subreddits import is_allowed_subreddit, is_bot_author, is_deleted_author


# Author-level fields preserved from the raw RS/RC records. Listed here so
# all records carry the same key set even when most values are null.
# Mirrors the may_22 multiturn dataset schema for cross-dataset consistency.
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


@dataclass(slots=True)
class PostMeta:
    post_id: str            # e.g. "abc123"
    op_author: str          # raw username, used internally only
    title: str
    selftext: str
    subreddit: str
    created_utc: int
    score: int
    author_fields: dict     # whitelisted author_* fields from the RS record


@dataclass(slots=True)
class ThanksRef:
    post_id: str            # link_id stripped of t3_
    parent_comment_id: str  # t1_xyz stripped of t1_
    thanks_reply_id: str
    thanks_reply_body: str
    thanks_reply_score: int
    thanks_reply_created: int


def _strip_kind(fullname: str | None) -> str | None:
    """'t3_abc123' -> 'abc123'. Pushshift sometimes stores the kind prefix,
    sometimes not, so normalize."""
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


# ── Pass 1: submissions ────────────────────────────────────────────────

def pass1_submissions(path: Path, stats: DumpReadStats) -> dict[str, PostMeta]:
    keep: dict[str, PostMeta] = {}
    for rec in iter_dump(path, stats):
        sub = rec.get("subreddit")
        if not is_allowed_subreddit(sub):
            continue
        author = rec.get("author")
        if is_deleted_author(author) or is_bot_author(author):
            continue
        title = rec.get("title") or ""
        selftext = rec.get("selftext") or ""
        if not is_question_post(title, selftext):
            continue
        # Drop posts the mods removed.
        if (rec.get("_meta") or {}).get("removal_type"):
            continue
        post_id = rec.get("id")
        if not post_id:
            continue
        keep[post_id] = PostMeta(
            post_id=post_id,
            op_author=author,
            title=title,
            selftext=selftext if _ok_text(selftext) else "",
            subreddit=sub,
            created_utc=int(rec.get("created_utc") or 0),
            score=int(rec.get("score") or 0),
            author_fields={k: rec.get(k) for k in _OP_FIELDS},
        )
    return keep


# ── Pass 2a: scan comments once for thanks-replies + per-post top comment ─

def pass2a_collect_signals(
    path: Path, keep_posts: dict[str, PostMeta], stats: DumpReadStats
) -> tuple[list[ThanksRef], dict[str, tuple[str, int]]]:
    """Single scan of RC that produces both signals we need:

      - thanks_refs: OP-thanks-reply records (post → which comment was thanked)
      - top_by_post: post_id → (best_comment_id, best_score) over all
        non-OP, non-bot, non-deleted, non-empty-body comments in the thread.

    Returning only ids+scores here keeps memory bounded (~24 bytes/post). The
    full bodies for those ids are materialized in pass 2b.
    """
    refs: list[ThanksRef] = []
    top_by_post: dict[str, tuple[str, int]] = {}

    for rec in iter_dump(path, stats):
        link_id = _strip_kind(rec.get("link_id"))
        if not link_id or link_id not in keep_posts:
            continue
        post = keep_posts[link_id]
        author = rec.get("author")
        body = rec.get("body") or ""
        score = int(rec.get("score") or 0)
        cid = rec.get("id") or ""

        if is_deleted_author(author) or is_bot_author(author):
            continue

        # Thanks-reply path: only OP-authored.
        if author == post.op_author:
            if is_thanks_reply(body):
                parent_id = _strip_kind(rec.get("parent_id"))
                if parent_id and parent_id != link_id:
                    refs.append(ThanksRef(
                        post_id=link_id,
                        parent_comment_id=parent_id,
                        thanks_reply_id=cid,
                        thanks_reply_body=body,
                        thanks_reply_score=score,
                        thanks_reply_created=int(rec.get("created_utc") or 0),
                    ))
            # OP comments are never the "top comment" of their own thread.
            continue

        # Top-comment candidate: not OP, body OK, not bot body.
        if not _ok_text(body) or looks_like_bot_body(body):
            continue
        cur = top_by_post.get(link_id)
        if cur is None or score > cur[1]:
            top_by_post[link_id] = (cid, score)

    return refs, top_by_post


# ── Pass 2b: materialize bodies for thanked-comments + top-comments ──

def pass2b_emit_pairs(
    path: Path,
    keep_posts: dict[str, PostMeta],
    thanks_by_parent: dict[str, ThanksRef],
    top_by_post: dict[str, tuple[str, int]],
    stats: DumpReadStats,
    out_fh,
) -> int:
    # Build the set of comment ids we need bodies for.
    need_ids: set[str] = set(thanks_by_parent.keys())
    for _post, (top_id, _s) in top_by_post.items():
        need_ids.add(top_id)

    # comment_id -> (body, author, score, author_fields)
    bodies: dict[str, tuple[str, str, int, dict]] = {}

    for rec in iter_dump(path, stats):
        cid = rec.get("id")
        if not cid or cid not in need_ids:
            continue
        author = rec.get("author")
        body = rec.get("body") or ""
        score = int(rec.get("score") or 0)
        author_fields = {k: rec.get(k) for k in _ANSWERER_FIELDS}
        bodies[cid] = (body, author, score, author_fields)

    emitted = 0
    for parent_cid, ref in thanks_by_parent.items():
        post = keep_posts.get(ref.post_id)
        if post is None:
            continue
        ans = bodies.get(parent_cid)
        if ans is None:
            continue
        ans_body, ans_author, ans_score, ans_author_fields = ans
        if is_deleted_author(ans_author) or is_bot_author(ans_author):
            continue
        if ans_author == post.op_author:
            continue  # OP thanking themselves
        if not _ok_text(ans_body) or looks_like_bot_body(ans_body):
            continue

        query = post.title.strip()
        if post.selftext:
            query = f"{query}\n\n{post.selftext.strip()}"

        top_info = top_by_post.get(ref.post_id)
        top_comment_body = None
        top_comment_id = None
        top_comment_score = None
        top_comment_anon_id = None
        if top_info is not None:
            top_id, _top_score_seen = top_info
            tb = bodies.get(top_id)
            if tb is not None:
                t_body, t_author, t_score, _t_fields = tb
                if (not is_deleted_author(t_author)
                        and not is_bot_author(t_author)
                        and t_author != post.op_author
                        and _ok_text(t_body)
                        and not looks_like_bot_body(t_body)):
                    top_comment_body = t_body.strip()
                    top_comment_id = top_id
                    top_comment_score = t_score
                    top_comment_anon_id = anon_user_id(t_author)

        op_anon = anon_user_id(post.op_author)
        ans_anon = anon_user_id(ans_author)
        op_metadata = {"user_id": op_anon}
        op_metadata.update(post.author_fields)
        answerer_metadata = {"user_id": ans_anon}
        answerer_metadata.update(ans_author_fields)

        pair = {
            "user_id": op_anon,
            "timestamp": _isoformat(post.created_utc),
            "subreddit": post.subreddit,
            "query": query,
            "preferred_answer": ans_body.strip(),
            "top_comment": top_comment_body,
            "op_metadata": op_metadata,
            "answerer_metadata": answerer_metadata,
            "metadata": {
                "post_id": post.post_id,
                "post_score": post.score,
                "answer_comment_id": parent_cid,
                "answer_score": ans_score,
                "answerer_anon_id": ans_anon,
                "top_comment_id": top_comment_id,
                "top_comment_score": top_comment_score,
                "top_comment_anon_id": top_comment_anon_id,
                "top_equals_preferred": top_comment_id == parent_cid,
                "thanks_reply_id": ref.thanks_reply_id,
                "thanks_reply_score": ref.thanks_reply_score,
                "thanks_reply_text": ref.thanks_reply_body,
                "thanks_reply_timestamp": _isoformat(ref.thanks_reply_created),
            },
        }
        out_fh.write(orjson.dumps(pair))
        out_fh.write(b"\n")
        emitted += 1
    return emitted


# ── Driver ────────────────────────────────────────────────────────────

def run_month(month: str, dumps_dir: Path, out_dir: Path) -> dict:
    rs = submission_path(dumps_dir, month)
    rc = comment_path(dumps_dir, month)

    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir = out_dir / "pairs"
    pairs_dir.mkdir(exist_ok=True)
    stats_dir = out_dir / "stats"
    stats_dir.mkdir(exist_ok=True)

    pair_path = pairs_dir / f"{month}.jsonl"
    stats_path = stats_dir / f"{month}.json"

    t0 = time.time()

    rs_stats = DumpReadStats()
    keep_posts = pass1_submissions(rs, rs_stats)
    t_rs = time.time()

    rc_a_stats = DumpReadStats()
    refs, top_by_post = pass2a_collect_signals(rc, keep_posts, rc_a_stats)
    t_rc_a = time.time()

    # Deduplicate: if OP posts more than one thanks-reply to the same comment,
    # keep the first one (chronologically — refs are dump-order, which is
    # roughly chronological per month).
    thanks_by_parent: dict[str, ThanksRef] = {}
    for r in refs:
        thanks_by_parent.setdefault(r.parent_comment_id, r)

    rc_b_stats = DumpReadStats()
    with open(pair_path, "wb") as fh:
        emitted = pass2b_emit_pairs(
            rc, keep_posts, thanks_by_parent, top_by_post, rc_b_stats, fh
        )
    t_rc_b = time.time()

    stats = {
        "month": month,
        "rs_records_scanned": rs_stats.records,
        "rs_bad_lines": rs_stats.bad_lines,
        "rc_records_scanned": rc_a_stats.records,
        "rc_bad_lines": rc_a_stats.bad_lines + rc_b_stats.bad_lines,
        "keep_posts": len(keep_posts),
        "thanks_refs": len(refs),
        "thanks_unique_parents": len(thanks_by_parent),
        "posts_with_top_comment": len(top_by_post),
        "pairs_emitted": emitted,
        "seconds": {
            "rs": round(t_rs - t0, 1),
            "rc_a": round(t_rc_a - t_rs, 1),
            "rc_b": round(t_rc_b - t_rc_a, 1),
            "total": round(t_rc_b - t0, 1),
        },
    }
    with open(stats_path, "wb") as fh:
        fh.write(orjson.dumps(stats, option=orjson.OPT_INDENT_2))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", help="YYYY-MM")
    ap.add_argument("--all", action="store_true", help="Process every month present in dumps-dir")
    ap.add_argument("--dumps-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    if args.all:
        months = discover_months(args.dumps_dir)
    elif args.month:
        months = [args.month]
    else:
        ap.error("must pass --month YYYY-MM or --all")

    for m in months:
        print(f"[extract] {m}")
        s = run_month(m, args.dumps_dir, args.out_dir)
        print(
            f"  posts={s['keep_posts']} thanks={s['thanks_unique_parents']} "
            f"pairs={s['pairs_emitted']} in {s['seconds']['total']}s"
        )


if __name__ == "__main__":
    main()
