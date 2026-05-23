"""Generate a per-subreddit users_<sub>.md report from a pair file.

Picks three users with distinct profiles:
  A) heaviest by interaction count, requiring >= 5 distinct posts
  B) longest time span, requiring >= 4 distinct posts
  C) highest divergence rate, requiring >= 5 distinct posts and >= 8 interactions

For each, renders 3-5 representative interactions (only on distinct posts so
the timeline reflects actual different threads), with the full Q, preferred,
top (if different), and OP thanks-reply text.

Usage:
  python generate_user_traces.py \\
      --pairs ../data/snapshot_xxx/extracted/pairs/sub-AskStatistics.jsonl \\
      --out ../data/users_AskStatistics.md
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

_ZW = re.compile(r"[​‌‍﻿]|&#x200B;|&amp;#x200B;")
_MULTI_WS = re.compile(r"\s+")


def clean(s: str) -> str:
    s = _ZW.sub("", s)
    s = s.replace("&amp;", "&").replace("&gt;", ">").replace("&lt;", "<")
    s = s.replace("\n\n", " ").replace("\n", " ")
    return _MULTI_WS.sub(" ", s).strip()


def truncate(s: str, n: int) -> str:
    s = clean(s)
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"


def pick_distinct_post_indices(history: list[dict], k: int = 5) -> list[int]:
    """Return up to k indices spanning distinct posts AND distinct timestamps,
    biased toward first/last with some middle picks."""
    seen = set()
    candidates = []
    for i, p in enumerate(history):
        pid = p["metadata"]["post_id"]
        if pid in seen:
            continue
        seen.add(pid)
        candidates.append(i)
    if len(candidates) <= k:
        return candidates
    # Spread: first, last, and (k-2) evenly between.
    out = [candidates[0]]
    step = (len(candidates) - 1) / (k - 1)
    for i in range(1, k - 1):
        out.append(candidates[round(i * step)])
    out.append(candidates[-1])
    # Dedup while preserving order
    return sorted(set(out))


def render_user(uid: str, history: list[dict], title: str, blurb: str,
                picks: list[int]) -> str:
    out: list[str] = []
    n = len(history)
    posts = len(set(p["metadata"]["post_id"] for p in history))
    span_start = history[0]["timestamp"][:10]
    span_end = history[-1]["timestamp"][:10]
    div = sum(1 for p in history if not p["metadata"]["top_equals_preferred"])
    out.append(f"## {title}")
    out.append("")
    out.append(f"**`{uid}`** — {n} interactions across {posts} distinct posts, "
               f"{span_start} → {span_end}, {div}/{n} divergent "
               f"({div*100//n}%).")
    out.append("")
    out.append(blurb)
    out.append("")
    for idx in picks:
        p = history[idx]
        md = p["metadata"]
        out.append(f"### Interaction {idx+1} of {n} — {p['timestamp'][:10]}")
        out.append("")
        out.append(f"> **Q:** {truncate(p['query'], 350)}")
        out.append("")
        out.append(f"- **preferred_answer** *(score = {md['answer_score']})*:")
        out.append(f"  > \"*{truncate(p['preferred_answer'], 450)}*\"")
        if md["top_equals_preferred"]:
            out.append(f"- **top_comment:** same as preferred (signals agree)")
        else:
            out.append(f"- **top_comment** *(score = {md['top_comment_score']})*:")
            out.append(f"  > \"*{truncate(p['top_comment'], 450)}*\"")
        out.append(f"- **OP's thanks-reply:** "
                   f"\"*{truncate(md['thanks_reply_text'], 250)}*\"")
        out.append("")
    return "\n".join(out)


def pick_three_users(by_user: dict[str, list[dict]]) -> list[tuple[str, str, list[int]]]:
    """Return [(uid, role, picks)] for three diverse users.

    Roles: 'heavy', 'longspan', 'highdiv'. The picks list is integer indices
    into the user's chronologically-sorted history.
    """
    rows = []
    for uid, hist in by_user.items():
        posts = set(p["metadata"]["post_id"] for p in hist)
        if len(posts) < 4:
            continue
        first = hist[0]["timestamp"][:4]
        last = hist[-1]["timestamp"][:4]
        span_yrs = int(last) - int(first)
        div = sum(1 for p in hist if not p["metadata"]["top_equals_preferred"])
        rows.append({
            "uid": uid,
            "n": len(hist),
            "n_posts": len(posts),
            "span_yrs": span_yrs,
            "div": div,
            "div_rate": div / len(hist),
        })

    # heavy: max n with n_posts >= 5
    heavy = max((r for r in rows if r["n_posts"] >= 5),
                key=lambda r: r["n"], default=None)
    # longspan: max span_yrs (tiebreak by n_posts) with n_posts >= 4
    longspan = max((r for r in rows if r["n_posts"] >= 4 and r["uid"] != (heavy or {}).get("uid")),
                   key=lambda r: (r["span_yrs"], r["n_posts"]), default=None)
    # highdiv: max div_rate with n_posts >= 5 and n >= 8, distinct from above
    used = {r["uid"] for r in (heavy, longspan) if r}
    highdiv = max((r for r in rows
                   if r["n_posts"] >= 5 and r["n"] >= 8 and r["uid"] not in used),
                  key=lambda r: (r["div_rate"], r["n_posts"]), default=None)

    out: list[tuple[str, str, list[int]]] = []
    if heavy:
        out.append((heavy["uid"], "heavy",
                    pick_distinct_post_indices(by_user[heavy["uid"]], 5)))
    if longspan:
        out.append((longspan["uid"], "longspan",
                    pick_distinct_post_indices(by_user[longspan["uid"]], 4)))
    if highdiv:
        out.append((highdiv["uid"], "highdiv",
                    pick_distinct_post_indices(by_user[highdiv["uid"]], 4)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, type=Path,
                    help="path to extracted/pairs/sub-<name>.jsonl")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sub-label", default=None,
                    help="subreddit name to show in headings; inferred from "
                         "filename if omitted")
    ap.add_argument("--note", default=None,
                    help="optional note prepended to the header (e.g. "
                         "'PARTIAL — fetch still in progress')")
    args = ap.parse_args()

    pairs = [json.loads(l) for l in args.pairs.read_bytes().splitlines() if l.strip()]
    if not pairs:
        raise SystemExit(f"no pairs in {args.pairs}")

    sub = args.sub_label or args.pairs.stem.removeprefix("sub-")

    by_user: dict[str, list[dict]] = defaultdict(list)
    for p in pairs:
        by_user[p["user_id"]].append(p)
    for uid in by_user:
        by_user[uid].sort(key=lambda p: p["timestamp"])

    total = len(pairs)
    users = len(by_user)
    both = sum(1 for p in pairs if p["top_comment"])
    agree = sum(1 for p in pairs if p["metadata"]["top_equals_preferred"])

    picks = pick_three_users(by_user)
    role_blurbs = {
        "heavy": ("User A — heavy user (most interactions)",
                  "Highest interaction count in this subreddit. Read top-to-"
                  "bottom for a snapshot of how this user engages over time — "
                  "what they ask, what answer style they thank, whether their "
                  "preferred answer matches the community top vote."),
        "longspan": ("User B — longest temporal span",
                     "The user whose interactions are spread over the most "
                     "years. Interesting for seeing how question topics evolve "
                     "while writing voice and answer-style preferences stay "
                     "stable."),
        "highdiv": ("User C — strongest individual signal (highest divergence)",
                    "Among users with enough activity (>=8 interactions, >=5 "
                    "distinct posts), this one most consistently thanks an "
                    "answer that wasn't the community-top — i.e. their "
                    "personal preference diverges from the population most "
                    "strongly. Useful as a personalization difficulty case."),
    }

    out: list[str] = []
    out.append(f"# User Traces — r/{sub}")
    out.append("")
    if args.note:
        out.append(f"> **{args.note}**")
        out.append("")
    out.append(f"*Source: `{args.pairs}`*")
    out.append("")
    out.append(f"- pairs: {total:,}")
    out.append(f"- unique users: {users:,}")
    out.append(f"- pairs with both signals attached: {both:,} "
               f"({both*100//total}%)")
    out.append(f"- top_comment == preferred_answer: {agree:,} "
               f"({agree*100//total}%)")
    out.append(f"- divergent: {total-agree:,} "
               f"({(total-agree)*100//total}%)")
    out.append("")
    out.append("Three users with distinct profiles are shown below. Interactions "
               "are sorted ascending by timestamp; picks are on distinct posts "
               "so the timeline reflects actual different threads.")
    out.append("")

    for uid, role, idx in picks:
        title, blurb = role_blurbs[role]
        out.append(render_user(uid, by_user[uid], title, blurb, idx))

    args.out.write_text("\n".join(out))
    print(f"wrote {args.out}  ({len(picks)} users, {total:,} pairs)")


if __name__ == "__main__":
    main()
