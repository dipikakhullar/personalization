"""Comment-tree indexing and OP↔answerer chain walking.

Pure data-structure code: takes already-parsed comments for a single post and
walks the parent→child tree to recover a contiguous, strictly alternating
two-party conversation. Kept separate from the dump-reading driver so the
logic is unit-testable without zstd fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Comment:
    id: str
    parent_id: str         # already stripped of t1_/t3_; equals post_id for top-level comments
    author: str
    body: str
    created_utc: int
    score: int
    author_fields: dict | None = None  # optional author_* fields preserved from raw rec


def build_children_index(comments: dict[str, Comment]) -> dict[str, list[str]]:
    """Return parent_id -> [child_id, ...] sorted ascending by created_utc.

    parent_id keys are comment_ids or the post_id (for top-level comments).
    """
    out: dict[str, list[str]] = {}
    for c in comments.values():
        out.setdefault(c.parent_id, []).append(c.id)
    for kids in out.values():
        kids.sort(key=lambda cid: comments[cid].created_utc)
    return out


def walk_linear(
    start_id: str,
    next_author: str,
    op_author: str,
    answerer_author: str,
    comments: dict[str, Comment],
    children_by_parent: dict[str, list[str]],
    is_valid_author,
) -> list[str]:
    """Walk the comment tree below ``start_id`` collecting an alternating chain.

    At each step we take the **earliest** child (by created_utc) whose author
    matches the expected next speaker. If no such child exists, or if the only
    candidates are filtered out by ``is_valid_author`` (e.g. deleted / bot
    accounts) or have an empty/removed body, the walk stops.

    Returns ordered comment_ids *after* ``start_id`` — the start itself is not
    included. The first id in the returned list is by ``next_author``.

    ``is_valid_author(author)`` must return True for acceptable authors.
    Empty/deleted-sentinel bodies also break the chain.
    """
    chain: list[str] = []
    current = start_id
    expected = next_author
    while True:
        kids = children_by_parent.get(current, [])
        # children are pre-sorted by created_utc; pick the earliest matching.
        nxt = None
        for cid in kids:
            c = comments[cid]
            if c.author != expected:
                continue
            if not is_valid_author(c.author):
                # The expected speaker themselves is filtered (bot/deleted) →
                # chain breaks; don't look at other expected-author candidates.
                return chain
            body = (c.body or "").strip()
            if not body or body in ("[deleted]", "[removed]"):
                return chain
            nxt = cid
            break
        if nxt is None:
            return chain
        chain.append(nxt)
        current = nxt
        expected = op_author if expected == answerer_author else answerer_author
