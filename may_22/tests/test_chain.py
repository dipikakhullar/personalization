"""Unit tests for chain.walk_linear and build_children_index.

Synthetic comment trees, no zst fixtures. Run with:
  python -m pytest may_22/tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from chain import Comment, build_children_index, walk_linear  # noqa: E402


OP = "op_user"
ANS = "answerer_user"
THIRD = "third_party"
BOT = "SomeBot"
DELETED = "[deleted]"


def _accept_real_users(author: str) -> bool:
    """Test stand-in for the real is_valid_author predicate."""
    if not author:
        return False
    if author in (DELETED, "[removed]", ""):
        return False
    if author.endswith("Bot"):
        return False
    return True


def _mk(id_, parent_id, author, t, body="x"):
    return Comment(id=id_, parent_id=parent_id, author=author,
                   body=body, created_utc=t, score=1)


def _run_walk(comments_list, start_id, post_id):
    comments = {c.id: c for c in comments_list}
    kids = build_children_index(comments)
    return walk_linear(
        start_id=start_id,
        next_author=ANS,
        op_author=OP,
        answerer_author=ANS,
        comments=comments,
        children_by_parent=kids,
        is_valid_author=_accept_real_users,
    )


def test_linear_chain_of_four():
    # post → A1 (ans) → OP1 (op,thanks) → A2 (ans) → OP2 (op) → A3 (ans)
    post_id = "p1"
    cs = [
        _mk("A1", post_id, ANS, 10),
        _mk("OP1", "A1",   OP,  20, body="thanks!"),
        _mk("A2", "OP1",   ANS, 30),
        _mk("OP2", "A2",   OP,  40),
        _mk("A3", "OP2",   ANS, 50),
    ]
    tail = _run_walk(cs, start_id="OP1", post_id=post_id)
    assert tail == ["A2", "OP2", "A3"]


def test_branch_picks_earliest_by_created_utc():
    # answerer wrote TWO children under OP1; we should pick the earlier one (A2a, t=25)
    # then continue normally.
    post_id = "p1"
    cs = [
        _mk("A1",  "p1",  ANS, 10),
        _mk("OP1", "A1",  OP,  20, body="thanks!"),
        _mk("A2b", "OP1", ANS, 35),
        _mk("A2a", "OP1", ANS, 25),
        _mk("OP2", "A2a", OP,  40),
    ]
    tail = _run_walk(cs, start_id="OP1", post_id=post_id)
    assert tail == ["A2a", "OP2"]


def test_third_party_breaks_chain():
    # After OP1, the only answerer-by-author reply is also followed by a third party
    # interleaving — but since walk requires answerer next, we just keep going
    # via the answerer's child. The third party's subtree is ignored.
    # Then under A2 there is ONLY a third-party reply (no OP), so chain stops at A2.
    post_id = "p1"
    cs = [
        _mk("A1",  "p1",  ANS,   10),
        _mk("OP1", "A1",  OP,    20, body="thanks!"),
        _mk("A2",  "OP1", ANS,   30),
        _mk("T1",  "A2",  THIRD, 40),     # third party replies to A2; no OP reply
    ]
    tail = _run_walk(cs, start_id="OP1", post_id=post_id)
    assert tail == ["A2"]


def test_no_tail_returns_empty():
    # Only the existing pair (A1 + OP1 thanks). Nothing past it.
    post_id = "p1"
    cs = [
        _mk("A1",  "p1", ANS, 10),
        _mk("OP1", "A1", OP,  20, body="thanks!"),
    ]
    tail = _run_walk(cs, start_id="OP1", post_id=post_id)
    assert tail == []


def test_bot_in_chain_truncates():
    # A2 is by a bot user → chain stops at OP1 (returns []).
    post_id = "p1"
    cs = [
        _mk("A1",  "p1",  ANS, 10),
        _mk("OP1", "A1",  OP,  20, body="thanks!"),
        # A bot impersonating the answerer slot — should be rejected.
        # But our walker expects next_author == ANS; the bot has a different
        # author name so it simply isn't picked. Test the case where the
        # answerer themselves disappears: the answerer's reply has an empty body.
        _mk("A2",  "OP1", ANS, 30, body=""),
        _mk("OP2", "A2",  OP,  40),
    ]
    tail = _run_walk(cs, start_id="OP1", post_id=post_id)
    assert tail == []


def test_deleted_body_truncates():
    post_id = "p1"
    cs = [
        _mk("A1",  "p1",  ANS, 10),
        _mk("OP1", "A1",  OP,  20, body="thanks!"),
        _mk("A2",  "OP1", ANS, 30, body="[deleted]"),
        _mk("OP2", "A2",  OP,  40),
    ]
    tail = _run_walk(cs, start_id="OP1", post_id=post_id)
    assert tail == []


def test_children_index_sorted_by_time():
    post_id = "p1"
    comments = {
        "c1": _mk("c1", "p1", ANS, 30),
        "c2": _mk("c2", "p1", ANS, 10),
        "c3": _mk("c3", "p1", ANS, 20),
    }
    kids = build_children_index(comments)
    assert kids["p1"] == ["c2", "c3", "c1"]
