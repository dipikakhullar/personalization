"""Question-shaping and thanks-reply signals.

Two classifiers, both deliberately conservative (favor precision over recall):

  is_question_post(title, selftext)
      Cheap surface check: ends in '?' OR begins with a wh-/aux question word.
      Removes obvious non-questions (announcements, link posts, rants).

  is_thanks_reply(body)
      Matches OP child-replies that read as positive acknowledgement.
      Rejects sarcastic / negated forms ("no thanks", "thanks for nothing").

Both are intentionally regex-based, not model-based: the pipeline streams
hundreds of millions of records and per-record latency matters. A model-based
re-rank pass can be layered on the extracted pairs later if precision needs
to go up.
"""

from __future__ import annotations

import re

# ── Question shape ─────────────────────────────────────────────────────

_QUESTION_PREFIX = re.compile(
    r"^\s*(how|what|why|when|where|which|who|whose|whom|"
    r"is|are|am|was|were|do|does|did|"
    r"can|could|should|would|will|may|might|must|shall|"
    r"has|have|had|"
    r"any(one|body)|"
    r"recommend|suggest|help|advice|tips?|"
    r"looking for|trying to|need to|need help|need advice"
    r")\b",
    re.IGNORECASE,
)


def is_question_post(title: str | None, selftext: str | None) -> bool:
    title = (title or "").strip()
    selftext = (selftext or "").strip()
    if not title:
        return False
    # Most reliable: ends in '?'. Selftext counts too — many askreddit-style
    # posts put the question in the title and the context in selftext.
    if title.endswith("?") or selftext.endswith("?"):
        return True
    # Fall back to lexical prefix on title only — selftext often opens with a
    # backstory paragraph, so a wh-word at its start is a weak signal.
    return bool(_QUESTION_PREFIX.match(title))


# ── Thanks-reply ───────────────────────────────────────────────────────
# We match short positive-acknowledgement replies from the post author.
# Tightening choices:
#   - Require the thanks token to appear early (within first 80 chars) so we
#     don't catch long replies that happen to contain "thanks" downstream.
#   - Reject if the same span is negated (preceded by 'no', 'not', "don't").
#   - Reject sarcasm-shaped phrases.

_THANKS_TOKENS = re.compile(
    r"\b("
    r"thanks?(?:\s*(?:a\s*lot|so\s*much|a\s*ton|a\s*bunch|heaps))?|"
    r"thank\s*(?:you|u)|"
    r"thx|tysm|ty\b|"
    r"appreciate(?:\s*(?:it|that|this))?|"
    r"this (?:worked|helped|did it|did the trick|is exactly|is perfect)|"
    r"that (?:worked|helped|did it|did the trick|was exactly|was perfect)|"
    r"exactly what (?:i|I) (?:needed|was looking for)|"
    r"perfect(?:,?\s*thanks?)?|"
    r"solved(?: it| the (?:problem|issue))?|"
    r"you(?:'re| are) (?:a (?:legend|lifesaver|hero|godsend|saint)|amazing|the best)|"
    r"legend(?:\s*,?\s*thanks?)?|"
    r"lifesaver"
    r")\b",
    re.IGNORECASE,
)

_NEGATORS = re.compile(
    r"\b(no thanks?|not? (?:really )?thank|don'?t (?:thank|appreciate)|"
    r"thanks?\s+for\s+nothing|thanks?\s+i\s+hate|sarcas)",
    re.IGNORECASE,
)

# Replies that are too long are probably not pure acknowledgement —
# they're follow-up questions or extended discussion. Cap to keep precision up.
_THANKS_MAX_LEN = 600


def is_thanks_reply(body: str | None) -> bool:
    if not body:
        return False
    body = body.strip()
    if not body or len(body) > _THANKS_MAX_LEN:
        return False
    if _NEGATORS.search(body):
        return False
    m = _THANKS_TOKENS.search(body)
    if not m:
        return False
    # Require the thanks token to appear in the opening of the reply.
    return m.start() <= 80


# ── Bot heuristics on body ─────────────────────────────────────────────

_BOT_BODY_MARKERS = re.compile(
    r"\b(I am a bot|I'?m a bot|beep boop|this action was performed automatically|"
    r"contact the moderators of this subreddit)\b",
    re.IGNORECASE,
)


def looks_like_bot_body(body: str | None) -> bool:
    return bool(body) and bool(_BOT_BODY_MARKERS.search(body))
