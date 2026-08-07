"""Shared loaders and theme labels for user-trend plots."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PAIRS_DIR = HERE.parent / "data" / "extracted_current" / "pairs"
USER_TRENDS_DIR = REPO / "plots" / "outputs" / "user_trends"

THEME_ORDER = [
    "home & crafts",
    "science & advice",
    "language",
    "programming",
    "hobbies & travel",
    "other",
]

THEME_COLORS = {
    "home & crafts": "#e76f51",
    "science & advice": "#2a9d8f",
    "language": "#9b5de5",
    "programming": "#457b9d",
    "hobbies & travel": "#f4a261",
    "other": "#94a3b8",
}

KIND: dict[str, str] = {
    "DIY": "home & crafts",
    "HomeImprovement": "home & crafts",
    "homeimprovement": "home & crafts",
    "woodworking": "home & crafts",
    "gardening": "home & crafts",
    "houseplants": "home & crafts",
    "Sewing": "home & crafts",
    "sewing": "home & crafts",
    "AskCulinary": "home & crafts",
    "AskBaking": "home & crafts",
    "Coffee": "home & crafts",
    "tea": "home & crafts",
    "askscience": "science & advice",
    "AskHistorians": "science & advice",
    "AskDocs": "science & advice",
    "AskEngineers": "science & advice",
    "AskStatistics": "science & advice",
    "AskAcademia": "science & advice",
    "askphilosophy": "science & advice",
    "languagelearning": "language",
    "LearnJapanese": "language",
    "German": "language",
    "LanguageTechnology": "language",
    "learnjavascript": "programming",
    "learnpython": "programming",
    "golang": "programming",
    "rust": "programming",
    "bicycling": "hobbies & travel",
    "solotravel": "hobbies & travel",
    "JapanTravel": "hobbies & travel",
    "Shoestring": "hobbies & travel",
}


def theme_of(sub: str) -> str:
    return KIND.get(sub, "other")


def load_user_sub_counts() -> dict[str, Counter[str]]:
    user_subs: dict[str, Counter[str]] = defaultdict(Counter)
    for fp in sorted(PAIRS_DIR.glob("sub-*.jsonl")):
        with fp.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sub = (rec.get("subreddit") or "").strip()
                if sub:
                    user_subs[rec["user_id"]][sub] += 1
    return dict(user_subs)


def subs_sorted_by_theme(all_subs: list[str]) -> list[str]:
    def key(s: str) -> tuple:
        t = theme_of(s)
        try:
            ti = THEME_ORDER.index(t)
        except ValueError:
            ti = len(THEME_ORDER)
        return (ti, s.lower())

    return sorted(all_subs, key=key)


def build_main_sub_flow(user_subs: dict[str, Counter[str]]) -> tuple[list[str], np.ndarray]:
    """Row = user's main sub; col = where pairs occur. Rows sum to 1."""
    flow: dict[str, Counter[str]] = defaultdict(Counter)
    for counts in user_subs.values():
        if not counts:
            continue
        main_sub, _ = counts.most_common(1)[0]
        for sub, n in counts.items():
            flow[main_sub][sub] += n

    subs = subs_sorted_by_theme(list(flow.keys()))
    n = len(subs)
    idx = {s: i for i, s in enumerate(subs)}
    mat = np.zeros((n, n))
    for src, dests in flow.items():
        i = idx[src]
        row_sum = sum(dests.values())
        if row_sum:
            for dst, c in dests.items():
                if dst in idx:
                    mat[i, idx[dst]] = c / row_sum
    return subs, mat


def off_main_destinations(
    user_subs: dict[str, Counter[str]],
    source_sub: str,
    *,
    top_k: int = 5,
) -> list[tuple[str, float, int]]:
    """Top destination subs for users whose main sub is source_sub (off-main only)."""
    dest_pairs = Counter()
    dest_users = Counter()
    n_users = 0
    for counts in user_subs.values():
        if source_sub not in counts:
            continue
        main_sub, _ = counts.most_common(1)[0]
        if main_sub != source_sub:
            continue
        n_users += 1
        for sub, n in counts.items():
            if sub != source_sub:
                dest_pairs[sub] += n
                dest_users[sub] += 1
    total = sum(dest_pairs.values())
    if not total:
        return []
    ranked = dest_pairs.most_common(top_k)
    return [
        (s, 100 * c / total, dest_users[s])
        for s, c in ranked
    ]
