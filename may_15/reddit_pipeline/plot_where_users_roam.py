"""One figure: which subreddits do multi-sub users post in besides their main one?

For each user with thanked threads in 2+ subs, "main sub" = where they have the most
pairs. Count all pairs on their other subs → bar chart of destinations.

Output: ../../plots/outputs/user_trends/where_multi_sub_users_post.png

Usage (from may_15/reddit_pipeline/):
  python plot_where_users_roam.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PAIRS_DIR = HERE.parent / "data" / "extracted_current" / "pairs"
OUT = REPO / "plots" / "outputs" / "user_trends" / "where_multi_sub_users_post.png"

# Loose grouping for bar colors (readable "kinds" of communities)
KIND = {
    "DIY": "home & crafts",
    "HomeImprovement": "home & crafts",
    "homeimprovement": "home & crafts",
    "woodworking": "home & crafts",
    "gardening": "home & crafts",
    "houseplants": "home & crafts",
    "Sewing": "home & crafts",
    "sewing": "home & crafts",
    "AskCulinary": "home & crafts",
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
    "AskBaking": "home & crafts",
}


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


def main() -> None:
    user_subs = load_user_sub_counts()
    multi = {u: c for u, c in user_subs.items() if len(c) >= 2}
    print(f"{len(user_subs):,} users, {len(multi):,} active in 2+ subs", flush=True)

    # Pairs that fall outside each user's main subreddit
    dest_pairs = Counter()
    dest_users = Counter()
    home_of_roamers = Counter()  # which main subs produce the most roaming

    for uid, counts in multi.items():
        main_sub, _ = counts.most_common(1)[0]
        home_of_roamers[main_sub] += 1
        for sub, n in counts.items():
            if sub != main_sub:
                dest_pairs[sub] += n
                dest_users[sub] += 1

    total_roam_pairs = sum(dest_pairs.values())
    ranked = dest_pairs.most_common()

    subs = [s for s, _ in ranked][::-1]
    vals = [100 * dest_pairs[s] / total_roam_pairs for s in subs]
    users_per_sub = [dest_users[s] for s in subs]
    kinds = [KIND.get(s, "other") for s in subs]
    kind_colors = {
        "home & crafts": "#e76f51",
        "science & advice": "#2a9d8f",
        "language": "#9b5de5",
        "programming": "#457b9d",
        "hobbies & travel": "#f4a261",
        "other": "#94a3b8",
    }
    colors = [kind_colors[k] for k in kinds]

    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(subs))
    ax.barh(y, vals, color=colors, edgecolor="white", height=0.78)
    ax.set_yticks(y)
    ax.set_yticklabels([f"r/{s}" for s in subs], fontsize=9)
    ax.set_xlabel("Share of thanked threads outside a user's main subreddit")
    ax.set_title(
        "Where do multi-sub users post besides their main community?\n"
        f"{len(multi):,} users have thanked answers in 2+ subs · "
        f"{total_roam_pairs:,} threads outside their main sub",
        fontsize=11,
    )
    for i, (s, pct) in enumerate(zip(subs, vals)):
        ax.text(
            pct + 0.35,
            i,
            f"{pct:.1f}%  ·  {dest_users[s]:,} users",
            va="center",
            fontsize=7.5,
        )
    ax.set_xlim(0, max(vals) * 1.28 if vals else 1)

    from matplotlib.patches import Patch
    seen = []
    legend_handles = []
    for k, col in kind_colors.items():
        if k in kinds and k not in seen:
            seen.append(k)
            legend_handles.append(Patch(facecolor=col, edgecolor="white", label=k))
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8, title="kind of sub")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")
    print("Top destinations (share of non-main pairs):")
    for s, n in ranked[:8]:
        print(f"  r/{s:20s}  {100*n/total_roam_pairs:5.1f}%  ({dest_users[s]:,} users)")


if __name__ == "__main__":
    main()
