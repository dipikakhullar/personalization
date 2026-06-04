"""User-trend figures for papers: themed flow matrix + small-multiple destinations.

All outputs under plots/outputs/user_trends/

Usage (from may_15/reddit_pipeline/):
  python plot_user_trends.py           # flow matrix + small multiples
  python plot_user_trends.py --all     # also regenerate activity / roam plots
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from user_trends_lib import (
    THEME_COLORS,
    THEME_ORDER,
    USER_TRENDS_DIR,
    build_main_sub_flow,
    load_user_sub_counts,
    off_main_destinations,
    subs_sorted_by_theme,
    theme_of,
)

# Six sources spanning themes + interesting cross-post patterns
SMALL_MULTIPLE_SOURCES = [
    "DIY",
    "HomeImprovement",
    "askscience",
    "languagelearning",
    "houseplants",
    "learnjavascript",
]


def _theme_boundaries(subs: list[str]) -> list[int]:
    """Indices after which to draw a grid line (between theme groups)."""
    bounds = []
    prev = theme_of(subs[0]) if subs else ""
    for i, s in enumerate(subs[1:], 1):
        t = theme_of(s)
        if t != prev:
            bounds.append(i - 0.5)
            prev = t
    return bounds


def plot_flow_matrix(subs: list[str], mat: np.ndarray, path) -> None:
    n = len(subs)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.42), max(8, n * 0.38)))
    pct = mat * 100
    im = ax.imshow(pct, cmap="YlOrRd", aspect="equal", vmin=0, vmax=min(40, pct.max() or 1))

    labels = [f"r/{s}" for s in subs]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Destination subreddit (where thanked threads occur)")
    ax.set_ylabel("Source subreddit (user's main community)")
    ax.set_title(
        "Thanked-thread flow by main subreddit (row-normalized %)\n"
        "Bright diagonal = users stay home; off-diagonal = roam",
        fontsize=11,
    )

    for i in range(n):
        for j in range(n):
            v = pct[i, j]
            if v < 1.0:
                continue
            color = "white" if v > 18 else "black"
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.5, color=color)

    for b in _theme_boundaries(subs):
        ax.axhline(b, color="#334155", lw=1.2)
        ax.axvline(b, color="#334155", lw=1.2)

    # Theme color strips on margins
    for i, s in enumerate(subs):
        c = THEME_COLORS[theme_of(s)]
        ax.plot(-0.85, i, "s", color=c, markersize=6, clip_on=False)
        ax.plot(i, n - 0.15, "s", color=c, markersize=6, clip_on=False)

    handles = [Patch(facecolor=THEME_COLORS[t], label=t) for t in THEME_ORDER if t in THEME_COLORS]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=7, title="theme")

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.12, label="% of row's thanked threads")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_small_multiples(user_subs: dict, path) -> None:
    sources = [
        s
        for s in SMALL_MULTIPLE_SOURCES
        if any(
            counts.most_common(1)[0][0] == s
            for counts in user_subs.values()
            if s in counts
        )
    ]

    n = len(sources)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.2 * nrows))
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, src in zip(axes_flat, sources):
        rows = off_main_destinations(user_subs, src, top_k=5)
        if not rows:
            ax.set_visible(False)
            continue
        dests, pcts, users = zip(*rows)
        y = np.arange(len(dests))
        colors = [THEME_COLORS[theme_of(d)] for d in dests]
        ax.barh(y, pcts, color=colors, edgecolor="white", height=0.65)
        ax.set_yticks(y)
        ax.set_yticklabels([f"r/{d}" for d in dests], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, max(pcts) * 1.35 if pcts else 1)
        ax.set_xlabel("% of away threads", fontsize=8)
        th = theme_of(src)
        ax.set_title(f"r/{src}\n(main sub · {th})", fontsize=9, color=THEME_COLORS.get(th, "#333"))
        for yi, pct, nu in zip(y, pcts, users):
            ax.text(pct + 0.5, yi, f"{pct:.0f}% · {nu} users", va="center", fontsize=7)

    for ax in axes_flat[len(sources):]:
        ax.set_visible(False)

    handles = [Patch(facecolor=THEME_COLORS[t], label=t) for t in THEME_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Top 5 destinations outside a user's main subreddit (row-normalized among away threads only)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_legacy_scripts() -> None:
    USER_TRENDS_DIR.mkdir(parents=True, exist_ok=True)
    pipe = Path(__file__).resolve().parent
    for name in (
        "plot_user_activity.py",
        "plot_where_users_roam.py",
        "plot_cross_sub_power_users.py",
    ):
        print(f"running {name}…", flush=True)
        subprocess.run([sys.executable, str(pipe / name)], check=True)


def cleanup_stale_outputs() -> None:
    """Remove user-trend PNGs/JSON left in plots/outputs/ (now under user_trends/)."""
    root = USER_TRENDS_DIR.parent
    stale = [
        "user_activity_distribution.png",
        "user_activity_history_summary.png",
        "user_activity_temporal_trends.png",
        "user_activity_threshold_sensitivity.png",
        "user_activity_threshold_sensitivity.json",
        "cross_sub_top100_power_users.png",
        "cross_sub_top100_power_users.json",
        "where_multi_sub_users_post.png",
    ]
    for name in stale:
        p = root / name
        if p.is_file():
            p.unlink()
            print(f"removed stale {p}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also run activity/roam/cross-sub scripts")
    args = ap.parse_args()

    USER_TRENDS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading pairs…", flush=True)
    user_subs = load_user_sub_counts()
    subs, mat = build_main_sub_flow(user_subs)
    print(f"  {len(user_subs):,} users, {len(subs)} subs in flow matrix", flush=True)

    flow_path = USER_TRENDS_DIR / "flow_matrix_by_theme.png"
    sm_path = USER_TRENDS_DIR / "roam_destinations_small_multiples.png"
    plot_flow_matrix(subs, mat, flow_path)
    plot_small_multiples(user_subs, sm_path)
    print(f"wrote {flow_path}")
    print(f"wrote {sm_path}")

    if args.all:
        run_legacy_scripts()
        cleanup_stale_outputs()
        print("done (--all)", flush=True)


if __name__ == "__main__":
    main()
