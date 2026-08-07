"""Where do each sub's top ~100 most active users also post?

For every subreddit, take the ~100 users with the most thanked threads IN that
sub. Measure how much of their full history lives on OTHER subs.

Writes: ../../plots/outputs/user_trends/cross_sub_top100_power_users.png
        ../../plots/outputs/user_trends/cross_sub_top100_power_users.json

Usage (from may_15/reddit_pipeline/):
  python plot_cross_sub_power_users.py
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
OUT_DIR = REPO / "plots" / "outputs" / "user_trends"

TOP_K = 100
HEATMAP_SUBS = 14  # largest subs for destination heatmap


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_activity() -> tuple[dict[str, dict[str, int]], dict[str, Counter[str]]]:
    """sub -> {user: pairs in sub}; user -> Counter(sub -> pairs)."""
    by_sub_user: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    user_subs: dict[str, Counter[str]] = defaultdict(Counter)
    for fp in sorted(PAIRS_DIR.glob("sub-*.jsonl")):
        for rec in iter_jsonl(fp):
            sub = (rec.get("subreddit") or "").strip()
            if not sub:
                continue
            uid = rec["user_id"]
            by_sub_user[sub][uid] += 1
            user_subs[uid][sub] += 1
    return dict(by_sub_user), dict(user_subs)


def cohort_metrics(
    home_sub: str,
    by_sub_user: dict[str, dict[str, int]],
    user_subs: dict[str, Counter[str]],
) -> dict | None:
    users_in_sub = by_sub_user.get(home_sub)
    if not users_in_sub:
        return None
    ranked = sorted(users_in_sub.items(), key=lambda x: -x[1])
    top = ranked[: min(TOP_K, len(ranked))]
    if not top:
        return None

    frac_other_list: list[float] = []
    n_other_subs_list: list[int] = []
    dest_counter: Counter[str] = Counter()
    home_pairs = 0
    other_pairs = 0

    for uid, home_count in top:
        all_counts = user_subs[uid]
        total = sum(all_counts.values())
        other = total - all_counts.get(home_sub, 0)
        home_pairs += all_counts.get(home_sub, 0)
        other_pairs += other
        frac_other_list.append(other / total if total else 0.0)
        n_other_subs_list.append(sum(1 for s, c in all_counts.items() if s != home_sub and c > 0))
        for s, c in all_counts.items():
            if s != home_sub:
                dest_counter[s] += c

    total_pairs = home_pairs + other_pairs
    return {
        "subreddit": home_sub,
        "cohort_size": len(top),
        "min_pairs_in_home_to_enter_top": top[-1][1] if top else 0,
        "max_pairs_in_home": top[0][1] if top else 0,
        "mean_frac_pairs_on_other_subs": float(np.mean(frac_other_list)),
        "median_frac_pairs_on_other_subs": float(np.median(frac_other_list)),
        "mean_other_subreddits_active": float(np.mean(n_other_subs_list)),
        "pct_users_with_any_other_sub": 100.0 * sum(1 for x in n_other_subs_list if x > 0) / len(top),
        "frac_all_pairs_on_other_subs": other_pairs / total_pairs if total_pairs else 0.0,
        "top_other_destinations": [
            {"subreddit": s, "pairs": int(c), "share_of_other_pairs": c / other_pairs if other_pairs else 0}
            for s, c in dest_counter.most_common(8)
        ],
    }


def build_flow_matrix(
    subs_order: list[str],
    by_sub_user: dict[str, dict[str, int]],
    user_subs: dict[str, Counter[str]],
) -> np.ndarray:
    """rows=home sub cohort, cols=dest sub; cell = share of cohort's total pairs in col."""
    n = len(subs_order)
    mat = np.zeros((n, n))
    sub_to_i = {s: i for i, s in enumerate(subs_order)}
    for i, home in enumerate(subs_order):
        ranked = sorted(by_sub_user[home].items(), key=lambda x: -x[1])[:TOP_K]
        cohort_total = 0
        col_counts = Counter()
        for uid, _ in ranked:
            for s, c in user_subs[uid].items():
                col_counts[s] += c
                cohort_total += c
        if cohort_total:
            for s, c in col_counts.items():
                j = sub_to_i.get(s)
                if j is not None:
                    mat[i, j] = c / cohort_total
    return mat


def plot(
    rows: list[dict],
    flow: np.ndarray,
    subs_order: list[str],
    path: Path,
) -> None:
    rows_sorted = sorted(rows, key=lambda r: -r["mean_frac_pairs_on_other_subs"])
    names = [r["subreddit"] for r in rows_sorted]
    fracs = [100 * r["mean_frac_pairs_on_other_subs"] for r in rows_sorted]
    n_other = [r["mean_other_subreddits_active"] for r in rows_sorted]
    pct_cross = [r["pct_users_with_any_other_sub"] for r in rows_sorted]

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], width_ratios=[1.1, 1],
                         hspace=0.38, wspace=0.28)

    # --- Panel A: which subs' power users roam most
    ax0 = fig.add_subplot(gs[0, :])
    y = np.arange(len(names))
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(names)))
    bars = ax0.barh(y, fracs, color=colors, edgecolor="white", height=0.72)
    ax0.set_yticks(y)
    ax0.set_yticklabels(names, fontsize=8)
    ax0.set_xlabel("% of each user's thanked threads that are NOT in this subreddit")
    ax0.set_title(
        f"Cross-subreddit activity among each sub's top {TOP_K} users "
        f"(ranked by activity in that sub only)",
        fontsize=11,
    )
    ax0.invert_yaxis()
    ax0.grid(axis="x", alpha=0.3)
    for bar, no, pc in zip(bars, n_other, pct_cross):
        ax0.text(
            bar.get_width() + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"avg {no:.1f} other subs · {pc:.0f}% use another sub",
            va="center",
            fontsize=6.5,
        )

    # --- Panel B: heatmap — where do their threads go (including home on diagonal)
    ax1 = fig.add_subplot(gs[1, 0])
    big_subs = subs_order
    im = ax1.imshow(flow, cmap="YlOrRd", aspect="auto", vmin=0, vmax=max(0.35, flow.max()))
    ax1.set_xticks(range(len(big_subs)))
    ax1.set_yticks(range(len(big_subs)))
    ax1.set_xticklabels(big_subs, rotation=55, ha="right", fontsize=7)
    ax1.set_yticklabels(big_subs, fontsize=7)
    ax1.set_xlabel("Where thanked threads fall")
    ax1.set_ylabel("Home sub (top 100 users here)")
    ax1.set_title("Share of each cohort's thanked threads by subreddit", fontsize=10)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="share of cohort pairs")

    # --- Panel C: off-diagonal only — secondary homes
    ax2 = fig.add_subplot(gs[1, 1])
    off = flow.copy()
    np.fill_diagonal(off, 0.0)
    row_sums = off.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    off_norm = off / row_sums  # among non-home pairs, where do they go?
    im2 = ax2.imshow(off_norm, cmap="BuPu", aspect="auto", vmin=0)
    ax2.set_xticks(range(len(big_subs)))
    ax2.set_yticks(range(len(big_subs)))
    ax2.set_xticklabels(big_subs, rotation=55, ha="right", fontsize=7)
    ax2.set_yticklabels(big_subs, fontsize=7)
    ax2.set_xlabel("Other subreddit")
    ax2.set_ylabel("Home sub")
    ax2.set_title("Among threads outside home sub — where they go", fontsize=10)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="share of non-home pairs")

    top3 = rows_sorted[:3]
    bot3 = rows_sorted[-3:]
    fig.suptitle(
        "Power users who roam the most: "
        + ", ".join(f"r/{r['subreddit']} ({100*r['mean_frac_pairs_on_other_subs']:.0f}% elsewhere)"
                    for r in top3)
        + "  ·  "
        "Most stay-at-home: "
        + ", ".join(f"r/{r['subreddit']} ({100*r['mean_frac_pairs_on_other_subs']:.0f}%)"
                    for r in bot3),
        fontsize=10,
        y=1.02,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("Loading pairs…", flush=True)
    by_sub_user, user_subs = load_activity()
    print(f"  {len(by_sub_user)} subreddits", flush=True)

    rows = []
    for sub in sorted(by_sub_user, key=lambda s: -sum(by_sub_user[s].values())):
        m = cohort_metrics(sub, by_sub_user, user_subs)
        if m:
            rows.append(m)

    rows_by_roam = sorted(rows, key=lambda r: -r["mean_frac_pairs_on_other_subs"])
    subs_by_volume = sorted(by_sub_user, key=lambda s: -sum(by_sub_user[s].values()))
    heat_subs = subs_by_volume[:HEATMAP_SUBS]
    flow = build_flow_matrix(heat_subs, by_sub_user, user_subs)

    out_png = OUT_DIR / "cross_sub_top100_power_users.png"
    out_json = OUT_DIR / "cross_sub_top100_power_users.json"
    plot(rows, flow, heat_subs, out_png)

    payload = {
        "top_k_per_sub": TOP_K,
        "definition": "Top K users by thanked-thread count within each sub; metrics use their full cross-sub history.",
        "most_roaming_top100_cohorts": rows_by_roam[:8],
        "most_homebound_top100_cohorts": rows_by_roam[-8:],
        "all_subreddits": rows_by_roam,
        "heatmap_subs": heat_subs,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"wrote {out_png}")
    print(f"wrote {out_json}")
    print("\nMost active on OTHER subreddits (top 100 cohort, mean % pairs elsewhere):")
    for r in rows_by_roam[:6]:
        dest = r["top_other_destinations"][0]["subreddit"] if r["top_other_destinations"] else "—"
        print(f"  r/{r['subreddit']:20s}  {100*r['mean_frac_pairs_on_other_subs']:5.1f}%  "
              f"avg {r['mean_other_subreddits_active']:.1f} other subs  "
              f"top elsewhere: r/{dest}")
    print("\nMost homebound:")
    for r in rows_by_roam[-5:]:
        print(f"  r/{r['subreddit']:20s}  {100*r['mean_frac_pairs_on_other_subs']:5.1f}%")


if __name__ == "__main__":
    main()
