"""Per-user activity distributions and threshold sensitivity analysis.

Reads all pairs in data/extracted_current/pairs/sub-*.jsonl, optionally
joins llm_judge for divergent counts (valid QA, thanked ≠ community top).

Writes to ../../plots/outputs/user_trends/:
  user_activity_distribution.png
  user_activity_history_summary.png
  user_activity_temporal_trends.png
  user_activity_threshold_sensitivity.png
  user_activity_threshold_sensitivity.json

Usage (from may_15/reddit_pipeline/):
  python plot_user_activity.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA = HERE.parent / "data"
PAIRS_DIR = DATA / "extracted_current" / "pairs"
JUDGE_DIR = DATA / "llm_judge"
OUT_DIR = REPO / "plots" / "outputs" / "user_trends"

# Reference cutoffs (annotation labels only — not file paths)
REF_MIN_PAIRS_SHARD = 3   # per-user shard aggregation
REF_MIN_PAIRS_TRACES = 5  # heavy-user case studies


@dataclass
class UserActivity:
    user_id: str
    n_pairs: int = 0
    n_posts: int = 0
    n_subreddits: int = 0
    span_days: int = 0
    divergent_pairs: int = 0  # valid QA ∧ thanked answer ≠ community top
    held_out_k_ge_1: int = 0  # threads that would be LOO eval with k>=1
    interaction_times: list[str] = field(default_factory=list)
    subs: set[str] = field(default_factory=set)
    posts: set[str] = field(default_factory=set)


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


def load_judge() -> dict[tuple[str, str], bool]:
    out: dict[tuple[str, str], bool] = {}
    if not JUDGE_DIR.is_dir():
        return out
    for p in JUDGE_DIR.glob("sub-*.jsonl"):
        for rec in iter_jsonl(p):
            key = (rec.get("post_id"), rec.get("answer_comment_id"))
            qa = (rec.get("is_qa_pair") or {}).get("question_answer_pair")
            if key[0] and key[1] and qa is not None:
                out[key] = bool(qa)
    return out


def build_users(judge: dict) -> dict[str, UserActivity]:
    users: dict[str, UserActivity] = {}
    for pf in sorted(PAIRS_DIR.glob("sub-*.jsonl")):
        for rec in iter_jsonl(pf):
            uid = rec["user_id"]
            u = users.setdefault(uid, UserActivity(user_id=uid))
            u.n_pairs += 1
            md = rec.get("metadata") or {}
            pid = md.get("post_id")
            if pid:
                u.posts.add(pid)
            sub = rec.get("subreddit") or ""
            if sub:
                u.subs.add(sub)
            ts = rec.get("timestamp") or ""
            if ts:
                u.interaction_times.append(ts)
            key = (pid, md.get("answer_comment_id"))
            if key in judge and judge[key] and not md.get("top_equals_preferred"):
                u.divergent_pairs += 1
    for u in users.values():
        u.n_posts = len(u.posts)
        u.n_subreddits = len(u.subs)
        if len(u.interaction_times) >= 2:
            seq = sorted(u.interaction_times)
            t0 = datetime.fromisoformat(seq[0].replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(seq[-1].replace("Z", "+00:00"))
            u.span_days = max(0, (t1 - t0).days)
            u.held_out_k_ge_1 = len(seq) - 1  # LOO rows with k >= 1
    return users


def build_pairs_per_user_by_subreddit() -> dict[str, dict[str, int]]:
    """subreddit -> {user_id: count of thanked threads in that sub only}."""
    by_sub: dict[str, dict[str, int]] = defaultdict(dict)
    for pf in sorted(PAIRS_DIR.glob("sub-*.jsonl")):
        for rec in iter_jsonl(pf):
            sub = (rec.get("subreddit") or "").strip() or "(unknown)"
            uid = rec["user_id"]
            by_sub[sub][uid] = by_sub[sub].get(uid, 0) + 1
    return dict(by_sub)


def plot_history_summary(
    users: dict[str, UserActivity],
    by_sub: dict[str, dict[str, int]],
    path: Path,
    *,
    top_n_subs: int = 22,
) -> None:
    """Plain summary: thanked threads per user (overall + by subreddit)."""
    pairs_per_user = np.array([u.n_pairs for u in users.values()])
    overall_avg = float(np.mean(pairs_per_user))
    overall_mid = float(np.median(pairs_per_user))

    sub_stats = []
    for sub, uid_counts in by_sub.items():
        counts = np.array(list(uid_counts.values()), dtype=float)
        sub_stats.append({
            "subreddit": sub,
            "n_users": len(counts),
            "n_pairs": int(counts.sum()),
            "avg_per_user": float(counts.mean()),
            "mid_per_user": float(np.median(counts)),
        })
    sub_stats.sort(key=lambda r: r["n_pairs"], reverse=True)
    top = sub_stats[:top_n_subs]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), gridspec_kw={"width_ratios": [1.05, 1.35]})

    ax = axes[0]
    max_show = 12
    hist_counts = pairs_per_user[pairs_per_user <= max_show]
    bins = np.arange(0.5, max_show + 1.5, 1)
    ax.hist(hist_counts, bins=bins, color="#2563eb", edgecolor="white", alpha=0.9)
    tail = int((pairs_per_user > max_show).sum())
    if tail:
        ax.bar(max_show + 1, tail, width=0.85, color="#94a3b8", edgecolor="white",
               label=f"{tail:,} users with >{max_show} threads")
    ax.axvline(overall_avg, color="#f59e0b", lw=2, ls="--",
               label=f"average = {overall_avg:.2f}")
    ax.axvline(overall_mid, color="#ef4444", lw=2, ls=":",
               label=f"typical user = {overall_mid:.0f} threads")
    ax.set_xlabel("Thanked threads per user (all subreddits combined)")
    ax.set_ylabel("Number of users")
    ax.set_title("Overall — how much history do users have?")
    ax.set_xticks(range(1, max_show + 2))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    names = [r["subreddit"] for r in top][::-1]
    avgs = [r["avg_per_user"] for r in top][::-1]
    y = np.arange(len(names))
    bars = ax.barh(y, avgs, color="#2563eb", edgecolor="white", height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Average thanked threads per user (in that subreddit only)")
    ax.set_title(f"By subreddit — top {len(top)} by pair count")
    xmax = max(avgs) * 1.35 if avgs else 1
    ax.set_xlim(0, xmax)
    for bar, row in zip(bars, top[::-1]):
        ax.text(
            bar.get_width() + 0.03,
            bar.get_y() + bar.get_height() / 2,
            f"avg {row['avg_per_user']:.1f} · {row['n_users']:,} users",
            va="center",
            fontsize=7,
        )
    ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        f"User history in the extracted corpus — {len(users):,} users, "
        f"{int(pairs_per_user.sum()):,} thanked threads · "
        f"overall average {overall_avg:.2f} threads/user, "
        f"typical user has {overall_mid:.0f}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def sensitivity_curve(users: dict[str, UserActivity], attr: str, thresholds: list[int]) -> list[dict]:
    vals = [getattr(u, attr) for u in users.values()]
    total_users = len(users)
    total_pairs = sum(u.n_pairs for u in users.values())
    total_divergent = sum(u.divergent_pairs for u in users.values())
    rows = []
    for t in thresholds:
        kept = [u for u in users.values() if getattr(u, attr) >= t]
        div_kept = sum(u.divergent_pairs for u in kept)
        rows.append({
            "threshold": t,
            "users_kept": len(kept),
            "users_pct": 100 * len(kept) / total_users if total_users else 0,
            "pairs_kept": sum(u.n_pairs for u in kept),
            "pairs_pct": 100 * sum(u.n_pairs for u in kept) / total_pairs if total_pairs else 0,
            "divergent_kept": div_kept,
            "divergent_pct": 100 * div_kept / total_divergent if total_divergent else 0,
            "held_out_k_ge_1": sum(u.held_out_k_ge_1 for u in kept),
        })
    return rows


def plot_distribution(users: dict[str, UserActivity], path: Path) -> None:
    pairs_per_user = np.array([u.n_pairs for u in users.values()])
    subs_per_user = np.array([u.n_subreddits for u in users.values()])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    ax = axes[0]
    bins = np.unique(np.concatenate([
        [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, 30, 50, 100],
        pairs_per_user[pairs_per_user <= 100],
    ]))
    ax.hist(pairs_per_user, bins=bins, color="#2563eb", edgecolor="white", alpha=0.85)
    for name, thr in [(f"shard filter (≥{REF_MIN_PAIRS_SHARD})", REF_MIN_PAIRS_SHARD),
                      (f"heavy users (≥{REF_MIN_PAIRS_TRACES})", REF_MIN_PAIRS_TRACES)]:
        if thr <= pairs_per_user.max():
            ax.axvline(thr, color="#f59e0b", ls="--", lw=1.5)
            ax.text(thr + 0.1, ax.get_ylim()[1] * 0.92, name, fontsize=7, rotation=90, va="top")
    ax.set_xlabel("Pairs per user (thank → preferred)")
    ax.set_ylabel("Users")
    ax.set_title("Activity depth (most users = 1 pair)")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 10, 20, 50, 100])
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))

    ax = axes[1]
    sorted_p = np.sort(pairs_per_user)
    y = np.arange(1, len(sorted_p) + 1) / len(sorted_p)
    ax.plot(sorted_p, y, color="#2563eb", lw=2)
    for thr, col in [(3, "#f59e0b"), (5, "#ef4444")]:
        share = (sorted_p >= thr).mean() * 100
        ax.axvline(thr, color=col, ls="--", lw=1.2)
        ax.text(thr, 0.05, f"≥{thr}: {share:.1f}% users", fontsize=8, color=col)
    ax.set_xlabel("Pairs per user")
    ax.set_ylabel("Fraction of users")
    ax.set_title("ECDF — choosing min_pairs cutoff")
    ax.set_xscale("log")
    ax.set_xlim(0.9, max(100, sorted_p.max()))
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.hist(subs_per_user, bins=np.arange(0.5, min(12, subs_per_user.max() + 2), 1),
            color="#64748b", edgecolor="white")
    ax.set_xlabel("Distinct subreddits per user")
    ax.set_ylabel("Users")
    ax.set_title("Cross-sub breadth (aggregate uses min_subreddits)")

    med = int(np.median(pairs_per_user))
    p90 = int(np.percentile(pairs_per_user, 90))
    fig.suptitle(
        f"User activity in extracted pairs (n={len(users):,} users, "
        f"{pairs_per_user.sum():,} pairs) · median {med} pair/user, p90={p90}",
        fontsize=11,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_temporal(users: dict[str, UserActivity], path: Path) -> None:
    year_counts: Counter[int] = Counter()
    cohort_depth: dict[int, list[int]] = defaultdict(list)

    for u in users.values():
        if not u.interaction_times:
            continue
        years = []
        for ts in u.interaction_times:
            try:
                y = datetime.fromisoformat(ts.replace("Z", "+00:00")).year
            except ValueError:
                continue
            years.append(y)
            year_counts[y] += 1
        if years:
            cohort_depth[min(years)].append(u.n_pairs)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ys = sorted(year_counts)
    ax = axes[0]
    ax.bar(ys, [year_counts[y] for y in ys], color="#2563eb", edgecolor="white")
    ax.set_xlabel("Year (pair timestamp)")
    ax.set_ylabel("Pairs")
    ax.set_title("Corpus activity by year")

    ax = axes[1]
    cx = sorted(cohort_depth)
    med = [float(np.median(cohort_depth[c])) for c in cx]
    p75 = [float(np.percentile(cohort_depth[c], 75)) for c in cx]
    ax.plot(cx, med, "o-", color="#2563eb", label="median pairs/user")
    ax.plot(cx, p75, "s--", color="#94a3b8", label="p75 pairs/user")
    ax.set_xlabel("User cohort (year of first pair)")
    ax.set_ylabel("Pairs per user")
    ax.set_title("Depth by first-seen cohort")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.35)

    fig.suptitle("Per-user activity trends over time", fontsize=11)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity(pairs_rows: list[dict], path: Path) -> None:
    """Single chart: retention vs min pairs, including divergent (personalization) rows."""
    xs = [r["threshold"] for r in pairs_rows]
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(xs, [r["users_pct"] for r in pairs_rows], "o-", color="#2563eb", lw=2.2,
            markersize=7, label="% users retained")
    ax.plot(xs, [r["pairs_pct"] for r in pairs_rows], "s--", color="#94a3b8", lw=1.8,
            markersize=6, label="% all pairs retained")
    ax.plot(xs, [r["divergent_pct"] for r in pairs_rows], "^-", color="#7c3aed", lw=2,
            markersize=7, label="% divergent pairs retained")

    for thr, label, color in [
        (REF_MIN_PAIRS_SHARD, f"≥{REF_MIN_PAIRS_SHARD} pairs (shard aggregation)", "#f59e0b"),
        (REF_MIN_PAIRS_TRACES, f"≥{REF_MIN_PAIRS_TRACES} pairs (heavy-user traces)", "#ef4444"),
    ]:
        if thr <= max(xs):
            ax.axvline(thr, color=color, ls=":", lw=1.4, alpha=0.85)
            row = next(r for r in pairs_rows if r["threshold"] == thr)
            ax.annotate(
                f"{label}\n{row['users_pct']:.0f}% users · "
                f"{row['divergent_pct']:.0f}% divergent",
                xy=(thr, row["divergent_pct"]),
                xytext=(thr + 0.8, min(98, row["divergent_pct"] + 12)),
                fontsize=8,
                color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
            )

    ax.set_xlabel("Minimum thanked threads per user")
    ax.set_ylabel("% of corpus retained")
    ax.set_ylim(0, 105)
    ax.set_xticks(xs)
    ax.legend(loc="center right", fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.35)

    r1 = pairs_rows[0]
    fig.suptitle(
        "Activity cutoff sensitivity — divergent = judge-valid QA and "
        "OP thanked ≠ community top",
        fontsize=11,
    )
    ax.set_title(
        f"At cutoff 1: {r1['divergent_kept']:,} divergent pairs "
        f"({r1['divergent_pct']:.0f}% of judged divergent corpus)",
        fontsize=9,
        pad=8,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print("Loading judge sidecars…")
    judge = load_judge()
    print("Scanning pairs…")
    users = build_users(judge)
    print(f"  {len(users):,} users")
    print("Counting per-subreddit history…")
    by_sub = build_pairs_per_user_by_subreddit()
    print(f"  {len(by_sub):,} subreddits")

    pair_thresholds = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
    sub_thresholds = [1, 2, 3, 4, 5]

    pairs_rows = sensitivity_curve(users, "n_pairs", pair_thresholds)
    subs_rows = sensitivity_curve(users, "n_subreddits", sub_thresholds)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dist_path = OUT_DIR / "user_activity_distribution.png"
    history_path = OUT_DIR / "user_activity_history_summary.png"
    sens_path = OUT_DIR / "user_activity_threshold_sensitivity.png"
    json_path = OUT_DIR / "user_activity_threshold_sensitivity.json"

    temporal_path = OUT_DIR / "user_activity_temporal_trends.png"
    plot_distribution(users, dist_path)
    plot_history_summary(users, by_sub, history_path)
    plot_temporal(users, temporal_path)
    plot_sensitivity(pairs_rows, sens_path)

    payload = {
        "n_users": len(users),
        "n_pairs": sum(u.n_pairs for u in users.values()),
        "n_divergent_pairs": sum(u.divergent_pairs for u in users.values()),
        "divergent_definition": "is_qa_pair true and top_equals_preferred false",
        "reference_cutoffs": {
            "min_pairs_shard_aggregation": REF_MIN_PAIRS_SHARD,
            "min_pairs_heavy_user_traces": REF_MIN_PAIRS_TRACES,
        },
        "min_pairs_sensitivity": pairs_rows,
        "min_subreddits_sensitivity": subs_rows,
        "pairs_per_user_percentiles": {
            "p50": float(np.median([u.n_pairs for u in users.values()])),
            "p75": float(np.percentile([u.n_pairs for u in users.values()], 75)),
            "p90": float(np.percentile([u.n_pairs for u in users.values()], 90)),
            "p99": float(np.percentile([u.n_pairs for u in users.values()], 99)),
            "max": max(u.n_pairs for u in users.values()),
        },
        "overall_avg_pairs_per_user": float(np.mean([u.n_pairs for u in users.values()])),
        "top_subreddits_by_avg_pairs_per_user": sorted(
            [
                {
                    "subreddit": r["subreddit"],
                    "avg_pairs_per_user": r["avg_per_user"],
                    "n_users": r["n_users"],
                    "n_pairs": r["n_pairs"],
                }
                for r in sorted(
                    (
                        {
                            "subreddit": sub,
                            "avg_per_user": float(np.mean(list(uc.values()))),
                            "n_users": len(uc),
                            "n_pairs": sum(uc.values()),
                        }
                        for sub, uc in by_sub.items()
                    ),
                    key=lambda x: x["n_pairs"],
                    reverse=True,
                )[:25]
            ],
            key=lambda x: x["avg_pairs_per_user"],
            reverse=True,
        ),
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"wrote {dist_path}")
    print(f"wrote {history_path}")
    print(f"wrote {temporal_path}")
    print(f"wrote {sens_path}")
    print(f"wrote {json_path}")
    for r in pairs_rows:
        if r["threshold"] in (3, 5):
            print(f"  min_pairs>={r['threshold']}: {r['users_kept']:,} users ({r['users_pct']:.1f}%), "
                  f"{r['pairs_kept']:,} pairs ({r['pairs_pct']:.1f}%)")


if __name__ == "__main__":
    main()
