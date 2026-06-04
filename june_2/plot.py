"""Plots for the june_2 experiment.

  exp_A_similarity.png : for each similarity metric (embedding cosine + LLM 1-5
      rubric), how close gpt-5-chat's generated answer is to the user's
      preferred_answer vs the top_comment.
  exp_B_choice.png : when gpt-5-chat must pick between the two answers, how often
      it picks the user's preferred_answer vs the top_comment (vs 50% baseline).

Also prints a numeric summary.
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RESULTS_PATH = HERE / "results" / "results.jsonl"
SAMPLES_PATH = HERE / "data" / "samples.jsonl"
PAIRS_DIR = REPO / "may_15" / "data" / "extracted_current" / "pairs"
JUDGE_DIR = REPO / "may_15" / "data" / "llm_judge"
PLOTS_DIR = HERE / "plots"

PREF_C = "#2a9d8f"  # preferred
TOP_C = "#e76f51"   # top-rated


def _qa_verdicts():
    """(post_id, answer_comment_id) -> is_qa_pair bool, from the LLM-judge sidecars."""
    verdict = {}
    for fp in glob.glob(str(JUDGE_DIR / "sub-*.jsonl")):
        with open(fp) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                q = d.get("is_qa_pair") or {}
                if "question_answer_pair" in q:
                    verdict[(d.get("post_id"), d.get("answer_comment_id"))] = q[
                        "question_answer_pair"
                    ]
    return verdict


def load(qa_only=False):
    rows = [json.loads(l) for l in open(RESULTS_PATH)]
    if not qa_only:
        return rows
    samples = {json.loads(l)["sample_id"]: json.loads(l) for l in open(SAMPLES_PATH)}
    verdict = _qa_verdicts()
    kept, judged = [], 0
    for r in rows:
        s = samples.get(r["sample_id"])
        if not s:
            continue
        v = verdict.get((s["post_id"], s["answer_comment_id"]))
        if v is None:
            continue
        judged += 1
        if v is True:
            kept.append(r)
    print(f"qa-filter: {judged}/{len(rows)} samples judged, {len(kept)} are is_qa_pair=True")
    return kept


def _scatter(ax, pref, top, lo, hi, title, unit):
    ax.scatter(pref, top, alpha=0.5, color="C0", edgecolor="none", s=55, zorder=3)
    ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1, zorder=1)
    win = np.mean(np.array(pref) > np.array(top)) * 100  # closer to preferred
    ax.set_xlabel(f"similarity to PREFERRED answer ({unit})")
    ax.set_ylabel(f"similarity to TOP-rated answer ({unit})")
    ax.set_title(
        f"{title}\nmean pref={np.mean(pref):.2f}  top={np.mean(top):.2f}   "
        f"closer-to-preferred: {win:.0f}% of samples",
        fontsize=10,
    )
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25, zorder=0)
    # shade region where gen is closer to preferred (below diagonal)
    ax.fill_between([lo, hi], [lo, lo], [lo, hi], color=PREF_C, alpha=0.05, zorder=0)


def _judge_region_ellipses(ax):
    """Highlight below-diagonal (preferred) vs above-diagonal (top) judge clusters."""
    ax.add_patch(
        Ellipse(
            (3.95, 2.4),
            width=2.4,
            height=2.0,
            fill=False,
            edgecolor=PREF_C,
            linewidth=2.2,
            zorder=5,
        )
    )
    ax.add_patch(
        Ellipse(
            (2.85, 4.35),
            width=2.6,
            height=2.2,
            fill=False,
            edgecolor=TOP_C,
            linewidth=2.2,
            zorder=5,
        )
    )


def plot_exp_a(rows, suffix="", title_tag=""):
    cos_p = [r["cos_preferred"] for r in rows]
    cos_t = [r["cos_top"] for r in rows]
    jr = [r for r in rows if r["judge_sim_preferred"] is not None
          and r["judge_sim_top"] is not None]
    j_p = [r["judge_sim_preferred"] for r in jr]
    j_t = [r["judge_sim_top"] for r in jr]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), gridspec_kw={"wspace": 0.22})
    _scatter(axes[0], cos_p, cos_t, 0.0, 1.0,
             "Exp A — embedding cosine similarity", "cosine")
    # jitter the discrete 1-5 judge scores so overlapping points are visible
    rng = np.random.default_rng(0)
    jx = np.array(j_p) + rng.uniform(-0.12, 0.12, len(j_p))
    jy = np.array(j_t) + rng.uniform(-0.12, 0.12, len(j_t))
    _scatter(axes[1], jx, jy, 0.5, 5.5,
             "Exp A — LLM judge similarity (1-5 rubric, jittered)", "1-5")
    axes[1].set_xticks(range(1, 6)); axes[1].set_yticks(range(1, 6))
    _judge_region_ellipses(axes[1])

    fig.suptitle(
        "Does the generated answer resemble what the user PREFERRED or the TOP-rated reply?"
        f"{title_tag}\n"
        "Points below the dashed line = generated answer is closer to the user's preferred answer",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.86], pad=0.15)
    fig.subplots_adjust(left=0.04, right=0.995, top=0.84, bottom=0.14, wspace=0.22)
    out = PLOTS_DIR / f"exp_A_similarity{suffix}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return out, (cos_p, cos_t, j_p, j_t)


def plot_exp_b(rows, suffix="", title_tag=""):
    cr = [r for r in rows if r["choice_picked_preferred"] is not None]
    n = len(cr)
    n_pref = sum(1 for r in cr if r["choice_picked_preferred"])
    pct_pref = 100 * n_pref / n
    pct_top = 100 - pct_pref
    # position-bias diagnostic: how often did it pick whichever was shown as "A"
    picked_A = sum(
        1 for r in cr
        if r["choice_picked_preferred"] == r["choice_preferred_was_A"]
    )
    pct_A = 100 * picked_A / n

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [2, 1]})
    bars = ax.bar(
        ["picked\nPREFERRED", "picked\nTOP-rated"], [pct_pref, pct_top],
        color=[PREF_C, TOP_C], edgecolor="black", width=0.6,
    )
    ax.axhline(50, ls="--", color="gray", lw=1)
    ax.text(1.45, 51, "50% chance", color="gray", fontsize=9)
    for b, v in zip(bars, [pct_pref, pct_top]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%",
                ha="center", fontweight="bold")
    ax.set_ylabel("% of samples")
    ax.set_ylim(0, 100)
    ax.set_title(f"Exp B — forced choice (n={n})")

    # position-bias bar
    pb = ax2.bar(["picked\nposition A"], [pct_A], color="#577590",
                 edgecolor="black", width=0.5)
    ax2.axhline(50, ls="--", color="gray", lw=1)
    ax2.text(pb[0].get_x() + pb[0].get_width() / 2, pct_A + 1.5, f"{pct_A:.0f}%",
             ha="center", fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.set_title("position-bias check\n(answers shown in random order)", fontsize=9)

    fig.suptitle("When forced to choose, which answer does gpt-5-chat pick for this user?"
                 f"{title_tag}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = PLOTS_DIR / f"exp_B_choice{suffix}.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    return out, (n, pct_pref, pct_A)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa-only", action="store_true",
                    help="only plot samples the LLM-judge marked is_qa_pair=True")
    args = ap.parse_args()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = load(qa_only=args.qa_only)
    suffix = "_qa" if args.qa_only else ""
    title_tag = "  (is_qa_pair=True only)" if args.qa_only else ""
    print(f"loaded {len(rows)} results{' (qa-only)' if args.qa_only else ''}\n")

    a_out, (cp, ct, jp, jt) = plot_exp_a(rows, suffix, title_tag)
    print("=== Exp A: generated-answer similarity ===")
    print(f"  embedding cosine : mean(pref)={np.mean(cp):.3f}  mean(top)={np.mean(ct):.3f}  "
          f"closer-to-preferred {100*np.mean(np.array(cp)>np.array(ct)):.0f}%")
    if jp:
        print(f"  LLM judge (1-5)  : mean(pref)={np.mean(jp):.2f}  mean(top)={np.mean(jt):.2f}  "
              f"closer-to-preferred {100*np.mean(np.array(jp)>np.array(jt)):.0f}%")
    print(f"  -> {a_out}\n")

    b_out, (n, pct_pref, pct_A) = plot_exp_b(rows, suffix, title_tag)
    print("=== Exp B: forced choice ===")
    print(f"  n={n}  picked PREFERRED {pct_pref:.0f}%  picked TOP {100-pct_pref:.0f}%")
    print(f"  position-bias: picked 'A' {pct_A:.0f}% (≈50% = unbiased)")
    print(f"  -> {b_out}")


if __name__ == "__main__":
    main()
