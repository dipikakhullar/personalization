"""Single combined figure: the 2x2 design as 3 panels (Choose, Generate-cosine,
Generate-judge). In every panel each model has a WITH-history bar and a
NO-history bar side by side. Dashed line = 50% (no-signal baseline).

Clean pilot data (results_clean/). Output: plots/cells/combined.png
"""
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
WITH_DIR, NO_DIR = "results_clean/with_history", "results_clean/no_user_context"
WITH_C, NO_C = "#3a6ea5", "#bdbdbd"


def load(d):
    out = {}
    for fp in sorted(glob.glob(str(HERE / d / "results_*.jsonl"))):
        if "_old_" in os.path.basename(fp):
            continue
        rows = [json.loads(l) for l in open(fp)]
        if rows:
            out[rows[0].get("model", os.path.basename(fp)).split("/")[-1]] = rows
    return out


def cos_win(rows):
    return 100 * np.mean([r["cos_preferred"] > r["cos_top"] for r in rows])


def avg_cos_pref(rows):
    return float(np.mean([r["cos_preferred"] for r in rows]))


def avg_cos_top(rows):
    return float(np.mean([r["cos_top"] for r in rows]))


def jud_win(rows):
    jr = [r for r in rows if r["judge_sim_preferred"] is not None]
    return 100 * np.mean([r["judge_sim_preferred"] > r["judge_sim_top"] for r in jr])


def pick(rows):
    cr = [r for r in rows if r["choice_picked_preferred"] is not None]
    return 100 * np.mean([r["choice_picked_preferred"] for r in cr])


def main():
    wh, nh = load(WITH_DIR), load(NO_DIR)
    models = [m for m in wh if m in nh]
    n = len(next(iter(wh.values())))

    x = np.arange(len(models)); w = 0.38
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.8))

    # --- Panels 1 & 3: percentage metrics (0-100, dashed 50% chance) ---
    pct_panels = [
        (axes[0], "TASK 1 — CHOOSE\nShown both answers, which does the model pick?\n"
         "% that picked the user's PREFERRED answer", pick),
        (axes[2], "TASK 2 — GENERATE (LLM judge 1-5)\nGenerated answer scored by a judge;\n"
         "% closer to PREFERRED than to top", jud_win),
    ]
    for ax, title, fn in pct_panels:
        b1 = ax.bar(x - w/2, [fn(wh[m]) for m in models], w,
                    label="WITH user history", color=WITH_C, edgecolor="black")
        b2 = ax.bar(x + w/2, [fn(nh[m]) for m in models], w,
                    label="NO user context", color=NO_C, edgecolor="black")
        ax.axhline(50, ls="--", color="gray", lw=1)
        ax.text(len(models) - 0.5, 51.5, "50% = chance", fontsize=7, color="gray", ha="right")
        for b in [*b1, *b2]:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                    f"{b.get_height():.0f}", ha="center", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_ylabel("% of samples favoring the user's PREFERRED answer")
        ax.set_title(title, fontsize=9.5); ax.legend(fontsize=8, loc="upper right")

    # --- Panel 2: average cosine similarity of the generated answer to PREFERRED
    #     (bars), with similarity to TOP shown as a reference marker ---
    ax = axes[1]
    b1 = ax.bar(x - w/2, [avg_cos_pref(wh[m]) for m in models], w,
                label="WITH user history", color=WITH_C, edgecolor="black")
    b2 = ax.bar(x + w/2, [avg_cos_pref(nh[m]) for m in models], w,
                label="NO user context", color=NO_C, edgecolor="black")
    # reference: avg cosine to TOP answer (horizontal ticks on each bar)
    for cond, off in [(wh, -w/2), (nh, +w/2)]:
        tops = [avg_cos_top(cond[m]) for m in models]
        ax.scatter(x + off, tops, marker="_", s=420, color="#e76f51", zorder=5,
                   linewidths=2.2)
    ax.scatter([], [], marker="_", s=200, color="#e76f51", linewidths=2.2,
               label="avg cosine to TOP answer")
    for b in [*b1, *b2]:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.008,
                f"{b.get_height():.2f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("avg cosine similarity of generated answer to PREFERRED")
    ax.set_title("TASK 2 — GENERATE (embedding cosine)\nHow similar is the model's own answer to the\n"
                 "user's PREFERRED answer? (bar) vs the TOP answer (—)", fontsize=9.5)
    ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Can LLMs recover the answer a Reddit user actually preferred — "
                 "and does that user's history help?", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.905,
             "Each sample: a Reddit question with two real answers — PREFERRED (the one the "
             "original poster thanked) vs TOP (the highest-voted community reply). "
             "Blue = model is given the user's 3 prior (question, preferred-answer) pairs as "
             f"context;  Gray = no user context.  Matched n={n} samples per model "
             "(genuine Q&A pairs, distinct posts). Bars above 50% favor the user's preferred "
             "answer; blue≈gray means history doesn't change the outcome.",
             ha="center", va="top", fontsize=9, wrap=True)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    out = HERE / "plots" / "cells" / "cleanpilot_summary_bars.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
