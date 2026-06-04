"""Exp A similarity scatters as one paneled figure (clean pilot), per metric.

One panel per model. Each point is a sample: x = similarity of the model's
GENERATED answer to the user's PREFERRED answer, y = similarity to the community
TOP answer. Points BELOW the diagonal => generation is closer to the preferred
answer. WITH-history (blue) and NO-context (orange) overlaid, big marker = centroid.

  python3 plot_scatters.py                 # cosine  -> plots/cells/scatters.png
  python3 plot_scatters.py --metric judge  # 1-5 judge -> plots/cells/scatters_judge.png
"""
import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import Ellipse
import numpy as np


def confidence_ellipse(x, y, ax, n_std=1.5, **kw):
    """Covariance ellipse (n_std) over the point cloud, centered at its mean."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    rx, ry = np.sqrt(1 + pearson), np.sqrt(1 - pearson)
    ell = Ellipse((0, 0), width=2 * rx, height=2 * ry, **kw)
    transf = (mtransforms.Affine2D()
              .rotate_deg(45)
              .scale(np.sqrt(cov[0, 0]) * n_std, np.sqrt(cov[1, 1]) * n_std)
              .translate(x.mean(), y.mean()))
    ell.set_transform(transf + ax.transData)
    return ax.add_patch(ell)

HERE = Path(__file__).resolve().parent
WITH_DIR, NO_DIR = "results_clean/with_history", "results_clean/no_user_context"
WITH_C, NO_C = "C0", "C1"          # point clouds: matplotlib default blue / orange
MED_WITH, MED_NO = "navy", "#cc3311"  # median dots: darker same-family (navy / red-orange)

METRICS = {
    "cosine": dict(pkey="cos_preferred", tkey="cos_top", lo=0.0, hi=1.0, jit=0.0,
                   central="mean",
                   out="cleanpilot_scatter_cosine.png", unit="cosine similarity",
                   sub="x = cosine similarity of the generated answer to the PREFERRED answer "
                       "(the one the original poster thanked); y = cosine to the TOP "
                       "(highest-voted) answer."),
    "judge": dict(pkey="judge_sim_preferred", tkey="judge_sim_top", lo=0.5, hi=5.5, jit=0.13,
                  central="median",
                  out="cleanpilot_scatter_judge.png", unit="LLM judge similarity (1-5)",
                  sub="x = LLM-judge similarity (1-5 rubric) of the generated answer to the "
                      "PREFERRED answer; y = judge similarity to the TOP (highest-voted) answer. "
                      "Points are jittered (scores are integers 1-5)."),
}


def load(d):
    out = {}
    for fp in sorted(glob.glob(str(HERE / d / "results_*.jsonl"))):
        if "_old_" in os.path.basename(fp):
            continue
        rows = [json.loads(l) for l in open(fp)]
        if rows:
            out[rows[0].get("model", os.path.basename(fp)).split("/")[-1]] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=list(METRICS), default="cosine")
    args = ap.parse_args()
    M = METRICS[args.metric]

    wh, nh = load(WITH_DIR), load(NO_DIR)
    models = [m for m in wh if m in nh]
    n = len(next(iter(wh.values())))
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, len(models), figsize=(4.1 * len(models), 5.0))
    for ax, m in zip(axes, models):
        for rows, c, mc, lbl in [(wh[m], WITH_C, MED_WITH, "WITH history"),
                                 (nh[m], NO_C, MED_NO, "NO context")]:
            pts = [(r[M["pkey"]], r[M["tkey"]]) for r in rows
                   if r[M["pkey"]] is not None and r[M["tkey"]] is not None]
            xp = np.array([p[0] for p in pts], float)
            yt = np.array([p[1] for p in pts], float)
            jx = xp + rng.uniform(-M["jit"], M["jit"], len(xp))
            jy = yt + rng.uniform(-M["jit"], M["jit"], len(yt))
            ax.scatter(jx, jy, s=14, alpha=0.6, color=c, edgecolor="none", zorder=2,
                       label=lbl)
            confidence_ellipse(xp, yt, ax, n_std=1.5, facecolor=c, alpha=0.12, zorder=3)
            confidence_ellipse(xp, yt, ax, n_std=1.5, edgecolor=c, facecolor="none",
                               linewidth=2.0, zorder=4)
            cfn = np.median if M["central"] == "median" else np.mean
            cx, cy = float(cfn(xp)), float(cfn(yt))
            ax.scatter(cx, cy, s=80, color=mc, edgecolor="none",
                       marker="o", alpha=1.0, zorder=6, label=f"{lbl} ({M['central']})")
            # annotate the dot with its label + (x, y) via a thin black leader line
            coord = (f"({cx:.1f}, {cy:.1f})" if M["central"] == "median"
                     else f"({cx:.2f}, {cy:.2f})")
            fmt = f"{lbl} ({M['central']})\n{coord}"
            off = (-52, 34) if lbl == "WITH history" else (40, -42)
            ax.annotate(fmt, (cx, cy), textcoords="offset points", xytext=off,
                        fontsize=6.5, color="black", ha="center", va="center", zorder=7,
                        arrowprops=dict(arrowstyle="-", color="black", lw=0.7,
                                        shrinkA=0, shrinkB=3),
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))
        ax.plot([M["lo"], M["hi"]], [M["lo"], M["hi"]], "--", color="gray", lw=1, zorder=1)
        ax.fill_between([M["lo"], M["hi"]], [M["lo"], M["lo"]], [M["lo"], M["hi"]],
                        color="#2a9d8f", alpha=0.05, zorder=0)
        ax.set_xlim(M["lo"], M["hi"]); ax.set_ylim(M["lo"], M["hi"])
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(m, fontsize=10, fontweight="bold")
        ax.set_xlabel(f"sim to PREFERRED ({M['unit']})", fontsize=8)
        ax.grid(alpha=0.2)
        if args.metric == "judge":
            ax.set_xticks(range(1, 6)); ax.set_yticks(range(1, 6))
    axes[0].set_ylabel(f"sim to TOP community answer ({M['unit']})", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Where does each model's GENERATED answer land — closer to the user's "
                 "PREFERRED answer or the community TOP answer?", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.905,
             f"One point per Reddit Q&A sample. {M['sub']} Points BELOW the diagonal (shaded) are "
             f"closer to the user's preferred answer. Blue = model given the user's history; "
             f"Orange = no context. Big dots = per-condition {M['central']}; rings = 1.5-SD "
             f"covariance ellipses over each cloud. n={n} per model.",
             ha="center", va="top", fontsize=8.5, wrap=True)
    fig.tight_layout(rect=[0, 0.12, 1, 0.86])
    out = HERE / "plots" / "cells" / M["out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
