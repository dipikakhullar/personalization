"""Per-model summary plots for the multi-model june_2 experiment.

For each results_<slug>.jsonl in a results dir, produce ONE figure with three
panels:
  1. embedding cosine similarity scatter (gen vs preferred / vs top)
  2. LLM-judge 1-5 similarity scatter (jittered)
  3. forced-choice bar (% picked preferred vs top) + position-bias check

Usage:
  python3 plot_models.py                              # with-history results
  python3 plot_models.py --dir no_user_context --out plots/no_user_context --tag "no history"
"""
import argparse
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
PREF_C = "#2a9d8f"
TOP_C = "#e76f51"


def _scatter(ax, pref, top, lo, hi, unit, title):
    ax.scatter(pref, top, alpha=0.5, color="C0", edgecolor="none", s=40, zorder=3)
    ax.plot([lo, hi], [lo, hi], "--", color="gray", lw=1)
    win = np.mean(np.array(pref) > np.array(top)) * 100
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"sim to PREFERRED ({unit})")
    ax.set_ylabel(f"sim to TOP-rated ({unit})")
    ax.set_title(f"{title}\nmean {np.mean(pref):.2f} vs {np.mean(top):.2f} | "
                 f"closer-to-pref {win:.0f}%", fontsize=9)
    ax.grid(alpha=0.25)


def plot_one(rows, model, out_path, tag):
    cos_p = [r["cos_preferred"] for r in rows]
    cos_t = [r["cos_top"] for r in rows]
    jr = [r for r in rows if r["judge_sim_preferred"] is not None
          and r["judge_sim_top"] is not None]
    j_p = [r["judge_sim_preferred"] for r in jr]
    j_t = [r["judge_sim_top"] for r in jr]
    cr = [r for r in rows if r["choice_picked_preferred"] is not None]
    n_pref = sum(1 for r in cr if r["choice_picked_preferred"])
    pct_pref = 100 * n_pref / len(cr) if cr else 0
    picked_A = sum(1 for r in cr if r["choice_picked_preferred"] == r["choice_preferred_was_A"])
    pct_A = 100 * picked_A / len(cr) if cr else 0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    _scatter(axes[0], cos_p, cos_t, 0.0, 1.0, "cosine", "Exp A — embedding cosine")
    rng = np.random.default_rng(0)
    jx = np.array(j_p) + rng.uniform(-0.12, 0.12, len(j_p))
    jy = np.array(j_t) + rng.uniform(-0.12, 0.12, len(j_t))
    _scatter(axes[1], jx, jy, 0.5, 5.5, "1-5", "Exp A — LLM judge (jittered)")
    axes[1].set_xticks(range(1, 6)); axes[1].set_yticks(range(1, 6))

    ax = axes[2]
    bars = ax.bar(["preferred", "top-rated"], [pct_pref, 100 - pct_pref],
                  color=[PREF_C, TOP_C], edgecolor="black", width=0.6)
    ax.axhline(50, ls="--", color="gray", lw=1)
    for b, v in zip(bars, [pct_pref, 100 - pct_pref]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center",
                fontweight="bold")
    ax.set_ylim(0, 100); ax.set_ylabel("% of samples")
    ax.set_title(f"Exp B — forced choice (n={len(cr)})\nposition-bias: picked A {pct_A:.0f}%",
                 fontsize=9)

    suptitle = f"{model}   (n={len(rows)})"
    if tag:
        suptitle += f"   [{tag}]"
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130); plt.close(fig)
    return {
        "model": model, "n": len(rows),
        "cos_pref": np.mean(cos_p), "cos_top": np.mean(cos_t),
        "cos_winpref": 100 * np.mean(np.array(cos_p) > np.array(cos_t)),
        "judge_pref": np.mean(j_p) if j_p else None,
        "judge_top": np.mean(j_t) if j_t else None,
        "choice_pref": pct_pref, "pos_bias_A": pct_A,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/models", help="dir with results_<slug>.jsonl")
    ap.add_argument("--out", default="plots/models", help="output dir for plots")
    ap.add_argument("--tag", default="", help="label appended to titles (e.g. 'no history')")
    args = ap.parse_args()

    files = sorted(glob.glob(str(HERE / args.dir / "results_*.jsonl")))
    files = [f for f in files if "_old_" not in os.path.basename(f)]
    if not files:
        raise SystemExit(f"no results_*.jsonl in {args.dir}")

    summary = []
    for fp in files:
        slug = os.path.basename(fp)[len("results_"):-len(".jsonl")]
        rows = [json.loads(l) for l in open(fp)]
        if not rows:
            print(f"  {slug}: empty, skipped"); continue
        model = rows[0].get("model", slug)
        out = HERE / args.out / f"{slug}.png"
        summary.append(plot_one(rows, model, out, args.tag))
        print(f"  -> {out}")

    print(f"\n{'model':32s} {'n':>5} {'cosP/cosT':>12} {'win%':>5} "
          f"{'judP/judT':>11} {'pick_pref%':>10} {'posA%':>6}")
    for s in summary:
        jp = f"{s['judge_pref']:.2f}/{s['judge_top']:.2f}" if s['judge_pref'] else "n/a"
        print(f"{s['model']:32s} {s['n']:5d} "
              f"{s['cos_pref']:.2f}/{s['cos_top']:.2f}".rjust(13) +
              f" {s['cos_winpref']:4.0f}% {jp:>11} {s['choice_pref']:9.0f}% {s['pos_bias_A']:5.0f}%")


if __name__ == "__main__":
    main()
