"""Matched with-history vs no-history comparison.

For each model, load both conditions, restrict to the INTERSECTION of sample_ids
both actually ran (a fair paired test), then compare:
  - Exp A: % of samples where the generated answer is closer to the user's
           preferred answer than to the top-rated one (cosine, and 1-5 judge)
  - Exp B: % of samples where the model picked the preferred answer

Outputs plots/compare/compare.png and a printed matched-sample table.
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
WITH_C = "#3a6ea5"   # with history
NO_C = "#c0c0c0"     # no history


def load_dir(d):
    out = {}
    for fp in glob.glob(str(HERE / d / "results_*.jsonl")):
        if "_old_" in os.path.basename(fp):
            continue
        slug = os.path.basename(fp)[len("results_"):-len(".jsonl")]
        out[slug] = {json.loads(l)["sample_id"]: json.loads(l) for l in open(fp)}
    return out


def metrics(rows):
    cos_win = np.mean([r["cos_preferred"] > r["cos_top"] for r in rows]) * 100
    jr = [r for r in rows if r["judge_sim_preferred"] is not None
          and r["judge_sim_top"] is not None]
    judge_win = (np.mean([r["judge_sim_preferred"] > r["judge_sim_top"] for r in jr]) * 100
                 if jr else float("nan"))
    cr = [r for r in rows if r["choice_picked_preferred"] is not None]
    pick = np.mean([r["choice_picked_preferred"] for r in cr]) * 100 if cr else float("nan")
    return cos_win, judge_win, pick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-dir", default="results/models")
    ap.add_argument("--no-dir", default="no_user_context")
    ap.add_argument("--out", default="plots/compare/compare.png")
    args = ap.parse_args()

    wh, nh = load_dir(args.with_dir), load_dir(args.no_dir)
    slugs = sorted(set(wh) & set(nh))
    if not slugs:
        raise SystemExit("no overlapping models between the two dirs")

    labels, w_pick, n_pick, w_cos, n_cos, w_jud, n_jud, ns = ([] for _ in range(8))
    print(f"{'model':24s} {'matched_n':>9} {'pick W/N':>12} {'cosWin W/N':>13} {'judWin W/N':>13}")
    for slug in slugs:
        common = sorted(set(wh[slug]) & set(nh[slug]))
        if not common:
            print(f"{slug:24s} no overlap yet"); continue
        wr = [wh[slug][i] for i in common]
        nr = [nh[slug][i] for i in common]
        wc, wj, wp = metrics(wr)
        nc, nj, npk = metrics(nr)
        model = wr[0].get("model", slug)
        labels.append(model.split("/")[-1]); ns.append(len(common))
        w_pick.append(wp); n_pick.append(npk)
        w_cos.append(wc); n_cos.append(nc)
        w_jud.append(wj); n_jud.append(nj)
        print(f"{model:24s} {len(common):9d} {wp:5.0f}/{npk:<5.0f} "
              f"{wc:5.0f}/{nc:<6.0f} {wj:5.0f}/{nj:<6.0f}")

    x = np.arange(len(labels)); width = 0.38
    panels = [("Exp B: picked PREFERRED %", w_pick, n_pick),
              ("Exp A: closer-to-preferred % (cosine)", w_cos, n_cos),
              ("Exp A: closer-to-preferred % (judge 1-5)", w_jud, n_jud)]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
    for ax, (title, wv, nv) in zip(axes, panels):
        ax.bar(x - width/2, wv, width, label="with history", color=WITH_C, edgecolor="black")
        ax.bar(x + width/2, nv, width, label="no history", color=NO_C, edgecolor="black")
        ax.axhline(50, ls="--", color="gray", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 100); ax.set_ylabel("%"); ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    fig.suptitle("Does user history steer models toward the PREFERRED answer? "
                 "(matched samples, dashed = 50% chance)", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = HERE / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"\n-> {out}   (matched n per model: {dict(zip(labels, ns))})")


if __name__ == "__main__":
    main()
