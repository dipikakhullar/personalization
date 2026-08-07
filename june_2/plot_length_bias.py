"""Length-bias check for the CHOOSE task: do models pick the user's PREFERRED
answer simply because it is LONGER than the community TOP answer?

For each sample we know len(preferred) and len(top) (words). We split samples
into "preferred LONGER than top" vs "preferred SHORTER" and compare the
pick-preferred rate per model. A big gap (high pick-rate only when preferred is
longer) = length/verbosity confound. Roughly equal = the choice isn't just length.

Output: plots/cells/cleanpilot_length_bias.png  +  printed table.
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
SAMPLES = HERE / "data" / "samples_qa1000_clean.jsonl"
WITH_DIR = HERE / "results_clean" / "with_history"
LONG_C, SHORT_C = "#3a6ea5", "#e76f51"


def wlen(t):
    return len((t or "").split())


def main():
    samples = {json.loads(l)["sample_id"]: json.loads(l) for l in open(SAMPLES)}
    pref_longer = {sid: wlen(s["preferred_answer"]) > wlen(s["top_comment"])
                   for sid, s in samples.items()}
    frac_longer = 100 * np.mean(list(pref_longer.values()))

    rows = {}
    for fp in sorted(glob.glob(str(WITH_DIR / "results_*.jsonl"))):
        if "_old_" in os.path.basename(fp):
            continue
        slug = os.path.basename(fp)[len("results_"):-len(".jsonl")]
        rows[slug] = [json.loads(l) for l in open(fp)]

    labels, longp, shortp, nlong, nshort = [], [], [], [], []
    print(f"(preferred is longer than top in {frac_longer:.0f}% of samples)\n")
    print(f"{'model':16s} {'pick-pref | pref LONGER':>24} {'pref SHORTER':>14}")
    for slug, rs in rows.items():
        cr = [r for r in rs if r["choice_picked_preferred"] is not None
              and r["sample_id"] in pref_longer]
        L = [r["choice_picked_preferred"] for r in cr if pref_longer[r["sample_id"]]]
        S = [r["choice_picked_preferred"] for r in cr if not pref_longer[r["sample_id"]]]
        lp, sp = 100*np.mean(L), 100*np.mean(S)
        labels.append(slug); longp.append(lp); shortp.append(sp)
        nlong.append(len(L)); nshort.append(len(S))
        print(f"{slug:16s} {lp:20.0f}% (n={len(L)})  {sp:8.0f}% (n={len(S)})")

    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.8))
    b1 = ax.bar(x - w/2, longp, w, label="preferred is LONGER than top", color=LONG_C,
                edgecolor="black")
    b2 = ax.bar(x + w/2, shortp, w, label="preferred is SHORTER than top", color=SHORT_C,
                edgecolor="black")
    ax.axhline(50, ls="--", color="gray", lw=1)
    for b in [*b1, *b2]:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1, f"{b.get_height():.0f}",
                ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 100); ax.set_ylabel("% that picked the user's PREFERRED answer")
    ax.set_title("Length-bias check (CHOOSE task): is picking the PREFERRED answer just "
                 "a length effect?\nPick-rate split by whether the preferred answer is longer "
                 "or shorter than the top answer.", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    fig.text(0.5, 0.005,
             f"If bars within a model are similar, the choice isn't driven by length. If "
             f"'longer' >> 'shorter', there's a verbosity confound. Dashed = 50%. "
             f"Preferred is longer than top in {frac_longer:.0f}% of samples. n=500/model "
             f"(with-history condition).", ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = HERE / "plots" / "cells" / "cleanpilot_length_bias.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
