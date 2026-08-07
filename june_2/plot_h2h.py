"""Head-to-head result: for each model, win-rate of the WITH-history generation
over the NO-context generation, judged by which the user would prefer (given
their history). Tested against the 50% no-effect baseline.

Output: plots/cells/cleanpilot_head2head.png  +  printed table with 95% CIs + p.
"""
import glob
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results_clean" / "head_to_head"

try:
    from scipy.stats import binomtest
    def pval(k, n):
        return binomtest(k, n, 0.5).pvalue
except Exception:  # normal approximation fallback
    def pval(k, n):
        if n == 0:
            return float("nan")
        z = (k - n / 2) / (math.sqrt(n) / 2)
        return math.erfc(abs(z) / math.sqrt(2))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return (100*(c-h), 100*(c+h))


def main():
    rows = {}
    for fp in sorted(glob.glob(str(RES / "results_*.jsonl"))):
        slug = os.path.basename(fp)[len("results_"):-len(".jsonl")]
        rows[slug] = [json.loads(l) for l in open(fp)]

    labels, wins, los, his, ks, ns, ps, posbias = ([] for _ in range(8))
    print(f"{'model':16s} {'n':>4} {'win%':>6} {'95% CI':>15} {'p vs 50%':>10} {'posbiasA%':>9}")
    for slug, rs in rows.items():
        cr = [r for r in rs if r["with_history_won"] is not None]
        n = len(cr); k = sum(1 for r in cr if r["with_history_won"])
        wr = 100 * k / n
        lo, hi = wilson(k, n); p = pval(k, n)
        pb = 100 * np.mean([r["with_was_A"] for r in cr])
        labels.append(slug); wins.append(wr); los.append(lo); his.append(hi)
        ks.append(k); ns.append(n); ps.append(p); posbias.append(pb)
        sig = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"
        print(f"{slug:16s} {n:4d} {wr:6.1f} {lo:6.1f}-{hi:5.1f}  {p:10.4f} {sig:>4} {pb:8.0f}")

    x = np.arange(len(labels))
    err = [[w - lo for w, lo in zip(wins, los)], [hi - w for w, hi in zip(wins, his)]]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(x, wins, 0.55, yerr=err, capsize=5, color="#3a6ea5", edgecolor="black")
    ax.axhline(50, ls="--", color="gray", lw=1)
    ax.text(len(labels)-0.5, 50.7, "50% = history made no difference", fontsize=8, color="gray", ha="right")
    for i, (w, p) in enumerate(zip(wins, ps)):
        sig = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"
        ax.text(i, his[i] + 1.2, f"{w:.0f}%\n{sig}", ha="center", fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of questions where the user prefers the WITH-history answer")
    ax.set_title("Does giving the model the user's history produce an answer the user prefers?\n"
                 "For each question the model writes two answers — one WITH the user's history, "
                 "one WITHOUT.\nA judge (shown the history) picks which the user would prefer; "
                 "bars = how often the WITH-history answer wins.", fontsize=10)
    fig.text(0.5, 0.005,
             f"The two answers are compared directly, one vs one, with their order randomized. "
             f"Error bars = 95% Wilson CI. Above 50% (and significant: * p<.05, ** p<.01, "
             f"*** p<.001) = the history helped. n={ns[0] if ns else 0} per model.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = HERE / "plots" / "cells" / "cleanpilot_head2head.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
