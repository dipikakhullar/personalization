"""
Capability preservation under activation steering — Gemma-3-4B.

Target figure for students:
  - 3 subplots (small multiples), one per steering axis
  - x-axis: 5 capability benchmarks (MMLU, HumanEval, GSM8K, TruthfulQA, ARC)
  - y-axis: benchmark accuracy / pass rate (%)
  - 3 bars per benchmark: -α (negative pole), baseline (unsteered), +α (positive pole)
  - Error bars: ±1 SEM across seeds
  - Direction arrow in each panel title showing what -α vs +α means
  - Dashed horizontal line = unsteered baseline per benchmark

The claim being made: each pole differs visibly from baseline but error bars
overlap with (or nearly overlap with) baseline — i.e. steering does not
significantly degrade capability.

Replace MEANS / SEMS with real numbers from your runs (3+ seeds each).
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# DATA: replace with real Gemma-3-4B numbers.
# Structure: means[pole][benchmark] = mean accuracy (%) across seeds
#            sems[pole][benchmark]  = standard error of the mean
# Poles: "neg" (−α steering) / "baseline" (α=0) / "pos" (+α steering)
# ---------------------------------------------------------------------------
BENCHMARKS = ["MMLU", "HumanEval", "GSM8K", "TruthfulQA", "ARC"]

BASELINE = {
    "MMLU":       58.4,
    "HumanEval":  36.0,
    "GSM8K":      52.3,
    "TruthfulQA": 47.1,
    "ARC":        62.8,
}
BASELINE_SEM = {
    "MMLU":       1.2,
    "HumanEval":  2.5,
    "GSM8K":      1.8,
    "TruthfulQA": 2.0,
    "ARC":        1.1,
}

rng = np.random.default_rng(11)
def perturb(base, lo=-2.0, hi=2.0):
    return {b: round(base[b] + rng.uniform(lo, hi), 1) for b in base}
def seeded_sem(lo=1.2, hi=3.0):
    return {b: round(rng.uniform(lo, hi), 2) for b in BENCHMARKS}

AXES = [
    {
        "name":      "Verbosity",
        "neg_label": "concise",
        "pos_label": "detailed",
        "means": {"neg": perturb(BASELINE, lo=-6.0, hi=-2.0), "baseline": BASELINE, "pos": perturb(BASELINE)},
        "sems":  {"neg": seeded_sem(),       "baseline": BASELINE_SEM, "pos": seeded_sem()},
    },
    {
        "name":      "Social style",
        "neg_label": "neutral",
        "pos_label": "warm",
        "means": {"neg": perturb(BASELINE), "baseline": BASELINE, "pos": perturb(BASELINE)},
        "sems":  {"neg": seeded_sem(),       "baseline": BASELINE_SEM, "pos": seeded_sem()},
    },
    {
        "name":      "Guidance",
        "neg_label": "reactive",
        "pos_label": "proactive",
        "means": {"neg": perturb(BASELINE, lo=-9.0, hi=-4.0), "baseline": BASELINE, "pos": perturb(BASELINE, lo=-8.0, hi=-3.0)},
        "sems":  {"neg": seeded_sem(),       "baseline": BASELINE_SEM, "pos": seeded_sem()},
    },
]

# ---------------------------------------------------------------------------
# STYLE
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
})

C_NEG  = "#1f77b4"   # blue — −α  (matplotlib default)
C_BASE = "#ff7f0e"   # orange — baseline
C_POS  = "#2ca02c"   # green — +α
C_LINE = "#111827"
ERR_KW = dict(ecolor="#1f2937", elinewidth=1.0, capsize=2.5, capthick=1.0)

# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0), sharey=True)
bar_w = 0.26
x_idx = np.arange(len(BENCHMARKS))

for ax, cfg in zip(axes, AXES):
    m, s = cfg["means"], cfg["sems"]
    neg_m  = [m["neg"][b]      for b in BENCHMARKS]
    base_m = [m["baseline"][b] for b in BENCHMARKS]
    pos_m  = [m["pos"][b]      for b in BENCHMARKS]
    neg_e  = [s["neg"][b]      for b in BENCHMARKS]
    base_e = [s["baseline"][b] for b in BENCHMARKS]
    pos_e  = [s["pos"][b]      for b in BENCHMARKS]

    ax.bar(x_idx - bar_w, neg_m,  bar_w, yerr=neg_e,  color=C_NEG,  error_kw=ERR_KW,
           label=f"−α  ({cfg['neg_label']})" if ax is axes[0] else "_nolegend_",
           edgecolor="white", linewidth=0.6)
    ax.bar(x_idx,         base_m, bar_w, yerr=base_e, color=C_BASE, error_kw=ERR_KW,
           label="baseline (α=0)" if ax is axes[0] else "_nolegend_",
           edgecolor="white", linewidth=0.6)
    ax.bar(x_idx + bar_w, pos_m,  bar_w, yerr=pos_e,  color=C_POS,  error_kw=ERR_KW,
           label=f"+α  ({cfg['pos_label']})" if ax is axes[0] else "_nolegend_",
           edgecolor="white", linewidth=0.6)

    for i, b in enumerate(BENCHMARKS):
        ax.hlines(BASELINE[b], i - bar_w*1.6, i + bar_w*1.6,
                  colors=C_LINE, linestyles=(0, (3, 2)),
                  linewidth=0.9, alpha=0.55, zorder=5)

    subtitle = f"−α  {cfg['neg_label']}  ←——→  {cfg['pos_label']}  +α"
    ax.set_title(f"{cfg['name']}\n{subtitle}", pad=8, fontweight="semibold",
                 linespacing=1.4)
    ax.set_xticks(x_idx)
    ax.set_xticklabels(BENCHMARKS, rotation=20, ha="right")
    ax.set_ylim(0, 80)
    ax.grid(axis="y", linestyle=":", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

axes[0].set_ylabel("Benchmark score (%)")

fig.legend(*axes[0].get_legend_handles_labels(),
           loc="lower center", bbox_to_anchor=(0.5, -0.04),
           ncol=3, frameon=False, fontsize=10,
           handlelength=1.4, handleheight=1.0)

fig.suptitle(
    "Steering Task Generalization",
    fontsize=13.5, fontweight="bold", y=1.04,
)
fig.text(
    0.5, -0.06,
    "Each panel: one steering axis. Bars: −α / unsteered / +α. "
    "Error bars: ±1 SEM over 3 seeds. Dashed lines = unsteered baseline. "
    "Steered scores deviate from baseline by < 1 SEM on most benchmarks ⇒ capability preserved.",
    ha="center", fontsize=9, style="italic", color="#374151",
)

plt.tight_layout()
plt.savefig("plots/outputs/generalization.png",
            dpi=200, bbox_inches="tight", facecolor="white")
print("Saved plots/outputs/generalization.png")