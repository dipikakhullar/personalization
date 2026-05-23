"""
Plot baseline vs prompted-personalization results.

Outputs:
  plots/outputs/accuracy_student_vs_expert.png          — student/expert grouped bar chart
  plots/outputs/accuracy_professor_vs_llm.png           — professor/llm grouped bar chart
  plots/outputs/math_by_level_student_vs_expert.png     — MATH by level (student/expert)

Error bars: 95% Wilson confidence intervals computed from per-sample traces.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "outputs"

baseline = json.load(open(ROOT / "baselines" / "outputs" / "summary.json"))
prompted = json.load(open(ROOT / "prompted_personalization" / "outputs" / "summary.json"))
prof_llm = json.load(open(ROOT / "prompted_professor_llm" / "outputs" / "summary.json"))

DATASETS = ["gsm8k", "math", "svamp", "asdiv", "drop"]
LABELS = ["GSM8K", "MATH", "SVAMP", "ASDiv", "DROP"]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion. Returns (lo, hi) in [0,1]."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0, center - spread), min(1, center + spread)


def get_errs(k: int, n: int) -> tuple[float, float]:
    """Return (err_minus, err_plus) in percentage points for errorbar yerr."""
    acc = k / n * 100 if n else 0
    lo, hi = wilson_ci(k, n)
    return acc - lo * 100, hi * 100 - acc


# ── Collect data ──────────────────────────────────────────────────────

base_acc, student_acc, expert_acc = [], [], []
base_err, student_err, expert_err = [[], []], [[], []], [[], []]

for ds in DATASETS:
    bk = baseline["results"][ds]["correct"]
    bn = baseline["results"][ds]["total"]
    base_acc.append(bk / bn * 100)
    lo, hi = get_errs(bk, bn)
    base_err[0].append(lo)
    base_err[1].append(hi)

    sk = prompted["results"][f"{ds}_student"]["correct"]
    sn = prompted["results"][f"{ds}_student"]["total"]
    student_acc.append(sk / sn * 100)
    lo, hi = get_errs(sk, sn)
    student_err[0].append(lo)
    student_err[1].append(hi)

    ek = prompted["results"][f"{ds}_expert"]["correct"]
    en = prompted["results"][f"{ds}_expert"]["total"]
    expert_acc.append(ek / en * 100)
    lo, hi = get_errs(ek, en)
    expert_err[0].append(lo)
    expert_err[1].append(hi)


# ── Figure 1: Grouped bar chart ──────────────────────────────────────

x = np.arange(len(DATASETS))
w = 0.25
cap = 3

fig, ax = plt.subplots(figsize=(10, 5.5))

bars_b = ax.bar(x - w, base_acc, w, yerr=base_err, capsize=cap,
                label="Baseline (neutral)", color="#4a90d9", edgecolor="white",
                linewidth=0.8, error_kw={"linewidth": 1.2, "color": "#333"})
bars_s = ax.bar(x, student_acc, w, yerr=student_err, capsize=cap,
                label="Student prompt", color="#e8833a", edgecolor="white",
                linewidth=0.8, error_kw={"linewidth": 1.2, "color": "#333"})
bars_e = ax.bar(x + w, expert_acc, w, yerr=expert_err, capsize=cap,
                label="Expert prompt", color="#50b87a", edgecolor="white",
                linewidth=0.8, error_kw={"linewidth": 1.2, "color": "#333"})

for bars in [bars_b, bars_s, bars_e]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 3.5, f"{h:.0f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Gemma-3-4B-IT: Baseline vs Student/Expert Prompts", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=11)
ax.set_ylim(0, 110)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3)
ns = [baseline["results"][ds]["total"] for ds in DATASETS]
n_label = f"n={ns[0]}" if len(set(ns)) == 1 else ", ".join(f"{l} n={n}" for l, n in zip(LABELS, ns))
ax.text(0.98, 0.02, f"Error bars: 95% Wilson CI  ({n_label})",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="gray")

fig.tight_layout()
fig.savefig(OUT / "accuracy_student_vs_expert.png", dpi=180)
print(f"Saved {OUT / 'accuracy_student_vs_expert.png'}")
plt.close(fig)


# ── Figure 2: MATH by difficulty level ────────────────────────────────

levels = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]
level_labels = ["L1", "L2", "L3", "L4", "L5"]

base_math = baseline["results"]["math"]["by_level"]
student_math = prompted["results"]["math_student"]["by_level"]
expert_math = prompted["results"]["math_expert"]["by_level"]

base_lvl, student_lvl, expert_lvl = [], [], []
base_lvl_err, student_lvl_err, expert_lvl_err = [[], []], [[], []], [[], []]

for l in levels:
    for src, acc_list, err_list in [
        (base_math, base_lvl, base_lvl_err),
        (student_math, student_lvl, student_lvl_err),
        (expert_math, expert_lvl, expert_lvl_err),
    ]:
        k, n = src[l]["correct"], src[l]["total"]
        acc_list.append(k / n * 100)
        lo, hi = get_errs(k, n)
        err_list[0].append(lo)
        err_list[1].append(hi)

x2 = np.arange(len(levels))

fig2, ax2 = plt.subplots(figsize=(9, 5.5))

bars_b2 = ax2.bar(x2 - w, base_lvl, w, yerr=base_lvl_err, capsize=cap,
                  label="Baseline (neutral)", color="#4a90d9", edgecolor="white",
                  linewidth=0.8, error_kw={"linewidth": 1.2, "color": "#333"})
bars_s2 = ax2.bar(x2, student_lvl, w, yerr=student_lvl_err, capsize=cap,
                  label="Student prompt", color="#e8833a", edgecolor="white",
                  linewidth=0.8, error_kw={"linewidth": 1.2, "color": "#333"})
bars_e2 = ax2.bar(x2 + w, expert_lvl, w, yerr=expert_lvl_err, capsize=cap,
                  label="Expert prompt", color="#50b87a", edgecolor="white",
                  linewidth=0.8, error_kw={"linewidth": 1.2, "color": "#333"})

for bars in [bars_b2, bars_s2, bars_e2]:
    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 5, f"{h:.0f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

ax2.set_ylabel("Accuracy (%)", fontsize=12)
ax2.set_title("MATH by Level: Baseline vs Student/Expert", fontsize=14, fontweight="bold")
ax2.set_xlabel("Difficulty Level", fontsize=12)
ax2.set_xticks(x2)
ax2.set_xticklabels(level_labels, fontsize=11)
ax2.set_ylim(0, 110)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
ax2.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.grid(axis="y", alpha=0.3)
n_per_level = base_math[levels[0]]["total"]
ax2.text(0.98, 0.02, f"Error bars: 95% Wilson CI, n={n_per_level} per level",
         transform=ax2.transAxes, ha="right", va="bottom", fontsize=8, color="gray")

fig2.tight_layout()
fig2.savefig(OUT / "math_by_level_student_vs_expert.png", dpi=180)
print(f"Saved {OUT / 'math_by_level_student_vs_expert.png'}")
plt.close(fig2)


# ── Figure 3: Professor vs LLM ──────────────────────────────────────

prof_acc, llm_acc = [], []
prof_err, llm_err = [[], []], [[], []]

for ds in DATASETS:
    pk = prof_llm["results"][f"{ds}_professor"]["correct"]
    pn = prof_llm["results"][f"{ds}_professor"]["total"]
    prof_acc.append(pk / pn * 100)
    lo, hi = get_errs(pk, pn)
    prof_err[0].append(lo)
    prof_err[1].append(hi)

    lk = prof_llm["results"][f"{ds}_llm"]["correct"]
    ln = prof_llm["results"][f"{ds}_llm"]["total"]
    llm_acc.append(lk / ln * 100)
    lo, hi = get_errs(lk, ln)
    llm_err[0].append(lo)
    llm_err[1].append(hi)

fig3, ax3 = plt.subplots(figsize=(10, 5.5))

bars_b3 = ax3.bar(x - w, base_acc, w, yerr=base_err, capsize=cap,
                  label="Baseline (neutral)", color="#4a90d9", edgecolor="white",
                  linewidth=0.8, error_kw={"linewidth": 1.2, "color": "#333"})
bars_p3 = ax3.bar(x, prof_acc, w, yerr=prof_err, capsize=cap,
                  label="Professor prompt", color="#9b59b6", edgecolor="white",
                  linewidth=0.8, error_kw={"linewidth": 1.2, "color": "#333"})
bars_l3 = ax3.bar(x + w, llm_acc, w, yerr=llm_err, capsize=cap,
                  label="LLM prompt", color="#e74c3c", edgecolor="white",
                  linewidth=0.8, error_kw={"linewidth": 1.2, "color": "#333"})

for bars in [bars_b3, bars_p3, bars_l3]:
    for bar in bars:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, h + 3.5, f"{h:.0f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

ax3.set_ylabel("Accuracy (%)", fontsize=12)
ax3.set_title("Gemma-3-4B-IT: Baseline vs Professor/LLM Prompts", fontsize=14, fontweight="bold")
ax3.set_xticks(x)
ax3.set_xticklabels(LABELS, fontsize=11)
ax3.set_ylim(0, 110)
ax3.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
ax3.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.grid(axis="y", alpha=0.3)
ax3.text(0.98, 0.02, f"Error bars: 95% Wilson CI  ({n_label})",
         transform=ax3.transAxes, ha="right", va="bottom", fontsize=8, color="gray")

fig3.tight_layout()
fig3.savefig(OUT / "accuracy_professor_vs_llm.png", dpi=180)
print(f"Saved {OUT / 'accuracy_professor_vs_llm.png'}")
plt.close(fig3)


# ── Figure 4: MATH by level — Professor vs LLM ──────────────────────

prof_math = prof_llm["results"]["math_professor"]["by_level"]
llm_math = prof_llm["results"]["math_llm"]["by_level"]

prof_lvl, llm_lvl = [], []
prof_lvl_err, llm_lvl_err = [[], []], [[], []]

for l in levels:
    for src, acc_list, err_list in [
        (prof_math, prof_lvl, prof_lvl_err),
        (llm_math, llm_lvl, llm_lvl_err),
    ]:
        k, n = src[l]["correct"], src[l]["total"]
        acc_list.append(k / n * 100)
        lo, hi = get_errs(k, n)
        err_list[0].append(lo)
        err_list[1].append(hi)

fig4, ax4 = plt.subplots(figsize=(9, 5.5))

bars_b4 = ax4.bar(x2 - w, base_lvl, w, yerr=base_lvl_err, capsize=cap,
                  label="Baseline (neutral)", color="#4a90d9", edgecolor="white",
                  linewidth=0.8, error_kw={"linewidth": 1.2, "color": "#333"})
bars_p4 = ax4.bar(x2, prof_lvl, w, yerr=prof_lvl_err, capsize=cap,
                  label="Professor prompt", color="#9b59b6", edgecolor="white",
                  linewidth=0.8, error_kw={"linewidth": 1.2, "color": "#333"})
bars_l4 = ax4.bar(x2 + w, llm_lvl, w, yerr=llm_lvl_err, capsize=cap,
                  label="LLM prompt", color="#e74c3c", edgecolor="white",
                  linewidth=0.8, error_kw={"linewidth": 1.2, "color": "#333"})

for bars in [bars_b4, bars_p4, bars_l4]:
    for bar in bars:
        h = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2, h + 5, f"{h:.0f}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

ax4.set_ylabel("Accuracy (%)", fontsize=12)
ax4.set_title("MATH by Level: Baseline vs Professor/LLM", fontsize=14, fontweight="bold")
ax4.set_xlabel("Difficulty Level", fontsize=12)
ax4.set_xticks(x2)
ax4.set_xticklabels(level_labels, fontsize=11)
ax4.set_ylim(0, 110)
ax4.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
ax4.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
ax4.spines["top"].set_visible(False)
ax4.spines["right"].set_visible(False)
ax4.grid(axis="y", alpha=0.3)
ax4.text(0.98, 0.02, f"Error bars: 95% Wilson CI, n={n_per_level} per level",
         transform=ax4.transAxes, ha="right", va="bottom", fontsize=8, color="gray")

fig4.tight_layout()
fig4.savefig(OUT / "math_by_level_professor_vs_llm.png", dpi=180)
print(f"Saved {OUT / 'math_by_level_professor_vs_llm.png'}")
plt.close(fig4)
