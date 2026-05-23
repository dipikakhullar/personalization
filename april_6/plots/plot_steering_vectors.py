"""Plot accuracy vs alpha for both steering vectors (expert-student & professor-llm)."""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(SCRIPT_DIR, "..")
OUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# --- Wilson confidence interval ---
def wilson_ci(k, n, z=1.96):
    if n == 0:
        return 0, 0, 0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return centre, centre - margin, centre + margin

# --- Load summaries ---
with open(os.path.join(BASE, "steering_vector_evals/expert_student_sv_layer_31_n100/summary.json")) as f:
    es_data = json.load(f)

with open(os.path.join(BASE, "steering_vector_evals/professor_vs_llm/summary.json")) as f:
    pl_data = json.load(f)

COLORS = {
    "gsm8k": "#4a90d9",
    "math": "#e8833a",
    "svamp": "#50b87a",
    "asdiv": "#9b59b6",
    "drop": "#e74c3c",
}
MARKERS = {"gsm8k": "o", "math": "s", "svamp": "D", "asdiv": "^", "drop": "v"}

def plot_sv(summary, title, filename, xlabel_note=""):
    results = summary["results"]
    layer = summary["layer"]
    pos = summary["pos_label"]
    neg = summary["neg_label"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    for dataset, by_alpha in sorted(results.items()):
        alphas = sorted(by_alpha.keys(), key=float)
        alpha_vals = [float(a) for a in alphas]
        accs, lo, hi = [], [], []
        avg_words = []
        for a in alphas:
            r = by_alpha[a]
            c, l, h = wilson_ci(r["num_correct"], r["num_items"])
            accs.append(r["accuracy"])
            lo.append(r["accuracy"] - l)
            hi.append(h - r["accuracy"])
            avg_words.append(r["avg_words"])

        color = COLORS.get(dataset, "#888")
        marker = MARKERS.get(dataset, "o")
        label = dataset.upper()

        ax1.errorbar(alpha_vals, accs, yerr=[lo, hi], label=label,
                     color=color, marker=marker, capsize=3, linewidth=2, markersize=7)
        ax2.plot(alpha_vals, avg_words, label=label,
                 color=color, marker=marker, linewidth=2, markersize=7)

    ax1.set_xlabel(f"Alpha (steering strength){xlabel_note}", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_title(f"Accuracy vs Alpha", fontsize=13)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel(f"Alpha (steering strength){xlabel_note}", fontsize=11)
    ax2.set_ylabel("Avg Words per Response", fontsize=11)
    ax2.set_title(f"Response Length vs Alpha", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"{title}\n(Layer {layer}, +\u03b1 \u2192 {pos}, \u2212\u03b1 \u2192 {neg}, n=25 per dataset)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=180, bbox_inches="tight")
    print(f"Saved {filename}")
    plt.close(fig)

# --- Plot both ---
plot_sv(es_data,
        "Expert\u2013Student Steering Vector",
        "steering_expert_student.png")

plot_sv(pl_data,
        "Professor\u2013LLM Steering Vector",
        "steering_professor_llm.png",
        xlabel_note=" (note: very small values)")

# --- Combined comparison figure ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, summary, label in zip(axes,
                                [es_data, pl_data],
                                ["Expert\u2013Student (Layer 31)", "Professor\u2013LLM (Layer 33)"]):
    results = summary["results"]
    for dataset, by_alpha in sorted(results.items()):
        alphas = sorted(by_alpha.keys(), key=float)
        alpha_vals = [float(a) for a in alphas]
        accs = [by_alpha[a]["accuracy"] for a in alphas]
        color = COLORS.get(dataset, "#888")
        marker = MARKERS.get(dataset, "o")
        ax.plot(alpha_vals, accs, label=dataset.upper(),
                color=color, marker=marker, linewidth=2, markersize=7)
    ax.set_title(label, fontsize=13, fontweight="bold")
    ax.set_xlabel("Alpha", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

fig.suptitle("Steering Vector Accuracy Comparison", fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "steering_comparison.png"), dpi=180, bbox_inches="tight")
print("Saved steering_comparison.png")
plt.close(fig)

print("Done.")
