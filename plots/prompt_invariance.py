"""
Construction-prompt invariance of steering directions — Gemma-3-4B.

For each axis we build the steering direction K=5 times from K disjoint
subsets of construction prompts, then ask two questions:

  (1) Are the K direction vectors themselves similar?
      → pairwise cosine similarity heatmap per axis (top row).

  (2) Do the K variants produce the same downstream effect?
      → benchmark scores under +α steering, 5 markers per benchmark per axis
        (one per variant). Tight clustering ⇒ prompt-invariant behaviour.

Replace COSINE_SIMS and BENCHMARK_SCORES with values measured from your runs.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# DATA — replace with measured values
# ---------------------------------------------------------------------------
AXES = ["Verbosity", "Social style", "Guidance"]
AXIS_LABELS = {
    "Verbosity":    "(detailed ↔ concise)",
    "Social style": "(warm ↔ neutral)",
    "Guidance":     "(proactive ↔ reactive)",
}
BENCHMARKS = ["MMLU", "HumanEval", "GSM8K", "TruthfulQA", "ARC"]
BASELINE = {
    "MMLU":       58.4,
    "HumanEval":  36.0,
    "GSM8K":      52.3,
    "TruthfulQA": 47.1,
    "ARC":        62.8,
}
K = 5  # number of construction-prompt subsets per axis

# Pairwise cosine similarity between K direction variants (5x5, symmetric,
# diagonal = 1). Off-diagonals ~0.92–0.98 → directions are nearly the same
# vector regardless of which prompt subset built them.
rng = np.random.default_rng(23)
def mock_cosine_matrix(low=0.91, high=0.98):
    M = rng.uniform(low, high, size=(K, K))
    M = (M + M.T) / 2
    np.fill_diagonal(M, 1.0)
    return np.round(M, 3)

COSINE_SIMS = {axis: mock_cosine_matrix() for axis in AXES}

# Benchmark scores under +α steering, K variants × N benchmarks per axis.
# Mock so variants cluster within ~1 point of each other (well within SEM)
# and sit close to baseline.
def mock_variant_scores(spread=4.0, drift=0.0):
    return {b: np.round(BASELINE[b] + drift + rng.uniform(-spread, spread, size=K), 2)
            for b in BENCHMARKS}

BENCHMARK_SCORES = {axis: mock_variant_scores() for axis in AXES}

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
})

C_DOT      = "#ef4444"   # individual variant markers
C_MEAN     = "#111827"   # mean marker
C_BASELINE = "#6b7280"
heat_cmap  = LinearSegmentedColormap.from_list(
    "sim", ["#fef3c7", "#fbbf24", "#dc2626"]
)

# ---------------------------------------------------------------------------
# FIGURE 1 — cosine similarity heatmaps
# ---------------------------------------------------------------------------
fig1, axes1 = plt.subplots(1, 3, figsize=(14.5, 4.0))

for col, axis in enumerate(AXES):
    ax = axes1[col]
    M = COSINE_SIMS[axis]
    im = ax.imshow(M, cmap=heat_cmap, vmin=0.85, vmax=1.0, aspect="equal")

    for i in range(K):
        for j in range(K):
            txt_color = "white" if M[i, j] > 0.95 else "#1f2937"
            ax.text(j, i, f"{M[i, j]:.2f}",
                    ha="center", va="center", fontsize=8.5, color=txt_color)

    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels([f"v{i+1}" for i in range(K)], fontsize=8.5)
    ax.set_yticklabels([f"v{i+1}" for i in range(K)], fontsize=8.5)
    title = f"{axis}\n{AXIS_LABELS[axis]}"
    ax.set_title(title, fontweight="semibold", pad=8, linespacing=1.4)
    if col == 0:
        ax.set_ylabel("Direction variant", fontsize=9.5)

fig1.suptitle(
    "Cosine similarity between construction-prompt variants  —  Gemma-3-4B",
    fontsize=13, fontweight="bold", y=1.02,
)
fig1.colorbar(im, ax=axes1.tolist(), shrink=0.75, pad=0.15, label="cosine similarity")
fig1.subplots_adjust(wspace=0.35, right=0.78)
fig1.savefig("plots/outputs/prompt_invariance_similarity.png",
             dpi=200, bbox_inches="tight", facecolor="white")
print("Saved plots/outputs/prompt_invariance_similarity.png")
plt.close(fig1)

# ---------------------------------------------------------------------------
# FIGURE 2 — per-variant benchmark scatter
# ---------------------------------------------------------------------------
x_idx = np.arange(len(BENCHMARKS))

V_SHADES = ["#08195e", "#1a56a0", "#2e86c1", "#5dade2", "#a9cce3"]
V_LABELS = [f"v{i+1}" for i in range(K)]

fig2, axes2 = plt.subplots(1, 3, figsize=(14.5, 5.0), sharey=True)

for col, axis in enumerate(AXES):
    ax = axes2[col]
    scores = BENCHMARK_SCORES[axis]

    for i, b in enumerate(BENCHMARKS):
        ax.hlines(BASELINE[b], i - 0.38, i + 0.38,
                  colors=C_BASELINE, linestyles=(0, (3, 2)),
                  linewidth=1.0, alpha=0.7, zorder=2)

    for i, b in enumerate(BENCHMARKS):
        ys = scores[b]
        for v in range(K):
            ax.scatter([i], [ys[v]], s=50, color=V_SHADES[v], alpha=0.8,
                       edgecolor="white", linewidth=0.8, zorder=4,
                       label=V_LABELS[v] if (i == 0 and col == 0) else None)
        ax.scatter([i], [np.mean(ys)], marker="_", s=380,
                   color=C_MEAN, linewidth=2.2, zorder=5,
                   label="variant mean" if (i == 0 and col == 0) else None)

    ax.set_xticks(x_idx)
    ax.set_xticklabels(BENCHMARKS, rotation=20, ha="right", fontsize=9.5)
    ax.set_ylim(30, 70)
    ax.set_title(f"{axis} — score under +α steering",
                 fontweight="semibold", pad=8)
    ax.grid(axis="y", linestyle=":", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    if col == 0:
        ax.set_ylabel("Benchmark score (%)")

handles = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=V_SHADES[v],
               markeredgecolor="white", markersize=7, alpha=0.8, label=V_LABELS[v])
    for v in range(K)
] + [
    plt.Line2D([0], [0], marker="_", color=C_MEAN, markersize=14,
               linewidth=2.2, linestyle="none", label="variant mean"),
    plt.Line2D([0], [0], color=C_BASELINE, linestyle=(0, (3, 2)),
               linewidth=1.2, label="unsteered baseline"),
]
fig2.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06),
            ncol=7, frameon=False, fontsize=9, handlelength=1.6)

fig2.suptitle(
    "Benchmark scores across construction-prompt variants  —  Gemma-3-4B",
    fontsize=13, fontweight="bold", y=1.02,
)
fig2.tight_layout()
fig2.savefig("plots/outputs/prompt_invariance_benchmarks.png",
             dpi=200, bbox_inches="tight", facecolor="white")
print("Saved plots/outputs/prompt_invariance_benchmarks.png")
plt.close(fig2)