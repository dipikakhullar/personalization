"""
Create contrastive steering vectors for Gemma-3-4B-IT.

Extracts activations for two contrasting persona prompts over GSM8K problems.
Each run gets its own uniquely named subdirectory under outputs/.

Outputs (all under outputs/{run_name}/):
  activations_layers_{range}_n{N}.pkl           — raw activations (reusable)
  sv_layer_{L}_n{N}.pt                          — steering vector per layer
  sv_best_layer_{L}_n{N}.pt                     — best layer copy
  trace.json                                    — full run metadata
  pca_layers_{range}_n{N}.png                   — PCA grid plot

Usage:
  # Expert vs Student (default)
  python create_steering_vector.py --num-problems 100 --layers 0-33

  # Custom personas
  python create_steering_vector.py --num-problems 100 --layers 0-33 \\
      --name professor_vs_llm \\
      --positive "I am a math professor. I verify answers before outputting my final answer." \\
      --negative "I am an LLM. I just do next token prediction." \\
      --pos-label professor --neg-label llm
"""

import argparse
import json
import os
import pickle
import time
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ── Defaults ─────────────────────────────────────────────────────────
MODEL_NAME = "google/gemma-3-4b-it"

DEFAULT_POS_PROMPT = (
    "You are helping an expert engineer/researcher who just needs the "
    "answer or a compact solution. Do not show unnecessary working or "
    "pedagogical explanation — they will verify the details themselves. "
    "Be direct and concise."
)
DEFAULT_NEG_PROMPT = (
    "You are helping a student who is learning this topic for the first "
    "time. They need to see every step explained clearly and thoroughly. "
    "Show all working, define terms, and walk through the reasoning. "
    "A bare answer with no explanation is useless to them."
)


# ── Helpers ──────────────────────────────────────────────────────────

def get_language_model_layers(model):
    if hasattr(model.model, "language_model"):
        return model.model.language_model.layers
    return model.model.layers


def load_model(model_name: str):
    print(f"Loading {model_name} …")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def extract_all_layers(model, tokenizer, system_prompt: str, question: str,
                       candidate_layers: list[int]) -> dict[int, torch.Tensor]:
    """Single forward pass; return last-token hidden state at every candidate layer."""
    messages = [
        {"role": "user", "content": f"{system_prompt}\n\n{question}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    layers = get_language_model_layers(model)
    captured: dict[int, torch.Tensor] = {}
    handles = []

    for layer_idx in candidate_layers:
        def make_hook(idx):
            def hook(_module, _input, output):
                h = output[0] if isinstance(output, tuple) else output
                captured[idx] = h[0, -1, :].detach().cpu().float()
            return hook
        handles.append(layers[layer_idx].register_forward_hook(make_hook(layer_idx)))

    with torch.no_grad():
        model(**inputs)

    for h in handles:
        h.remove()

    return captured


def collect_activations(model, tokenizer, candidate_layers: list[int],
                        problems: list[str], pos_prompt: str, neg_prompt: str):
    """
    2 forward passes per problem (positive + negative persona).
    Returns dict: layer_idx -> {"positive": [N, hidden_dim], "negative": [N, hidden_dim]}
    """
    layer_acts: dict[int, dict[str, list[torch.Tensor]]] = {
        l: {"positive": [], "negative": []} for l in candidate_layers
    }

    for i, q in enumerate(problems):
        print(f"  [{i+1:>3}/{len(problems)}] {q[:60]}")
        pos_hiddens = extract_all_layers(model, tokenizer, pos_prompt, q, candidate_layers)
        neg_hiddens = extract_all_layers(model, tokenizer, neg_prompt, q, candidate_layers)
        for layer_idx in candidate_layers:
            layer_acts[layer_idx]["positive"].append(pos_hiddens[layer_idx])
            layer_acts[layer_idx]["negative"].append(neg_hiddens[layer_idx])

    for layer_idx in candidate_layers:
        layer_acts[layer_idx]["positive"] = torch.stack(layer_acts[layer_idx]["positive"])
        layer_acts[layer_idx]["negative"] = torch.stack(layer_acts[layer_idx]["negative"])

    return layer_acts


def plot_pca_grid(layer_acts: dict, candidate_layers: list[int],
                  n_problems: int, pos_label: str, neg_label: str,
                  title: str, save_path: str):
    """PCA scatter grid, one subplot per layer."""
    n_layers = len(candidate_layers)
    cols = min(n_layers, 6)
    rows = (n_layers + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows))
    if n_layers == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, layer_idx in enumerate(candidate_layers):
        ax = axes[i]
        pos = layer_acts[layer_idx]["positive"]
        neg = layer_acts[layer_idx]["negative"]
        all_acts = torch.cat([pos, neg], dim=0).numpy()
        n = pos.shape[0]

        pca = PCA(n_components=2)
        projected = pca.fit_transform(all_acts)
        var = pca.explained_variance_ratio_

        ax.scatter(projected[:n, 0], projected[:n, 1],
                   c="#2196F3", alpha=0.6, s=20, label=pos_label, edgecolors="none")
        ax.scatter(projected[n:, 0], projected[n:, 1],
                   c="#FF5722", alpha=0.6, s=20, label=neg_label, edgecolors="none")

        ax.set_xlabel(f"PC1 ({var[0]:.1%})")
        ax.set_ylabel(f"PC2 ({var[1]:.1%})")
        ax.set_title(f"Layer {layer_idx}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for j in range(n_layers, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {save_path}")


def build_steering_vectors(layer_acts: dict, candidate_layers: list[int]):
    """Mean-difference vector per layer (raw, not normalized).

    α=1.0 at eval time means "add the full mean difference".
    α=0.5 means "half the difference", etc.
    Returns (results, best_layer).
    """
    results = {}
    best_layer, best_norm = -1, -1.0

    for layer_idx in candidate_layers:
        pos_mean = layer_acts[layer_idx]["positive"].mean(dim=0)
        neg_mean = layer_acts[layer_idx]["negative"].mean(dim=0)
        raw_diff = pos_mean - neg_mean
        norm = raw_diff.norm().item()
        unit = F.normalize(raw_diff, dim=0)
        results[layer_idx] = {"raw_diff": raw_diff, "unit": unit, "norm": norm}
        print(f"  Layer {layer_idx:>2}: mean diff norm = {norm:.4f}")
        if norm > best_norm:
            best_norm = norm
            best_layer = layer_idx

    print(f"\n  → Best layer: {best_layer} (norm {best_norm:.4f})")
    return results, best_layer


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Create contrastive steering vectors")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--num-problems", type=int, default=100)
    parser.add_argument("--layers", default="0-33",
                        help="Layer range to scan, e.g. '0-33' or '10,14,18'")
    parser.add_argument("--name", default="expert_vs_student",
                        help="Run name — used for output directory")
    parser.add_argument("--positive", default=DEFAULT_POS_PROMPT,
                        help="Positive direction system prompt")
    parser.add_argument("--negative", default=DEFAULT_NEG_PROMPT,
                        help="Negative direction system prompt")
    parser.add_argument("--pos-label", default="expert",
                        help="Label for positive direction in plots")
    parser.add_argument("--neg-label", default="student",
                        help="Label for negative direction in plots")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract activations even if cached")
    args = parser.parse_args()

    # Parse layer range
    if "-" in args.layers and "," not in args.layers:
        lo, hi = map(int, args.layers.split("-"))
        candidate_layers = list(range(lo, hi + 1))
    else:
        candidate_layers = [int(x) for x in args.layers.split(",")]

    n = args.num_problems
    layers_tag = f"{candidate_layers[0]}-{candidate_layers[-1]}"

    # Create unique run directory
    run_dir = os.path.join("outputs", args.name)
    os.makedirs(run_dir, exist_ok=True)

    activations_path = os.path.join(run_dir, f"activations_layers_{layers_tag}_n{n}.pkl")

    # Try loading cached activations
    if os.path.exists(activations_path) and not args.force:
        print(f"\n═══ Loading cached activations from {activations_path} ═══")
        with open(activations_path, "rb") as f:
            cached = pickle.load(f)
        layer_acts = cached["layer_acts"]
        problems = cached["problems"]
        print(f"Loaded {len(problems)} problems, layers {candidate_layers}")
    else:
        model, tokenizer = load_model(args.model)

        print(f"\n═══ Loading {n} GSM8K problems ═══")
        gsm8k = load_dataset("gsm8k", "main")
        problems = gsm8k["train"]["question"][:n]
        print(f"Loaded {len(problems)} problems")

        print("\n═══ Collecting activations ═══")
        layer_acts = collect_activations(
            model, tokenizer, candidate_layers, problems,
            args.positive, args.negative,
        )

        print(f"\n═══ Saving activations to {activations_path} ═══")
        with open(activations_path, "wb") as f:
            pickle.dump({
                "layer_acts": layer_acts,
                "problems": list(problems),
                "model_name": args.model,
                "candidate_layers": candidate_layers,
                "positive_prompt": args.positive,
                "negative_prompt": args.negative,
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Saved ({os.path.getsize(activations_path) / 1e6:.1f} MB)")

    # PCA plot — always in plots/outputs/ with run name
    print("\n═══ Generating PCA plot ═══")
    plot_dir = "plots/outputs"
    os.makedirs(plot_dir, exist_ok=True)
    plot_title = (f"{args.pos_label} vs {args.neg_label} activations — "
                  f"PCA (n={n})")
    plot_path = os.path.join(plot_dir, f"pca_{args.name}_layers_{layers_tag}_n{n}.png")
    plot_pca_grid(layer_acts, candidate_layers, n,
                  args.pos_label, args.neg_label, plot_title, plot_path)

    # Build steering vectors
    print("\n═══ Building steering vectors ═══")
    sv_results, best_layer = build_steering_vectors(layer_acts, candidate_layers)

    # Save vectors
    print("\n═══ Saving steering vectors ═══")
    sv_paths = {}
    for layer_idx in candidate_layers:
        save_path = os.path.join(run_dir, f"sv_layer_{layer_idx}_n{n}.pt")
        torch.save({
            "steering_vector": sv_results[layer_idx]["raw_diff"],
            "steering_vector_unit": sv_results[layer_idx]["unit"],
            "raw_diff_norm": sv_results[layer_idx]["norm"],
            "model_name": args.model,
            "target_layer": layer_idx,
            "candidate_layers": candidate_layers,
            "positive_prompt": args.positive,
            "negative_prompt": args.negative,
            "pos_label": args.pos_label,
            "neg_label": args.neg_label,
            "num_contrastive_pairs": n,
            "dataset": "gsm8k",
        }, save_path)
        sv_paths[layer_idx] = save_path
        marker = " ← best" if layer_idx == best_layer else ""
        print(f"  {save_path}{marker}")

    best_save = os.path.join(run_dir, f"sv_best_layer_{best_layer}_n{n}.pt")
    torch.save({
        "steering_vector": sv_results[best_layer]["raw_diff"],
        "steering_vector_unit": sv_results[best_layer]["unit"],
        "raw_diff_norm": sv_results[best_layer]["norm"],
        "model_name": args.model,
        "target_layer": best_layer,
        "candidate_layers": candidate_layers,
        "positive_prompt": args.positive,
        "negative_prompt": args.negative,
        "pos_label": args.pos_label,
        "neg_label": args.neg_label,
        "num_contrastive_pairs": n,
        "dataset": "gsm8k",
    }, best_save)
    print(f"  {best_save} (best copy)")

    # Save trace
    trace = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_name": args.name,
        "model": args.model,
        "num_problems": n,
        "candidate_layers": candidate_layers,
        "best_layer": best_layer,
        "layer_norms": {str(l): round(sv_results[l]["norm"], 4)
                        for l in candidate_layers},
        "positive_prompt": args.positive,
        "negative_prompt": args.negative,
        "pos_label": args.pos_label,
        "neg_label": args.neg_label,
        "dataset": "gsm8k",
        "outputs": {
            "activations": activations_path,
            "steering_vectors": {str(l): p for l, p in sv_paths.items()},
            "best_vector": best_save,
            "pca_plot": plot_path,
        },
    }
    trace_path = os.path.join(run_dir, "trace.json")
    with open(trace_path, "w") as f:
        json.dump(trace, f, indent=2)
    print(f"\nTrace saved to {trace_path}")
    print("Done.")


if __name__ == "__main__":
    main()
