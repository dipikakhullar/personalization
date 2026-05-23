"""
Build a steering vector that moves Gemma-3-4B-IT along the
expert ↔ student axis.

Positive direction  → expert  (concise, answer-only)
Negative direction  → student (step-by-step, pedagogical)

Method: contrastive activation extraction.
  1. Run diverse prompts through the model twice — once with a student
     system prompt, once with an expert system prompt.
  2. Collect the residual-stream hidden state at a target layer
     (last-token position, which conditions the upcoming generation).
  3. steering_vector = normalize(mean(h_expert) − mean(h_student))
  4. At inference, add  α · steering_vector  to the target layer
     to nudge the model toward either end of the axis.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────
MODEL_NAME = "google/gemma-3-4b-it"
CANDIDATE_LAYERS = list(range(10, 15))  # layers 10-14
STEER_ALPHA = 4.0          # steering strength for the demo
SAVE_PATH = "expert_student_steering_vector.pt"
NUM_PROBLEMS = 5           # number of GSM8K problems to use

STUDENT_SYSTEM = (
    "You are helping a student who is learning this topic for the first "
    "time. They need to see every step explained clearly and thoroughly. "
    "Show all working, define terms, and walk through the reasoning. "
    "A bare answer with no explanation is useless to them."
)

EXPERT_SYSTEM = (
    "You are helping an expert engineer/researcher who just needs the "
    "answer or a compact solution. Do not show unnecessary working or "
    "pedagogical explanation — they will verify the details themselves. "
    "Be direct and concise."
)

def load_gsm8k_problems(n: int = NUM_PROBLEMS) -> list[str]:
    """Load the first n problems from GSM8K train split."""
    gsm8k = load_dataset("gsm8k", "main")
    return gsm8k["train"]["question"][:n]


# ── Helpers ───────────────────────────────────────────────────────────

def load_model():
    print(f"Loading {MODEL_NAME} …")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def get_language_model_layers(model):
    """Return the transformer layer list, handling Gemma 3's nested architecture."""
    # Gemma 3: model.model.language_model.layers
    if hasattr(model.model, "language_model"):
        return model.model.language_model.layers
    # Gemma 2 / standard: model.model.layers
    return model.model.layers


def extract_hidden(model, tokenizer, system_prompt: str, question: str,
                   layer_idx: int) -> torch.Tensor:
    """Forward-pass only; return the last-token hidden state at `layer_idx`.

    Uses a forward hook on the target layer to capture the hidden state
    directly, which works reliably across both standard and multimodal
    (Gemma 3) architectures.
    """
    messages = [
        {"role": "user", "content": f"{system_prompt}\n\n{question}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Use a hook to capture the layer output directly
    captured = {}
    target_layer = get_language_model_layers(model)[layer_idx]

    def capture_hook(_module, _input, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = h.detach()

    handle = target_layer.register_forward_hook(capture_hook)
    with torch.no_grad():
        model(**inputs)
    handle.remove()

    h = captured["hidden"]                 # [1, seq_len, hidden_dim]
    last_tok = h[0, -1, :].cpu().float()
    return last_tok


def build_steering_vector(model, tokenizer, candidate_layers: list[int],
                          problems: list[str]):
    """
    Contrastive extraction over GSM8K problems.

    For each candidate layer, compute the mean difference vector.
    Pick the layer whose mean difference has the largest norm
    (i.e. the layer where expert vs student is most separable).
    Return the normalized steering vector and chosen layer index.
    """
    # Collect per-layer differences
    layer_diffs: dict[int, list[torch.Tensor]] = {l: [] for l in candidate_layers}

    for i, q in enumerate(problems):
        print(f"  [{i+1:>3}/{len(problems)}] {q[:60]}")
        for layer_idx in candidate_layers:
            h_expert = extract_hidden(model, tokenizer, EXPERT_SYSTEM, q, layer_idx)
            h_student = extract_hidden(model, tokenizer, STUDENT_SYSTEM, q, layer_idx)
            layer_diffs[layer_idx].append(h_expert - h_student)

    # Pick the layer with largest mean-difference norm
    best_layer = -1
    best_norm = -1.0
    best_sv = None

    for layer_idx in candidate_layers:
        mean_diff = torch.stack(layer_diffs[layer_idx]).mean(dim=0)
        norm = mean_diff.norm().item()
        print(f"  Layer {layer_idx:>2}: mean diff norm = {norm:.4f}")
        if norm > best_norm:
            best_norm = norm
            best_layer = layer_idx
            best_sv = mean_diff

    sv = F.normalize(best_sv, dim=0)
    print(f"\n  → Best layer: {best_layer} (norm {best_norm:.4f})")
    return sv, best_layer


def make_steering_hook(steering_vec: torch.Tensor, alpha: float):
    """Forward hook that adds α·v to the hidden states."""
    def hook(_module, _input, output):
        if isinstance(output, torch.Tensor):
            h = output
        else:
            h = output[0]

        v = steering_vec.to(device=h.device, dtype=h.dtype)
        h_steered = h + alpha * v
        if isinstance(output, torch.Tensor):
            return h_steered
        return (h_steered,) + tuple(output[i] for i in range(1, len(output)))
    return hook


def generate(model, tokenizer, question: str, layer_idx: int,
             steering_hook=None, max_new_tokens=300):
    """Generate a response, optionally with a steering hook applied."""
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    target_layer = get_language_model_layers(model)[layer_idx]
    handle = None
    if steering_hook is not None:
        handle = target_layer.register_forward_hook(steering_hook)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    if handle is not None:
        handle.remove()

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    model, tokenizer = load_model()

    # 1. Load GSM8K problems
    print("\n═══ Loading GSM8K problems ═══")
    problems = load_gsm8k_problems(NUM_PROBLEMS)
    print(f"Loaded {len(problems)} problems")

    # 2. Build the steering vector (scans layers 10-14, picks best)
    print("\n═══ Building steering vector ═══")
    sv, best_layer = build_steering_vector(
        model, tokenizer, CANDIDATE_LAYERS, problems,
    )
    print(f"\nSteering vector shape: {sv.shape}")
    print(f"Norm (should be ~1): {sv.norm().item():.4f}")
    print(f"Selected layer: {best_layer}")

    torch.save({
        "steering_vector": sv,
        "model_name": MODEL_NAME,
        "target_layer": best_layer,
        "candidate_layers": CANDIDATE_LAYERS,
        "expert_system": EXPERT_SYSTEM,
        "student_system": STUDENT_SYSTEM,
        "num_contrastive_pairs": len(problems),
        "dataset": "gsm8k",
    }, SAVE_PATH)
    print(f"Saved to {SAVE_PATH}")

    # 3. Demo: generate with and without steering
    test_questions = [
        "Explain how backpropagation works.",
        "What is the difference between a process and a thread?",
        "How do you invert a binary tree?",
    ]

    for q in test_questions:
        print(f"\n{'═'*70}")
        print(f"Q: {q}")

        print(f"\n{'─'*70}")
        print("▸ BASELINE (no steering):")
        print(generate(model, tokenizer, q, best_layer))

        print(f"\n{'─'*70}")
        print(f"▸ EXPERT steering (α = +{STEER_ALPHA}):")
        hook = make_steering_hook(sv, alpha=+STEER_ALPHA)
        print(generate(model, tokenizer, q, best_layer, steering_hook=hook))

        print(f"\n{'─'*70}")
        print(f"▸ STUDENT steering (α = −{STEER_ALPHA}):")
        hook = make_steering_hook(sv, alpha=-STEER_ALPHA)
        print(generate(model, tokenizer, q, best_layer, steering_hook=hook))


if __name__ == "__main__":
    main()
