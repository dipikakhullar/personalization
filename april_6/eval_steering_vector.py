"""
Evaluate a steering vector on math benchmarks at varying alpha values.

Runs baseline + steered generation across GSM8K, MATH, SVAMP, ASDiv.
Extracts numerical answers and checks correctness.

Outputs go to: steering_vector_evals/{vector_name}/
  eval_{dataset}_{alpha}.json   — per-problem results
  summary.json                  — aggregate metrics across all runs

Usage:
  python eval_steering_vector.py \\
      --vector outputs/expert_vs_student/sv_best_layer_33_n100.pt \\
      --alpha 0,2,4,8 --samples 25

  python eval_steering_vector.py \\
      --vector outputs/professor_vs_llm/sv_best_layer_33_n100.pt \\
      --alpha 0,2,4,8,-2,-4,-8 --samples 25
"""

import argparse
import json
import os
import re
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

MAX_NEW_TOKENS = 4096


# ── Dataset loaders ──────────────────────────────────────────────────

def load_gsm8k(n: int):
    ds = load_dataset("gsm8k", "main")["test"]
    items = []
    for row in ds:
        # Answer is after "####"
        answer_str = row["answer"].split("####")[-1].strip().replace(",", "")
        items.append({
            "question": row["question"],
            "gold": answer_str,
            "dataset": "gsm8k",
            "meta": {},
        })
        if len(items) >= n:
            break
    return items


def load_math(n: int):
    ds = load_dataset("qwedsacf/competition_math")["train"]
    items = []
    for row in ds:
        gold = extract_boxed(row["solution"])
        if gold is None:
            continue
        items.append({
            "question": row["problem"],
            "gold": gold,
            "dataset": "math",
            "meta": {"level": row["level"], "type": row["type"]},
        })
        if len(items) >= n:
            break
    return items


def load_svamp(n: int):
    ds = load_dataset("ChilleD/SVAMP")["test"]
    items = []
    for row in ds:
        items.append({
            "question": row["Body"] + " " + row["Question"],
            "gold": str(row["Answer"]),
            "dataset": "svamp",
            "meta": {},
        })
        if len(items) >= n:
            break
    return items


def load_asdiv(n: int):
    ds = load_dataset("EleutherAI/asdiv")["validation"]
    items = []
    for row in ds:
        # Parse numeric answer from strings like "9 (apples)"
        gold = parse_numeric(row["answer"])
        if gold is None:
            continue
        items.append({
            "question": row["body"] + " " + row["question"],
            "gold": gold,
            "dataset": "asdiv",
            "meta": {},
        })
        if len(items) >= n:
            break
    return items


DATASET_LOADERS = {
    "gsm8k": load_gsm8k,
    "math": load_math,
    "svamp": load_svamp,
    "asdiv": load_asdiv,
}


# ── Answer extraction helpers ────────────────────────────────────────

def extract_boxed(text: str) -> str | None:
    """Extract content from the last \\boxed{...} in text, handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start:i-1].strip()


def parse_numeric(text: str) -> str | None:
    """Extract a number from text like '9 (apples)' or '32 cents'."""
    m = re.search(r"[-+]?\d*\.?\d+(?:/\d+)?", text)
    return m.group(0) if m else None


def extract_answer_from_response(response: str) -> str | None:
    """Try to extract a final numerical answer from model output."""
    # Try \boxed{} first
    boxed = extract_boxed(response)
    if boxed is not None:
        num = parse_numeric(boxed)
        if num is not None:
            return num
        return boxed

    # Try "#### answer" format
    if "####" in response:
        after = response.split("####")[-1].strip()
        num = parse_numeric(after)
        if num is not None:
            return num

    # Try "the answer is X" patterns
    for pattern in [
        r"(?:the\s+)?answer\s+is\s*:?\s*([-+]?\d*\.?\d+)",
        r"(?:=\s*)([-+]?\d*\.?\d+)\s*$",
        r"\*\*([-+]?\d*\.?\d+)\*\*\s*$",
    ]:
        m = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1)

    # Last number in the response
    nums = re.findall(r"[-+]?\d*\.?\d+", response)
    return nums[-1] if nums else None


def normalize_answer(ans: str) -> str:
    """Normalize for comparison: strip whitespace, remove trailing .0, commas."""
    ans = ans.strip().replace(",", "").replace(" ", "")
    # Remove trailing .0 / .00 etc
    if re.match(r"^-?\d+\.0+$", ans):
        ans = ans.split(".")[0]
    return ans


def check_correct(predicted: str | None, gold: str) -> bool:
    if predicted is None:
        return False
    pred_norm = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)
    if pred_norm == gold_norm:
        return True
    # Try numeric comparison
    try:
        # Handle fractions
        if "/" in pred_norm:
            parts = pred_norm.split("/")
            pred_val = float(parts[0]) / float(parts[1])
        else:
            pred_val = float(pred_norm)
        if "/" in gold_norm:
            parts = gold_norm.split("/")
            gold_val = float(parts[0]) / float(parts[1])
        else:
            gold_val = float(gold_norm)
        return abs(pred_val - gold_val) < 1e-5
    except (ValueError, ZeroDivisionError):
        return False


# ── Model / generation ───────────────────────────────────────────────

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


def make_steering_hook(steering_vec: torch.Tensor, alpha: float):
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


def generate_batch(model, tokenizer, questions: list[str], layer_idx: int,
                   steering_hook=None, max_new_tokens=MAX_NEW_TOKENS) -> list[str]:
    """Generate responses for a batch of questions."""
    prompts = []
    for q in questions:
        messages = [{"role": "user", "content": q}]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        ))

    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True,
    ).to(model.device)

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

    # Decode only the new tokens for each sequence
    responses = []
    input_len = inputs["input_ids"].shape[1]
    for i in range(len(questions)):
        new_tokens = output_ids[i][input_len:]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return responses


# ── Main eval loop ──────────────────────────────────────────────────

BATCH_SIZE = 8


def run_eval_dataset(model, tokenizer, items: list[dict], sv: torch.Tensor,
                     layer_idx: int, alpha: float, dataset_name: str,
                     eval_path: str):
    """Evaluate one dataset at one alpha with batched generation.
    Writes results incrementally to eval_path after each batch."""
    hook = make_steering_hook(sv, alpha) if alpha != 0.0 else None
    results = []

    for batch_start in range(0, len(items), BATCH_SIZE):
        batch_items = items[batch_start:batch_start + BATCH_SIZE]
        questions = [item["question"] for item in batch_items]

        t0 = time.time()
        responses = generate_batch(model, tokenizer, questions, layer_idx,
                                   steering_hook=hook)
        elapsed = time.time() - t0

        for j, (item, response) in enumerate(zip(batch_items, responses)):
            idx = batch_start + j
            predicted = extract_answer_from_response(response)
            correct = check_correct(predicted, item["gold"])
            word_count = len(response.split())

            results.append({
                "question": item["question"],
                "gold": item["gold"],
                "predicted": predicted,
                "correct": correct,
                "word_count": word_count,
                "response": response,
                "meta": item.get("meta", {}),
            })

            status = "✓" if correct else "✗"
            print(f"  [{idx+1:>3}/{len(items)}] {status} pred={predicted} "
                  f"gold={item['gold']} words={word_count}")

        # Incremental write after each batch
        accuracy_so_far = sum(r["correct"] for r in results) / len(results)
        avg_words_so_far = sum(r["word_count"] for r in results) / len(results)
        with open(eval_path, "w") as f:
            json.dump({
                "dataset": dataset_name,
                "alpha": alpha,
                "layer": layer_idx,
                "accuracy": round(accuracy_so_far, 4),
                "avg_words": round(avg_words_so_far, 1),
                "num_items": len(results),
                "complete": batch_start + len(batch_items) >= len(items),
                "items": results,
            }, f, indent=2)

        print(f"    batch {batch_start//BATCH_SIZE + 1} "
              f"({len(batch_items)} items, {elapsed:.1f}s) "
              f"acc={accuracy_so_far:.1%}")

    accuracy = sum(r["correct"] for r in results) / len(results) if results else 0
    avg_words = sum(r["word_count"] for r in results) / len(results) if results else 0
    print(f"  → {dataset_name} α={alpha}: accuracy={accuracy:.1%} "
          f"avg_words={avg_words:.0f}")

    return results, accuracy, avg_words


def main():
    parser = argparse.ArgumentParser(description="Evaluate steering vector on math benchmarks")
    parser.add_argument("--vector", required=True,
                        help="Path to .pt file")
    parser.add_argument("--alpha", default="0,2,4,8",
                        help="Comma-separated alpha values (0 = baseline)")
    parser.add_argument("--samples", type=int, default=25,
                        help="Number of samples per dataset")
    parser.add_argument("--model", default=None)
    parser.add_argument("--datasets", default="gsm8k,math,svamp,asdiv",
                        help="Comma-separated dataset names")
    args = parser.parse_args()

    alphas = [float(x) for x in args.alpha.split(",")]
    dataset_names = [d.strip() for d in args.datasets.split(",")]

    # Load steering vector
    sv_data = torch.load(args.vector, weights_only=True)
    sv = sv_data["steering_vector"]
    layer_idx = sv_data["target_layer"]
    model_name = args.model or sv_data["model_name"]
    pos_label = sv_data.get("pos_label", "positive")
    neg_label = sv_data.get("neg_label", "negative")

    # Derive vector name from path for output dir
    vector_name = os.path.basename(os.path.dirname(args.vector))
    if not vector_name or vector_name == "outputs":
        vector_name = os.path.splitext(os.path.basename(args.vector))[0]

    eval_dir = os.path.join("steering_vector_evals", vector_name)
    os.makedirs(eval_dir, exist_ok=True)

    print(f"Vector: {args.vector}")
    print(f"  layer={layer_idx}, pos={pos_label}, neg={neg_label}")
    print(f"  alphas={alphas}")
    print(f"  datasets={dataset_names}, samples={args.samples}")
    print(f"  output dir: {eval_dir}")

    model, tokenizer = load_model(model_name)

    # Load all datasets
    print("\n═══ Loading datasets ═══")
    all_items = {}
    for ds_name in dataset_names:
        loader = DATASET_LOADERS.get(ds_name)
        if loader is None:
            print(f"  Unknown dataset: {ds_name}, skipping")
            continue
        items = loader(args.samples)
        all_items[ds_name] = items
        print(f"  {ds_name}: {len(items)} items loaded")

    # Run evals
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "vector": args.vector,
        "model": model_name,
        "layer": layer_idx,
        "pos_label": pos_label,
        "neg_label": neg_label,
        "samples_per_dataset": args.samples,
        "alphas": alphas,
        "results": {},
    }

    summary_path = os.path.join(eval_dir, "summary.json")

    for ds_name, items in all_items.items():
        print(f"\n{'═'*70}")
        print(f"  DATASET: {ds_name} ({len(items)} items)")
        print(f"{'═'*70}")

        summary["results"][ds_name] = {}

        for alpha in alphas:
            alpha_label = f"a{alpha}" if alpha >= 0 else f"a_neg{abs(alpha)}"
            print(f"\n  ── α = {alpha} ──")

            eval_path = os.path.join(eval_dir, f"eval_{ds_name}_{alpha_label}.json")
            results, accuracy, avg_words = run_eval_dataset(
                model, tokenizer, items, sv, layer_idx, alpha, ds_name,
                eval_path,
            )

            summary["results"][ds_name][str(alpha)] = {
                "accuracy": round(accuracy, 4),
                "avg_words": round(avg_words, 1),
                "num_correct": sum(r["correct"] for r in results),
                "num_items": len(results),
                "eval_file": eval_path,
            }

            # Incremental summary write
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

    # Print summary table
    print(f"\n{'═'*70}")
    print("  SUMMARY")
    print(f"{'═'*70}")
    print(f"  {'dataset':<12s}", end="")
    for a in alphas:
        print(f"  {'α='+str(a):>10s}", end="")
    print()

    for ds_name in all_items:
        print(f"  {ds_name:<12s}", end="")
        for a in alphas:
            acc = summary["results"][ds_name][str(a)]["accuracy"]
            print(f"  {acc:>9.1%}", end="")
        print()

    print(f"\n  {'dataset':<12s}", end="")
    for a in alphas:
        print(f"  {'α='+str(a):>10s}", end="")
    print("   (avg words)")

    for ds_name in all_items:
        print(f"  {ds_name:<12s}", end="")
        for a in alphas:
            words = summary["results"][ds_name][str(a)]["avg_words"]
            print(f"  {words:>10.0f}", end="")
        print()

    print(f"\nSummary saved to {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
