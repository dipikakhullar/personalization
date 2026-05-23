"""
Run Gemma-3-4B-IT on GSM8K, MATH, SVAMP, ASDiv, DROP with professor/llm
system prompts via OpenRouter.

Same samples as baselines (seed=42, ~500 per dataset).
Full message traces (system, user, assistant) saved per dataset per persona.

Usage:
  python run_prompted.py                    # run all datasets
  python run_prompted.py --datasets drop    # run only DROP
  python run_prompted.py --datasets drop gsm8k
"""

import argparse
import asyncio
import json
import os
import random
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# ── Config ────────────────────────────────────────────────────────────
MODEL = "google/gemma-3-4b-it"
TEMPERATURE = 0.6
MAX_TOKENS = 4096
N_SAMPLES = 500
CONCURRENCY = 20
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

MATH_ANSWER_INSTRUCTION = (
    "\n\nSolve the problem. Show your reasoning, then put your "
    "final numerical answer after #### on its own line at the end."
)

DROP_ANSWER_INSTRUCTION = (
    "\n\nAnswer based on the passage. Show brief reasoning, then put "
    "your final answer after #### on its own line at the end."
)

PERSONAS_MATH = {
    "professor": (
        "I am a math professor. I verify answers before outputting my final answer."
        + MATH_ANSWER_INSTRUCTION
    ),
    "llm": (
        "I am an LLM. I just do next token prediction."
        + MATH_ANSWER_INSTRUCTION
    ),
}

PERSONAS_DROP = {
    "professor": (
        "I am a math professor. I verify answers before outputting my final answer."
        + DROP_ANSWER_INSTRUCTION
    ),
    "llm": (
        "I am an LLM. I just do next token prediction."
        + DROP_ANSWER_INSTRUCTION
    ),
}

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

# ── Dataset loaders (identical to baselines — same seed, same indices) ─

def load_gsm8k(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    indices = sorted(random.sample(range(len(ds)), min(n, len(ds))))
    samples = []
    for i in indices:
        row = ds[i]
        answer_text = row["answer"]
        final_answer = answer_text.split("####")[-1].strip().replace(",", "")
        samples.append({
            "idx": i,
            "question": row["question"],
            "gold_solution": answer_text,
            "gold_answer": final_answer,
        })
    return samples


def load_math(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="test")
    by_level = {}
    for i, row in enumerate(ds):
        by_level.setdefault(row["level"], []).append(i)
    per_level = max(1, n // len(by_level))
    indices = []
    for level in sorted(by_level):
        pool = by_level[level]
        indices.extend(sorted(random.sample(pool, min(per_level, len(pool)))))
    indices = sorted(indices)[:n]
    samples = []
    for i in indices:
        row = ds[i]
        boxed = extract_boxed(row["solution"])
        samples.append({
            "idx": i,
            "question": row["problem"],
            "level": row["level"],
            "type": row["type"],
            "gold_solution": row["solution"],
            "gold_answer": boxed,
        })
    return samples


def load_svamp(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("ChilleD/SVAMP", split="test")
    indices = sorted(random.sample(range(len(ds)), min(n, len(ds))))
    samples = []
    for i in indices:
        row = ds[i]
        question = row["Body"] + " " + row["Question"]
        samples.append({
            "idx": i,
            "question": question,
            "gold_equation": row["Equation"],
            "gold_answer": str(row["Answer"]),
        })
    return samples


def load_asdiv(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("MU-NLPC/Calc-asdiv_a", split="test")
    indices = sorted(random.sample(range(len(ds)), min(n, len(ds))))
    samples = []
    for i in indices:
        row = ds[i]
        samples.append({
            "idx": i,
            "question": row["question"],
            "gold_answer": row["result"].strip(),
            "gold_answer_float": row["result_float"],
        })
    return samples


def load_drop(n: int) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("ucinlp/drop", split="validation")
    indices = sorted(random.sample(range(len(ds)), min(n, len(ds))))
    samples = []
    for i in indices:
        row = ds[i]
        spans = row["answers_spans"]["spans"]
        types = row["answers_spans"]["types"]
        samples.append({
            "idx": i,
            "passage": row["passage"],
            "question": row["question"],
            "gold_spans": spans,
            "gold_types": types,
            "gold_answer": spans[0] if spans else "",
            "is_drop": True,
        })
    return samples


# ── Answer extraction ─────────────────────────────────────────────────

def extract_boxed(text: str) -> str:
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return ""
    depth, start = 0, idx + 7
    for j in range(start, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            if depth == 0:
                return text[start:j]
            depth -= 1
    return text[start:]


def extract_final_number(text: str) -> str | None:
    m = re.search(r"####\s*([+-]?[\d,]+\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "")
    boxed = extract_boxed(text)
    if boxed:
        nums = re.findall(r"[+-]?[\d,]+\.?\d*", boxed)
        if nums:
            return nums[-1].replace(",", "")
        return boxed
    nums = re.findall(r"[+-]?[\d,]+\.?\d*", text)
    if nums:
        return nums[-1].replace(",", "")
    return None


def normalize_answer(s: str | None) -> str:
    if s is None:
        return ""
    s = s.strip().rstrip(".").replace(",", "")
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return str(f)
    except ValueError:
        return s


def answers_match(pred: str | None, gold: str) -> bool:
    p = normalize_answer(pred)
    g = normalize_answer(gold)
    if not p:
        return False
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-3
    except ValueError:
        return p.strip() == g.strip()


def normalize_drop_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return " ".join(s.split())


def drop_answer_match(pred_text: str, gold_spans: list[str]) -> bool:
    if not pred_text:
        return False
    pred_norm = normalize_drop_text(pred_text)
    for span in gold_spans:
        if normalize_drop_text(span) == pred_norm:
            return True
    try:
        pred_f = float(pred_text.replace(",", ""))
        for span in gold_spans:
            try:
                if abs(pred_f - float(span.replace(",", ""))) < 1e-3:
                    return True
            except ValueError:
                continue
    except ValueError:
        pass
    return False


def extract_drop_answer(text: str) -> str:
    m = re.search(r"####\s*(.+)", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"\*\*(?:Answer|ANSWER|Final Answer)[:\s]*\*\*\s*(.+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(".")
    m = re.search(r"(?:answer|Answer|ANSWER)\s*(?:is|:)\s*(.+?)(?:\.|$)", text)
    if m:
        return m.group(1).strip().rstrip(".")
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if lines:
        return lines[-1].rstrip(".")
    return ""


# ── API call ──────────────────────────────────────────────────────────

sem = asyncio.Semaphore(CONCURRENCY)


async def call_model(system_prompt: str, question: str) -> dict:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    async with sem:
        for attempt in range(5):
            try:
                t0 = time.time()
                resp = await client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                elapsed = time.time() - t0
                content = resp.choices[0].message.content or ""
                return {
                    "messages": messages + [
                        {"role": "assistant", "content": content}
                    ],
                    "response": content,
                    "model": resp.model,
                    "elapsed_s": round(elapsed, 2),
                    "usage": {
                        "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
                        "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
                    },
                }
            except Exception as e:
                wait = 2 ** attempt
                print(f"  Retry {attempt+1}/5 after {wait}s: {e}")
                await asyncio.sleep(wait)
    return {
        "messages": messages + [{"role": "assistant", "content": ""}],
        "response": "", "model": MODEL, "elapsed_s": 0, "error": "max retries",
    }


# ── Run one dataset x one persona ────────────────────────────────────

async def run_dataset_persona(
    dataset_name: str,
    persona_name: str,
    system_prompt: str,
    samples: list[dict],
) -> dict:
    tag = f"{dataset_name}/{persona_name}"
    print(f"\n{'─'*60}")
    print(f"  {tag}: {len(samples)} samples")
    print(f"{'─'*60}")

    is_drop = samples and samples[0].get("is_drop", False)

    if is_drop:
        user_inputs = [
            f"Passage: {s['passage']}\n\nQuestion: {s['question']}"
            for s in samples
        ]
    else:
        user_inputs = [s["question"] for s in samples]

    tasks = [call_model(system_prompt, q) for q in user_inputs]
    results = await asyncio.gather(*tasks)

    correct = 0
    traces = []
    for sample, result in zip(samples, results):
        if is_drop:
            pred = extract_drop_answer(result["response"])
            is_correct = drop_answer_match(pred, sample["gold_spans"])
        else:
            pred = extract_final_number(result["response"])
            is_correct = answers_match(pred, sample["gold_answer"])
        if is_correct:
            correct += 1

        trace = {
            **sample,
            "persona": persona_name,
            "messages": result["messages"],
            "model_answer": pred,
            "correct": is_correct,
            "elapsed_s": result["elapsed_s"],
            "usage": result.get("usage"),
        }
        traces.append(trace)

    accuracy = correct / len(samples) if samples else 0
    print(f"  {correct}/{len(samples)} = {accuracy:.1%}")

    output = {
        "dataset": dataset_name,
        "persona": persona_name,
        "system_prompt": system_prompt,
        "model": MODEL,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "n_samples": len(samples),
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": len(samples),
        "traces": traces,
    }

    if dataset_name == "math" and samples and "level" in samples[0]:
        by_level = {}
        for t in traces:
            lvl = t.get("level", "?")
            by_level.setdefault(lvl, {"correct": 0, "total": 0})
            by_level[lvl]["total"] += 1
            if t["correct"]:
                by_level[lvl]["correct"] += 1
        for lvl in sorted(by_level):
            b = by_level[lvl]
            pct = b["correct"] / b["total"] if b["total"] else 0
            print(f"    {lvl}: {b['correct']}/{b['total']} = {pct:.1%}")
            b["accuracy"] = round(pct, 4)
        output["by_level"] = by_level

    out_path = OUTPUT_DIR / f"{dataset_name}_{persona_name}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved → {out_path}")

    return output


# ── Main ──────────────────────────────────────────────────────────────

ALL_DATASETS = {
    "gsm8k": load_gsm8k,
    "math": load_math,
    "svamp": load_svamp,
    "asdiv": load_asdiv,
    "drop": load_drop,
}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=list(ALL_DATASETS.keys()),
                        help="Datasets to run (default: all)")
    args = parser.parse_args()

    random.seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Model: {MODEL}")
    print(f"Temperature: {TEMPERATURE}, Max tokens: {MAX_TOKENS}")
    print(f"Samples per dataset: ~{N_SAMPLES}")
    print(f"Personas: professor, llm")

    to_run = args.datasets if args.datasets else list(ALL_DATASETS.keys())
    datasets = {}
    for name, loader in ALL_DATASETS.items():
        loaded = loader(N_SAMPLES)
        if name in to_run:
            datasets[name] = loaded

    summary_path = OUTPUT_DIR / "summary.json"
    if summary_path.exists():
        existing = json.load(open(summary_path))
        summaries = existing.get("results", {})
    else:
        summaries = {}

    for ds_name, samples in datasets.items():
        print(f"\n{'═'*60}")
        print(f"  DATASET: {ds_name} ({len(samples)} samples)")
        print(f"{'═'*60}")

        is_drop = samples and samples[0].get("is_drop", False)
        personas = PERSONAS_DROP if is_drop else PERSONAS_MATH

        for persona_name, system_prompt in personas.items():
            result = await run_dataset_persona(
                ds_name, persona_name, system_prompt, samples,
            )
            key = f"{ds_name}_{persona_name}"
            summaries[key] = {
                "dataset": ds_name,
                "persona": persona_name,
                "accuracy": result["accuracy"],
                "correct": result["correct"],
                "total": result["total"],
            }
            if "by_level" in result:
                summaries[key]["by_level"] = result["by_level"]

    with open(summary_path, "w") as f:
        json.dump({
            "model": MODEL,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "results": summaries,
        }, f, indent=2)

    print(f"\n{'═'*60}")
    print("  SUMMARY — prompted personalization")
    print(f"{'═'*60}")
    print(f"  {'dataset':>8}  {'professor':>10}  {'llm':>10}")
    print(f"  {'─'*8}  {'─'*10}  {'─'*10}")
    for ds_name in datasets:
        p = summaries.get(f"{ds_name}_professor", {})
        l = summaries.get(f"{ds_name}_llm", {})
        pa = f"{p.get('correct','?')}/{p.get('total','?')}"
        la = f"{l.get('correct','?')}/{l.get('total','?')}"
        print(f"  {ds_name:>8}  {pa:>10}  {la:>10}")
    print(f"\nSaved summary → {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
