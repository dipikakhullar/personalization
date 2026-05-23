"""Generate a behavioral persona-style Q&A dataset via few-shot prompting.

For each (axis, pole) bucket we expand 3 hand-written seeds into ~30 records by
having three OpenRouter models (Kimi-K2.5, Claude-Sonnet-4.6, GPT-5.5) each
contribute ~10 records across two few-shot calls, sampling 4 prior examples
per call to drive diversity.

Output: outputs/dataset.jsonl (pooled) + outputs/by_axis/<axis>.jsonl.

Usage:
  python generate_dataset.py
  python generate_dataset.py --axes verbosity creativity   # subset
  python generate_dataset.py --calls-per-model 1           # cheaper smoke test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import string
import time
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from axes import AXES, SEEDS

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# ── Config ────────────────────────────────────────────────────────────
MODELS = [
    "moonshotai/Kimi-K2.5",
    "anthropic/claude-sonnet-4.6",
    "openai/gpt-5.5",
]
RECORDS_PER_CALL = 5
CALLS_PER_MODEL_DEFAULT = 2
FEWSHOT_K = 4
TEMPERATURE = 0.9
MAX_TOKENS = 6000
CONCURRENCY = 8
RETRY_ATTEMPTS = 5
SEED_TAG = "human_seed"

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DATASET_PATH = OUTPUT_DIR / "dataset.jsonl"
BY_AXIS_DIR = OUTPUT_DIR / "by_axis"

REQUIRED_FIELDS = (
    "user_prompt",
    "assistant_response_a",
    "assistant_response_b",
    "correct_response",
)

client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

sem = asyncio.Semaphore(CONCURRENCY)


# ── Trait words to filter from user_prompt (soft filter) ─────────────
# Words that would directly name the axis the model is supposed to be inferring.
# If any appear in a generated user_prompt, we drop the record.
TRAIT_WORDS = {
    "verbosity": [
        "concise", "concisely", "brief", "briefly", "short answer", "in short",
        "detailed", "in detail", "comprehensive", "thoroughly", "long answer",
        "tldr", "tl;dr", "in depth", "in-depth", "verbose",
    ],
    "social_style": [
        "warmly", "be warm", "be empathetic", "encouraging", "be supportive",
        "no fluff", "matter-of-fact", "clinical", "be neutral", "without emotion",
    ],
    "guidance": [
        "anticipate", "proactive", "proactively", "warn me about", "tell me everything",
        "just answer", "don't elaborate", "no extra info", "only what i ask",
    ],
    "confidence": [
        "be decisive", "pick one", "no hedging", "don't hedge",
        "hedge", "be cautious", "trade-offs", "list pros and cons",
    ],
    "teaching_style": [
        "ask me questions", "socratic", "lead me", "don't just tell me",
        "just tell me", "don't ask me", "give me the answer directly",
    ],
    "technicality": [
        "use jargon", "be technical", "expert level", "be advanced",
        "explain like i'm five", "eli5", "no jargon", "in plain english",
        "simple terms", "for a beginner",
    ],
    "planning": [
        "in bullet points", "as a list", "use bullets", "step by step",
        "numbered list", "in prose", "no bullets", "no list", "as a paragraph",
    ],
    "creativity": [
        "be creative", "brainstorm", "weird ideas", "many ideas",
        "boring is fine", "the obvious one", "the standard answer", "the practical one",
    ],
}


def _normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def _has_trait_word(prompt: str, axis: str) -> str | None:
    p = " " + prompt.lower() + " "
    for w in TRAIT_WORDS.get(axis, []):
        if " " + w + " " in p or p.startswith(w + " ") or p.endswith(" " + w):
            return w
    return None


# ── Prompt construction ──────────────────────────────────────────────

SYSTEM_PROMPT = """You generate evaluation items for a benchmark that tests \
whether language models can infer a user's latent communication preferences \
from behavioral signals. Each item is a forced-choice question:

- A `user_prompt` (a realistic message a user might send to an assistant).
- Two candidate assistant responses, A and B, that BOTH correctly answer the \
prompt on substance.
- The two responses differ ONLY along the named personality axis. All other \
attributes (length where length is not the axis, factual content, register \
where register is not the axis) should be held roughly constant.
- A `correct_response` field naming whichever of A or B matches the target \
persona pole.

Hard rules:
1. The `user_prompt` must NEVER name the axis or instruct the assistant on \
how to respond. No 'be brief', 'in detail', 'be friendly', 'list the steps', \
'use bullets', 'in plain English', etc. The preference must be inferable only \
from the response styles, never from the prompt.
2. The two responses must differ on the specified axis in a clear, \
distinguishable way. The non-target response must be a plausible, competent \
answer that a different user would prefer — not a strawman, not bad, not \
obviously wrong.
3. Vary the domain widely across records: cooking, code, travel, fitness, \
finance, relationships, science, hobbies, work, parenting, art, etc.
4. Surface paraphrases of the provided examples are forbidden. Pick fresh \
domains and prompts each time.
5. Keep `user_prompt` to 1–3 sentences, written in a natural conversational \
voice — not stiff, not formal.

Output a JSON array of records, no prose around it. Each record has exactly \
these fields:
  user_prompt, assistant_response_a, assistant_response_b, correct_response

`correct_response` is "A" or "B".
"""


def build_user_prompt(axis_spec: dict, target_pole: str, examples: list[dict],
                      n: int, force_label: str | None) -> str:
    pos = axis_spec["positive_pole"]
    neg = axis_spec["negative_pole"]
    target_desc = (axis_spec["positive_desc"] if target_pole == pos
                   else axis_spec["negative_desc"])
    other_pole = neg if target_pole == pos else pos
    other_desc = (axis_spec["negative_desc"] if target_pole == pos
                  else axis_spec["positive_desc"])

    parts = [
        f"AXIS: {axis_spec['name']}",
        f"  Pole '{pos}': {axis_spec['positive_desc']}",
        f"  Pole '{neg}': {axis_spec['negative_desc']}",
        "",
        f"TARGET PERSONA for this batch: {target_pole}",
        f"  → The CORRECT response in each item should match: {target_desc}",
        f"  → The OTHER response should match: {other_desc}",
        "",
        f"EXAMPLES ({len(examples)} sampled from existing items in this bucket):",
    ]
    for i, ex in enumerate(examples, 1):
        parts.append(f"\nExample {i}:")
        parts.append(json.dumps({
            "user_prompt": ex["user_prompt"],
            "assistant_response_a": ex["assistant_response_a"],
            "assistant_response_b": ex["assistant_response_b"],
            "correct_response": ex["correct_response"],
        }, ensure_ascii=False, indent=2))

    parts.append("")
    parts.append(
        f"Now produce a JSON array of {n} NEW records for axis "
        f"'{axis_spec['name']}', target persona '{target_pole}'. Domains must "
        f"differ from the examples and from each other. Do not name the trait "
        f"in user_prompt."
    )
    if force_label:
        parts.append(
            f"\nIMPORTANT: For this batch, set correct_response to \"{force_label}\" "
            f"on EVERY record (we are rebalancing A/B in this bucket). The "
            f"target-persona content stays the same; just put it in slot "
            f"{force_label} and the other-persona content in the other slot."
        )

    parts.append(
        "\nReturn ONLY the JSON array. No commentary before or after. No code "
        "fences. The first character of your response must be '[' and the last "
        "must be ']'."
    )
    return "\n".join(parts)


# ── API call ─────────────────────────────────────────────────────────

def _extract_json_array(text: str):
    """Try to parse a JSON array from the model's response."""
    text = text.strip()
    if text.startswith("```"):
        # strip code fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


async def call_model(model: str, system: str, user: str) -> tuple[list[dict] | None, str]:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    async with sem:
        for attempt in range(RETRY_ATTEMPTS):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                content = resp.choices[0].message.content or ""
                arr = _extract_json_array(content)
                if isinstance(arr, list):
                    return arr, content
                # fall through to retry on parse failure
                last_err = f"could not parse JSON array (got {content[:200]!r})"
            except Exception as e:
                last_err = repr(e)
            wait = 2 ** attempt
            print(f"    [{model}] retry {attempt+1}/{RETRY_ATTEMPTS} after {wait}s — {last_err}")
            await asyncio.sleep(wait)
    return None, ""


# ── Per-bucket expansion loop ────────────────────────────────────────

async def expand_bucket(axis_spec: dict, target_pole: str,
                        calls_per_model: int) -> list[dict]:
    axis = axis_spec["name"]
    bucket = list(SEEDS[(axis, target_pole)])  # start with seeds
    seen_norms = {_normalize_text(r["user_prompt"]) for r in bucket}
    print(f"  ▶ {axis}/{target_pole}: starting with {len(bucket)} seeds")

    # Round-robin: m1c1, m2c1, m3c1, m1c2, m2c2, m3c2
    for call_idx in range(calls_per_model):
        for model in MODELS:
            # A/B balance check: if skewed beyond 65/35, force the underrepresented label
            label_counts = Counter(r["correct_response"] for r in bucket)
            total = label_counts["A"] + label_counts["B"]
            force_label = None
            if total >= 6:
                a_frac = label_counts["A"] / total
                if a_frac < 0.35:
                    force_label = "A"
                elif a_frac > 0.65:
                    force_label = "B"

            examples = random.sample(bucket, min(FEWSHOT_K, len(bucket)))
            user_msg = build_user_prompt(
                axis_spec, target_pole, examples, RECORDS_PER_CALL, force_label,
            )
            arr, _raw = await call_model(model, SYSTEM_PROMPT, user_msg)
            if arr is None:
                print(f"    ✗ {model} call {call_idx+1}: failed all retries")
                continue

            kept = 0
            for rec in arr:
                if not isinstance(rec, dict):
                    continue
                if not all(k in rec for k in REQUIRED_FIELDS):
                    continue
                if rec["correct_response"] not in {"A", "B"}:
                    continue
                up = rec["user_prompt"].strip()
                if not up:
                    continue
                bad_word = _has_trait_word(up, axis)
                if bad_word:
                    continue
                norm = _normalize_text(up)
                if norm in seen_norms:
                    continue
                seen_norms.add(norm)
                bucket.append({
                    "axis": axis,
                    "target_persona": target_pole,
                    "user_prompt": up,
                    "assistant_response_a": rec["assistant_response_a"].strip(),
                    "assistant_response_b": rec["assistant_response_b"].strip(),
                    "correct_response": rec["correct_response"],
                    "generated_by": model,
                })
                kept += 1
            print(f"    + {model} call {call_idx+1}: kept {kept}/{len(arr)}"
                  + (f" (forced label {force_label})" if force_label else ""))

    return bucket


# ── Main ─────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--axes", nargs="+", default=None,
                        help="Subset of axes to run (default: all)")
    parser.add_argument("--calls-per-model", type=int, default=CALLS_PER_MODEL_DEFAULT,
                        help=f"Calls per model per bucket (default: {CALLS_PER_MODEL_DEFAULT})")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    BY_AXIS_DIR.mkdir(parents=True, exist_ok=True)

    selected_axes = AXES if not args.axes else [a for a in AXES if a["name"] in args.axes]
    if not selected_axes:
        raise SystemExit(f"No axes match {args.axes}")

    print(f"Models: {MODELS}")
    print(f"Calls/model/bucket: {args.calls_per_model} × {RECORDS_PER_CALL} records = "
          f"~{args.calls_per_model * RECORDS_PER_CALL} per model per pole")
    print(f"Axes: {[a['name'] for a in selected_axes]}")
    print(f"Concurrency: {CONCURRENCY}")
    print()

    # Run all buckets in parallel; calls within a bucket are sequential so the
    # few-shot pool grows iteratively.
    t0 = time.time()
    tasks = []
    for ax in selected_axes:
        for pole in (ax["positive_pole"], ax["negative_pole"]):
            tasks.append(expand_bucket(ax, pole, args.calls_per_model))
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - t0
    print(f"\nGeneration finished in {elapsed:.1f}s")

    # ── Write outputs ────────────────────────────────────────────────
    all_records: list[dict] = []
    by_axis: dict[str, list[dict]] = defaultdict(list)
    for bucket_records in results:
        for r in bucket_records:
            all_records.append(r)
            by_axis[r["axis"]].append(r)

    with open(DATASET_PATH, "w") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for axis_name, recs in by_axis.items():
        with open(BY_AXIS_DIR / f"{axis_name}.jsonl", "w") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Report ───────────────────────────────────────────────────────
    print(f"\nWrote {len(all_records)} records → {DATASET_PATH}")

    counts = Counter()
    by_model = defaultdict(Counter)
    label_dist = defaultdict(Counter)
    for r in all_records:
        key = (r["axis"], r["target_persona"])
        counts[key] += 1
        by_model[key][r["generated_by"]] += 1
        label_dist[key][r["correct_response"]] += 1

    print("\nCounts per bucket:")
    print(f"  {'axis':<16} {'pole':<20} {'total':>5}  {'seeds':>5}  "
          f"{'kimi':>5}  {'claude':>6}  {'gpt':>5}  A/B")
    for ax in selected_axes:
        for pole in (ax["positive_pole"], ax["negative_pole"]):
            k = (ax["name"], pole)
            n = counts[k]
            seed_n = by_model[k][SEED_TAG]
            kimi = by_model[k]["moonshotai/Kimi-K2.5"]
            claude = by_model[k]["anthropic/claude-sonnet-4.6"]
            gpt = by_model[k]["openai/gpt-5.5"]
            a = label_dist[k]["A"]
            b = label_dist[k]["B"]
            print(f"  {ax['name']:<16} {pole:<20} {n:>5}  {seed_n:>5}  "
                  f"{kimi:>5}  {claude:>6}  {gpt:>5}  {a}/{b}")


if __name__ == "__main__":
    asyncio.run(main())
