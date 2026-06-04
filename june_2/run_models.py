"""Multi-model version of the june_2 experiment, run on is_qa_pair=True samples.

Same two sub-experiments as run_experiment.py, but parametrized by model:

  --model       the model that GENERATES the answer (Exp A) and makes the
                forced CHOICE (Exp B).  This is the variable under test.
  --judge-model the FIXED model that scores generated-vs-reference similarity
                (1-5 rubric).  Kept constant across all runs so the similarity
                metric does not confound the model comparison.

Embedding cosine similarity (sentence-transformers) is model-agnostic.

Results stream to results/<results-name> (resumable per sample_id).
"""
import argparse
import json
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
load_dotenv(REPO_ROOT / ".env")

EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_JUDGE = "openai/gpt-5-chat"
_write_lock = threading.Lock()


def client() -> OpenAI:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not set in .env")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def chat(cl, model, messages, max_retries=8, **kw):
    """Returns message content. Retries transient errors AND empty content
    (reasoning models occasionally spend the whole budget on hidden reasoning)."""
    last = None
    for _ in range(max_retries):
        try:
            resp = cl.chat.completions.create(model=model, messages=messages, **kw)
            content = resp.choices[0].message.content
            if content and content.strip():
                return content
            last = "empty content"
        except Exception as e:  # noqa: BLE001 - retry transient API errors
            last = e
    raise RuntimeError(f"chat({model}) failed after {max_retries} retries: {last!r}")


def history_block(history):
    return "\n".join(
        f"[Past thread {i}]\n"
        f"Question they asked: {h['query']}\n"
        f"Answer they thanked / preferred: {h['preferred_answer']}\n"
        for i, h in enumerate(history, 1)
    )


# ---------------------------------------------------------------- Exp A: generate
def generate_answer(cl, model, sample, extra=None, no_history=False):
    if no_history:
        # control condition: no user context at all
        sys_prompt = (
            "You are answering a question on Reddit. Write a helpful answer. "
            "Reply with only the answer text."
        )
        user_prompt = f"Question:\n\n{sample['query']}\n\nWrite the best answer."
    else:
        sys_prompt = (
            "You are answering questions on Reddit. You are shown a particular user's "
            "history of past questions together with the answers they personally found "
            "most helpful (the ones they thanked). Infer this user's preferences in "
            "tone, length, and style from that history, then answer their new question "
            "the way THIS user would most appreciate. Reply with only the answer text."
        )
        user_prompt = (
            f"Here is the user's history:\n\n{history_block(sample['history'])}\n"
            f"Now the same user asks a new question:\n\n{sample['query']}\n\n"
            "Write the answer this user would most prefer."
        )
    return chat(
        cl, model,
        [{"role": "system", "content": sys_prompt},
         {"role": "user", "content": user_prompt}],
        temperature=0.7, max_tokens=1500,  # headroom for reasoning models (gpt-5)
        **(extra or {}),
    )


SIM_RE = re.compile(r'"?sim_preferred"?\s*[:=]\s*([1-5]).*?"?sim_top"?\s*[:=]\s*([1-5])', re.S)

SIM_RUBRIC = (
    "Similarity rubric (how close the CANDIDATE's advice is to a REFERENCE):\n"
    "  5 = Same core recommendation and main reasoning; a reader would act the same way.\n"
    "  4 = Largely the same recommendation; minor differences in detail/caveats/emphasis.\n"
    "  3 = Partial overlap; shares some points but also gives notably different advice.\n"
    "  2 = Mostly different advice; only superficial or topical overlap.\n"
    "  1 = Unrelated or contradictory advice.\n"
)


def judge_similarity(cl, judge_model, generated, preferred, top):
    prompt = (
        "Rate how similar a CANDIDATE answer is to each of two REFERENCE answers "
        "on a 1-5 scale. Judge the substance of the advice, not length or wording.\n\n"
        f"{SIM_RUBRIC}\n"
        f"CANDIDATE:\n{generated}\n\n"
        f"REFERENCE_PREFERRED:\n{preferred}\n\n"
        f"REFERENCE_TOP:\n{top}\n\n"
        'Respond with ONLY JSON: {"sim_preferred": <1-5>, "sim_top": <1-5>}'
    )
    raw = chat(cl, judge_model, [{"role": "user", "content": prompt}],
               temperature=0, max_tokens=60)
    m = SIM_RE.search(raw or "")
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


# ------------------------------------------------------------------ Exp B: choose
CHOICE_RE = re.compile(r"\b([AB])\b")


def choose(cl, model, sample, seed, extra=None, no_history=False):
    rng = random.Random(seed)
    preferred_is_A = rng.random() < 0.5
    if preferred_is_A:
        answer_A, answer_B = sample["preferred_answer"], sample["top_comment"]
    else:
        answer_A, answer_B = sample["top_comment"], sample["preferred_answer"]

    if no_history:
        # control: pick the better answer with no user context
        sys_prompt = (
            "You are shown a question and two candidate answers. Pick the single "
            "answer that is the better, more helpful response to the question."
        )
        user_prompt = (
            f"Question:\n\n{sample['query']}\n\n"
            f"Answer A:\n{answer_A}\n\nAnswer B:\n{answer_B}\n\n"
            'Which answer is better? Respond with ONLY JSON: {"choice": "A" or "B"}'
        )
    else:
        sys_prompt = (
            "You are shown a user's history of past questions and the answers they "
            "personally preferred, then a new question and two candidate answers. "
            "Pick the single answer THIS user would most prefer. Consider their "
            "inferred style/tone preferences, not just generic quality."
        )
        user_prompt = (
            f"User history:\n\n{history_block(sample['history'])}\n"
            f"New question:\n\n{sample['query']}\n\n"
            f"Answer A:\n{answer_A}\n\nAnswer B:\n{answer_B}\n\n"
            'Which answer would this user prefer? Respond with ONLY JSON: '
            '{"choice": "A" or "B"}'
        )
    raw = chat(cl, model,
               [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}],
               temperature=0, max_tokens=800,  # headroom for reasoning models (gpt-5)
               **(extra or {}))
    m = CHOICE_RE.search((raw or "").split("choice")[-1]) or CHOICE_RE.search(raw or "")
    if not m:
        return None, preferred_is_A
    picked_A = m.group(1) == "A"
    return picked_A == preferred_is_A, preferred_is_A


# --------------------------------------------------------------------- per-sample
def process(sample, cl, embedder, gen_model, judge_model, gen_extra=None, no_history=False):
    import numpy as np
    generated = generate_answer(cl, gen_model, sample, gen_extra, no_history=no_history)
    embs = embedder.encode(
        [generated, sample["preferred_answer"], sample["top_comment"]],
        normalize_embeddings=True,
    )
    cos_pref = float(np.dot(embs[0], embs[1]))
    cos_top = float(np.dot(embs[0], embs[2]))
    sim_pref, sim_top = judge_similarity(
        cl, judge_model, generated, sample["preferred_answer"], sample["top_comment"]
    )
    picked_preferred, preferred_is_A = choose(
        cl, gen_model, sample, seed=sample["sample_id"], extra=gen_extra, no_history=no_history
    )
    return {
        "sample_id": sample["sample_id"],
        "model": gen_model,
        "subreddit": sample["subreddit"],
        "generated_answer": generated,
        "cos_preferred": cos_pref,
        "cos_top": cos_top,
        "judge_sim_preferred": sim_pref,
        "judge_sim_top": sim_top,
        "choice_picked_preferred": picked_preferred,
        "choice_preferred_was_A": preferred_is_A,
    }


def load_done(path):
    if not path.exists():
        return set()
    done = set()
    for line in open(path):
        try:
            done.add(json.loads(line)["sample_id"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="generation/choice model id")
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE)
    ap.add_argument("--samples", default=str(HERE / "data" / "samples_qa1000.jsonl"))
    ap.add_argument("--out", required=True, help="results jsonl path")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reasoning-effort", default="",
                    help="for reasoning models (e.g. gpt-5): minimal|low|medium|high")
    ap.add_argument("--no-history", action="store_true",
                    help="control condition: omit user history from prompts")
    args = ap.parse_args()

    gen_extra = (
        {"extra_body": {"reasoning": {"effort": args.reasoning_effort}}}
        if args.reasoning_effort else None
    )

    from sentence_transformers import SentenceTransformer

    samples = [json.loads(l) for l in open(args.samples)]
    if args.limit:
        samples = samples[: args.limit]
    out = Path(args.out)
    done = load_done(out)
    todo = [s for s in samples if s["sample_id"] not in done]
    mode = "NO-history" if args.no_history else "with-history"
    print(f"[{args.model}] {mode}: {len(samples)} samples, {len(done)} done, "
          f"{len(todo)} to run", flush=True)
    if not todo:
        print(f"[{args.model}] nothing to do")
        return

    cl = client()
    embedder = SentenceTransformer(EMBED_MODEL)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process, s, cl, embedder, args.model, args.judge_model,
                          gen_extra, args.no_history): s
                for s in todo}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  ! [{args.model}] sample {s['sample_id']} failed: {e!r}", flush=True)
                continue
            with _write_lock, open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            n_ok += 1
            if n_ok % 25 == 0:
                print(f"  [{args.model}] {n_ok}/{len(todo)}", flush=True)
    print(f"[{args.model}] done: {n_ok} new -> {out}", flush=True)


if __name__ == "__main__":
    main()
