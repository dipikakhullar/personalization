"""june_2 personalization experiment.

Two sub-experiments per sample, both conditioned on the user's history
(prior question + preferred-answer pairs), using openai/gpt-5-chat:

  A) GENERATE: model answers the new question. We then measure how similar its
     answer is to the user's preferred_answer vs the top_comment, via
       - embedding cosine similarity (sentence-transformers, all-mpnet-base-v2)
       - an LLM-judge 1-5 rubric similarity score (gpt-5-chat)

  B) CHOOSE: model is shown both candidate answers (order randomized per sample)
     and picks the one the user would most prefer. We record whether it picked
     the preferred_answer or the top_comment.

Results stream to june_2/results/results.jsonl (resumable).
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

SAMPLES_PATH = HERE / "data" / "samples.jsonl"
RESULTS_PATH = HERE / "results" / "results.jsonl"

MODEL = "openai/gpt-5-chat"
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
_write_lock = threading.Lock()


def client() -> OpenAI:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not set in .env")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def chat(cl: OpenAI, messages, max_retries=4, **kw):
    last = None
    for attempt in range(max_retries):
        try:
            resp = cl.chat.completions.create(model=MODEL, messages=messages, **kw)
            return resp.choices[0].message.content
        except Exception as e:  # noqa: BLE001 - retry transient API errors
            last = e
    raise RuntimeError(f"chat failed after {max_retries} retries: {last!r}")


def history_block(history):
    lines = []
    for i, h in enumerate(history, 1):
        lines.append(
            f"[Past thread {i}]\n"
            f"Question they asked: {h['query']}\n"
            f"Answer they thanked / preferred: {h['preferred_answer']}\n"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------- Exp A: generate
def generate_answer(cl, sample):
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
        cl,
        [{"role": "system", "content": sys_prompt},
         {"role": "user", "content": user_prompt}],
        temperature=0.7,
        max_tokens=600,
    )


SIM_RE = re.compile(r'"?sim_preferred"?\s*[:=]\s*([1-5]).*?"?sim_top"?\s*[:=]\s*([1-5])', re.S)

SIM_RUBRIC = (
    "Similarity rubric (how close the CANDIDATE's advice is to a REFERENCE):\n"
    "  5 = Same core recommendation and main reasoning; a reader would act the "
    "same way.\n"
    "  4 = Largely the same recommendation; minor differences in detail, caveats, "
    "or emphasis.\n"
    "  3 = Partial overlap; shares some points but also gives notably different "
    "advice.\n"
    "  2 = Mostly different advice; only superficial or topical overlap.\n"
    "  1 = Unrelated or contradictory advice.\n"
)


def judge_similarity(cl, generated, preferred, top):
    """gpt-5-chat rates semantic similarity (1-5 rubric) of generated vs each reference."""
    prompt = (
        "Rate how similar a CANDIDATE answer is to each of two REFERENCE answers "
        "on a 1-5 scale. Judge the substance of the advice, not length or wording.\n\n"
        f"{SIM_RUBRIC}\n"
        f"CANDIDATE:\n{generated}\n\n"
        f"REFERENCE_PREFERRED:\n{preferred}\n\n"
        f"REFERENCE_TOP:\n{top}\n\n"
        'Respond with ONLY JSON: {"sim_preferred": <1-5>, "sim_top": <1-5>}'
    )
    raw = chat(
        cl,
        [{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=60,
    )
    m = SIM_RE.search(raw or "")
    if not m:
        return None, None, raw
    return int(m.group(1)), int(m.group(2)), raw


# ------------------------------------------------------------------ Exp B: choose
CHOICE_RE = re.compile(r"\b([AB])\b")


def choose(cl, sample, seed):
    """Show both answers in randomized order; model picks the better one."""
    rng = random.Random(seed)
    preferred_is_A = rng.random() < 0.5
    if preferred_is_A:
        answer_A, answer_B = sample["preferred_answer"], sample["top_comment"]
    else:
        answer_A, answer_B = sample["top_comment"], sample["preferred_answer"]

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
    raw = chat(
        cl,
        [{"role": "system", "content": sys_prompt},
         {"role": "user", "content": user_prompt}],
        temperature=0,
        max_tokens=20,
    )
    m = CHOICE_RE.search((raw or "").split("choice")[-1]) or CHOICE_RE.search(raw or "")
    if not m:
        return None, preferred_is_A, raw
    picked_A = m.group(1) == "A"
    picked_preferred = picked_A == preferred_is_A
    return picked_preferred, preferred_is_A, raw


# --------------------------------------------------------------------- per-sample
def process(sample, cl, embedder):
    generated = generate_answer(cl, sample)

    import numpy as np
    embs = embedder.encode(
        [generated, sample["preferred_answer"], sample["top_comment"]],
        normalize_embeddings=True,
    )
    cos_pref = float(np.dot(embs[0], embs[1]))
    cos_top = float(np.dot(embs[0], embs[2]))

    sim_pref, sim_top, sim_raw = judge_similarity(
        cl, generated, sample["preferred_answer"], sample["top_comment"]
    )
    picked_preferred, preferred_is_A, choice_raw = choose(
        cl, sample, seed=sample["sample_id"]
    )

    return {
        "sample_id": sample["sample_id"],
        "subreddit": sample["subreddit"],
        "preferred_score": sample["preferred_score"],
        "top_score": sample["top_score"],
        "generated_answer": generated,
        # Exp A
        "cos_preferred": cos_pref,
        "cos_top": cos_top,
        "judge_sim_preferred": sim_pref,
        "judge_sim_top": sim_top,
        # Exp B
        "choice_picked_preferred": picked_preferred,
        "choice_preferred_was_A": preferred_is_A,
    }


def load_done():
    if not RESULTS_PATH.exists():
        return set()
    done = set()
    with open(RESULTS_PATH) as f:
        for line in f:
            try:
                done.add(json.loads(line)["sample_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="cap #samples (0=all)")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    samples = [json.loads(l) for l in open(SAMPLES_PATH)]
    if args.limit:
        samples = samples[: args.limit]
    done = load_done()
    todo = [s for s in samples if s["sample_id"] not in done]
    print(f"{len(samples)} samples, {len(done)} done, {len(todo)} to run", flush=True)
    if not todo:
        print("nothing to do")
        return

    cl = client()
    print(f"loading embedder {EMBED_MODEL} ...", flush=True)
    embedder = SentenceTransformer(EMBED_MODEL)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process, s, cl, embedder): s for s in todo}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  ! sample {s['sample_id']} failed: {e!r}", flush=True)
                continue
            with _write_lock, open(RESULTS_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            n_ok += 1
            print(
                f"  [{n_ok}/{len(todo)}] sample {res['sample_id']} "
                f"cos(pref/top)={res['cos_preferred']:.3f}/{res['cos_top']:.3f} "
                f"judge={res['judge_sim_preferred']}/{res['judge_sim_top']} "
                f"chose_preferred={res['choice_picked_preferred']}",
                flush=True,
            )
    print(f"done: {n_ok} new results -> {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
