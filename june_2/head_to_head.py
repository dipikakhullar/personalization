"""Head-to-head test: does giving a model the user's history produce an answer
THIS user prefers, vs the same model with no context?

For each (model, sample): take the model's WITH-history generation and its
NO-context generation (already produced in results_clean/), show a fixed neutral
judge the user's history + the new question + both answers (A/B order randomized),
and ask which one THIS user would prefer. The with-history generation's win-rate
vs 50% is the measurement — no embeddings, no topic-dominance confound. The
preferred/top reference answers are NOT used here (they're validation anchors
elsewhere).

Results stream to results_clean/head_to_head/results_<slug>.jsonl (resumable).
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
load_dotenv(HERE.parent / ".env")

SAMPLES = HERE / "data" / "samples_qa1000_clean.jsonl"
WITH_DIR = HERE / "results_clean" / "with_history"
NO_DIR = HERE / "results_clean" / "no_user_context"
OUT_DIR = HERE / "results_clean" / "head_to_head"
JUDGE_MODEL = "openai/gpt-5-chat"
_lock = threading.Lock()
CHOICE_RE = re.compile(r"\b([AB])\b")


def client():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not set in .env")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def chat(cl, model, messages, max_retries=6, **kw):
    last = None
    for _ in range(max_retries):
        try:
            r = cl.chat.completions.create(model=model, messages=messages, **kw)
            c = r.choices[0].message.content
            if c and c.strip():
                return c
            last = "empty"
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"judge failed: {last!r}")


def history_block(history):
    return "\n".join(
        f"[Past thread {i}]\nQuestion: {h['query']}\n"
        f"Answer they preferred: {h['preferred_answer']}\n"
        for i, h in enumerate(history, 1)
    )


def judge_pair(cl, sample, gen_with, gen_no, seed, use_history=True):
    rng = random.Random(seed)
    with_is_A = rng.random() < 0.5
    answer_A, answer_B = (gen_with, gen_no) if with_is_A else (gen_no, gen_with)
    if use_history:
        sys_prompt = (
            "You are shown a particular Reddit user's history of past questions and the "
            "answers they personally preferred (thanked). Then a new question that user "
            "asked, with two candidate answers, A and B. Choose the single answer THIS "
            "user would most prefer, judging by the style, tone, length, and substance "
            "they have shown they like. Pick the better-personalized answer for this user, "
            "not the generically 'better' one."
        )
        user_prompt = (
            f"User history:\n\n{history_block(sample['history'])}\n"
            f"New question:\n\n{sample['query']}\n\n"
            f"Answer A:\n{answer_A}\n\nAnswer B:\n{answer_B}\n\n"
            'Which answer would THIS user prefer? Respond with ONLY JSON: {"choice": "A" or "B"}'
        )
    else:
        # CONTROL: judge sees NO user history — just picks the generically better answer.
        sys_prompt = (
            "You are shown a Reddit question and two candidate answers, A and B. Choose "
            "the single answer that is the better, more helpful response to the question."
        )
        user_prompt = (
            f"Question:\n\n{sample['query']}\n\n"
            f"Answer A:\n{answer_A}\n\nAnswer B:\n{answer_B}\n\n"
            'Which answer is better? Respond with ONLY JSON: {"choice": "A" or "B"}'
        )
    raw = chat(cl, JUDGE_MODEL,
               [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}],
               temperature=0, max_tokens=600)
    m = CHOICE_RE.search((raw or "").split("choice")[-1]) or CHOICE_RE.search(raw or "")
    if not m:
        return None, with_is_A
    picked_A = m.group(1) == "A"
    with_history_won = picked_A == with_is_A
    return with_history_won, with_is_A


def load_gen(d, slug):
    fp = d / f"results_{slug}.jsonl"
    return {json.loads(l)["sample_id"]: json.loads(l)["generated_answer"] for l in open(fp)}


def process(sample, cl, gen_with, gen_no, use_history=True):
    won, with_is_A = judge_pair(cl, sample, gen_with, gen_no, seed=sample["sample_id"],
                                use_history=use_history)
    return {"sample_id": sample["sample_id"], "with_history_won": won, "with_was_A": with_is_A}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slug", required=True, help="model slug, e.g. claude46")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--judge-no-history", action="store_true",
                    help="CONTROL: judge does NOT see the user history (picks generically better)")
    args = ap.parse_args()
    use_history = not args.judge_no_history
    out_dir = OUT_DIR if use_history else (HERE / "results_clean" / "head_to_head_nohistjudge")

    samples = {json.loads(l)["sample_id"]: json.loads(l) for l in open(SAMPLES)}
    gen_with = load_gen(WITH_DIR, args.slug)
    gen_no = load_gen(NO_DIR, args.slug)
    common = sorted(set(gen_with) & set(gen_no) & set(samples))

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"results_{args.slug}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["sample_id"] for l in open(out)}
    todo = [s for s in common if s not in done]
    print(f"[{args.slug}] {len(common)} paired samples, {len(done)} done, {len(todo)} to run",
          flush=True)
    if not todo:
        return

    cl = client()
    n = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process, samples[sid], cl, gen_with[sid], gen_no[sid], use_history): sid
                for sid in todo}
        for fut in as_completed(futs):
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  ! {futs[fut]} failed: {e!r}", flush=True); continue
            with _lock, open(out, "a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
            n += 1
            if n % 50 == 0:
                print(f"  [{args.slug}] {n}/{len(todo)}", flush=True)
    print(f"[{args.slug}] done: {n} new -> {out}", flush=True)


if __name__ == "__main__":
    main()
