"""LLM top-1 ranker for LOO benchmark (OpenRouter / gpt-5-chat)."""

from __future__ import annotations

import json
import os
import re
import threading
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from load_threads import ThreadExample

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env")

DEFAULT_MODEL = "openai/gpt-5-chat"
LABELS = "ABCDEFGH"
CHOICE_RE = re.compile(r'"choice"\s*:\s*"([A-H])"', re.I)
_write_lock = threading.Lock()


def openrouter_client() -> OpenAI:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY not set (repo .env)")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


def chat(cl: OpenAI, messages: list[dict], *, model: str, max_tokens: int = 40) -> str:
    resp = cl.chat.completions.create(
        model=model, messages=messages, temperature=0, max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def history_block(history: list[ThreadExample], *, max_items: int = 6) -> str:
    lines = []
    for i, h in enumerate(history[-max_items:], 1):
        body = ""
        for c in h.candidates:
            if c.comment_id == h.thanked_id:
                body = c.body[:500]
                break
        lines.append(
            f"[Past {i}] r/{h.subreddit}\n"
            f"Question: {h.query[:450]}\n"
            f"Answer they thanked: {body}\n"
        )
    return "\n".join(lines) if lines else "(no prior history)"


def rank_thread(
    cl: OpenAI,
    held: ThreadExample,
    history: list[ThreadExample],
    *,
    model: str = DEFAULT_MODEL,
    max_candidates: int = 8,
) -> tuple[str | None, str]:
    """Return (picked_comment_id, raw_response)."""
    cands = held.candidates[:max_candidates]
    if len(cands) < 2:
        return None, ""
    labels = LABELS[: len(cands)]
    blocks = "\n\n".join(
        f"Answer {lab}:\n{c.body[:700]}" for lab, c in zip(labels, cands)
    )
    sys_prompt = (
        "You pick which existing Reddit comment a specific user would most likely "
        "thank as the best answer. Use their past thanked answers to infer tone, "
        "depth, and style preferences — not generic upvote popularity."
    )
    user_prompt = (
        f"User history (earlier questions and answers they thanked):\n\n"
        f"{history_block(history)}\n\n"
        f"New question on r/{held.subreddit}:\n{held.query[:600]}\n\n"
        f"Candidate answers on this post:\n\n{blocks}\n\n"
        f"Which single answer would THIS user thank? "
        f'Reply with ONLY JSON: {{"choice": "{labels[0]}"}} '
        f"using one letter from {', '.join(labels)}."
    )
    raw = chat(
        cl,
        [{"role": "system", "content": sys_prompt},
         {"role": "user", "content": user_prompt}],
        model=model,
    )
    m = CHOICE_RE.search(raw) or re.search(r"\b([A-H])\b", raw.upper())
    if not m:
        return None, raw
    letter = m.group(1).upper()
    idx = labels.find(letter)
    if idx < 0:
        return None, raw
    return cands[idx].comment_id, raw


def load_cache(path: Path) -> dict[str, bool]:
    if not path.is_file():
        return {}
    out: dict[str, bool] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["post_id"]] = bool(rec["correct"])
    return out


def append_cache(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


def enrich_records_with_llm(
    threads: list[ThreadExample],
    records: list[dict],
    *,
    max_calls: int = 120,
    cache_path: Path | None = None,
    model: str = DEFAULT_MODEL,
) -> int:
    """Set llm_ranker on records (in place). Returns new API calls made."""
    by_post = {t.post_id: t for t in threads}
    by_asker: dict[str, list[ThreadExample]] = defaultdict(list)
    for t in threads:
        by_asker[t.asker_id].append(t)
    for seq in by_asker.values():
        seq.sort(key=lambda x: x.timestamp)

    cache_path = cache_path or (
        Path(__file__).resolve().parent / "outputs" / "llm_ranker_cache.jsonl"
    )
    cache = load_cache(cache_path)

    for r in records:
        pid = r.get("post_id")
        if pid and pid in cache:
            r["llm_ranker"] = cache[pid]

    pending = [r for r in records if r.get("llm_ranker") is None and r.get("post_id")]
    pending.sort(
        key=lambda r: (
            0 if r.get("divergent") and r.get("cross_sub") else 1,
            -r.get("k_raw", 0),
        ),
    )
    budget = max(0, max_calls - sum(1 for r in records if r.get("llm_ranker") is not None))
    pending = pending[:budget]

    if not pending:
        return 0

    cl = openrouter_client()
    n_calls = 0
    for r in pending:
        pid = r["post_id"]
        held = by_post.get(pid)
        if not held:
            continue
        seq = by_asker[r["asker_id"]]
        history = [h for h in seq if h.timestamp < held.timestamp]
        pick, _raw = rank_thread(cl, held, history, model=model)
        correct = bool(pick and pick == held.thanked_id)
        r["llm_ranker"] = correct
        cache[pid] = correct
        append_cache(
            cache_path,
            {"post_id": pid, "pick_id": pick, "thanked_id": held.thanked_id, "correct": correct},
        )
        n_calls += 1
    return n_calls
