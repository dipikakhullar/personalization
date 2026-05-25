"""LLM-as-judge over (query, preferred_answer) pairs.

For every record in `may_15/data/extracted_current/pairs/sub-*.jsonl` ask
GPT-4o via OpenRouter whether the `preferred_answer` actually resolves the
OP's question/concern. Writes the boolean + one-sentence reason to a sidecar
JSONL per subreddit, and re-pushes the merged dataset to HuggingFace every
1000 successful judgements.

Identity key for a pair: (metadata.post_id, metadata.answer_comment_id).

Usage:
  python judge_qa_pairs.py --smoke 50           # 50 records, stdout only
  python judge_qa_pairs.py --smoke 50 --persist # writes sidecars, no HF push
  python judge_qa_pairs.py                      # full run, push every 1000
  python judge_qa_pairs.py --retry-failures     # re-run failures.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

HERE = Path(__file__).resolve().parent
DATA_DIR = (HERE / ".." / "data").resolve()
PAIRS_DIR = DATA_DIR / "extracted_current" / "pairs"
JUDGE_DIR = DATA_DIR / "llm_judge"
MERGED_DIR = JUDGE_DIR / "merged_for_push"
MERGED_PAIRS_DIR = MERGED_DIR / "pairs"
FAILURES_PATH = JUDGE_DIR / "failures.jsonl"

DEFAULT_MODEL = "openai/gpt-4o"
DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"
# Records written before `judge_model` was added are all from this model.
LEGACY_DEFAULT_JUDGE_MODEL = "openai/gpt-4o"
TEMPERATURE = 0.0
MAX_TOKENS = 500
RETRY_ATTEMPTS = 5
TRUNCATE_CHARS = 6000
FSYNC_EVERY = 100
PUSH_EVERY = 1000

HF_REPO_ID = "dipikakhullar/personalization-reddit"
HF_REPO_TYPE = "dataset"

SYSTEM_PROMPT = (
    "You evaluate Reddit threads. Given an original post and a reply, decide "
    "whether the reply actually resolves the question or concern raised in "
    "the post. Output ONLY a JSON object with exactly two keys:\n"
    '  "question_answer_pair": boolean (true if the reply resolves the OP\'s '
    "question/concern, false otherwise)\n"
    '  "explanation": a one-sentence string explaining why.\n'
    "No prose outside the JSON. No code fences."
)


def _client(api_key_env: str) -> OpenAI:
    key = os.environ.get(api_key_env)
    if not key:
        raise SystemExit(f"{api_key_env} not set in .env")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)


# ── JSON parsing ─────────────────────────────────────────────────────────

_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


def _coerce_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "yes", "1", "y", "t"}:
            return True
        if s in {"false", "no", "0", "n", "f"}:
            return False
    return None


def _first_sentence(s: str) -> str:
    s = s.strip()
    m = re.search(r"[.!?]\s", s)
    if m:
        s = s[: m.end() - 1]
    return s.strip()


def parse_judgement(content: str):
    """Return ({"question_answer_pair": bool, "explanation": str}, None) on
    success, or (None, reason_str) on failure."""
    if content is None:
        return None, "empty content"
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    obj = None
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        pass
    if obj is None:
        m = _BRACE_RE.search(text)
        if m:
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    if not isinstance(obj, dict):
        return None, f"not a JSON object: {content[:200]!r}"
    if "question_answer_pair" not in obj or "explanation" not in obj:
        return None, f"missing keys: {sorted(obj.keys())}"
    b = _coerce_bool(obj["question_answer_pair"])
    if b is None:
        return None, f"non-bool question_answer_pair: {obj['question_answer_pair']!r}"
    expl = obj["explanation"]
    if not isinstance(expl, str) or not expl.strip():
        return None, f"bad explanation: {expl!r}"
    return {"question_answer_pair": b, "explanation": _first_sentence(expl)}, None


# ── OpenRouter call ──────────────────────────────────────────────────────


def _truncate(s, n=TRUNCATE_CHARS):
    if s is None:
        return ""
    if len(s) <= n:
        return s
    return s[:n] + "…[truncated]"


def build_user_prompt(query: str, preferred_answer: str) -> str:
    return (
        f"ORIGINAL POST:\n{_truncate(query)}\n\n"
        f"REPLY:\n{_truncate(preferred_answer)}\n\n"
        "Respond with the JSON object now."
    )


def judge_one(client: OpenAI, model: str, query: str, preferred_answer: str):
    """Returns (judgement_dict, last_raw_content_or_err). On total failure
    returns (None, raw_or_err). The returned judgement carries judge_model."""
    user_msg = build_user_prompt(query, preferred_answer)
    last_raw = ""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},
            )
            last_raw = resp.choices[0].message.content or ""
            judgement, err = parse_judgement(last_raw)
            if judgement is not None:
                judgement["judge_model"] = model
                return judgement, last_raw
            # Log raw length so we can tell at-a-glance whether the response
            # was truncated (short) vs malformed (long but unparseable).
            print(
                f"    parse retry {attempt + 1}/{RETRY_ATTEMPTS} "
                f"(raw len={len(last_raw)}): {err}",
                flush=True,
            )
        except Exception as e:
            last_raw = f"<exception>{e!r}"
            print(
                f"    api retry {attempt + 1}/{RETRY_ATTEMPTS}: {e!r}",
                flush=True,
            )
        if attempt < RETRY_ATTEMPTS - 1:
            time.sleep(2 ** attempt)
    return None, last_raw


# ── Sidecar I/O ──────────────────────────────────────────────────────────


def sidecar_path(sub: str) -> Path:
    return JUDGE_DIR / f"sub-{sub}.jsonl"


def load_seen_keys(include_failures: bool = True) -> set[tuple[str, str]]:
    """Load (post_id, answer_comment_id) for every already-judged record
    across all sidecar files. If include_failures is True (default), also
    include records that have already failed all retries — they would just
    fail again deterministically and waste API time. Set include_failures
    to False in `--retry-failures` mode to actually re-attempt them."""
    seen: set[tuple[str, str]] = set()
    if not JUDGE_DIR.exists():
        return seen
    for p in JUDGE_DIR.glob("sub-*.jsonl"):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = rec.get("post_id")
                aid = rec.get("answer_comment_id")
                if pid and aid:
                    seen.add((pid, aid))
    if include_failures and FAILURES_PATH.exists():
        with open(FAILURES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = rec.get("post_id")
                aid = rec.get("answer_comment_id")
                if pid and aid:
                    seen.add((pid, aid))
    return seen


def write_failure(record_key: dict, raw: str, sub: str) -> None:
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    with open(FAILURES_PATH, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "subreddit": sub,
                    "post_id": record_key.get("post_id"),
                    "answer_comment_id": record_key.get("answer_comment_id"),
                    "raw": raw[:2000],
                },
                ensure_ascii=False,
            )
            + "\n"
        )


# ── HF push (merge + upload) ─────────────────────────────────────────────


def _load_sidecar_map(sub: str) -> dict[tuple[str, str], dict]:
    p = sidecar_path(sub)
    out: dict[tuple[str, str], dict] = {}
    if not p.exists():
        return out
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = rec.get("post_id")
            aid = rec.get("answer_comment_id")
            j = rec.get("is_qa_pair")
            if pid and aid and isinstance(j, dict):
                # Backfill judge_model for records written before the field
                # was added — they were all from the legacy default model.
                if "judge_model" not in j:
                    j = dict(j)
                    j["judge_model"] = LEGACY_DEFAULT_JUDGE_MODEL
                out[(pid, aid)] = j
    return out


def merge_sub(sub: str) -> int:
    """Stream pairs/sub-<sub>.jsonl, attach is_qa_pair where we have one, write
    to merged_for_push/pairs/sub-<sub>.jsonl. Returns number of judged rows."""
    src = PAIRS_DIR / f"sub-{sub}.jsonl"
    if not src.exists():
        return 0
    judge_map = _load_sidecar_map(sub)
    MERGED_PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    dst = MERGED_PAIRS_DIR / f"sub-{sub}.jsonl"
    judged = 0
    with open(src, "r", encoding="utf-8") as fin, open(
        dst, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                fout.write(line + "\n")
                continue
            md = rec.get("metadata") or {}
            key = (md.get("post_id"), md.get("answer_comment_id"))
            j = judge_map.get(key)
            if j is not None:
                rec["is_qa_pair"] = j
                judged += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return judged


def _upload_one_sub(sub: str, cumulative_total: int) -> None:
    """Merge a single sub and upload only that file. Runs on push worker."""
    if not os.environ.get("HF_TOKEN"):
        print(f"  ! HF_TOKEN not set; skipping push for sub-{sub}", flush=True)
        return
    n = merge_sub(sub)
    src = MERGED_PAIRS_DIR / f"sub-{sub}.jsonl"
    try:
        api = HfApi(token=os.environ["HF_TOKEN"])
        api.upload_file(
            path_or_fileobj=str(src),
            path_in_repo=f"extracted/pairs/sub-{sub}.jsonl",
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            commit_message=(
                f"judge: is_qa_pair sub-{sub} ({n} judged; cumulative {cumulative_total})"
            ),
        )
        print(f"  ✓ HF upload sub-{sub} ({n} judged rows)", flush=True)
    except Exception as e:
        print(f"  ✗ HF upload sub-{sub} failed: {e!r}", flush=True)
        traceback.print_exc()


class PushWorker:
    """Background uploader: a single thread that drains a dirty-subs set.

    The main API loop adds a sub to `dirty` and calls `nudge()`. The worker
    pops one sub at a time, merges + uploads it. If the same sub gets dirtied
    again while it's uploading, the next loop iteration picks it back up.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._dirty: set[str] = set()
        self._cumulative = 0
        self._cv = threading.Condition(self._lock)
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="hf-push", daemon=True)

    def start(self):
        self._thread.start()

    def mark_dirty(self, sub: str, cumulative_total: int):
        with self._cv:
            self._dirty.add(sub)
            self._cumulative = cumulative_total

    def nudge(self):
        with self._cv:
            self._cv.notify()

    def stop_and_drain(self, timeout: float | None = 1800.0):
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._thread.join(timeout=timeout)

    def _run(self):
        while True:
            with self._cv:
                while not self._dirty and not self._stop:
                    self._cv.wait()
                if self._stop and not self._dirty:
                    return
                sub = next(iter(sorted(self._dirty)))
                self._dirty.discard(sub)
                cumulative = self._cumulative
            _upload_one_sub(sub, cumulative)


# ── Iteration over pair records ──────────────────────────────────────────


def iter_pair_files(reverse: bool = False, subs_filter: set[str] | None = None):
    """Yield (sub, path) for each sub-*.jsonl. Forward = alphabetical;
    reverse = reverse alphabetical. If subs_filter is given, only yield subs
    in that set — used for explicit per-worker partitioning."""
    paths = sorted(PAIRS_DIR.glob("sub-*.jsonl"), reverse=reverse)
    for p in paths:
        sub = p.stem[len("sub-"):]
        if subs_filter is not None and sub not in subs_filter:
            continue
        yield sub, p


def iter_records(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield rec


# ── Main ─────────────────────────────────────────────────────────────────


def run(
    smoke: int,
    persist: bool,
    retry_failures: bool,
    model: str,
    api_key_env: str,
    reverse: bool,
    subs_filter: set[str] | None,
) -> int:
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)

    client = _client(api_key_env)
    print(
        f"config: model={model} api_key_env={api_key_env} "
        f"reverse={reverse} subs={sorted(subs_filter) if subs_filter else 'ALL'}",
        flush=True,
    )

    if retry_failures:
        return run_retry_failures(client, model)

    seen = load_seen_keys(include_failures=True)
    print(f"resume: {len(seen)} records already judged", flush=True)

    total_judged_run = 0
    total_failed_run = 0
    open_sub_file = None
    open_sub_name = None
    smoke_remaining = smoke if smoke > 0 else None

    push_worker = None
    if smoke == 0:
        push_worker = PushWorker()
        push_worker.start()

    try:
        for sub, path in iter_pair_files(reverse=reverse, subs_filter=subs_filter):
            if smoke_remaining is not None and smoke_remaining <= 0:
                break
            print(f"\n=== sub-{sub}.jsonl ===", flush=True)

            if open_sub_file is not None:
                open_sub_file.close()
                open_sub_file = None
                open_sub_name = None

            if smoke == 0 or persist:
                JUDGE_DIR.mkdir(parents=True, exist_ok=True)
                open_sub_file = open(sidecar_path(sub), "a", encoding="utf-8")
                open_sub_name = sub

            for rec in iter_records(path):
                if smoke_remaining is not None and smoke_remaining <= 0:
                    break
                md = rec.get("metadata") or {}
                pid = md.get("post_id")
                aid = md.get("answer_comment_id")
                if not pid or not aid:
                    continue
                if (pid, aid) in seen:
                    continue
                query = rec.get("query") or ""
                preferred = rec.get("preferred_answer") or ""
                if not query.strip() or not preferred.strip():
                    continue

                judgement, raw = judge_one(client, model, query, preferred)
                if judgement is None:
                    total_failed_run += 1
                    write_failure(
                        {"post_id": pid, "answer_comment_id": aid}, raw, sub
                    )
                    print(
                        f"  ✗ failed {pid}/{aid} after {RETRY_ATTEMPTS} retries",
                        flush=True,
                    )
                    continue

                line = (
                    json.dumps(
                        {
                            "post_id": pid,
                            "answer_comment_id": aid,
                            "is_qa_pair": judgement,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                if smoke > 0 and not persist:
                    # smoke without persist → stdout only
                    qhead = (query[:80] + "…") if len(query) > 80 else query
                    qhead = qhead.replace("\n", " ")
                    print(
                        f"  [{pid}/{aid}] {judgement['question_answer_pair']!s:>5} "
                        f"| {qhead} | {judgement['explanation']}",
                        flush=True,
                    )
                else:
                    open_sub_file.write(line)
                    open_sub_file.flush()
                    if push_worker is not None:
                        push_worker.mark_dirty(sub, cumulative_total=len(seen) + 1)

                seen.add((pid, aid))
                total_judged_run += 1
                if smoke_remaining is not None:
                    smoke_remaining -= 1

                if total_judged_run % FSYNC_EVERY == 0 and open_sub_file is not None:
                    os.fsync(open_sub_file.fileno())
                    print(
                        f"  · {total_judged_run} judged this run "
                        f"(fsync; failed={total_failed_run})",
                        flush=True,
                    )

                if (
                    push_worker is not None
                    and total_judged_run > 0
                    and total_judged_run % PUSH_EVERY == 0
                ):
                    if open_sub_file is not None:
                        open_sub_file.flush()
                        os.fsync(open_sub_file.fileno())
                    push_worker.nudge()
                    print(
                        f"  ↗ push nudge at {total_judged_run} this run",
                        flush=True,
                    )

    finally:
        if open_sub_file is not None:
            try:
                open_sub_file.flush()
                os.fsync(open_sub_file.fileno())
            except Exception:
                pass
            open_sub_file.close()
        if push_worker is not None:
            push_worker.nudge()
            print("  ↗ final push drain…", flush=True)
            push_worker.stop_and_drain()

    print(
        f"\ndone: judged {total_judged_run}, failed {total_failed_run}, "
        f"cumulative seen {len(seen)}",
        flush=True,
    )
    return 0


def run_retry_failures(client: OpenAI, model: str) -> int:
    if not FAILURES_PATH.exists():
        print("no failures.jsonl", flush=True)
        return 0
    entries = []
    with open(FAILURES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    print(f"retrying {len(entries)} failures", flush=True)

    seen = load_seen_keys(include_failures=False)
    by_sub: dict[str, list[dict]] = {}
    for e in entries:
        by_sub.setdefault(e["subreddit"], []).append(e)

    new_judged = 0
    still_failed: list[dict] = []
    dirty: set[str] = set()
    for sub, fails in by_sub.items():
        wanted = {(e["post_id"], e["answer_comment_id"]) for e in fails}
        wanted -= seen
        if not wanted:
            continue
        src = PAIRS_DIR / f"sub-{sub}.jsonl"
        if not src.exists():
            continue
        with open(sidecar_path(sub), "a", encoding="utf-8") as out:
            for rec in iter_records(src):
                md = rec.get("metadata") or {}
                pid = md.get("post_id")
                aid = md.get("answer_comment_id")
                if (pid, aid) not in wanted:
                    continue
                query = rec.get("query") or ""
                preferred = rec.get("preferred_answer") or ""
                judgement, raw = judge_one(client, model, query, preferred)
                if judgement is None:
                    still_failed.append(
                        {
                            "subreddit": sub,
                            "post_id": pid,
                            "answer_comment_id": aid,
                            "raw": raw[:2000],
                        }
                    )
                    continue
                out.write(
                    json.dumps(
                        {
                            "post_id": pid,
                            "answer_comment_id": aid,
                            "is_qa_pair": judgement,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                out.flush()
                new_judged += 1
                seen.add((pid, aid))
                dirty.add(sub)
    # Rewrite failures.jsonl with anything still unresolved.
    with open(FAILURES_PATH, "w", encoding="utf-8") as f:
        for s in still_failed:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(
        f"retry done: +{new_judged} judged, {len(still_failed)} still failing",
        flush=True,
    )
    if dirty:
        for sub in sorted(dirty):
            _upload_one_sub(sub, cumulative_total=len(seen))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--smoke",
        type=int,
        default=0,
        help="Run only this many records then stop (0 = full run).",
    )
    ap.add_argument(
        "--persist",
        action="store_true",
        help="In smoke mode, still write to sidecar files (no HF push).",
    )
    ap.add_argument(
        "--retry-failures",
        action="store_true",
        help="Re-run records in failures.jsonl and rewrite the file.",
    )
    ap.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter model id (default: {DEFAULT_MODEL}).",
    )
    ap.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help=(
            f"Env var holding the OpenRouter key (default: {DEFAULT_API_KEY_ENV}). "
            "Use a second key for a second parallel worker."
        ),
    )
    ap.add_argument(
        "--reverse",
        action="store_true",
        help=(
            "Iterate subreddit files in reverse alphabetical order. Use for "
            "the second worker so it starts from the opposite end and avoids "
            "colliding with the forward worker."
        ),
    )
    ap.add_argument(
        "--subs",
        default="",
        help=(
            "Comma-separated subreddit names (no 'sub-' prefix) restricting "
            "this worker to that set. Empty = all subs."
        ),
    )
    args = ap.parse_args()

    if not PAIRS_DIR.exists():
        raise SystemExit(f"pairs dir missing: {PAIRS_DIR}")
    subs_filter = (
        {s.strip() for s in args.subs.split(",") if s.strip()}
        if args.subs
        else None
    )
    sys.exit(
        run(
            smoke=args.smoke,
            persist=args.persist,
            retry_failures=args.retry_failures,
            model=args.model,
            api_key_env=args.api_key_env,
            reverse=args.reverse,
            subs_filter=subs_filter,
        )
    )


if __name__ == "__main__":
    main()
