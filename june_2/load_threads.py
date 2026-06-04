"""Build thread-level selection examples for the personalization benchmark.

Unit: one post (OP question + candidate answers).
Label: which candidate OP thanked (answer_comment_id).
Candidates: comments on that post with is_qa_pair == true (LLM judge).

Uses pairs + llm_judge sidecars; optionally scans RC dump for candidate bodies.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MAY15 = REPO / "may_15"
PIPE = MAY15 / "reddit_pipeline"
if str(PIPE) not in sys.path:
    sys.path.insert(0, str(PIPE))

from dump_reader import iter_dump  # noqa: E402
from subreddits import is_bot_author, is_deleted_author  # noqa: E402
from signals import looks_like_bot_body  # noqa: E402


def _strip_kind(fullname: str | None) -> str | None:
    if not fullname:
        return None
    if "_" in fullname and len(fullname) > 3 and fullname[2] == "_":
        return fullname.split("_", 1)[1]
    return fullname


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


@dataclass
class Candidate:
    comment_id: str
    body: str
    score: int
    position: int = 0  # order among top-level candidates by created_utc


@dataclass
class ThreadExample:
    post_id: str
    asker_id: str
    timestamp: str
    subreddit: str
    query: str
    op_flair: str | None
    thanked_id: str
    candidates: list[Candidate] = field(default_factory=list)
    top_comment_id: str | None = None
    answer_score: int = 0
    top_equals_preferred: bool = False

    @property
    def is_agreement(self) -> bool:
        return self.top_equals_preferred

    @property
    def is_divergent(self) -> bool:
        return not self.top_equals_preferred


def load_judge_by_post(judge_path: Path) -> dict[str, dict[str, bool]]:
    """post_id -> {comment_id -> is_qa_pair bool}."""
    out: dict[str, dict[str, bool]] = {}
    for rec in iter_jsonl(judge_path):
        qa = (rec.get("is_qa_pair") or {}).get("question_answer_pair")
        if qa is None:
            continue
        pid = rec["post_id"]
        cid = rec["answer_comment_id"]
        out.setdefault(pid, {})[cid] = bool(qa)
    return out


def _thread_from_pair(
    rec: dict,
    judge_by_post: dict[str, dict[str, bool]],
    seen_posts: set[str],
    *,
    contrastive_only: bool,
) -> ThreadExample | None:
    md = rec.get("metadata") or {}
    post_id = md.get("post_id")
    thanked = md.get("answer_comment_id")
    if not post_id or not thanked or post_id in seen_posts:
        return None
    if contrastive_only and md.get("top_equals_preferred"):
        return None
    qa_map = judge_by_post.get(post_id, {})
    j = qa_map.get(thanked)
    if j is None:
        j = (rec.get("is_qa_pair") or {}).get("question_answer_pair")
    if not j:
        return None
    if sum(1 for ok in qa_map.values() if ok) < 2:
        return None
    seen_posts.add(post_id)
    op_meta = rec.get("op_metadata") or {}
    return ThreadExample(
        post_id=post_id,
        asker_id=rec["user_id"],
        timestamp=rec["timestamp"],
        subreddit=rec.get("subreddit") or "",
        query=rec.get("query") or "",
        op_flair=op_meta.get("author_flair_text"),
        thanked_id=thanked,
        candidates=[],
        top_comment_id=md.get("top_comment_id"),
        answer_score=int(md.get("answer_score") or 0),
        top_equals_preferred=bool(md.get("top_equals_preferred")),
    )


def load_selection_threads(
    pairs_path: Path,
    judge_path: Path,
    *,
    contrastive_only: bool = False,
) -> list[ThreadExample]:
    """Valid-QA threads with >=2 candidates; optional contrastive-only filter."""
    judge_by_post = load_judge_by_post(judge_path)
    seen_posts: set[str] = set()
    threads: list[ThreadExample] = []
    for rec in iter_jsonl(pairs_path):
        t = _thread_from_pair(rec, judge_by_post, seen_posts, contrastive_only=contrastive_only)
        if t is not None:
            threads.append(t)
    return threads


def load_contrastive_threads(
    pairs_path: Path,
    judge_path: Path,
    **kwargs,
) -> list[ThreadExample]:
    return load_selection_threads(pairs_path, judge_path, contrastive_only=True)


def attach_comment_bodies(
    threads: list[ThreadExample],
    rc_path: Path,
    judge_by_post: dict[str, dict[str, bool]],
) -> list[ThreadExample]:
    """Scan RC once; fill candidate bodies for QA-valid comment ids."""
    post_ids = {t.post_id for t in threads}
    need_ids: set[str] = set()
    for t in threads:
        for cid in judge_by_post.get(t.post_id, {}):
            if judge_by_post[t.post_id][cid]:
                need_ids.add(cid)

    bodies: dict[str, tuple[str, int, int]] = {}  # id -> body, score, created
    for rec in iter_dump(rc_path):
        link = _strip_kind(rec.get("link_id"))
        if link not in post_ids:
            continue
        cid = rec.get("id") or ""
        if cid not in need_ids:
            continue
        author = rec.get("author")
        body = (rec.get("body") or "").strip()
        if is_deleted_author(author) or is_bot_author(author):
            continue
        if not body or body in ("[deleted]", "[removed]") or looks_like_bot_body(body):
            continue
        score = int(rec.get("score") or 0)
        created = int(rec.get("created_utc") or 0)
        bodies[cid] = (body, score, created)

    out: list[ThreadExample] = []
    for t in threads:
        qa_ids = [cid for cid, ok in judge_by_post.get(t.post_id, {}).items() if ok]
        cands: list[Candidate] = []
        for cid in qa_ids:
            if cid not in bodies:
                continue
            body, score, created = bodies[cid]
            cands.append(Candidate(comment_id=cid, body=body, score=score, position=created))
        if len(cands) < 2 or not any(c.comment_id == t.thanked_id for c in cands):
            continue
        cands.sort(key=lambda c: (c.position, c.comment_id))
        for i, c in enumerate(cands):
            c.position = i
        t.candidates = cands
        out.append(t)
    return out


# High-judge subs for ~10 min pooled pilot (pairs + judge + RC all present)
FAST_BENCHMARK_SUBS = [
    "AskAcademia", "AskBaking", "AskCulinary", "AskDocs", "AskEngineers",
    "AskHistorians", "AskStatistics", "askphilosophy", "askscience",
    "DIY", "Coffee", "Sewing", "homeimprovement", "houseplants",
]


def discover_benchmark_subs(data_root: Path | None = None) -> list[str]:
    """Subs with extracted pairs, non-empty judge sidecar, and RC comment dump."""
    data = data_root or (MAY15 / "data")
    pairs_dir = data / "extracted_current" / "pairs"
    ready: list[str] = []
    for pf in sorted(pairs_dir.glob("sub-*.jsonl")):
        sub = pf.stem.replace("sub-", "")
        judge = data / "llm_judge" / f"sub-{sub}.jsonl"
        dump = data / "dumps" / f"sub-{sub}"
        if not judge.is_file() or judge.stat().st_size < 100:
            continue
        if not dump.is_dir() or not list(dump.glob("RC_*")):
            continue
        ready.append(sub)
    return ready


def _paths_for_sub(sub: str, data: Path) -> tuple[Path, Path, Path]:
    pairs_path = data / "extracted_current" / "pairs" / f"sub-{sub}.jsonl"
    judge_path = data / "llm_judge" / f"sub-{sub}.jsonl"
    dump_dir = data / "dumps" / f"sub-{sub}"
    rc_files = list(dump_dir.glob("RC_*"))
    if not rc_files:
        raise FileNotFoundError(f"No RC dump under {dump_dir}")
    return pairs_path, judge_path, rc_files[0]


def _select_heldout_corpus(
    threads: list[ThreadExample],
    *,
    max_threads: int | None,
    min_prior: int,
) -> list[ThreadExample]:
    """Pick held-outs (k >= min_prior), prefer high k; include prefixes + k=0 baselines."""
    by_asker: dict[str, list[ThreadExample]] = defaultdict(list)
    for t in threads:
        by_asker[t.asker_id].append(t)
    for seq in by_asker.values():
        seq.sort(key=lambda x: x.timestamp)

    # (k, asker_id, index_in_seq, thread) for held-outs with k >= min_prior
    eligible: list[tuple[int, str, int, ThreadExample]] = []
    for uid, seq in by_asker.items():
        for i, t in enumerate(seq):
            if i >= min_prior:
                eligible.append((i, uid, i, t))

    # Prefer high-k held-outs (more history → where personalization is testable)
    eligible.sort(key=lambda x: (-x[0], x[1]))

    if max_threads and len(eligible) > max_threads:
        eligible = eligible[:max_threads]

    # Minimal superset: temporal prefix through each selected held-out
    seen_posts: set[str] = set()
    out: list[ThreadExample] = []
    for k, uid, idx, held in eligible:
        for t in by_asker[uid][: idx + 1]:
            if t.post_id not in seen_posts:
                seen_posts.add(t.post_id)
                out.append(t)

    # Include each selected asker's k=0 held-out for cold-start baseline contrast
    selected_askers = {uid for _, uid, _, _ in eligible}
    for uid in selected_askers:
        seq = by_asker[uid]
        if not seq:
            continue
        t0 = seq[0]
        if t0.post_id not in seen_posts:
            seen_posts.add(t0.post_id)
            out.append(t0)

    out.sort(key=lambda t: t.timestamp)
    return out


def filter_users_min_pairs(threads: list[ThreadExample], min_pairs: int) -> list[ThreadExample]:
    counts: dict[str, int] = defaultdict(int)
    for t in threads:
        counts[t.asker_id] += 1
    ok = {u for u, c in counts.items() if c >= min_pairs}
    return [t for t in threads if t.asker_id in ok]


def load_corpus(
    subs: list[str] | None = None,
    *,
    max_threads: int | None = None,
    min_prior: int = 1,
    min_pairs_per_user: int = 1,
    data_root: Path | None = None,
) -> tuple[list[ThreadExample], list[str], dict[str, int]]:
    """Load valid-QA threads across one or many subreddits.

    Askers are keyed globally (same user_id across subs share one temporal history).
    Returns (threads_with_bodies, subs_used, asker_id -> total valid-QA pair count).
    """
    data = data_root or (MAY15 / "data")
    used = subs if subs is not None else discover_benchmark_subs(data)
    if not used:
        raise FileNotFoundError("No subreddits with pairs + judge + RC dump")

    skeleton: list[ThreadExample] = []
    for sub in used:
        pairs_path, judge_path, _ = _paths_for_sub(sub, data)
        if not pairs_path.is_file():
            continue
        skeleton.extend(load_selection_threads(pairs_path, judge_path, contrastive_only=False))

    user_pair_counts = dict(Counter(t.asker_id for t in skeleton))

    skel_for_select = skeleton
    if min_pairs_per_user > 1:
        skel_for_select = filter_users_min_pairs(skeleton, min_pairs_per_user)

    selected = _select_heldout_corpus(skel_for_select, max_threads=max_threads, min_prior=min_prior)

    by_sub: dict[str, list[ThreadExample]] = defaultdict(list)
    for t in selected:
        by_sub[t.subreddit or ""].append(t)

    out: list[ThreadExample] = []
    for sub in used:
        chunk = by_sub.get(sub, [])
        if not chunk:
            continue
        _, judge_path, rc_path = _paths_for_sub(sub, data)
        judge_by_post = load_judge_by_post(judge_path)
        out.extend(attach_comment_bodies(chunk, rc_path, judge_by_post))

    out.sort(key=lambda t: t.timestamp)
    return out, used, user_pair_counts


def load_subset(
    sub: str = "AskStatistics",
    *,
    max_threads: int | None = 400,
    min_prior: int = 1,
    data_root: Path | None = None,
) -> list[ThreadExample]:
    """Load valid-QA threads for one subreddit (or use load_corpus for all)."""
    if sub.lower() in ("all", "*", "corpus"):
        threads, _, _ = load_corpus(
            max_threads=max_threads, min_prior=min_prior, data_root=data_root,
        )
        return threads

    data = data_root or (MAY15 / "data")
    pairs_path, judge_path, rc_path = _paths_for_sub(sub, data)
    judge_by_post = load_judge_by_post(judge_path)
    threads = load_selection_threads(pairs_path, judge_path, contrastive_only=False)
    selected = _select_heldout_corpus(threads, max_threads=max_threads, min_prior=min_prior)

    by_sub: dict[str, list[ThreadExample]] = defaultdict(list)
    for t in selected:
        by_sub[t.subreddit or sub].append(t)
    out: list[ThreadExample] = []
    for s, chunk in by_sub.items():
        _, jp, rp = _paths_for_sub(s if s else sub, data)
        out.extend(attach_comment_bodies(chunk, rp, load_judge_by_post(jp)))
    out.sort(key=lambda t: t.timestamp)
    return out
