# Reddit personalization pipeline — data flow & roadmap

This documents how raw Reddit data becomes the `(query, preferred_answer, top_comment)`
dataset used by the personalization experiments (`june_2/`), what is automatic vs.
manual today, the **one-pair-per-post** data model, and the changes still needed.

Last updated: 2026-06-03.

---

## 1. The stages (fetch → extract → judge/push → experiment)

```
arctic_shift API
      │  fetch_subreddit.py            (per-sub dumps; resumable via fetch_state.json)
      ▼
data/dumps/sub-<Name>/RS_<Name>.ndjson , RC_<Name>.ndjson
      │  extract.py                    (OP-thanks-reply signal; allowlist-filtered)
      ▼
data/extracted_current/pairs/sub-<Name>.jsonl     ← ONE ROW PER THANKED ANSWER
      │  merge_pairs_by_post.py        (collapse to one row per post)
      ▼
data/pairs_by_post/sub-<Name>.jsonl               ← ONE ROW PER POST (canonical, target state)
      │  judge_qa_pairs.py             (is_qa_pair verdict; auto-push to HF)
      ▼
data/llm_judge/sub-<Name>.jsonl   +   HF dataset dipikakhullar/personalization-reddit
      │  june_2/build_samples.py       (history + target sampling)
      ▼
june_2/data/samples_*.jsonl  →  run_models.py  →  results / plots
```

### Stage 1 — Fetch (`fetch_subreddit.py`)
- Downloads submissions (RS) + comments (RC) for one subreddit into `data/dumps/sub-<Name>/`.
- Drivers run it N-at-a-time: `queue_fetchers.sh`, `run_remaining_pairs.sh`,
  `parallel_runner.sh`, `fetch_diverse20.sh` (parallelism 2–3).
- Resumable via each sub's `fetch_state.json` (`rs.done` / `rc.done`).
- **Does NOT extract or upload anything.** Dumps only.

### Stage 2 — Extract (`extract.py`)
- Per "batch" (a `sub-<Name>` dir): keep question-posts whose subreddit is in the
  `subreddits.py` **allowlist**, detect OP "thanks" replies, emit one
  `(query, preferred_answer)` row **per thanked answer**, attaching `top_comment`.
- Output: `data/extracted_current/pairs/sub-<Name>.jsonl`.
- Drivers: `extract_new_subs.sh`, `extract_diverse20.sh` (parallelism 3).
- **A subreddit must be in the allowlist or extract emits nothing.**
- **NOT automatic after fetch** — a driver/human runs it.

### Stage 3 — By-post merge (`merge_pairs_by_post.py`)  ← the fix
- Collapses the per-answer rows into **one row per `(subreddit, post_id)`**:
  - `first_preferred` = answer of the **earliest** thanks-reply (tiebreak: `answer_score`)
  - `other_preferred[]` = the remaining thanked answers (nothing dropped)
  - `is_qa_pair` verdicts attached per answer.
- Output: `data/pairs_by_post/sub-<Name>.jsonl`.
- As of 2026-06-03 the extraction drivers (`extract_new_subs.sh`, `extract_diverse20.sh`)
  run this automatically as their final step (**Option A**, below).

### Stage 4 — Judge + HF push (`judge_qa_pairs.py`)
- Reads `extracted_current/pairs/`, asks an LLM whether each `(query, preferred_answer)`
  is a genuine Q&A pair → `data/llm_judge/sub-<Name>.jsonl`
  (`is_qa_pair.question_answer_pair` bool + `judge_model`).
- **Auto-pushes** the merged (`pairs + is_qa_pair`) file to HF dataset
  `dipikakhullar/personalization-reddit` every `PUSH_EVERY=5000` judged rows —
  but **only for subs in a worker's `--subs` set**, and only while running.
- Run distributed across models via `launch_distributed_judge.py` (disjoint sub
  partitions so workers never double-judge). Resumable by `(post_id, answer_comment_id)`.

### Stage 5 — Experiment sampling (`june_2/build_samples.py`)
- Builds per-user `{history, target}` samples from `extracted_current/pairs/`.
- **Guard added 2026-06-03:** history must be 3 *distinct posts* and the target a
  *distinct* post — without it, users whose pairs all came from one popular thread
  produced degenerate/leaky samples (see §3).

---

## 2. What is automatic vs. manual (today)

| Step | Automatic? |
|------|-----------|
| fetch → dumps | manual (run a driver) |
| dumps → pairs (extract) | manual (run a driver) |
| pairs → pairs_by_post (merge) | **auto** at end of the extraction drivers |
| pairs → is_qa_pair + HF push | semi — auto **only** for subs assigned to a running judge worker |
| is_qa_pair for a newly-extracted sub | manual — must re-partition the judge fleet to include it |

So a brand-new subreddit needs, in order: run a fetch driver → run an extract driver
(now also merges) → add it to a judge worker's `--subs` (re-partition) so it gets
judged and pushed to HF.

---

## 3. Why the by-post model exists (the bug it fixes)

`extract.py` emits one row per *thanked answer*, so a single post with several
OP-thanked replies became several rows with an **identical query**. Consequences:
- In the 1000-sample experiment set: 48.7% had the target question also in the
  history (leakage), 80.9% had duplicate history questions, **86.4% degenerate**.
- The first "history hurts alignment" result was largely an **artifact** of this; on
  clean samples it disappears (history ≈ neutral).

Fix = one row per post (`first_preferred` + `other_preferred`), no data loss. Source
collapse magnitude: 3,020,654 pairs → 1,900,400 posts (**37.1%** were duplicate-post rows).

---

## 4. Roadmap — how we still need to change this

**Option A (in progress, recommended): standardize the merge + make `pairs_by_post/` canonical.**
- [x] Extraction drivers run `merge_pairs_by_post.py` as their final step.
- [ ] **Migrate the HF push** (`judge_qa_pairs.py: merge_sub` / uploader) to publish
      `pairs_by_post/` (path `extracted/pairs_by_post/`) instead of / in addition to the
      per-answer files. *Deferred:* the judge fleet is currently running on the
      per-answer format; do this at a fleet restart to avoid disruption.
- [ ] **Migrate `june_2/build_samples.py`** to read `pairs_by_post/` directly. Each row
      is already a distinct post, so the post_id-dedup guard becomes unnecessary and
      history is clean by construction. Do this on the next clean full re-run.
- [ ] Backfill: run a one-shot HF push of the existing `pairs_by_post/` to the new path.

**Option B (later, cleanest): emit one-per-post natively in `extract.py`.**
- Rewrite `pass2b_emit_pairs` to group by post and write `first_preferred`/`other_preferred`
  directly, removing the separate merge entirely.
- **Blocked on** updating every consumer that reads per-answer rows: `judge_qa_pairs.py`
  (read + HF push), `build_samples.py`, and the judge's `(post_id, answer_comment_id)`
  resume key. Requires a coordinated cutover + judge-fleet restart.

**Other known gaps**
- New subs are not judged/pushed until added to the fleet partition (re-partition needed).
- `medicine`, `Ask_Lawyers` fetched as 34K stubs (arctic_shift HTTP 500) — need re-fetch.
- `subreddits.py` mixes the allowlist with a bot-author blocklist; the `diversity_v1`
  category holds the deliberately-different subs added for the diversity study.
