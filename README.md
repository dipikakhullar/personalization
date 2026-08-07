# personalization

Research code for inferring latent communication preferences (verbosity, technicality, planning style, etc.) from behavioral signals rather than explicit user instructions.

## Layout

- `april_6/`, `may_8/`, `may_15/` — dated experiment directories, newest is most relevant.
- `may_15/reddit_pipeline/` — Reddit `(query, preferred_answer)` extraction → aggregation → LLM judging → HuggingFace dataset.

## Reddit pipeline — end-to-end

Five stages, run in order. Outputs at each step are inputs to the next.

```
fetch  →  extract  →  aggregate     →  push to HF  →  judge
(API)     (pairs)     (multi-turn)     (dataset)      (is_qa_pair)
```

All commands are run from `may_15/reddit_pipeline/`. Setup:

```bash
cd may_15/reddit_pipeline
pip install -r requirements.txt
# Secrets at repo root .env (gitignored): HF_TOKEN, OPENROUTER_API_KEY[, OPENROUTERAPIkey2]
```

### 1. Fetch raw dumps

`fetch_subreddit.py` paginates the arctic_shift HTTP API for one subreddit, writing NDJSON to `data/dumps/sub-<name>/`:

```bash
python fetch_subreddit.py --sub AskHistorians --out-dir ../data/dumps --rate-seconds 0.05
# resumable: re-running picks up at fetch_state.json cursor
```

Bulk drivers (each runs N fetchers in parallel under `nohup`, each fetcher logs to `/tmp/fetch_<sub>.log`):

```bash
# fixed list (edit script to change subs):
nohup bash queue_fetchers.sh > /tmp/queue_fetchers.log 2>&1 &

# allowlist of remaining subs, resumable; skips subs whose fetch_state.json says done:
nohup bash run_remaining_pairs.sh > ../data/pairs.log 2>&1 &
```

Output per sub: `data/dumps/sub-<name>/{RS_<name>.ndjson, RC_<name>.ndjson, fetch_state.json}`.

### 2. Extract `(query, preferred_answer)` pairs

`extract.py` reads one sub's dumps and emits pairs using the OP-thanks-reply heuristic (when the original poster replies "thanks" to a comment, that parent comment is the preferred answer):

```bash
python extract.py --month sub-AskHistorians --dumps-dir ../data/dumps --out-dir ../data/extracted_current
```

Or, batch over every sub that has substantial data (≥100 submissions and ≥1000 comments):

```bash
bash auto_users_docs.sh
# wraps extract.py + generate_user_traces.py per sub
# writes data/extracted_current/pairs/sub-<name>.jsonl + data/users_<name>.md
```

Per-pair schema (one JSON object per line): `user_id` (HMAC-anonymized), `timestamp`, `subreddit`, `query`, `preferred_answer`, `top_comment`, `op_metadata`, `answerer_metadata`, `metadata` (post_id, scores, thanks-reply context, etc.).

### 3. Aggregate into per-user multi-turn histories

`aggregate.py` groups pairs by `user_id`, sorts each user's interactions by timestamp, and shards across N files for streaming-friendly training:

```bash
python aggregate.py \
    --extracted-dir ../data/extracted_current \
    --out-dir ../data/users \
    --min-pairs-per-user 5 \
    --min-subreddits-per-user 2
```

Output: `data/users/users.shard-NNN.jsonl` (one user per line, `{user_id, interactions: [...]}`), `data/users/subreddits.csv`, `data/users/aggregate_stats.json`.

### 4. Push pairs dataset to HuggingFace

`push_to_hf.py` uploads `data/extracted_current/` to `dipikakhullar/personalization-reddit` (one HF split per subreddit). Uses `HF_TOKEN` from `.env`.

```bash
python push_to_hf.py
```

### 5. LLM-as-judge → `is_qa_pair`

`judge_qa_pairs.py` runs every pair through OpenRouter (GPT-4o / claude-opus-4.6-fast / claude-haiku-4.5) and asks whether the preferred answer actually resolves the OP's question. Adds a top-level field per record:

```json
{"is_qa_pair": {"question_answer_pair": true, "explanation": "...", "judge_model": "openai/gpt-4o"}}
```

Smoke test first:

```bash
python judge_qa_pairs.py --smoke 50               # 50 records, stdout only
python judge_qa_pairs.py --smoke 50 --persist     # writes sidecar, no HF push
```

Full run (use `--subs` to partition work across multiple parallel workers; each worker uses one OpenRouter key + one model):

```bash
# worker 1 — gpt-4o, partition A
nohup python -u judge_qa_pairs.py \
  --model openai/gpt-4o --api-key-env OPENROUTER_API_KEY \
  --subs AskAcademia,AskBaking,AskCulinary,... \
  >> ../data/judge.log 2>&1 &

# worker 2 — claude-haiku-4.5, partition B
nohup python -u judge_qa_pairs.py \
  --model anthropic/claude-haiku-4.5 --api-key-env OPENROUTERAPIkey2 \
  --subs AskDocs,Coffee,DIY,... \
  > ../data/judge_haiku.log 2>&1 &

# worker 3 — claude-opus-4.6-fast, partition C
nohup python -u judge_qa_pairs.py \
  --model anthropic/claude-opus-4.6-fast --api-key-env OPENROUTERAPIkey2 \
  --subs AskEngineers,German,LanguageTechnology,... \
  >> ../data/judge_claude.log 2>&1 &
```

Resume is free across restarts: sidecars at `data/llm_judge/sub-<name>.jsonl` + `failures.jsonl` are loaded into a seen-set on startup and skipped. Every 1000 judgements per worker, that worker merges `is_qa_pair` into the affected `sub-<name>.jsonl` and uploads it to HF in a background thread (does not block API calls).

Records that fail all 5 retries are appended to `data/llm_judge/failures.jsonl`. Recover with:

```bash
python judge_qa_pairs.py --retry-failures --model openai/gpt-4o --api-key-env OPENROUTER_API_KEY
```

## Secrets

Put `HF_TOKEN`, `OPENROUTER_API_KEY`, and any extra OpenRouter keys (e.g. `OPENROUTERAPIkey2`) in `.env` at the repo root. The file is gitignored.
