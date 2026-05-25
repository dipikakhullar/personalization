# personalization

Research code for inferring latent communication preferences (verbosity, technicality, planning style, etc.) from behavioral signals rather than explicit user instructions.

## Layout

- `april_6/`, `may_8/`, `may_15/` — dated experiment directories, newest is most relevant.
- `may_15/reddit_pipeline/` — extracts `(query, preferred_answer)` pairs from Reddit using an OP-thanks-reply heuristic, pushes the dataset to HuggingFace at `dipikakhullar/personalization-reddit`.

## Reddit pipeline (`may_15/reddit_pipeline/`)

| script | purpose |
| --- | --- |
| `fetch_subreddit.py` | downloads an arctic_shift dump for one subreddit |
| `queue_fetchers.sh`, `run_remaining_pairs.sh` | nohup drivers for many subs |
| `extract.py` | finds OP-thanks replies, emits `sub-<name>.jsonl` of pairs |
| `signals.py`, `anon.py`, `subreddits.py` | filters, HMAC anonymization, allowlist |
| `push_to_hf.py` | uploads `data/extracted_current/` to the HF dataset repo |
| `judge_qa_pairs.py` | LLM-as-judge: GPT-4o / Claude opus / haiku via OpenRouter label each pair with `is_qa_pair: {question_answer_pair, explanation, judge_model}` and stream updates back to HF |

### Running the judge

```bash
# smoke test
python judge_qa_pairs.py --smoke 50

# one model on a partition (run multiple in parallel, one per partition)
nohup python -u judge_qa_pairs.py \
  --model openai/gpt-4o \
  --api-key-env OPENROUTER_API_KEY \
  --subs AskAcademia,AskBaking,... \
  > judge.log 2>&1 &

# retry records that failed all 5 retries
python judge_qa_pairs.py --retry-failures
```

Sidecars land at `data/llm_judge/sub-<name>.jsonl`; the judge merges + pushes each affected sub-file to HF every 1000 judgements (background thread, never blocks API calls). Resume is free — every restart skips records already in the sidecars and in `failures.jsonl`.

## Secrets

Put `HF_TOKEN`, `OPENROUTER_API_KEY`, and any extra OpenRouter keys in a `.env` at the repo root (gitignored).
