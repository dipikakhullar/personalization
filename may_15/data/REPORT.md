# Reddit Personalization Dataset — Interim Report

*Snapshot taken from a partial AskStatistics fetch (~50% of the available
2008–2026 history) at `data/snapshot_20260515T1912/`. Numbers are interim and
expected to grow ~2× when AskStatistics RC completes, and 8× when all eight
curated subs are pulled.*

## What we are doing

We are building a temporally-ordered, user-level dataset of `(query,
preferred_answer)` pairs from Reddit to study whether language models can
infer latent communication preferences from behavioral signal alone — that
is, *without* explicit instructions from the user. The hypothesis is that
how a user reacts to answers (what they upvote, what they thank) reveals a
stable preference profile, and that this profile can be learned over time
from history rather than being elicited up front.

Reddit has no native "accepted answer" field, so we proxy with two signals:

1. **`preferred_answer`** — the comment that the **original poster (OP)
   explicitly thanked.** We detect this by scanning the OP's child-replies
   in their own thread for "thank you / this worked / perfect" language
   (with negation-aware filtering — "no thanks" doesn't count). When such a
   reply exists, its *parent comment* is the OP's preferred answer.
2. **`top_comment`** — the highest-scored non-OP, non-bot comment in the
   same thread. This is the **community** signal: what got upvoted by the
   wider audience.

Both signals are emitted side-by-side per `(query, answer)` pair, so each
record captures both a personal preference (`preferred_answer`) and a
public preference (`top_comment`). The rows we use for personalization are
**contrastive pairs**: `is_qa_pair` = true (judge says the thanked comment
is a real answer) **and** `top_equals_preferred` = false (community top ≠
thanked answer). Everything else is filtered out or used only as a null
baseline check. See *Data retention funnel* below and
[`plots/outputs/retention_funnel.png`](../../plots/outputs/retention_funnel.png).

**Keep rule:** a post is included *only* if OP wrote a thanks-shaped reply
to some comment. Posts where OP never thanked anyone are silently dropped
at extraction time. This is the entire selection criterion — there is no
upvote threshold, score floor, or subreddit-specific override.

**Answer-quality filter (`is_qa_pair`):** after extraction, every
`(query, preferred_answer)` pair is run through an LLM judge
(`judge_qa_pairs.py`, GPT-4o via OpenRouter) that asks whether the
thanked comment actually resolves the OP's question. This is the cheap
non-answer filter — OP thanks are often polite acknowledgments of partial
or off-topic replies, not genuine answers. The judge fires **false** on a
large share of pairs; use only survivors for contrastive / personalization
training.

| Judge metric (757,878 judged of 1.41M pairs, Jun 2026) | Count | % |
|---|---:|---:|
| `is_qa_pair` = true (survive) | 459,335 | **60.6%** |
| `is_qa_pair` = false (reject) | 298,543 | 39.4% |

Survival varies by sub (among judged pairs): AskHistorians ~56%, AskDocs
~57%, AskStatistics ~51%, askphilosophy ~65%. Roughly **40% of thanked
comments are not real answers** under the judge's criteria.

## Where the data comes from

We pull from **arctic-shift**, a free, independently-run mirror of Reddit's
full historical archive. A bit of background on why we're using this and
not Reddit's own API:

For years, Reddit researchers used a service called **Pushshift** that
archived every post and comment on Reddit and made monthly bulk files
freely available. In 2023, Reddit changed its API terms and Pushshift's
public access was locked down — the archives are now only available to
verified Reddit moderators. **arctic-shift** is what filled the gap. It's
maintained by one developer (Arthur Heitmann) and provides:

1. **The bulk archive** — every Reddit post and comment from June 2005
   through the present, published as monthly torrents on Academic
   Torrents (totals roughly 2–3 TB compressed for the full history).
2. **A free HTTP query API** at `arctic-shift.photon-reddit.com` that
   lets you fetch from those archives without downloading terabytes.

There is **no signup, no API key, no payment, and no quota** beyond a
polite rate limit. The only constraints are:

- **Rate limit:** 2,000 requests per ~40 seconds per IP address, or about
  **50 requests per second**. The server returns HTTP 429 if you cross
  it.
- **Page size:** each request returns at most **100 records** of either
  posts or comments. To pull a full subreddit you paginate forward by
  passing `after=<last_timestamp>` on each successive request.
- **Single source of truth:** this is not Reddit's own API. Reddit's
  official API caps historical browsing at the most recent ~1,000 items
  per user or subreddit, which makes the kind of multi-year history we
  want impossible to assemble directly. arctic-shift is precisely the
  tool you reach for when Reddit's own API can't do the job.

**Our use of it in this project:** `fetch_subreddit.py` issues paginated
GET requests like `…/api/posts/search?subreddit=AskStatistics&sort=asc
&after=<timestamp>&limit=100` and writes each batch of records to a
local newline-delimited JSON file. We pace requests at 5 per second — about
**2% of the published rate-limit ceiling** — which is conservative on
purpose: a single connection's throughput is already bottlenecked by the
~1 second per request the server takes to return, so being polite costs
us nothing in speed. Once a subreddit is fully pulled, the rest of the
pipeline (`extract.py`, `aggregate.py`, the user-trace generator) runs
entirely on the local files — no further network calls are made.

## Pipeline at a glance

```
arctic-shift API → fetch_subreddit.py → ndjson dumps (RS_*, RC_*)
                            │
                            ▼
                    extract.py (3-pass)
                       │
                       ├ pass 1: keep question-shaped posts in allowlist subs
                       ├ pass 2: scan comments once for (a) OP thanks-replies
                       │         and (b) per-post top non-OP comment
                       └ pass 3: materialize bodies → emit pair JSONL
                            │
                            ▼
                    aggregate.py → users.shard-NNN.jsonl + subreddits.csv
                                   (group by user, sort by timestamp, shard)
                            │
                            ▼
                    judge_qa_pairs.py → is_qa_pair on each pair
                            │
                            ▼
                    push_to_hf.py (merged pairs + judge field)
```

Usernames are SHA-256 hashed (with optional salt) to `anon_<16hex>` before
anything reaches disk — no raw Reddit usernames appear in the output.

## Numbers from the current snapshot

Partial AskStatistics fetch (RC ~50% pulled, RS complete):

| Funnel stage | Count |
|---|---:|
| Submissions scanned | 65,674 |
| Comments scanned | 194,400 |
| Question-shaped posts kept | 32,820 |
| Posts with an OP thanks-reply | 10,199 |
| Posts with at least one non-OP top-comment candidate | 21,374 |
| **Pairs emitted** | **9,919** |
| **Unique users** | **5,973** |

Yield: ~**30% of question-posts** had an OP-thanked reply. This is higher
than expected — AskStatistics is unusually answer-oriented; smoke testing
on AskBaking gave ~14%.

Both signals attached to every pair: **100%**.

## Schema

Each line of `extracted/pairs/sub-<name>.jsonl`:

```json
{
  "user_id": "anon_fb93fae33aca2a5c",
  "timestamp": "2018-04-08T13:42:11+00:00",
  "subreddit": "AskStatistics",
  "query": "Looking for techniques to do linear regression or ML model of sparse, discrete time series events…",
  "preferred_answer": "Paired t tests?",
  "top_comment": "Paired t tests?",
  "metadata": {
    "post_id": "8b9...",
    "post_score": 7,
    "answer_comment_id": "dx0...",
    "answer_score": 2,
    "answerer_anon_id": "anon_…",
    "top_comment_id": "dx0...",
    "top_comment_score": 2,
    "top_comment_anon_id": "anon_…",
    "top_equals_preferred": true,
    "thanks_reply_id": "dx1...",
    "thanks_reply_score": 0,
    "thanks_reply_text": "Thanks, but could you clarify further?…",
    "thanks_reply_timestamp": "2018-04-08T14:05:00+00:00"
  },
  "is_qa_pair": {
    "question_answer_pair": true,
    "explanation": "…",
    "judge_model": "openai/gpt-4o"
  }
}
```

`is_qa_pair.question_answer_pair` is the survival bit. Present on HF uploads
and in `data/llm_judge/merged_for_push/pairs/` once judged; sidecars live
at `data/llm_judge/sub-<name>.jsonl` before merge.

After aggregation, each user record is:

```json
{
  "user_id": "anon_fb93fae33aca2a5c",
  "interactions": [
    {...pair as above, without the redundant "user_id" key...},
    {...}, ...
  ]
}
```

Interactions inside a user record are sorted ascending by `timestamp`, so
reading top-to-bottom reconstructs that user's chronological Q&A history.

## Temporal histories work — one user, five years

User `anon_fb93fae33aca2a5c` has three interactions in the snapshot,
spanning **2013 → 2015 → 2018**. The questions are consistently
applied-statistics in nature, the writing voice is consistent, and the
thanks-replies show a consistent pattern: this user wants **concrete,
formula-level help** and pushes back when the response is too abstract or
too vague.

### Interaction 1 — 2013-12-24

> **Q:** *Hypothesis test for comparing two samples based on categorical
> variables / proportions?* — for a marketing analytics project, comparing
> ~5–10M sample to ~35M population by means; needs a proportions test.

- **preferred_answer** *(score = 2)* — the user thanked a concrete pointer:
  > "*Aside from the issues others have raised about doing significance
  > testing with such a large sample, the R command you are looking for is
  > prop.test —* `http://stat.ethz.ch/R-manual/R-patched/library/stats/html/prop.test.html`"
- **top_comment** *(score = 3)* — community voted up a philosophical reply
  about why the *whole frame* of hypothesis testing is wrong at that scale:
  > "*with millions of observations, it seems ludicrous to use hypothesis
  > testing at all; almost any interesting hypothesis should be rejected.
  > Estimates (both point and interval) may be interesting, but I can't see
  > what value there could be in hypothesis tests.*"
- **OP's thanks-reply:** "*Yes! This is exactly the kind of test that I
  was looking for. Thank you. I understand significance testing with large
  samples is not needed, but it should still be valid, no?*"

**This pair diverges**, and the divergence is *informative*: the community
preferred the meta-critique, the OP preferred the concrete tool. The
thanks-reply text even acknowledges the meta-critique but explicitly opts
out of it.

### Interaction 2 — 2015-02-14

> **Q:** *What distribution would the number of sales of a product in a
> given time period follow?* — looking for an alternative to the normal
> distribution for event counts per time.

- **preferred_answer = top_comment** *(score = 9, both signals agree)* —
  > "*Poisson distribution or negative binomial distribution are the best
  > options. Poisson distributions model event per time period while the
  > exponential distribution models time between events.*"
- **OP's thanks-reply:** "*Thank you for the straightforward answer!*"

The user's preference for **concrete, named tools** stays consistent — and
the thanks-reply explicitly calls out *straightforwardness* as the
desirable property. This kind of meta-comment is *exactly* the signal we
hope LLMs can learn to internalize.

### Interaction 3 — 2018-04-08

> **Q:** *Looking for techniques to do linear regression or ML model of
> sparse, discrete time series events (interventions) versus a continuous
> outcome.*

- **preferred_answer = top_comment** *(score = 2, both signals agree)*:
  > "*Paired t tests?*"
- **OP's thanks-reply:** "*Thanks, but could you clarify further? I'm not
  following how you'd apply a paired t-test here. What are the two
  means/two samples under test?*"

A terse suggestion, both the OP and community ranked it the most useful
reply. The thanks-reply here is *partially negative* — OP thanks the
suggestion but pushes back for specifics. This shows the signal isn't
"OP loved this answer" but rather "OP found this the most useful starting
point" — sometimes that's an answer they need to refine further.

### More multi-trace examples

Per-subreddit user-trace files live alongside this report and are
regenerated from the completed pair file via
`reddit_pipeline/generate_user_traces.py`. Each picks three users — heaviest
by interactions, longest temporal span, and highest divergence rate — and
renders 4–5 representative interactions per user on distinct posts.

Currently available:

- **[`users_AskStatistics.md`](users_AskStatistics.md)** — ✓ complete.
  12,811 pairs / 7,586 users / 56% divergent. Featured: a 6-year heavy
  user, a 10-year-span user, and an 87%-divergence personal-signal case.
- **[`users_AskAcademia.md`](users_AskAcademia.md)** — ⚠ partial (missing
  last ~8 months of comments). 23,823 pairs / 10,841 users / **73%
  divergent** — much higher than the other subs, plausibly because
  AskAcademia threads ask for advice on the OP's specific situation while
  the community upvotes general hot takes.
- **[`users_askphilosophy.md`](users_askphilosophy.md)** — ⚠ partial (RC
  about half done). 32,364 pairs / 16,073 users / 57% divergent.

Files for the remaining 5 curated subs will appear as each finishes.
Partial files are regenerated automatically when more data lands (just
re-run `reddit_pipeline/auto_users_docs.sh`).

## Data retention funnel

How much data survives each pipeline stage. Regenerate the plot with
`python plot_retention_funnel.py` from `reddit_pipeline/`.

![Retention funnel](../../plots/outputs/retention_funnel.png)

| Stage | Count | Kept (vs previous) | Kept (vs pairs emitted) |
|---|---:|---:|---:|
| Question-shaped posts | 5,670,576 | — | — |
| Unique thanked comments (pre-pair) | 1,454,730 | 25.7% of posts | — |
| **Pairs emitted** (thanks → preferred) | 1,410,273 | 97.0% of thanked comments | **100%** (baseline) |
| LLM judged | 757,867 | **53.7%** | 53.7% |
| Valid QA (`is_qa_pair` true) | 459,328 | **60.6%** | 32.6% |
| **Contrastive** (valid QA ∧ top ≠ preferred) | **232,947** | **50.7%** | **16.5%** |

**Target rows:** contrastive only — judge-valid answer, community top is a
different comment. **232,947** rows = **16.5%** of emitted pairs (**4.1%** of
question-shaped posts). Judge coverage is incomplete (~46% of emitted pairs
not judged yet); contrastive count will grow as judging finishes.

Among **valid QA** (459,328 rows), contrastive is **50.7%**; the other
49.3% are **agreement** rows (`top == preferred` on a real answer) — useful
for null-baseline evaluation, not contrastive training.

## Per-user activity and minimum thresholds

The extracted corpus has **645,073** askers and **1,410,284** emitted pairs.
Activity is extremely sparse: **median 1 pair/user**, p90 = 4, p99 = 13
(max 587). Regenerate with `python plot_user_activity.py` from
`reddit_pipeline/`.

| Plot | Path |
|---|---|
| User-trend figures (all) | [`plots/outputs/user_trends/`](../../plots/outputs/user_trends/) |
| Distribution + ECDF | [`user_activity_distribution.png`](../../plots/outputs/user_trends/user_activity_distribution.png) |
| Average history per user (overall + by sub) | [`user_activity_history_summary.png`](../../plots/outputs/user_trends/user_activity_history_summary.png) |
| Themed cross-sub flow matrix | [`flow_matrix_by_theme.png`](../../plots/outputs/user_trends/flow_matrix_by_theme.png) |
| Roam destinations (small multiples) | [`roam_destinations_small_multiples.png`](../../plots/outputs/user_trends/roam_destinations_small_multiples.png) |
| Activity by year / cohort depth | [`user_activity_temporal_trends.png`](../../plots/outputs/user_trends/user_activity_temporal_trends.png) |
| Threshold sensitivity | [`user_activity_threshold_sensitivity.png`](../../plots/outputs/user_trends/user_activity_threshold_sensitivity.png) |
| Numeric curves | [`user_activity_threshold_sensitivity.json`](../../plots/outputs/user_trends/user_activity_threshold_sensitivity.json) |

The sensitivity plot is one chart: as **min thanked threads per user** rises, it
shows % of users, all pairs, and **divergent pairs** retained (judge-valid QA
where OP’s thanked answer ≠ community top — the rows we train/evaluate on).

**Why these cutoffs:**

| Threshold | Use | Users kept | All pairs | Divergent pairs |
|---|---|---:|---:|---:|
| `min_pairs ≥ 3` | Per-user shard aggregation | 21.9% | 54.7% | see JSON |
| `min_pairs ≥ 5` | Heavy-user qualitative traces | 8.3% | 34.1% | see JSON |
| `min_subreddits ≥ 2` | Cross-sub only (in JSON) | 8.1% | 21.2% | — |

Raising `min_pairs` beyond 5 quickly erodes coverage (e.g. ≥10 keeps 2% of users,
16% of pairs). LOO benchmarks use **per-thread** prior count (`k`), not this
global user filter.

### Per curated sub — contrastive count (judged subs)

| Subreddit | Contrastive | % of sub's valid QA |
|---|---:|---:|
| AskDocs | 21,978 | 34.4% |
| AskCulinary | 17,469 | 60.5% |
| askphilosophy | 11,456 | 46.9% |
| AskEngineers | 14,345 | 64.7% |
| askscience | 12,646 | 33.0% |
| AskHistorians | 8,998 | 21.5% |
| AskAcademia | 9,189 | 65.3% |
| AskStatistics | 2,951 | 45.3% |

### Contrastive example #1 — joke vs. informative

> **Q:** *What is this kind of graph called and how does it work?* (image of
> a flow diagram)

- **preferred_answer** *(score = 1)*:
  > "*Is there any context or description given with the graphic? I agree
  > the vertical ordering seems like it should mean something but it is
  > not immediately clear.*"
- **top_comment** *(score = 103)*:
  > "*I call this a diarrhea graph and hope that it causes my students to
  > never use it.*"

The community top-voted a joke (103 upvotes). The OP thanked someone
genuinely engaging with what context might be missing. Two completely
different things "got the prize" — community optimized for entertainment,
OP optimized for help. This is a paradigmatic divergence case.

### Contrastive example #2 — concise vs. exhaustive

> **Q:** *Why is model "overfitting" bad? Shouldn't that be a good thing?*
> — first-year undergrad asking a basic stats question.

- **preferred_answer** *(score = 8)*:
  > "*Elements of statistical learning, pages 10 to 18, there you will find
  > one of the better example for the argument*"
- **top_comment** *(score = 156)*:
  > "*To put it simple, imagine you are preparing for an exam. You memorize
  > all the past papers down to every word and number. Now your professor
  > gives you a new exam paper and you are screwed because…*"

The community wrote a beloved, intuitive analogy (156 upvotes). The OP
thanked a curt textbook pointer. Different preferences for *explanation
style*: concrete-and-exhaustive vs. terse-and-authoritative. The OP's
thanks-reply ("*Thanks ill check it out! I'm not really versed in stats
myself; so my questions sounds dumb*") suggests they trusted authority over
metaphor.

## Agreement examples

When `top_equals_preferred`, both signals confirm the same answer — useful
training data for "this is high-quality regardless of who you are."

Example (interaction 2 above, repeated):

> **Q:** *What distribution would the number of sales of a product in a
> given time period follow?*
>
> **Both:** "*Poisson distribution or negative binomial distribution…*"
> *(score = 9, OP thanked it explicitly)*

For research:

- **Train / evaluate personalization on contrastive rows only** (valid QA,
  `top ≠ preferred`).
- **Agreement rows** (valid QA, `top == preferred`) — check whether a
  personalized model beats the null policy "always pick top comment"; it
  often should not.

## Caveats and pending work

- **Partial fetch.** Only AskStatistics is partially pulled. The other 7
  curated subs (AskAcademia, AskCulinary, AskDocs, AskEngineers,
  AskHistorians, askscience, askphilosophy) are queued. Full numbers will
  arrive once the background fetch finishes (~1–2 days).
- **Single-sub histories.** Most users currently have only 1 pair (5,973
  users / 9,919 pairs ≈ 1.66 pairs/user). The cross-sub histories — the
  thing that makes this dataset valuable for personalization — only
  materialize after multiple subs are pulled.
- **Thanks-reply precision is regex-based.** It catches the typical "thanks
  / perfect / this worked / exactly what I needed" phrasings, rejects
  negators ("no thanks", "thanks for nothing"), and requires the thanks
  token in the opening of the reply. Precision is high (manual eyeballing
  of the smoke set found 0 false positives in 29 pairs). Non-answer thanks
  are handled downstream by `is_qa_pair` (~39% rejection rate).
- **Judge coverage is partial.** ~54% of extracted pairs are judged so far;
  the rest lack `is_qa_pair` until `judge_qa_pairs.py` catches up.
- **`top_comment` is whole-thread.** It's the top-scored non-OP comment
  anywhere in the tree, not necessarily at top-level. For most posts the
  top comment *is* top-level, but for very deep threads this distinction
  matters. Configurable in a future revision if needed.
- **No subreddit identity beyond the `subreddit` field.** We deliberately
  don't filter by community norms; a user's preferences might be
  consistent across subreddits, and that's an empirical question we want
  to leave open.

## Where to find things

```
/workspace/personalization/may_15/
├── reddit_pipeline/                              # source code
│   ├── fetch_subreddit.py
│   ├── extract.py
│   ├── aggregate.py
│   ├── judge_qa_pairs.py     # is_qa_pair LLM judge
│   ├── plot_retention_funnel.py
│   ├── signals.py            # thanks-reply + question-shape regexes
│   └── subreddits.py         # curated allowlist + bot blocklist
├── plots/outputs/            # figures (retention_funnel.png, …)
└── data/
    ├── REPORT.md             # ← this file
    ├── fetch.log             # live fetcher log
    ├── llm_judge/            # judge sidecars + merged_for_push/
    ├── dumps/sub-<name>/     # raw API data, live and growing
    ├── smoke/                # frozen 300-record AskBaking probe
    └── snapshot_<ts>/        # point-in-time extracted snapshots
        ├── extracted/pairs/<sub>.jsonl
        ├── extracted/stats/<sub>.json
        └── users/users.shard-*.jsonl + subreddits.csv
```
