"""Three-panel personalization benchmark (no model training).

Panel A: LOO temporal top-1 accuracy vs k (faceted: divergent / agreement).
Panel B: ICC waterfall with confound decrements.
Panel C: Permutation histogram of within-asker similarity.

Run:
  python benchmark.py --sub AskStatistics --max-threads 800
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from load_threads import (
    FAST_BENCHMARK_SUBS,
    ThreadExample,
    discover_benchmark_subs,
    load_corpus,
    load_subset,
)

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "plots" / "outputs"

METHODS = [
    ("chance", "#94a3b8", "Chance (1/n candidates)"),
    ("baseline", "#334155", "Non-personalized baseline"),
    ("per_asker_raw", "#ea580c", "TF-IDF + raw history"),
    ("per_asker_resid", "#2563eb", "TF-IDF + residualized history"),
    ("llm_ranker", "#059669", "LLM ranker (history-conditioned)"),
]
SCORE_FIELDS = [m[0] for m in METHODS if m[0] != "chance"]
PLOT_BAR_FIELDS = [m for m in METHODS if m[0] not in ("chance",)]
MIN_K_FOR_CURVE = 20  # smallest k bucket needs this many threads to show on curve


# ── Features / scoring ───────────────────────────────────────────────────


def fit_vectorizer(threads: list[ThreadExample]) -> TfidfVectorizer:
    texts = [t.query for t in threads]
    for t in threads:
        texts.extend(c.body for c in t.candidates)
    vec = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2)
    vec.fit(texts)
    return vec


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return (an @ bn.T).ravel()


def topic_clusters(vec: TfidfVectorizer, threads: list[ThreadExample], k: int = 12) -> dict[str, int]:
    n = min(k, max(2, len(threads) // 15))
    Q = vec.transform([t.query for t in threads])
    labels = KMeans(n_clusters=n, n_init=10, random_state=0).fit_predict(Q)
    return {t.post_id: int(lab) for t, lab in zip(threads, labels)}


def subreddit_index(threads: list[ThreadExample]) -> dict[str, int]:
    subs = sorted({t.subreddit for t in threads})
    return {s: i for i, s in enumerate(subs)}


def flair_hash(flair: str | None) -> int:
    return hash(flair or "") % 997


def history_embedding(vec: TfidfVectorizer, bodies: list[str]) -> np.ndarray:
    if not bodies:
        return np.zeros(vec.transform(["x"]).shape[1])
    M = vec.transform(bodies)
    v = np.asarray(M.mean(axis=0)).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def residualize_1d(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    if X.size == 0 or X.shape[0] < X.shape[1] + 2:
        return y - y.mean()
    Xd = np.column_stack([np.ones(len(y)), X])
    coef, _, _, _ = np.linalg.lstsq(Xd, y, rcond=None)
    return y - Xd @ coef


def confound_matrix(
    thread: ThreadExample,
    topic_map: dict[str, int],
    sub_map: dict[str, int],
    n_topics: int,
    n_subs: int,
) -> np.ndarray:
    row = []
    t = topic_map.get(thread.post_id, 0)
    row.extend(1.0 if i == t else 0.0 for i in range(n_topics))
    s = sub_map.get(thread.subreddit, 0)
    row.extend(1.0 if i == s else 0.0 for i in range(n_subs))
    row.append(float(flair_hash(thread.op_flair) % 17) / 17.0)
    return np.array(row, dtype=float)


@dataclass
class Scores:
    chance: float
    baseline: float
    per_asker_raw: float
    per_asker_resid: float


def score_candidates(
    thread: ThreadExample,
    history: list[ThreadExample],
    vec: TfidfVectorizer,
    topic_map: dict[str, int],
    sub_map: dict[str, int],
    n_topics: int,
    n_subs: int,
    *,
    global_score_mean: float,
    global_score_std: float,
) -> dict[str, Scores]:
    q = np.asarray(vec.transform([thread.query]).todense()).ravel()
    conf = confound_matrix(thread, topic_map, sub_map, n_topics, n_subs)
    n_c = len(thread.candidates)
    out: dict[str, Scores] = {}

    for c in thread.candidates:
        cv = np.asarray(vec.transform([c.body]).todense()).ravel()
        sem = float(cosine_rows(q.reshape(1, -1), cv.reshape(1, -1))[0])
        sc = (c.score - global_score_mean) / (global_score_std + 1e-9)
        base = 0.45 * sem + 0.35 * sc + 0.05 * np.log1p(len(c.body)) - 0.02 * c.position
        raw_parts: list[float] = []
        hist_confs: list[np.ndarray] = []
        for h in history:
            hb = []
            for hc in h.candidates:
                if hc.comment_id == h.thanked_id:
                    hb.append(hc.body)
                    break
            if not hb:
                continue
            he = history_embedding(vec, hb)
            raw_parts.append(float(cosine_rows(he.reshape(1, -1), cv.reshape(1, -1))[0]))
            hist_confs.append(confound_matrix(h, topic_map, sub_map, n_topics, n_subs))
        raw_hist = float(np.mean(raw_parts)) if raw_parts else 0.0
        # Residualize per-history similarities (need ≥3 points; 1-row lstsq zeros out)
        if len(raw_parts) >= 3:
            y = np.array(raw_parts)
            X = np.vstack(hist_confs)
            resid_hist = float(np.mean(residualize_1d(y, X)))
        else:
            resid_hist = raw_hist
        out[c.comment_id] = Scores(
            chance=1.0 / n_c,
            baseline=base,
            per_asker_raw=base + 0.55 * raw_hist,
            per_asker_resid=base + 0.55 * resid_hist,
        )
    return out


def predict_top1(scores: dict[str, Scores], field: str) -> str:
    return max(scores.keys(), key=lambda cid: getattr(scores[cid], field))


def ranking(scores: dict[str, Scores], field: str) -> list[str]:
    return sorted(scores.keys(), key=lambda cid: (-getattr(scores[cid], field), cid))


GUARDRAIL_MIN_REORDER = 0.05


def guardrail_reorders(
    threads: list[ThreadExample],
    vec: TfidfVectorizer,
    topic_map: dict[str, int],
    sub_map: dict[str, int],
    k_cap: int,
    *,
    min_prior: int = 1,
) -> dict:
    """Fraction of held-outs where raw/resid change the full ranking vs baseline."""
    n_topics = max(topic_map.values()) + 1 if topic_map else 1
    n_subs = len(sub_map)
    scores_all = [c.score for t in threads for c in t.candidates]
    sm, ss = float(np.mean(scores_all)), float(np.std(scores_all) + 1e-9)

    by_asker: dict[str, list[ThreadExample]] = defaultdict(list)
    for t in threads:
        by_asker[t.asker_id].append(t)
    for seq in by_asker.values():
        seq.sort(key=lambda x: x.timestamp)

    n = raw_diff = resid_diff = 0
    for uid, seq in by_asker.items():
        for i, held in enumerate(seq):
            if i < min_prior:
                continue
            history = seq[:i]
            sc = score_candidates(
                held, history, vec, topic_map, sub_map, n_topics, n_subs,
                global_score_mean=sm, global_score_std=ss,
            )
            rb = ranking(sc, "baseline")
            if rb != ranking(sc, "per_asker_raw"):
                raw_diff += 1
            if rb != ranking(sc, "per_asker_resid"):
                resid_diff += 1
            n += 1

    frac_raw = raw_diff / n if n else 0.0
    frac_resid = resid_diff / n if n else 0.0
    return {
        "n_held_out": n,
        "frac_raw_reorders": frac_raw,
        "frac_resid_reorders": frac_resid,
        "passed": frac_raw >= GUARDRAIL_MIN_REORDER and frac_resid >= GUARDRAIL_MIN_REORDER,
    }


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def bootstrap_pool_stats(
    records: list[dict],
    *,
    n_boot: int,
    rng: np.random.Generator,
    fields: list[str] | None = None,
) -> dict:
    """Mean accuracy + asker-bootstrap CI per method; delta CIs vs baseline."""
    empty = {
        "n_threads": 0,
        "n_askers": 0,
        "mean": {f: float("nan") for f in SCORE_FIELDS},
        "ci": {f: (float("nan"), float("nan")) for f in SCORE_FIELDS},
        "wilson": {f: (float("nan"), float("nan")) for f in SCORE_FIELDS},
        "chance_mean": float("nan"),
        "delta_mean": {},
        "delta_ci": {},
    }
    if not records:
        return empty

    fields = fields or list(SCORE_FIELDS)
    if records and "llm_ranker" in fields:
        fields = [
            f for f in fields
            if f != "llm_ranker" or any(r.get("llm_ranker") is not None for r in records)
        ]

    by_asker: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_asker[r["asker_id"]].append(r)
    askers = list(by_asker.keys())
    n = len(records)

    mean = {}
    ci = {}
    wilson = {}
    for field in fields:
        if field == "llm_ranker":
            sub = [r for r in records if r.get("llm_ranker") is not None]
            hits = sum(1 for r in sub if r["llm_ranker"])
            n_f = len(sub)
        else:
            hits = sum(1 for r in records if r[field])
            n_f = n
        mean[field] = hits / n_f if n_f else float("nan")
        wilson[field] = wilson_ci(hits, n_f) if n_f else (float("nan"), float("nan"))
        boots = []
        for _ in range(n_boot):
            picked = rng.choice(askers, size=len(askers), replace=True)
            vals = [
                (r.get("llm_ranker") if field == "llm_ranker" else r[field])
                for a in picked for r in by_asker[a]
                if field != "llm_ranker" or r.get("llm_ranker") is not None
            ]
            if vals:
                boots.append(float(np.mean(vals)))
        ci[field] = (
            (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
            if len(boots) >= 30
            else (float("nan"), float("nan"))
        )

    delta_mean = {}
    delta_ci = {}
    delta_fields = [f for f in ("per_asker_raw", "per_asker_resid", "llm_ranker") if f in fields]
    for field in delta_fields:
        delta_mean[field] = mean[field] - mean["baseline"]
        boots = []
        for _ in range(n_boot):
            picked = rng.choice(askers, size=len(askers), replace=True)
            vals = [
                (r.get(field) if field == "llm_ranker" else r[field]) - r["baseline"]
                for a in picked for r in by_asker[a]
                if field != "llm_ranker" or r.get("llm_ranker") is not None
            ]
            if vals:
                boots.append(float(np.mean(vals)))
        delta_ci[field] = (
            (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
            if len(boots) >= 30
            else (float("nan"), float("nan"))
        )

    return {
        "n_threads": n,
        "n_askers": len(askers),
        "mean": mean,
        "ci": ci,
        "wilson": wilson,
        "chance_mean": float(np.mean([r["chance_expect"] for r in records])),
        "delta_mean": delta_mean,
        "delta_ci": delta_ci,
    }


def collect_records(
    threads: list[ThreadExample],
    vec: TfidfVectorizer,
    topic_map: dict[str, int],
    sub_map: dict[str, int],
    k_cap: int,
    *,
    min_prior: int = 0,
) -> list[dict]:
    n_topics = max(topic_map.values()) + 1 if topic_map else 1
    n_subs = len(sub_map)
    scores_all = [c.score for t in threads for c in t.candidates]
    sm, ss = float(np.mean(scores_all)), float(np.std(scores_all) + 1e-9)

    by_asker: dict[str, list[ThreadExample]] = defaultdict(list)
    for t in threads:
        by_asker[t.asker_id].append(t)
    for seq in by_asker.values():
        seq.sort(key=lambda x: x.timestamp)

    records = []
    for uid, seq in by_asker.items():
        for i, held in enumerate(seq):
            if i < min_prior:
                continue
            history = [h for h in seq if h.timestamp < held.timestamp]
            k = min(len(history), k_cap)
            hist_subs = {h.subreddit for h in history}
            cross_sub = held.subreddit not in hist_subs
            sc = score_candidates(
                held, history, vec, topic_map, sub_map, n_topics, n_subs,
                global_score_mean=sm, global_score_std=ss,
            )
            correct = held.thanked_id
            rec = {
                "asker_id": uid,
                "post_id": held.post_id,
                "k": k,
                "k_raw": len(history),
                "n_candidates": len(held.candidates),
                "chance_expect": 1.0 / len(held.candidates),
                "divergent": held.is_divergent,
                "cross_sub": cross_sub,
                "same_sub": not cross_sub,
                "held_subreddit": held.subreddit,
            }
            for field in ("baseline", "per_asker_raw", "per_asker_resid"):
                rec[field] = predict_top1(sc, field) == correct
            records.append(rec)
    return records


def panel_a_by_k(
    records: list[dict],
    *,
    k_cap: int,
    min_prior: int,
    n_boot: int,
    rng: np.random.Generator,
) -> dict:
    """Per-k discrete estimates (appendix only when n is tiny)."""
    ks = list(range(min_prior, k_cap + 1))
    by_k: dict[int, list[dict]] = {k: [] for k in ks}
    for r in records:
        if min_prior <= r["k"] <= k_cap:
            by_k[r["k"]].append(r)

    per_k = {}
    for k in ks:
        sub = by_k[k]
        if not sub:
            per_k[k] = {"n_threads": 0, "n_askers": 0}
            continue
        stats = bootstrap_pool_stats(sub, n_boot=n_boot, rng=rng)
        stats["chance_mean"] = float(np.mean([r["chance_expect"] for r in sub]))
        per_k[k] = stats

    return {
        "k_values": ks,
        "per_k": per_k,
        "k_cap": k_cap,
        "min_prior": min_prior,
        "min_n_for_curve": MIN_K_FOR_CURVE,
    }


def panel_a(
    threads: list[ThreadExample],
    vec: TfidfVectorizer,
    topic_map: dict[str, int],
    sub_map: dict[str, int],
    *,
    k_cap: int = 8,
    min_prior: int = 1,
    n_boot: int = 400,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    all_recs = collect_records(
        threads, vec, topic_map, sub_map, k_cap, min_prior=0,
    )

    eval_recs = [r for r in all_recs if r["k"] >= min_prior]
    div_eval = [r for r in eval_recs if r["divergent"]]
    agr_eval = [r for r in eval_recs if not r["divergent"]]
    div_k0 = [r for r in all_recs if r["divergent"] and r["k"] == 0]
    agr_k0 = [r for r in all_recs if not r["divergent"] and r["k"] == 0]

    def facet_bundle(div_pool: list[dict], agr_pool: list[dict], div_k0_pool, agr_k0_pool):
        return {
            "pooled_k_ge_1": bootstrap_pool_stats(div_pool, n_boot=n_boot, rng=rng),
            "pooled_k_0": bootstrap_pool_stats(div_k0_pool, n_boot=n_boot, rng=rng),
        }

    return {
        "min_prior": min_prior,
        "k_cap": k_cap,
        "n_total_eval": len(eval_recs),
        "divergent": {
            "pooled_k_ge_1": bootstrap_pool_stats(div_eval, n_boot=n_boot, rng=rng),
            "pooled_k_0": bootstrap_pool_stats(div_k0, n_boot=n_boot, rng=rng),
            "by_k": panel_a_by_k(
                [r for r in all_recs if r["divergent"]],
                k_cap=k_cap, min_prior=min_prior, n_boot=n_boot, rng=rng,
            ),
        },
        "agreement": {
            "pooled_k_ge_1": bootstrap_pool_stats(agr_eval, n_boot=n_boot, rng=rng),
            "pooled_k_0": bootstrap_pool_stats(agr_k0, n_boot=n_boot, rng=rng),
            "by_k": panel_a_by_k(
                [r for r in all_recs if not r["divergent"]],
                k_cap=k_cap, min_prior=min_prior, n_boot=n_boot, rng=rng,
            ),
        },
    }


# ── Panel B ───────────────────────────────────────────────────────────────


def icc_oneway(embeddings: np.ndarray, groups: np.ndarray) -> float:
    uniq = np.unique(groups)
    if len(uniq) < 2 or len(embeddings) < 3:
        return 0.0
    grand = embeddings.mean(axis=0)
    n = len(embeddings)
    ss_between = ss_within = 0.0
    for g in uniq:
        idx = groups == g
        m = embeddings[idx].mean(axis=0)
        ss_between += idx.sum() * float(np.sum((m - grand) ** 2))
        for row in embeddings[idx]:
            ss_within += float(np.sum((row - m) ** 2))
    df_b, df_w = len(uniq) - 1, n - len(uniq)
    if df_b < 1 or df_w < 1:
        return 0.0
    ms_b, ms_w = ss_between / df_b, ss_within / df_w
    denom = ms_b + (n / len(uniq) - 1) * ms_w
    return max(0.0, (ms_b - ms_w) / denom) if denom > 0 else 0.0


def embed_thanked(threads: list[ThreadExample], vec: TfidfVectorizer) -> tuple[np.ndarray, np.ndarray, list[ThreadExample]]:
    bodies, used = [], []
    for t in threads:
        for c in t.candidates:
            if c.comment_id == t.thanked_id:
                bodies.append(c.body)
                used.append(t)
                break
    E = np.asarray(vec.transform(bodies).todense())
    g = np.array([hash(t.asker_id) % (2**31) for t in used], dtype=np.int64)
    return E, g, used


def residualize_embeddings(E: np.ndarray, C: np.ndarray, cols: slice) -> np.ndarray:
    out = E.copy()
    for j in range(E.shape[1]):
        out[:, j] = residualize_1d(E[:, j], C[:, cols])
    return out


def panel_b(
    threads: list[ThreadExample],
    vec: TfidfVectorizer,
    topic_map: dict[str, int],
    sub_map: dict[str, int],
    *,
    n_boot: int = 200,
    seed: int = 0,
) -> dict:
    E, g, used = embed_thanked(threads, vec)
    n_topics = max(topic_map.values()) + 1 if topic_map else 1
    n_subs = len(sub_map)
    C = np.vstack([confound_matrix(t, topic_map, sub_map, n_topics, n_subs) for t in used])

    icc_raw = icc_oneway(E, g)
    E_t = residualize_embeddings(E, C, slice(0, n_topics))
    icc_t = icc_oneway(E_t, g)
    E_s = residualize_embeddings(E_t, C, slice(n_topics, n_topics + n_subs))
    icc_s = icc_oneway(E_s, g)
    E_f = residualize_embeddings(E_s, C, slice(-1, None))
    icc_f = icc_oneway(E_f, g)
    icc_res = icc_f

    steps = ["raw", "after_topic", "after_subreddit", "after_flair", "residual"]
    levels = [icc_raw, icc_t, icc_s, icc_f, icc_res]
    decrements = [levels[i] - levels[i + 1] for i in range(4)]

    # Bootstrap ICC raw & residual (resample askers)
    by_asker: dict[int, list[int]] = defaultdict(list)
    for i, t in enumerate(used):
        by_asker[g[i]].append(i)
    asker_keys = list(by_asker.keys())
    rng = np.random.default_rng(seed)
    boot_raw, boot_res = [], []
    for _ in range(n_boot):
        picked = rng.choice(asker_keys, size=len(asker_keys), replace=True)
        idx = [i for a in picked for i in by_asker[a]]
        if len(idx) < 10:
            continue
        boot_raw.append(icc_oneway(E[idx], g[idx]))
        boot_res.append(icc_oneway(E_f[idx], g[idx]))

    def ci(arr):
        if len(arr) < 20:
            return None
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    return {
        "steps": steps,
        "levels": levels,
        "decrements": decrements,
        "icc_raw": icc_raw,
        "icc_residual": icc_res,
        "ci_raw": ci(boot_raw),
        "ci_residual": ci(boot_res),
        "n_threads": len(used),
        "n_askers": len(asker_keys),
    }


# ── Panel C ───────────────────────────────────────────────────────────────


def panel_c(
    threads: list[ThreadExample],
    vec: TfidfVectorizer,
    topic_map: dict[str, int],
    sub_map: dict[str, int],
    *,
    n_perm: int = 2000,
    seed: int = 1,
) -> dict:
    E, g, used = embed_thanked(threads, vec)
    n_topics = max(topic_map.values()) + 1 if topic_map else 1
    n_subs = len(sub_map)
    C = np.vstack([confound_matrix(t, topic_map, sub_map, n_topics, n_subs) for t in used])
    E_r = residualize_embeddings(E, C, slice(0, C.shape[1]))

    def within_stat(M: np.ndarray, groups: np.ndarray) -> float:
        by: dict[int, list[int]] = defaultdict(list)
        for i, gi in enumerate(groups):
            by[gi].append(i)
        sims = []
        for idxs in by.values():
            if len(idxs) < 2:
                continue
            V = M[idxs]
            V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
            S = V @ V.T
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    sims.append(S[i, j])
        return float(np.mean(sims)) if sims else 0.0

    obs_raw = within_stat(E, g)
    obs_res = within_stat(E_r, g)
    uniq = np.unique(g)
    rng = np.random.default_rng(seed)
    null_raw, null_res = [], []
    for _ in range(n_perm):
        perm = rng.permutation(uniq)
        mp = {a: perm[i] for i, a in enumerate(uniq)}
        g_p = np.array([mp[gi] for gi in g])
        null_raw.append(within_stat(E, g_p))
        null_res.append(within_stat(E_r, g_p))

    null_raw = np.array(null_raw)
    null_res = np.array(null_res)
    return {
        "observed_raw": obs_raw,
        "observed_residual": obs_res,
        "p_raw": float((null_raw >= obs_raw).mean()),
        "p_residual": float((null_res >= obs_res).mean()),
        "null_raw": null_raw.tolist(),
        "null_residual": null_res.tolist(),
        "null_p95_raw": float(np.percentile(null_raw, 95)),
        "null_p95_residual": float(np.percentile(null_res, 95)),
        "n_perm": n_perm,
        "n_askers_multi": sum(1 for a in uniq if (g == a).sum() >= 2),
    }


# ── Pooled cross-sub benchmark ────────────────────────────────────────────


def corpus_stats(
    threads: list[ThreadExample],
    eval_records: list[dict],
    *,
    subs_used: list[str],
    user_pair_counts: dict[str, int],
) -> dict:
    k_dist: dict[int, int] = defaultdict(int)
    for r in eval_records:
        k_dist[int(r["k"])] += 1
    return {
        "n_subs": len(subs_used),
        "subs_used": subs_used,
        "n_threads_with_bodies": len(threads),
        "n_askers_in_threads": len({t.asker_id for t in threads}),
        "n_eval_held_outs_k_ge_1": len(eval_records),
        "n_divergent_eval": sum(1 for r in eval_records if r["divergent"]),
        "n_agreement_eval": sum(1 for r in eval_records if not r["divergent"]),
        "n_cross_sub": sum(1 for r in eval_records if r["cross_sub"]),
        "n_same_sub": sum(1 for r in eval_records if r["same_sub"]),
        "k_distribution": {str(k): v for k, v in sorted(k_dist.items())},
        "users_by_pair_count": {
            str(c): sum(1 for u, n in user_pair_counts.items() if n == c)
            for c in sorted(set(user_pair_counts.values()))[-12:]
        },
        "median_pairs_per_user": float(np.median(list(user_pair_counts.values())))
        if user_pair_counts else 0,
    }


def build_pooled_metrics(
    threads: list[ThreadExample],
    vec: TfidfVectorizer,
    topic_map: dict[str, int],
    sub_map: dict[str, int],
    user_pair_counts: dict[str, int],
    *,
    k_cap: int,
    min_prior: int,
    n_boot: int,
    seed: int,
    eval_recs: list[dict] | None = None,
) -> tuple[dict, list[dict]]:
    rng = np.random.default_rng(seed)
    guard = guardrail_reorders(
        threads, vec, topic_map, sub_map, k_cap, min_prior=min_prior,
    )
    if not guard["passed"]:
        raise SystemExit(
            f"GUARDRAIL FAILED: raw_reorders={guard['frac_raw_reorders']:.3f} "
            f"resid_reorders={guard['frac_resid_reorders']:.3f} "
            f"(need both >= {GUARDRAIL_MIN_REORDER}). "
            "Fix residualization before plotting."
        )
    print(
        f"  Guardrail OK: raw_reorders={guard['frac_raw_reorders']:.3f} "
        f"resid_reorders={guard['frac_resid_reorders']:.3f} (n={guard['n_held_out']})"
    )

    if eval_recs is None:
        all_recs = collect_records(
            threads, vec, topic_map, sub_map, k_cap, min_prior=min_prior,
        )
        eval_recs = [r for r in all_recs if r["k"] >= min_prior]

    sweep: dict = {}
    for mp in (2, 3, 5):
        filt = [r for r in eval_recs if user_pair_counts.get(r["asker_id"], 0) >= mp]
        sweep[str(mp)] = {}
        for facet, is_div in (("divergent", True), ("agreement", False)):
            sweep[str(mp)][facet] = {}
            facet_recs = [r for r in filt if r["divergent"] == is_div]
            for stratum in ("cross_sub", "same_sub"):
                want_cross = stratum == "cross_sub"
                sub = [r for r in facet_recs if r["cross_sub"] == want_cross]
                sweep[str(mp)][facet][stratum] = bootstrap_pool_stats(
                    sub, n_boot=n_boot, rng=rng,
                    fields=_fields_for_records(sub),
                )


def _fields_for_records(records: list[dict]) -> list[str]:
    fields = list(SCORE_FIELDS)
    if records and not any(r.get("llm_ranker") is not None for r in records):
        fields = [f for f in fields if f != "llm_ranker"]
    return fields

    payload = {
        "guardrail": guard,
        "min_prior": min_prior,
        "k_cap": k_cap,
        "min_pairs_sweep": sweep,
        "headline": sweep.get("3", {}).get("divergent", {}).get("cross_sub", {}),
    }
    return payload, eval_recs


def _plot_accuracy_panel(
    ax,
    stats: dict,
    *,
    panel_title: str,
    note: str = "",
    llm_n: int | None = None,
) -> None:
    n = stats.get("n_threads", 0)
    ch = stats.get("chance_mean", np.nan)
    if np.isfinite(ch):
        ax.axhline(ch, color="#94a3b8", ls="--", lw=1.2, zorder=1,
                   label=f"Chance ({ch:.0%})")

    bars = [m for m in PLOT_BAR_FIELDS if m[0] in stats.get("mean", {})]
    x = np.arange(len(bars))
    for i, (field, label, col) in enumerate(bars):
        y = stats["mean"].get(field, np.nan)
        ax.bar(i, y, width=0.62, color=col, edgecolor="white", zorder=3, label=label)
        _err_y(ax, i, y, _interval(stats, field), col)
        if np.isfinite(y):
            ax.text(i, y + 0.02, f"{y:.0%}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([b[2].replace("TF-IDF + ", "") for b in bars], fontsize=8, rotation=12, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Top-1 accuracy")
    ax.grid(axis="y", alpha=0.35)
    ax.set_title(panel_title, fontsize=11, loc="left", fontweight="bold")
    n_label = f"n = {n}" if llm_n is None else f"n = {n} ({llm_n} with LLM ranker)"
    ax.text(0.5, 0.97, n_label, transform=ax.transAxes, ha="center", va="top",
            fontsize=10, fontweight="bold")
    if note:
        ax.text(0.5, 0.02, note, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8, color="#64748b", style="italic")


def plot_pooled_panel_a(
    sweep: dict,
    corpus: dict,
    path: Path,
    *,
    min_pairs_show: int = 3,
    llm_n: int | None = None,
) -> None:
    """Side by side: divergent vs agreement (cross-sub, min_pairs slice)."""
    block = sweep[str(min_pairs_show)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)

    _plot_accuracy_panel(
        axes[0], block["divergent"]["cross_sub"],
        panel_title="Divergent (community top ≠ thanked)",
        note="Prior thanks in other subreddits",
        llm_n=llm_n,
    )
    _plot_accuracy_panel(
        axes[1], block["agreement"]["cross_sub"],
        panel_title="Agreement (community top = thanked)",
        note="Baseline often high — sanity check",
        llm_n=llm_n,
    )

    div = block["divergent"]["cross_sub"]
    llm_acc = div["mean"].get("llm_ranker")
    llm_str = f" · LLM {llm_acc:.0%}" if llm_acc is not None and np.isfinite(llm_acc) else ""
    title = (
        f"Pooled LOO — {corpus['n_subs']} subreddits · min_pairs ≥ {min_pairs_show} · cross-sub"
    )
    subtitle = (
        f"Pick thanked comment among judge-valid answers · Wilson bars · "
        f"eval n={corpus['n_eval_held_outs_k_ge_1']}{llm_str}"
    )
    _add_figure_caption(fig, title, subtitle)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.12, 1, 0.88])
    _save_fig(fig, path)


# ── Plotting ──────────────────────────────────────────────────────────────


def _err_y(
    ax, x: float, y: float, interval: tuple[float, float], color: str,
) -> None:
    if not interval or not all(np.isfinite(interval)):
        return
    lo, hi = interval
    ax.errorbar(
        x, y, yerr=[[max(0, y - lo)], [max(0, hi - y)]],
        fmt="none", ecolor=color, capsize=4, lw=1.2, zorder=4,
    )


def _interval(stats: dict, field: str, *, prefer_wilson: bool = True) -> tuple[float, float]:
    if prefer_wilson and field in stats.get("wilson", {}):
        return stats["wilson"][field]
    return stats.get("ci", {}).get(field, (float("nan"), float("nan")))


def _plot_pooled_accuracy(ax, pool_k0: dict, pool_k1: dict, *, title: str, facet_note: str) -> None:
    """Discrete points: cold-start (k=0) vs pooled k≥1."""
    clusters = [("k = 0\ncold start", pool_k0), (f"k ≥ 1\npooled", pool_k1)]
    offsets = np.linspace(-0.28, 0.28, len(METHODS))
    x_centers = [0, 1.35]

    for xi, (clabel, pool) in enumerate(clusters):
        n = pool.get("n_threads", 0)
        ax.text(
            x_centers[xi], 1.06, f"{clabel}\nn = {n} threads",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            transform=ax.get_xaxis_transform(),
        )
        chance = pool.get("chance_mean", np.nan)
        if np.isfinite(chance):
            ax.hlines(chance, x_centers[xi] - 0.42, x_centers[xi] + 0.42,
                      colors="#94a3b8", linestyles="--", lw=1.2, zorder=1)
        for j, (field, color, label) in enumerate(METHODS):
            if field == "chance":
                continue
            x = x_centers[xi] + offsets[j]
            y = pool["mean"].get(field, np.nan)
            ax.scatter(x, y, s=70, color=color, edgecolors="white", linewidths=0.6,
                       zorder=5, label=label if xi == 1 else None)
            _err_y(ax, x, y, _interval(pool, field), color)

    ax.set_xticks(x_centers)
    ax.set_xticklabels([c[0].split("\n")[0] for c in clusters], fontsize=9)
    ax.set_xlim(-0.55, 1.9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title(title, fontsize=10, loc="left")
    if facet_note:
        ax.text(0.02, 0.04, facet_note, transform=ax.transAxes, fontsize=7.5,
                va="bottom", ha="left", color="#64748b", style="italic")
    ax.grid(axis="y", alpha=0.35)
    ax.axhline(0, color="#e2e8f0", lw=0.5)


def _plot_pooled_delta(ax, pool: dict, *, title: str) -> None:
    """Lift vs baseline; zero line = no gain over non-personalized."""
    x0 = 0.0
    colors = {"per_asker_raw": "#ea580c", "per_asker_resid": "#2563eb"}
    labels = {"per_asker_raw": "Raw − baseline", "per_asker_resid": "Residualized − baseline"}
    for i, field in enumerate(("per_asker_raw", "per_asker_resid")):
        x = x0 + i * 0.55
        d = pool["delta_mean"].get(field, np.nan)
        ax.scatter(x, d, s=80, color=colors[field], zorder=5, label=labels[field])
        _err_y(ax, x, d, pool["delta_ci"].get(field), colors[field])  # bootstrap on delta
    n = pool.get("n_threads", 0)
    ax.text(0.28, 0.92, f"n = {n}", transform=ax.transAxes, ha="center", fontsize=9,
            fontweight="bold")
    ax.axhline(0, color="#334155", lw=1.2, zorder=2)
    ax.set_xlim(-0.35, 0.85)
    ax.set_ylim(-0.55, 0.55)
    ax.set_xticks([0, 0.55])
    ax.set_xticklabels(["raw", "resid."], fontsize=8)
    ax.set_ylabel("Δ accuracy vs baseline")
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(axis="y", alpha=0.35)


def _plot_by_k_appendix(ax, by_k_facet: dict, title: str) -> None:
    ks = [k for k in by_k_facet["k_values"]
          if by_k_facet["per_k"].get(k, {}).get("n_threads", 0) > 0]
    if not ks:
        ax.text(0.5, 0.5, "No k buckets", ha="center", va="center", transform=ax.transAxes)
        return

    x_pos = np.arange(len(ks))
    max_n = max(by_k_facet["per_k"][k]["n_threads"] for k in ks)
    bar_h = [by_k_facet["per_k"][k]["n_threads"] / max(max_n, 1) * 0.35 for k in ks]
    ax.bar(x_pos, bar_h, width=0.72, color="#cbd5e1", edgecolor="white", zorder=1)
    for i, k in enumerate(ks):
        n = by_k_facet["per_k"][k]["n_threads"]
        ax.text(i, bar_h[i] + 0.02, f"k={k}\nn={n}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", zorder=6)

    offsets = np.linspace(-0.22, 0.22, len(SCORE_FIELDS))
    method_style = {f: (c, l) for f, c, l in METHODS}
    for j, field in enumerate(SCORE_FIELDS):
        color, label = method_style[field]
        xs, ys, yerr_lo, yerr_hi = [], [], [], []
        for i, k in enumerate(ks):
            st = by_k_facet["per_k"][k]
            m = st["mean"].get(field, np.nan)
            if not np.isfinite(m):
                continue
            xs.append(x_pos[i] + offsets[j])
            ys.append(m)
            iv = _interval(st, field)
            if iv and all(np.isfinite(iv)):
                yerr_lo.append(max(0.0, m - iv[0]))
                yerr_hi.append(max(0.0, iv[1] - m))
            else:
                yerr_lo.append(0)
                yerr_hi.append(0)
        ax.errorbar(xs, ys, yerr=[yerr_lo, yerr_hi], fmt="o", color=color, ms=7,
                    capsize=3, lw=1.1, label=label, zorder=4)

    for i, k in enumerate(ks):
        ch = by_k_facet["per_k"][k].get("chance_mean", np.nan)
        if np.isfinite(ch):
            ax.hlines(ch, x_pos[i] - 0.35, x_pos[i] + 0.35, colors="#94a3b8",
                      linestyles="--", lw=1, zorder=2)

    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("k = prior thanked threads (separate populations per k)")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=10, loc="left")
    ax.grid(axis="y", alpha=0.35)


def _sub_label(config: dict) -> str:
    sub = config.get("sub", "?")
    if str(sub).lower() in ("all", "*", "corpus"):
        n = len(config.get("subs_used") or [])
        return f"{n} subreddits" if n else "all available subs"
    return f"r/{sub}"


def _experiment_caption(config: dict, pa: dict) -> tuple[str, str]:
    scope = _sub_label(config)
    n = pa.get("n_total_eval", "?")
    min_prior = pa.get("min_prior", config.get("min_prior", 1))
    k_cap = pa.get("k_cap", config.get("k_cap", 8))
    title = (
        f"Temporal LOO personalization benchmark — {scope} "
        f"(valid QA threads, TF-IDF rankers, no training)"
    )
    subtitle = (
        f"Predict thanked comment (top-1) · eval n={n} held-outs with k≥{min_prior} "
        f"· k capped at {k_cap} · primary contrast: pooled k≥{min_prior} vs k=0 cold start "
        f"· 95% CIs = asker bootstrap"
    )
    return title, subtitle


def _by_k_caption(
    config: dict,
    pa: dict,
    *,
    facet: str,
    n_facet: int,
) -> tuple[str, str]:
    scope = _sub_label(config)
    min_prior = pa.get("min_prior", 1)
    title = (
        f"LOO top-1 accuracy by prior history k — {scope} · {facet}"
    )
    subtitle = (
        f"Separate population per k (not connected) · {n_facet} eval threads in facet "
        f"· min_prior={min_prior} · dashed = chance (1/n candidates)"
    )
    return title, subtitle


def _add_figure_caption(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)
    fig.text(0.5, 0.935, subtitle, ha="center", va="top", fontsize=9, color="#475569")


def plot_panel_a_summary(pa: dict, path: Path, config: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.8))
    div = pa["divergent"]
    agr = pa["agreement"]

    _plot_pooled_accuracy(
        axes[0, 0], div["pooled_k_0"], div["pooled_k_ge_1"],
        title="Divergent — accuracy (community top ≠ thanked)",
        facet_note="Personalization can only help here.",
    )
    _plot_pooled_delta(
        axes[0, 1], div["pooled_k_ge_1"],
        title="Divergent — lift vs baseline (k ≥ 1 pooled)",
    )
    _plot_pooled_accuracy(
        axes[1, 0], agr["pooled_k_0"], agr["pooled_k_ge_1"],
        title="Agreement — accuracy (community top = thanked)",
        facet_note="Baseline ceiling ≈ 1.0 by construction; not a personalization test.",
    )
    _plot_pooled_delta(
        axes[1, 1], agr["pooled_k_ge_1"],
        title="Agreement — lift vs baseline (k ≥ 1 pooled)",
    )

    _add_figure_caption(fig, *_experiment_caption(config, pa))
    fig.tight_layout(rect=[0, 0.26, 1, 0.88])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    chance_handle = Line2D([0], [0], color="#94a3b8", ls="--", lw=1.2,
                           label="Chance (1/n candidates)")
    handles = [chance_handle] + handles
    labels = ["Chance (1/n candidates)"] + labels
    fig.legend(
        handles, labels, loc="lower center", ncol=4, fontsize=8,
        frameon=False, bbox_to_anchor=(0.5, 0.02),
    )
    _save_fig(fig, path)


def plot_panel_a_by_k(
    facet: dict,
    facet_label: str,
    path: Path,
    *,
    config: dict,
    pa: dict,
) -> None:
    by_k = facet["by_k"]
    n_facet = facet["pooled_k_ge_1"]["n_threads"]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    _plot_by_k_appendix(ax, by_k, facet_label)
    _add_figure_caption(fig, *_by_k_caption(config, pa, facet=facet_label, n_facet=n_facet))
    fig.tight_layout(rect=[0, 0.30, 1, 0.86])
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=2, fontsize=8,
        frameon=False, bbox_to_anchor=(0.5, 0.02),
    )
    _save_fig(fig, path)


def _plot_panel_b(ax, pb: dict) -> None:
    labels = ["Raw", "− topic", "− subreddit", "− flair", "Residual"]
    levels = pb["levels"]
    decs = pb["decrements"]
    x = np.arange(5)

    ax.axhline(0, color="#94a3b8", lw=0.8)
    ax.set_ylim(0, max(0.05, max(levels) * 1.4 + 0.02))

    # solid anchors
    ax.bar(0, levels[0], color="#2563eb", width=0.55, zorder=3)
    ax.bar(4, levels[4], color="#1e40af", width=0.55, zorder=3)
    def _errbar(xi, y, ci):
        if not ci:
            return
        lo, hi = max(0.0, ci[0]), ci[1]
        ax.errorbar(xi, y, yerr=[[max(0, y - lo)], [max(0, hi - y)]],
                    fmt="none", ecolor="black", capsize=4, zorder=4)

    _errbar(0, levels[0], pb.get("ci_raw"))
    _errbar(4, levels[4], pb.get("ci_residual"))

    # floating decrements between anchor levels
    for i, d in enumerate(decs, start=1):
        bottom = levels[i]
        ax.bar(i, d, bottom=bottom, width=0.45, color="#93c5fd", edgecolor="#3b82f6",
               linewidth=0.8, zorder=2)
        ax.annotate(f"−{d:.3f}", (i, bottom + d / 2), ha="center", va="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("ICC(1) between-asker / total")
    ax.set_title("B — ICC waterfall (thanked answers)", fontsize=10)


def _plot_panel_c(ax, pc: dict) -> None:
    null = np.array(pc["null_residual"])
    ax.hist(null, bins=40, color="#cbd5e1", edgecolor="white", density=True, label="Null (shuffled askers)")
    p95 = pc["null_p95_residual"]
    ax.axvline(pc["observed_raw"], color="#f59e0b", lw=2, label=f"Observed raw ({pc['p_raw']:.3f})")
    ax.axvline(pc["observed_residual"], color="#2563eb", lw=2,
               label=f"Observed residual ({pc['p_residual']:.3f})")
    ax.axvline(p95, color="#64748b", ls="--", lw=1.2, label="Null 95th pct")
    ax.set_xlabel("Mean within-asker pairwise cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title("C — Permutation null (residual embeddings)", fontsize=10)


def _save_fig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_panel_b_single(pb: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _plot_panel_b(ax, pb)
    fig.suptitle(
        "Panel B — ICC waterfall on thanked-answer embeddings\n"
        f"between-asker share of variance · {pb['n_threads']} posts, {pb['n_askers']} askers",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save_fig(fig, path)


def plot_panel_c_single(pc: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _plot_panel_c(ax, pc)
    fig.suptitle(
        "Panel C — Permutation null (asker labels shuffled)\n"
        f"{pc['n_perm']} permutations · {pc['n_askers_multi']} askers with ≥2 threads",
        fontsize=10,
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0.1, 1, 0.92])
    _save_fig(fig, path)


def plot_results(
    pa: dict, pb: dict, pc: dict, out_stem: Path, *, config: dict | None = None,
) -> list[Path]:
    """Write one PNG per panel (and per Panel A facet). Returns paths written."""
    stem = out_stem.with_suffix("")  # drop .png if passed
    out_dir = stem.parent if str(stem.parent) != "." else OUT_DIR
    base = stem.name

    paths = [
        out_dir / f"{base}_panel_a_summary.png",
        out_dir / f"{base}_panel_a_divergent_by_k.png",
        out_dir / f"{base}_panel_a_agreement_by_k.png",
        out_dir / f"{base}_panel_b_icc.png",
        out_dir / f"{base}_panel_c_permutation.png",
    ]
    cfg = config or {}
    plot_panel_a_summary(pa, paths[0], cfg)
    plot_panel_a_by_k(
        pa["divergent"],
        "Divergent (community top ≠ thanked)",
        paths[1],
        config=cfg,
        pa=pa,
    )
    plot_panel_a_by_k(
        pa["agreement"],
        "Agreement (community top = thanked)",
        paths[2],
        config=cfg,
        pa=pa,
    )
    if pb:
        plot_panel_b_single(pb, paths[3])
    else:
        paths = paths[:3]
    if pc:
        plot_panel_c_single(pc, paths[4])
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sub", default="all",
        help="sub name, 'all' (21 subs), or 'fast' (14 high-judge subs, ~10 min)",
    )
    ap.add_argument("--max-threads", type=int, default=None,
                    help="cap held-outs (default: no cap, all eligible)")
    ap.add_argument("--pooled-json", type=Path,
                    default=REPO / "june_2" / "outputs" / "benchmark_pooled.json")
    ap.add_argument("--skip-bc", action="store_true", default=True,
                    help="skip Panels B/C (default: on)")
    ap.add_argument("--extra-plots", action="store_true",
                    help="also write summary/by-k/B/C PNGs")
    ap.add_argument("--plot-only", action="store_true",
                    help="regenerate plot from --pooled-json (no reload)")
    ap.add_argument(
        "--out-plot",
        type=Path,
        default=OUT_DIR / "personalization_benchmark.png",
        help="single figure output path",
    )
    ap.add_argument("--llm-sample", type=int, default=120,
                    help="max held-out threads to score with LLM ranker (0=skip)")
    ap.add_argument("--llm-cache", type=Path,
                    default=REPO / "june_2" / "outputs" / "llm_ranker_cache.jsonl")
    ap.add_argument("--min-prior", type=int, default=1,
                    help="min prior thanked threads (k); 1 excludes cold start")
    ap.add_argument("--k-cap", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=400)
    ap.add_argument("--out-json", type=Path, default=REPO / "june_2" / "outputs" / "benchmark_results.json")
    ap.add_argument(
        "--out-stem",
        type=Path,
        default=OUT_DIR / "june_2_personalization_benchmark",
        help="output stem (writes *_panel_a_summary.png, *_by_k.png, B, C)",
    )
    args = ap.parse_args()

    if args.plot_only:
        pooled = json.loads(args.pooled_json.read_text())
        corpus = pooled["corpus_stats"]
        plot_pooled_panel_a(
            pooled["min_pairs_sweep"], corpus, args.out_plot,
            llm_n=pooled.get("llm_n_scored"),
        )
        print(f"wrote {args.out_plot}")
        return

    subs_used: list[str] = []
    user_pair_counts: dict[str, int] = {}
    if args.sub.lower() == "fast":
        subs_used = list(FAST_BENCHMARK_SUBS)
    elif args.sub.lower() in ("all", "*", "corpus"):
        subs_used = discover_benchmark_subs()

    if subs_used or args.sub.lower() in ("all", "*", "corpus", "fast"):
        if not subs_used:
            subs_used = discover_benchmark_subs()
        cap = args.max_threads if args.max_threads else "all"
        print(f"Loading pooled corpus ({len(subs_used)} subs, min_prior k>={args.min_prior}, "
              f"max held-outs={cap})…")
        threads, subs_used, user_pair_counts = load_corpus(
            subs_used, max_threads=args.max_threads, min_prior=args.min_prior,
        )
    else:
        print(f"Loading valid-QA (sub={args.sub}, min_prior k>={args.min_prior}, "
              f"max held-outs={args.max_threads})…")
        threads = load_subset(
            args.sub, max_threads=args.max_threads, min_prior=args.min_prior,
        )
        subs_used = [args.sub]
    # Held-outs = threads that are not first for their asker (when min_prior>=1)
    by_a: dict[str, list] = defaultdict(list)
    for t in threads:
        by_a[t.asker_id].append(t)
    for s in by_a.values():
        s.sort(key=lambda x: x.timestamp)
    n_held = sum(
        1 for seq in by_a.values() for i, _ in enumerate(seq) if i >= args.min_prior
    )
    n_div = sum(
        1 for seq in by_a.values() for i, t in enumerate(seq)
        if i >= args.min_prior and t.is_divergent
    )
    print(f"  {len(threads)} posts in context ({n_held} held-outs with k>={args.min_prior}), "
          f"{n_div} divergent held-outs, {len(by_a)} askers")

    vec = fit_vectorizer(threads)
    topic_map = topic_clusters(vec, threads)
    sub_map = subreddit_index(threads)

    print("Collecting TF-IDF LOO records…")
    eval_recs = collect_records(
        threads, vec, topic_map, sub_map, args.k_cap, min_prior=args.min_prior,
    )
    llm_n = 0
    if args.llm_sample > 0:
        from llm_ranker import enrich_records_with_llm

        print(f"LLM ranker (max {args.llm_sample} threads, cached)…")
        llm_n = enrich_records_with_llm(
            threads, eval_recs, max_calls=args.llm_sample, cache_path=args.llm_cache,
        )
        print(f"  {llm_n} new LLM calls · {sum(1 for r in eval_recs if r.get('llm_ranker') is not None)} scored")

    print("Guardrail + pooled metrics (min_pairs 2/3/5)…")
    pooled, _ = build_pooled_metrics(
        threads, vec, topic_map, sub_map, user_pair_counts,
        k_cap=args.k_cap, min_prior=args.min_prior, n_boot=args.n_boot, seed=0,
        eval_recs=eval_recs,
    )
    corpus = corpus_stats(
        threads, eval_recs, subs_used=subs_used, user_pair_counts=user_pair_counts,
    )
    pooled["corpus_stats"] = corpus
    pooled["llm_n_scored"] = sum(1 for r in eval_recs if r.get("llm_ranker") is not None)

    def _jd(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(type(o))

    args.pooled_json.parent.mkdir(parents=True, exist_ok=True)
    args.pooled_json.write_text(json.dumps(pooled, indent=2, default=_jd) + "\n")
    print(f"wrote {args.pooled_json}")
    print(
        f"  eval n={corpus['n_eval_held_outs_k_ge_1']} "
        f"divergent={corpus['n_divergent_eval']} cross_sub={corpus['n_cross_sub']}"
    )

    plot_pooled_panel_a(
        pooled["min_pairs_sweep"], corpus, args.out_plot, llm_n=pooled.get("llm_n_scored"),
    )
    print(f"wrote {args.out_plot}")

    if args.extra_plots:
        pa = panel_a(
            threads, vec, topic_map, sub_map,
            k_cap=args.k_cap, min_prior=args.min_prior, n_boot=args.n_boot,
        )
        pb, pc = {}, {}
        if not args.skip_bc:
            print("Panel B…")
            pb = panel_b(threads, vec, topic_map, sub_map, n_boot=min(200, args.n_boot))
            print("Panel C…")
            pc = panel_c(threads, vec, topic_map, sub_map, n_perm=min(500, args.n_perm))
        config = {
            **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "subs_used": subs_used,
            "n_subs": len(subs_used),
        }
        results = {"config": config, "panel_a": pa, "panel_b": pb, "panel_c": pc, "pooled": pooled}
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(results, indent=2, default=_jd) + "\n")
        pngs = plot_results(pa, pb, pc, args.out_stem, config=config)
        print(f"wrote {args.out_json}")
        for p in pngs:
            print(f"wrote {p}")


if __name__ == "__main__":
    main()
