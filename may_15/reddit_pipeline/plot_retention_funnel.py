"""Plot corpus retention funnel for the Reddit personalization pipeline.

Writes:
  ../../plots/outputs/retention_funnel.png
  ../../plots/outputs/retention_funnel.json

Usage (from may_15/reddit_pipeline/):
  python plot_retention_funnel.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterSciNotation

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DATA = HERE.parent / "data"
STATS_DIR = DATA / "extracted_current" / "stats"
PAIRS_DIR = DATA / "extracted_current" / "pairs"
JUDGE_DIR = DATA / "llm_judge"
OUT_DIR = REPO / "plots" / "outputs"
OUT_PNG = OUT_DIR / "retention_funnel.png"
OUT_JSON = OUT_DIR / "retention_funnel.json"


def iter_jsonl(path: Path):
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_judge_map() -> dict[tuple[str, str], bool]:
    out: dict[tuple[str, str], bool] = {}
    for p in sorted(JUDGE_DIR.glob("sub-*.jsonl")):
        for rec in iter_jsonl(p):
            key = (rec.get("post_id"), rec.get("answer_comment_id"))
            b = (rec.get("is_qa_pair") or {}).get("question_answer_pair")
            if key[0] and key[1] and b is not None:
                out[key] = bool(b)
    return out


def aggregate_upstream() -> dict[str, int]:
    totals = {
        "keep_posts": 0,
        "thanks_unique_parents": 0,
        "pairs_emitted_stats": 0,
    }
    for f in STATS_DIR.glob("sub-*.json"):
        s = json.loads(f.read_text())
        totals["keep_posts"] += s.get("keep_posts", 0)
        totals["thanks_unique_parents"] += s.get("thanks_unique_parents", 0)
        totals["pairs_emitted_stats"] += s.get("pairs_emitted", 0)
    return totals


def count_pairs(jmap: dict) -> dict[str, int]:
    emitted = judged = valid_qa = contrastive = 0
    for pf in sorted(PAIRS_DIR.glob("sub-*.jsonl")):
        for rec in iter_jsonl(pf):
            emitted += 1
            md = rec.get("metadata") or {}
            key = (md.get("post_id"), md.get("answer_comment_id"))
            j = jmap.get(key)
            if j is None:
                continue
            judged += 1
            if not j:
                continue
            valid_qa += 1
            if not md.get("top_equals_preferred"):
                contrastive += 1
    return {
        "pairs_emitted": emitted,
        "judged": judged,
        "valid_qa": valid_qa,
        "contrastive": contrastive,
    }


def build_funnel() -> list[dict]:
    up = aggregate_upstream()
    jmap = load_judge_map()
    down = count_pairs(jmap)

    raw = [
        ("Question-shaped posts", up["keep_posts"]),
        ("Unique thanked comments (pre-pair)", up["thanks_unique_parents"]),
        ("Pairs emitted (thanks → preferred)", down["pairs_emitted"]),
        ("LLM judged", down["judged"]),
        ("Valid QA (`is_qa_pair` true)", down["valid_qa"]),
        (
            "Contrastive (valid QA ∧ top ≠ preferred)",
            down["contrastive"],
        ),
    ]

    pairs_idx = 2  # "Pairs emitted" — baseline for downstream % only
    base = raw[pairs_idx][1]
    funnel: list[dict] = []
    prev = None
    for i, (label, n) in enumerate(raw):
        row = {"step": label, "count": n, "stage": "upstream" if i < pairs_idx else "downstream"}
        if prev is not None:
            row["pct_of_previous"] = round(100 * n / prev, 2) if prev else None
        else:
            row["pct_of_previous"] = None
        row["pct_of_pairs_emitted"] = (
            round(100 * n / base, 2) if (base and i >= pairs_idx) else None
        )
        funnel.append(row)
        prev = n
    funnel[-1]["is_target"] = True
    return funnel


def _bar_annotation(idx: int, n: int, row: dict, prev_n: int | None, pairs_idx: int) -> str:
    parts = [f"{n:,}"]
    if idx > 0 and prev_n and row.get("pct_of_previous") is not None:
        parts.append(f"{row['pct_of_previous']:.1f}% of prev")
    if idx == pairs_idx:
        parts.append("pair baseline")
    elif idx > pairs_idx and row.get("pct_of_pairs_emitted") is not None:
        parts.append(f"{row['pct_of_pairs_emitted']:.1f}% of pairs")
    return "  ·  ".join(parts)


def plot(funnel: list[dict]) -> None:
    labels = [r["step"] for r in funnel]
    counts = [r["count"] for r in funnel]
    pairs_idx = next(
        i for i, r in enumerate(funnel) if "Pairs emitted" in r["step"]
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    y = range(len(labels))
    colors = ["#94a3b8"] * (len(labels) - 1) + ["#2563eb"]

    bars = ax.barh(y, counts, color=colors, height=0.62, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Row count (log scale)")
    ax.set_xscale("log")
    lo = min(c for c in counts if c > 0)
    ax.set_xlim(lo * 0.35, max(counts) * 18)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=14))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10)), numticks=100))
    ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=10, labelOnlyBase=False))
    ax.tick_params(axis="x", which="major", labelsize=8)
    ax.tick_params(axis="x", which="minor", length=3)
    ax.grid(axis="x", which="both", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.set_title(
        "Reddit personalization pipeline — retention by stage\n"
        "(target: contrastive = valid QA and community top ≠ thanked answer)",
        fontsize=11,
    )

    for idx, (bar, n) in enumerate(zip(bars, counts)):
        x = bar.get_width()
        prev_n = counts[idx - 1] if idx > 0 else None
        ax.text(
            x * 1.08,
            bar.get_y() + bar.get_height() / 2,
            _bar_annotation(idx, n, funnel[idx], prev_n, pairs_idx),
            va="center",
            fontsize=8,
            color="#1e293b",
        )

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    funnel = build_funnel()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"funnel": funnel}, indent=2) + "\n")
    plot(funnel)
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_PNG}")
    pairs_idx = next(
        i for i, r in enumerate(funnel) if "Pairs emitted" in r["step"]
    )
    for i, row in enumerate(funnel):
        parts = []
        if row["pct_of_previous"] is not None:
            parts.append(f"{row['pct_of_previous']:.1f}% of prev")
        if i == pairs_idx:
            parts.append("pair baseline")
        elif row.get("pct_of_pairs_emitted") is not None:
            parts.append(f"{row['pct_of_pairs_emitted']:.1f}% of pairs")
        mark = " ← target" if row.get("is_target") else ""
        suffix = f" ({', '.join(parts)})" if parts else ""
        print(f"  {row['step']}: {row['count']:,}{suffix}{mark}")


if __name__ == "__main__":
    main()
