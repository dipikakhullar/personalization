"""Dump clean JSON examples from the corners of the Exp A scatter
(x = generated-answer cosine to PREFERRED, y = cosine to TOP-rated community answer).

Corners are defined on the average cosine across the 5 models:
  top-right = high sim to BOTH  (avg_pref + avg_top largest)
  top-left  = high sim to TOP, low to PREFERRED  (avg_top - avg_pref largest)

Each emitted record aggregates all 5 models for that sample:
  sample_id, subreddit, op_query, preferred_answer, top_community_answer,
  avg_cos_to_preferred, avg_cos_to_top,
  models: { generated: {model: text}, mcq: {model: "preferred"|"top"} }

Usage:
  python3 build_corner_examples.py                       # with-history results
  python3 build_corner_examples.py --dir no_user_context --out-tag no_history
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_SAMPLES = HERE / "data" / "samples_qa1000_clean.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results_clean/with_history", help="results dir")
    ap.add_argument("--samples", default=str(DEFAULT_SAMPLES), help="samples jsonl")
    ap.add_argument("--out-dir", default="corner_examples")
    ap.add_argument("--out-tag", default="clean_with_history")
    ap.add_argument("--n", type=int, default=12, help="examples per corner")
    args = ap.parse_args()

    samples = {json.loads(l)["sample_id"]: json.loads(l) for l in open(args.samples)}

    # per-model results keyed by sample_id
    models = {}
    for fp in sorted(glob.glob(str(HERE / args.dir / "results_*.jsonl"))):
        if "_old_" in os.path.basename(fp):
            continue
        rows = [json.loads(l) for l in open(fp)]
        if not rows:
            continue
        model = rows[0].get("model", os.path.basename(fp))
        models[model] = {r["sample_id"]: r for r in rows}

    model_names = sorted(models)
    # samples present in ALL models
    common = sorted(set.intersection(*[set(m) for m in models.values()]))
    print(f"{len(model_names)} models, {len(common)} samples common to all")

    agg = []
    for sid in common:
        cp = np.mean([models[m][sid]["cos_preferred"] for m in model_names])
        ct = np.mean([models[m][sid]["cos_top"] for m in model_names])
        agg.append((sid, cp, ct))

    top_right = sorted(agg, key=lambda x: -(x[1] + x[2]))[: args.n]
    top_left = sorted(agg, key=lambda x: -(x[2] - x[1]))[: args.n]

    def record(sid, cp, ct):
        s = samples[sid]
        return {
            "sample_id": sid,
            "subreddit": s["subreddit"],
            "avg_cos_to_preferred": round(float(cp), 4),
            "avg_cos_to_top": round(float(ct), 4),
            "user_history": s["history"],
            "op_query": s["query"],
            "preferred_answer": s["preferred_answer"],
            "top_community_answer": s["top_comment"],
            "judge_models": {
                "generated": {
                    m: models[m][sid]["generated_answer"] for m in model_names
                },
                "mcq": {
                    m: ("preferred" if models[m][sid]["choice_picked_preferred"]
                        else "top" if models[m][sid]["choice_picked_preferred"] is False
                        else None)
                    for m in model_names
                },
                "similarity": {
                    m: {
                        "cos_to_preferred": round(models[m][sid]["cos_preferred"], 4),
                        "cos_to_top": round(models[m][sid]["cos_top"], 4),
                        "judge_to_preferred": models[m][sid]["judge_sim_preferred"],
                        "judge_to_top": models[m][sid]["judge_sim_top"],
                    }
                    for m in model_names
                },
            },
        }

    out_dir = HERE / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, corner in [("top_right", top_right), ("top_left", top_left)]:
        recs = [record(sid, cp, ct) for sid, cp, ct in corner]
        out = out_dir / f"{name}_{args.out_tag}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False, indent=2)
        print(f"  wrote {len(recs)} examples -> {out}")
        print(f"    cos(pref/top) range: "
              f"{recs[0]['avg_cos_to_preferred']:.2f}/{recs[0]['avg_cos_to_top']:.2f} ... "
              f"{recs[-1]['avg_cos_to_preferred']:.2f}/{recs[-1]['avg_cos_to_top']:.2f}")


if __name__ == "__main__":
    main()
