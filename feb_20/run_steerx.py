"""
SteerX experiment: compare our original mean-pooled steering
vs SteerX's causal-token-focused steering.

For each response, instead of mean-pooling all hidden states,
we:
  1. Compute per-token causal effects (factual vs counterfactual)
  2. Identify anchor tokens (preference-driven)
  3. Extract hidden states only at anchor positions
  4. Use those for the steering vector update
"""

import json, os, time, random
import torch
from config import Config
from models import load_model, generate_response
from tasks import load_prefeval, get_user_samples, sample_queries, check_alignment
from steering import SteeringVector
from causal_tokens import (
    compute_token_causal_effects, identify_anchor_tokens,
    get_anchor_hidden_states, smooth_anchor_tokens,
)


def run():
    config = Config()
    config.num_rounds = 4
    config.queries_per_round = 8
    config.num_responses_per_query = 2

    print("=" * 60)
    print("STEERX EXPERIMENT: Causal Token Identification")
    print("=" * 60)

    models = load_model(config)
    model = models["model"]
    tok = models["tokenizer"]
    hidden_dim = model.config.hidden_size

    print("\nLoading PrefEval dataset...")
    ds = load_prefeval()
    user_samples = get_user_samples(ds, config)

    # Two steering vectors per user: original vs steerx
    vectors_original = {}
    vectors_steerx = {}
    for uid in config.user_profiles:
        vectors_original[uid] = SteeringVector(f"{uid}_orig", hidden_dim, config)
        vectors_steerx[uid] = SteeringVector(f"{uid}_steerx", hidden_dim, config)

    logs = {"rounds": []}
    causal_threshold = 0.5

    for round_idx in range(config.num_rounds):
        round_log = {"round": round_idx, "users": {}}
        print(f"\n{'='*60}")
        print(f"ROUND {round_idx + 1} / {config.num_rounds}")
        print(f"{'='*60}")

        for uid in config.user_profiles:
            profile = config.user_profiles[uid]
            print(f"\n--- User: {uid} ({profile['description']}) ---")

            queries = sample_queries(
                user_samples[uid], config.queries_per_round,
                seed=round_idx * 100 + hash(uid) % 100,
            )

            # Get some user history for causal comparison
            rng = random.Random(round_idx + hash(uid))
            history_pool = [s for s in user_samples[uid] if s not in queries]
            user_history = rng.sample(history_pool, min(5, len(history_pool)))

            orig_pos, orig_neg = [], []
            steerx_pos, steerx_neg = [], []
            n_aligned_orig = 0
            n_aligned_steerx = 0
            n_total = 0
            total_anchor_pct = 0

            t0 = time.time()

            for sample in queries:
                q = sample["implicit_query"]
                aligned_op = sample["aligned_op"]
                all_options = sample["options"]

                for _ in range(config.num_responses_per_query):
                    # Generate response (using steerx vector if available)
                    hook = None
                    if vectors_steerx[uid].num_updates > 0:
                        hook = vectors_steerx[uid].make_hook()
                    response, h_full = generate_response(
                        q, models, config, steering_hook=hook
                    )

                    alignment = check_alignment(response, aligned_op, all_options)
                    is_aligned = alignment["is_aligned"]
                    n_total += 1

                    # ── Original approach: mean-pooled h ──
                    h_orig = h_full.flatten()
                    if is_aligned:
                        orig_pos.append(h_orig)
                        n_aligned_orig += 1
                    else:
                        orig_neg.append(h_orig)

                    # ── SteerX approach: causal token identification ──
                    causal = compute_token_causal_effects(
                        model, tok, q, response,
                        user_history=user_history, config=config,
                    )
                    anchors = identify_anchor_tokens(causal, threshold=causal_threshold)
                    total_anchor_pct += anchors["pct_anchor"]

                    h_anchor = get_anchor_hidden_states(
                        model, tok, response, anchors["anchor_positions"],
                        layer=config.steer_layer, config=config,
                    )

                    if h_anchor is not None:
                        h_anchor = h_anchor.flatten()
                        if is_aligned:
                            steerx_pos.append(h_anchor)
                            n_aligned_steerx += 1
                        else:
                            steerx_neg.append(h_anchor)

            elapsed = time.time() - t0
            avg_anchor = total_anchor_pct / max(n_total, 1)

            print(f"  Aligned: {n_aligned_orig}/{n_total} "
                  f"({100*n_aligned_orig/max(n_total,1):.0f}%)")
            print(f"  Avg anchor tokens: {avg_anchor:.1f}% of response")
            print(f"  Took {elapsed:.1f}s")

            # Update both vectors
            vectors_original[uid].update(orig_pos, orig_neg)
            vectors_steerx[uid].update(steerx_pos, steerx_neg)

            # Show sample anchor tokens
            if causal["tokens"]:
                top_idx = sorted(range(len(causal["deltas"])),
                                 key=lambda i: causal["deltas"][i],
                                 reverse=True)[:10]
                top_tokens = [(causal["tokens"][i], f"{causal['deltas'][i]:.2f}")
                              for i in top_idx]
                print(f"  Top anchor tokens (last response): {top_tokens}")

            # Smooth anchor tokens into description
            if anchors["anchor_tokens"]:
                desc = smooth_anchor_tokens(
                    model, tok, anchors["anchor_tokens"],
                    sample["preference"], config,
                )
                print(f"  Smoothed preference: {desc[:150]}")

            round_log["users"][uid] = {
                "n_aligned": n_aligned_orig,
                "n_total": n_total,
                "pct": 100 * n_aligned_orig / max(n_total, 1),
                "avg_anchor_pct": avg_anchor,
                "orig_pos": len(orig_pos), "orig_neg": len(orig_neg),
                "steerx_pos": len(steerx_pos), "steerx_neg": len(steerx_neg),
            }

        logs["rounds"].append(round_log)

    # Save
    os.makedirs("results", exist_ok=True)
    for uid in config.user_profiles:
        vectors_original[uid].save(f"results/{uid}_orig_sv.pt")
        vectors_steerx[uid].save(f"results/{uid}_steerx_sv.pt")

    # Cosine similarity between original and steerx vectors
    print(f"\n{'='*60}")
    print("STEERING VECTOR COMPARISON")
    print(f"{'='*60}")
    for uid in config.user_profiles:
        u_orig = vectors_original[uid].u
        u_steerx = vectors_steerx[uid].u
        cos = (u_orig * u_steerx).sum() / (u_orig.norm() * u_steerx.norm() + 1e-8)
        print(f"  {uid}: cos_sim(original, steerx) = {cos.item():.4f}")
        print(f"    original  norm={u_orig.norm():.4f}, updates={vectors_original[uid].num_updates}")
        print(f"    steerx    norm={u_steerx.norm():.4f}, updates={vectors_steerx[uid].num_updates}")

    with open("results/steerx_log.json", "w") as f:
        json.dump(logs, f, indent=2, default=str)
    print("\nSaved results to results/steerx_log.json")

    # Summary
    print(f"\n{'='*60}")
    print("ROUND-BY-ROUND ALIGNMENT")
    print(f"{'='*60}")
    for rl in logs["rounds"]:
        print(f"\nRound {rl['round']+1}:")
        for uid, stats in rl["users"].items():
            print(f"  {uid}: {stats['n_aligned']}/{stats['n_total']} aligned "
                  f"({stats['pct']:.0f}%), avg {stats['avg_anchor_pct']:.1f}% anchor tokens")


if __name__ == "__main__":
    run()
