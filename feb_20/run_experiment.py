"""
Per-sample preference steering experiment.

No fake user profiles. Each dataset sample has its own preference.
We maintain ONE steering vector and update it after every batch
using EMA. The vector learns the general direction of
"preference-aligned" vs "preference-misaligned" responses.

Flow:
  For each batch of samples:
    1. Generate responses (with current steering if available)
    2. Check alignment against that sample's own aligned_op
    3. Collect hidden states: aligned -> positive, misaligned -> negative
    4. Update steering vector (diff-of-means + EMA)
    5. Log alignment rate
"""

import json, os, time, random
from config import Config
from models import load_model, generate_response
from tasks import load_prefeval, check_alignment
from steering import SteeringVector


def run():
    config = Config()

    print("=" * 60)
    print("PER-SAMPLE PREFERENCE STEERING")
    print("=" * 60)

    models = load_model(config)
    hidden_dim = models["model"].config.hidden_size
    print(f"Hidden dim: {hidden_dim}")

    print("\nLoading PrefEval dataset...")
    ds = load_prefeval()
    samples = list(ds)
    random.seed(42)
    random.shuffle(samples)
    print(f"Total samples: {len(samples)}")

    sv = SteeringVector("preference", hidden_dim, config)

    all_logs = []
    total_aligned = 0
    total_seen = 0
    batch_idx = 0

    for start in range(0, len(samples), config.batch_size):
        batch = samples[start : start + config.batch_size]
        batch_idx += 1

        pos_h, neg_h = [], []
        batch_aligned = 0
        batch_total = 0

        t0 = time.time()

        for sample in batch:
            for _ in range(config.num_responses):
                hook = None
                if sv.num_updates > 0:
                    hook = sv.make_hook()

                response, h = generate_response(
                    sample, models, config, steering_hook=hook
                )

                aligned = check_alignment(
                    response, sample["aligned_op"], sample["options"]
                )

                h_flat = h.flatten()
                if aligned:
                    pos_h.append(h_flat)
                    batch_aligned += 1
                else:
                    neg_h.append(h_flat)
                batch_total += 1

        elapsed = time.time() - t0
        total_aligned += batch_aligned
        total_seen += batch_total
        batch_pct = 100 * batch_aligned / max(batch_total, 1)
        running_pct = 100 * total_aligned / max(total_seen, 1)

        print(f"\nBatch {batch_idx} (samples {start+1}-{start+len(batch)}):")
        print(f"  Aligned: {batch_aligned}/{batch_total} ({batch_pct:.0f}%)")
        print(f"  Running total: {total_aligned}/{total_seen} ({running_pct:.0f}%)")
        print(f"  Took {elapsed:.1f}s")

        # Show a sample
        s = batch[0]
        print(f"  Sample preference: {s['preference'][:100]}")
        print(f"  Sample aligned_op: {s['aligned_op'][:100]}")

        # Update steering vector
        sv.update(pos_h, neg_h)

        all_logs.append({
            "batch": batch_idx,
            "start": start,
            "n_aligned": batch_aligned,
            "n_total": batch_total,
            "batch_pct": batch_pct,
            "running_pct": running_pct,
            "sv_updates": sv.num_updates,
        })

    # Save
    os.makedirs("results", exist_ok=True)
    sv.save("results/steering_vector.pt")
    print(f"\nSaved steering vector ({sv.num_updates} updates)")

    with open("results/log.json", "w") as f:
        json.dump(all_logs, f, indent=2)
    print("Saved log")

    # Summary
    print(f"\n{'='*60}")
    print("ALIGNMENT OVER TIME")
    print(f"{'='*60}")
    print(f"{'Batch':<8}{'Samples':<16}{'Batch %':<12}{'Running %':<12}")
    print("-" * 48)
    for entry in all_logs:
        print(f"{entry['batch']:<8}"
              f"{entry['start']+1}-{entry['start']+entry['n_total']//config.num_responses:<12}"
              f"{entry['batch_pct']:<12.0f}"
              f"{entry['running_pct']:<12.0f}")

    print(f"\nFinal: {total_aligned}/{total_seen} "
          f"({100*total_aligned/max(total_seen,1):.1f}% aligned)")


if __name__ == "__main__":
    run()
