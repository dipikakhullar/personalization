# June 2 — personalization benchmark

Three-panel readout on **valid QA** threads (`is_qa_pair` true, ≥2 judge-valid candidates). No model training — TF-IDF rankers, asker-grouped temporal LOO.

## Run

```bash
cd june_2
pip install -r requirements.txt
python benchmark.py --sub fast --skip-bc --min-prior 1 --k-cap 6
```

**Pooled by `user_id` across subs** (one global temporal history per asker, not per-sub loops).

| `--sub` | Scope |
|---|---|
| `fast` | 14 high-judge subs (~10 min load) |
| `all` | 21 subs with pairs + judge + RC dump |
| `AskStatistics` | single sub (legacy pilot) |

Outputs: `outputs/benchmark_pooled.json` and **one figure**: `plots/outputs/personalization_benchmark.png`. Use `--extra-plots` for legacy multi-PNG output. Regenerate the figure only: `python benchmark.py --plot-only`.

Guardrail must pass (`frac_*_reorders` ≥ 0.05) before plots. Default **`--min-prior 1`** skips k=0 in Panel A. **`--max-threads`** caps held-outs globally (default: no cap).

Outputs: `outputs/benchmark_results.json` and PNGs under `../plots/outputs/`:

- `june_2_personalization_benchmark_panel_a_summary.png` — **primary** (pooled k≥1 vs k=0, deltas)
- `june_2_personalization_benchmark_panel_a_divergent_by_k.png` — appendix (discrete k, n labeled)
- `june_2_personalization_benchmark_panel_a_agreement_by_k.png`
- `june_2_personalization_benchmark_panel_b_icc.png`
- `june_2_personalization_benchmark_panel_c_permutation.png`

## Panel A — LOO top-1 accuracy

- **Primary figure:** two facets (divergent / agreement) × accuracy + **Δ vs baseline** (zero = no gain).
- **Pools:** **k≥1** (any history, default eval set) vs **k=0** cold start (first thanked thread per asker in sample).
- **Methods:** chance (dashed), baseline, per-asker raw, per-asker residualized — discrete points + asker-bootstrap CIs (no lines across k).
- **Appendix:** per-k breakdown with **n threads** prominent; populations at each k are separate (not connected).
- Agreement facet notes baseline ceiling ≈1.0 by construction; signal is expected in divergent only.

## Panel B — ICC waterfall

- Ordered steps: raw → −topic → −subreddit → −flair → residual.
- Solid bars at raw & residual; middle steps show floating decrements.
- Error bars on raw & residual (asker bootstrap).

## Panel C — Permutation histogram

- Histogram = null within-asker similarity after **shuffling asker labels**.
- Vertical lines: observed raw, observed residual, null 95th percentile; p-values annotated.

## Limits

- Candidates = comments with `is_qa_pair` in judge sidecar only (not all RC comments).
- Pilot sub / thread cap; high-k bars rest on few askers.
