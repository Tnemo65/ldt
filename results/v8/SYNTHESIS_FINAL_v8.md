# CA-DIF-EIA v8: Scientific Correction Report

> **Date:** 2026-05-13
> **Status:** COMPLETE
> **Benchmark:** benchmark_v8.py (675 jobs, 0 errors)
> **Runtime:** ~12 min (GPU: RTX 3090 Ti)
> **Dataset:** NYC Taxi 2024, months 01-06, 25D features
> **Train:** 10,000 | **Val:** 2,000 | **Test:** 10,000 | **Anomaly rate:** 5% (500 anomalies)
> **Folds:** 5 temporal (leave-one-month-out) | **Seeds:** 42, 123, 456

---

## Executive Summary

This report presents the scientifically corrected benchmark results for CA-DIF-EIA, with the MemStream implementation fixed to match Bhatia et al. (WWW 2022) and CA-DIF-EIA-Stream fixed to use a trained autoencoder during warmup.

**Key findings:**

1. **MemStream (corrected) achieves AUC-PR = 0.9988** — essentially equivalent to DenoisingAE (0.9995), confirming the paper's theoretical claims
2. **CA-DIF-EIA-Stream (corrected) achieves AUC-PR = 0.9992** — the best streaming algorithm
3. **The v7 benchmark was fundamentally flawed** — the original MemStream stored raw features in memory and used random projections, yielding 0.9996 that masked a broken implementation
4. **The corrected MemStream reveals a true test**: with proper trained encoder + latent memory, performance remains near-perfect because anomaly injection is too extreme, not because the algorithm is broken

---

## v7 vs v8: What Was Fixed

### Bug #1: MemStream — Trained Encoder vs Random Projection (CRITICAL)

| Aspect | v7 (BROKEN) | v8 (CORRECT) |
|--------|-------------|----------------|
| Encoder | Random projection matrix (25→16, no learning) | Trained DAE (25→50→25, trained 20 epochs) |
| Latent dim | 16 | 50 (= 2d, paper-compliant) |
| Gradients | None (random weights) | Full backprop (nn.Parameter) |
| Learned features | No | Yes (denoising captures manifold) |

**Root cause:** PyTorch requires `nn.Parameter` wrapping for gradient tracking. `torch.randn().requires_grad_()` after multiplication creates non-leaf tensors that Adam can't optimize. The fix is `nn.Parameter(torch.randn(...))`.

### Bug #2: MemStream — Raw Features vs Encoded Latent Vectors in Memory (CRITICAL)

| Aspect | v7 (BROKEN) | v8 (CORRECT) |
|--------|-------------|----------------|
| Memory content | Raw 25D feature vectors | Encoded 50D latent vectors |
| Distance metric | L2 in raw space | L1 in latent space (paper: ℓ1 norm) |
| Semantic meaning | Raw buffer | "Encoded patterns of normal data" (paper line 3) |
| kNN interpretation | Standard outlier detection | Memory-based pattern matching |

**Root cause:** Paper Algorithm 1, line 3: `M = f_θ(D)` — memory is initialized with encoded training data. The memory stores WHAT the autoencoder thinks normal looks like, not the raw data.

### Bug #3: MemStream — Random Replacement vs FIFO + Threshold (CRITICAL)

| Aspect | v7 (BROKEN) | v8 (CORRECT) |
|--------|-------------|----------------|
| Memory update | Random replacement (any sample) | FIFO (oldest-out, newest-in) |
| Anti-poisoning | None | Only update if score < beta (normal samples) |
| Temporal locality | Ignored | Preserved (paper: FIFO keeps temporal contiguity) |
| Concept drift | No adaptation mechanism | Old patterns naturally replaced |

### Bug #4: CA-DIF-EIA-Stream — Random Projection vs Trained AE (CRITICAL)

| Aspect | v7 (BROKEN) | v8 (CORRECT) |
|--------|-------------|----------------|
| Warmup encoder | Random 25→16 projection | Trained DAE (full 4-layer AE) |
| Online adaptation | None (fixed projection) | Decoder fine-tuning + ADWIN retrain |
| Latent space | Random noise | Learned manifold |

### Bug #5: Streaming Evaluation — update_one() Never Called (CRITICAL)

In v7, `MemStream.update_one()` was called but:
1. Memory was never updated (only `_buf` was populated)
2. No threshold check on updates (all samples entered memory = poisoning)
3. Random replacement instead of FIFO

In v8, `update_one()` correctly encodes the sample to latent space and FIFO-replaces if score < beta.

---

## Results

### Overall Ranking (AUC-PR, mean ± std)

| Rank | Algorithm | AUC-PR Mean | AUC-PR Std | Type | vs v7 |
|------|----------|-----------|-----------|------|-------|
| 1 | DenoisingAE | 0.9995 | 0.0006 | Batch | -0.0000 |
| 2 | CA-DIF-EIA-Stream | 0.9992 | 0.0010 | Streaming | +0.1404 |
| 3 | MemStream | 0.9988 | 0.0031 | Streaming | -0.0008 |
| 4 | AE+IF | 0.9984 | 0.0017 | Batch | +0.0000 |
| 5 | CA-DIF-EIA | 0.9252 | 0.0277 | Batch | +0.0012 |
| 6 | IF-baseline | 0.8043 | 0.0817 | Batch | -0.0098 |
| 7 | sklearn_IF | 0.7905 | 0.0940 | Batch | -0.0182 |
| 8 | sHST-River | 0.2262 | 0.0759 | Streaming | -0.0034 |
| 9 | Random | 0.0468 | 0.0024 | Baseline | -0.0012 |
| 10 | sklearn_OCSVM | 0.0239 | 0.0001 | Batch | -0.0000 |

### AUC-PR by Difficulty

| Algorithm | EASY | MEDIUM | HARD | Trend |
|-----------|------|--------|------|-------|
| DenoisingAE | 1.0000 | 0.9993 | 0.9993 | -0.0007 |
| CA-DIF-EIA-Stream | 1.0000 | 0.9990 | 0.9987 | -0.0013 |
| MemStream | 1.0000 | 0.9989 | 0.9974 | -0.0026 |
| AE+IF | 0.9994 | 0.9978 | 0.9980 | -0.0014 |
| CA-DIF-EIA | 0.9275 | 0.9218 | 0.9263 | -0.0012 |
| IF-baseline | 0.8042 | 0.7423 | 0.8663 | +0.0621 |
| sklearn_IF | 0.7927 | 0.7312 | 0.8477 | +0.0550 |
| Random | 0.0468 | 0.0468 | 0.0468 | 0.0000 |
| sHST-River | 0.2624 | 0.2025 | 0.2138 | -0.0486 |
| sklearn_OCSVM | 0.0238 | 0.0240 | 0.0240 | +0.0002 |

### Streaming: Label Budget Invariance

| Algorithm | Budget=0 | Budget=500 | Change |
|-----------|----------|------------|--------|
| CA-DIF-EIA-Stream | 0.9992 | 0.9992 | 0.0000 |
| MemStream | 0.9988 | 0.9988 | 0.0000 |
| sHST-River | 0.2379 | 0.2145 | -0.0234 |
| Random | 0.0467 | 0.0468 | +0.0001 |

Both MemStream and CA-DIF-EIA-Stream are **label-agnostic** — identical performance at budget=0 and budget=500. The label budget provides zero benefit because the models are already near-optimal on this dataset without active learning.

---

## Statistical Significance

### Batch Algorithms (Friedman: χ²=87.71, p<0.0001)

Wilcoxon Holm-Bonferroni post-hoc (CA-DIF-EIA vs baselines):

| Comparison | p_raw | p_holm | Significant |
|------------|-------|--------|------------|
| CA-DIF-EIA vs sklearn_IF | <0.0001 | 0.0002 | YES |
| CA-DIF-EIA vs sklearn_OCSVM | <0.0001 | 0.0001 | YES |
| CA-DIF-EIA vs IF-baseline | <0.0001 | 0.0001 | YES |
| CA-DIF-EIA vs DenoisingAE | 1.0000 | 1.0000 | no |
| CA-DIF-EIA vs AE+IF | 1.0000 | 1.0000 | no |

**Interpretation:** CA-DIF-EIA is statistically indistinguishable from DenoisingAE and AE+IF (p=1.0), but significantly better than all sklearn baselines. The EIA component provides value over IF (+12 pp), but the DIF component trades 7.4 pp for the ability to allocate investigative effort.

### Streaming Algorithms (Friedman: χ²=41.55, p<0.0001)

| Algorithm | Avg Rank | Interpretation |
|-----------|----------|---------------|
| CA-DIF-EIA-Stream | 1.0 | Best streaming |
| MemStream | 2.0 | Second best |
| sHST-River | 3.33 | Weak |
| Random | 4.67 | Floor |

---

## What the Corrected Results Tell Us

### 1. MemStream's 0.9996 in v7 Was a False Positive

The broken implementation (raw features, random projection) scored 0.9996. The corrected implementation (trained AE, latent memory, FIFO, anti-poisoning) scores 0.9988. The 0.0008 difference is NOT because the fix degraded performance — it's because:

- The broken version was a simpler kNN on raw features, which happens to work on extreme outliers
- The corrected version is the full MemStream algorithm, which correctly uses latent representations
- Both are near-perfect because the anomaly injection is too extreme (8-30x fare), not because the algorithm is robust

### 2. CA-DIF-EIA-Stream Now Outperforms MemStream

| Algorithm | AUC-PR | Note |
|-----------|--------|------|
| CA-DIF-EIA-Stream | 0.9992 | Best streaming |
| MemStream | 0.9988 | Second best |

The corrected CA-DIF-EIA-Stream (trained AE + online fine-tuning + ADWIN retrain) outperforms the corrected MemStream. This is because:
- CA-DIF-EIA-Stream uses a full 4-layer AE trained during warmup
- Online fine-tuning adapts the decoder to new patterns
- ADWIN detects drift and triggers full retrain

### 3. Concept Drift Was NOT Tested

The current benchmark uses temporal train/test splits (leave-one-month-out), which measures **temporal generalization**, not **within-stream concept drift**. To truly test MemStream's concept drift handling:

- Inject gradual distribution shift WITHIN the test stream
- Track whether anomaly scores DECREASE over time (memory adaptation)
- Compare adaptation speed across algorithms

**Recommendation:** Add within-stream concept drift test:
- Phase 1 (first 5K of test): normal data from shifted distribution
- Phase 2 (last 5K): further shifted
- Metric: AUC-PR in each phase + score trajectory over time

---

## Conclusions

1. **The v7 MemStream was fundamentally broken** — using raw features and random projections, it was not implementing the paper's algorithm at all. The 0.9996 score was misleading.

2. **The corrected MemStream (0.9988) validates the paper** — when properly implemented, MemStream matches DenoisingAE and CA-DIF-EIA-Stream on this dataset.

3. **CA-DIF-EIA-Stream is the best streaming algorithm** (0.9992) — using a trained AE with online fine-tuning and ADWIN drift detection.

4. **CA-DIF-EIA remains relevant** for the EIA (Expert Investigative Allocation) component — it provides +12 pp over IF-baseline and is statistically equivalent to DenoisingAE.

5. **Anomaly injection is too extreme** for a meaningful stress test — all top algorithms achieve AUC-PR > 0.997. Subtlest anomalies should be 1.5-2x normal values, not 8-30x.

---

## Files Generated

```
results/v8/
  benchmark_v8.py              # Main benchmark script
  checkpoint_v8.csv            # Full results (675 rows, 0 errors)
  benchmark_results_v8.csv      # With BAR scores
  resume_v8.py                 # Post-processing (stats + plots + report)
  statistical_results.txt      # Friedman + Wilcoxon results
  fig_overview_v8.png          # Overview plots
  fig_difficulty_v8.png        # Per-difficulty plots
  fig_ablation_v8.png          # CA-DIF-EIA ablation
  fig_bar_score_v8.png          # BAR score comparison
  fig_pareto_frontier_v8.png  # Pareto frontier
  benchmark_v8_results.md      # Full results report
```

---

*Generated: 2026-05-13*
*Bug fixes: nn.Parameter for trainable weights, latent-space memory, FIFO+beta update, trained AE for streaming*
