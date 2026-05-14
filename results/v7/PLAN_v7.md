# PLAN v7 — Corrected Benchmark

**Date:** 2026-05-12
**Author:** Claude (based on peer review of v6)

## Motivation

v6 benchmark has **3 critical bugs** and **6 high-severity issues** identified by two expert peer reviewers (code review + methodology review). These invalidate the comparative conclusions. v7 fixes all of them.

---

## Critical Bugs Fixed

### C1: Anomaly injection components overwrite each other (HARD tier is NOT hard)
**File:** `inject_anomalies()`, lines 204-217
**Problem:** In the `combined` type, three components are applied sequentially to the same rows. The last component (`slow_crawl`) overwrites all previous modifications. All "hard" anomalies are pure `slow_crawl`. The difficulty tiers produce **identical AUC-PR** (0.129/0.128/0.129).
**Fix:** Partition 1500 anomalies into 3 disjoint subsets. Each subset gets exactly ONE component. Result: hard tier has 1/3 meter_mult + 1/3 gps_spoof + 1/3 slow_crawl.

### C2: Threshold calibration is unfair between batch and streaming
**File:** `evaluate_batch()` vs `evaluate_streaming()`
**Problem:** Batch (`CADIFEiaBatch`) calibrates threshold on `X_val` (validation set, 15% anomalies). Streaming (`CADIFEiaStream`) calibrates threshold on first 1000 training records (100% normal). These are categorically different score distributions. Any F1/Precision/Recall/FPR comparison is invalid.
**Fix:** Both use the same `X_val` for threshold calibration, scoring without updating the model.

### C3: Duplicate `decision_function` in `CADIFEiaStream`
**File:** Lines 749 and 799
**Problem:** The class defines `decision_function` twice. Python uses the second (lines 799-810), making the first (lines 749-763) dead code.
**Fix:** Delete the first definition (lines 749-763).

---

## High-Severity Fixes

### H1: Streaming budget query is not monotonically consumed
**File:** `evaluate_streaming()`, lines 1117-1124
**Problem:** Each chunk picks `n_queries = min(label_budget, total_processed)` from a growing range `[0, chunk_start+CHUNK)`. Early chunks consume disproportionate budget. Later chunks waste queries on out-of-range positions.
**Fix:** Pre-generate ALL query positions from one RNG with fixed seed before the loop. Consume exactly `label_budget` positions across the full stream.

### H2: ContextFeatureWeighting has dimension mismatch
**File:** `ContextFeatureWeighting`, lines 470, 482, 505
**Problem:** `hour * 7 + dow` produces 168 unique IDs, but the weight matrix has only 24 rows. The clip `context_ids -> [0, 23]` silently reduces the model to hour-only (dow information discarded).
**Fix:** Set `n_contexts=168`. Increase minimum count per context from 50 to 30.

### H3: Streaming CA-DIF-EIA uses random projection vs batch trained AE
**File:** `CADIFEiaStream.fit()`, lines 723-725 vs `CADIFEiaBatch.fit()`, lines 634-636
**Problem:** Streaming uses `np.random.randn` projection matrix. Batch uses trained autoencoder. These are fundamentally different projections — batch and streaming CA-DIF-EIA are not comparable.
**Fix:** Document this as intentional design difference (streaming cannot train AE online). Rename streaming to `CA-DIF-EIA-Stream (random proj)` to make the difference explicit.

### H4: `MemStream.score_one()` returns 0.5 during warmup
**File:** `MemStream`, lines 841-843
**Problem:** When `len(memory) < k`, returns 0.5. These warmup scores pollute the threshold calibration buffer, biasing the 95th percentile low.
**Fix:** Remove the 0.5 fallback. Initialize `k` to 1 so the first point gets a real Mahalanobis score. Use `np.finfo(float).eps` for edge cases.

### H5: Sklearn baselines never set `thresh_`
**File:** `SklearnIF.fit()`, `SklearnOCSVM.fit()`, `SklearnLOF.fit()`
**Problem:** These algorithms never set `self.thresh_`. When `evaluate_batch` calls `predict(..., threshold=algo.thresh_)`, it falls back to sklearn's internal contamination threshold (5%). Batch methods use contamination-based threshold while CA-DIF-EIA uses validation-derived threshold. Unfair comparison.
**Fix:** Compute `self.thresh_` from validation scores in all `fit()` methods. Alternatively, remove `thresh_` fallback and always use the percentile-based threshold from `X_val`.

### H6: Friedman's test on temporally-nested correlated folds
**File:** Fold structure, lines 1776-1798
**Problem:** Fold 1 trains on month 1 (10K records). Fold 5 trains on months 1-5 (50K records). All folds share Jan-Feb data. Treating 5 temporally-nested folds as independent in Friedman violates the independence assumption.
**Fix:** Use **leave-one-month-out** cross-validation: each fold uses months {1..5} train, one month val, next month test. All folds have comparable training sizes (~20-50K) and are temporally independent.

---

## Medium-Severity Fixes

### M1: Lower anomaly rate from 15% to 2%
**Problem:** AUC-PR ~0.13 at 15% anomaly rate is near or below the prevalence baseline (~0.15). The algorithms appear to perform at or near random. NYC TLC fraud rate is <1%.
**Fix:** Set `ANOMALY_RATE = 0.02` (200 anomalies / 10,000 test). AUC-PR baseline drops to ~0.02, making differences meaningful.

### M2: Anomaly injection ranges are too extreme for medium
**Problem:** "Medium" (4-8x meter) still creates very obvious outliers. Only "Easy" is distinguishable.
**Fix:** Tighten ranges: easy=5-10x, medium=2-4x, hard=1.2-2x. Add composite injection for hard: slight fare deviation + slight speed deviation simultaneously.

### M3: Add random baseline algorithm
**Problem:** No AUC-PR calibration. Cannot tell if 0.13 is good or random.
**Fix:** Add `RandomBaseline` that returns uniform random scores [0,1]. This should achieve AUC-PR ≈ anomaly_rate (0.02). Any algorithm beating this significantly demonstrates non-random detection.

### M4: Remove unused `y_val` parameter from `CADIFEiaBatch.fit()`
**File:** Lines 629, 653-658
**Problem:** `y_val` is accepted but never used in threshold computation.
**Fix:** Remove `y_val` parameter from `CADIFEiaBatch.fit()` signature. Document that threshold is unsupervised (percentile-based).

### M5: Label budget parameter passed to `CADIFEiaStream` constructor but unused
**File:** Line 1091
**Problem:** `label_budget` is passed to constructor but `_budget_used` is never checked internally.
**Fix:** Either enforce budget internally in `update_one()`, or remove the parameter from constructor and document that budget is enforced externally.

---

## New Experimental Design

### Datasets
- NYC Yellow Taxi: January-June 2024 (6 months)
- Train: 100% normal (no injection)
- Validation: last 2000 records of last training month (no injection)
- Test: 10000 records with injected anomalies

### Folds (Leave-One-Month-Out)
| Fold | Train | Val | Test | Train Size |
|------|-------|-----|------|-----------|
| 1 | Jan | last 2K of Jan | Feb | ~2.5M |
| 2 | Jan+Feb | last 2K of Feb | Mar | ~5.1M |
| 3 | Jan-Mar | last 2K of Mar | Apr | ~7.7M |
| 4 | Jan-Apr | last 2K of Apr | May | ~10.3M |
| 5 | Jan-May | last 2K of May | Jun | ~13.0M |

**Key improvement:** All folds have ~2.5M+ training records (no tiny fold 1). Folds are temporally ordered but use disjoint test sets.

### Anomaly Injection (Corrected)
- **Rate:** 5% (500 anomalies / 10,000 test) — balances statistical power with realism vs 15% in v6 (too high) and <1% in real taxi fraud (too low for stable AUC-PR)
- **Easy (5-10x):** `fare_amount = trip_distance * 2.5 * uniform(5, 10)`
- **Medium (2-4x):** `fare_amount = trip_distance * 2.5 * uniform(2, 4)`
- **Hard (1.2-2x):** Partition into 3 disjoint subsets:
  - Subset A (67 pts): `fare_amount = trip_distance * 2.5 * uniform(1.2, 2)`
  - Subset B (67 pts): `trip_distance = dur_min * uniform(1.5, 3) / 60`, `fare_amount = trip_distance * 2.5`
  - Subset C (67 pts): `dur_min = uniform(30, 90)`, `trip_distance = uniform(0.5, 3)`, `fare_amount = uniform(5, 20)`

### Algorithms
**Batch:**
1. `RandomBaseline` — uniform random scores (calibration floor)
2. `sklearn_IF` — IsolationForest, n_estimators=200
3. `DenoisingAE` — autoencoder reconstruction error
4. `CA-DIF-EIA` — trained AE + IF + context weighting
5. `AE+IF` — trained AE + IF (no context weighting)
6. `IF-baseline` — IF on raw features

**Streaming:**
1. `RandomBaseline` — same as above
2. `sHST-River` — HalfSpaceTrees
3. `MemStream` — Mahalanobis distance with reservoir
4. `CA-DIF-EIA-Stream` — random projection + IF + context (explicitly labeled as random proj)

### Statistical Analysis
- Friedman test across algorithms (6 folds × 3 difficulties = 18 data points per algorithm)
- Holm-Bonferroni post-hoc with Wilcoxon signed-rank
- Bootstrap CIs (1000 iterations, BCa method)
- All comparisons include RandomBaseline to establish floor

### Metrics
- **Primary:** AUC-PR (threshold-independent, meaningful at low anomaly rates)
- **Secondary:** AUC-ROC, F1, Precision, Recall, FPR (with calibrated threshold from validation)
- **Efficiency:** BAR Score (AUC-PR / log(budget + 1)) for streaming

---

## Expected Outcomes

1. **Difficulty tiers should be discriminative:** Easy should show highest AUC-PR, hard lowest. The partition-based injection ensures each anomaly type is distinct.

2. **RandomBaseline establishes floor:** AUC-PR ≈ 0.02 (anomaly rate). Algorithms beating this significantly demonstrate real detection capability.

3. **Friedman test gains power:** With 6 independent folds (vs 2 in v6), the test has more statistical power to detect real algorithm differences.

4. **Batch vs streaming comparison is fair:** Both use identical validation-based threshold calibration.

5. **CA-DIF-EIA can be validated:** If CA-DIF-EIA outperforms IF-baseline significantly, it validates the context-weighting contribution. If not, the ablation study is informative either way.

---

## Timeline Estimate

- Data loading: ~2 min per month × 6 = 12 min
- Batch job: ~15s each × 5 algos × 6 folds × 3 difficulties × 3 seeds = ~40 min
- Streaming job: ~2s each × 3 algos × 6 folds × 3 difficulties × 3 seeds × 4 budgets = ~7 min
- Statistical analysis: ~2 min
- **Total: ~60 min** (with vectorized streaming optimizations from v6)

Config for fast run: 2 folds (Feb+Mar), 2 seeds (42, 123), 2 budgets (0, 500) → ~15 min.
