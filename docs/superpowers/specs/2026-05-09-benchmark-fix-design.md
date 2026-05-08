# Benchmark Fix Design — Thesis Defense Validation

**Date:** 2026-05-09
**Goal:** Fix Kaggle benchmark notebook to produce scientifically meaningful results that prove 3 hypotheses for thesis defense.

## Problem Statement

Current notebook (`kaggle_benchmark_FINAL/notebooka66121ef2a.ipynb`) has 4 critical bugs:

1. **Scenario-layer mismatch:** 4/5 anomaly scenarios get caught by L1+L2 rules before reaching ML. ML only detects 2-6% of remaining anomalies.
2. **Label alignment bug:** ML evaluation reports 4,779 anomalies when only 1,000 remain after L1+L2 filtering.
3. **No variant comparison:** Only 1 ML variant tested instead of 5.
4. **No statistical testing:** No significance tests, confidence intervals, or effect sizes.

## 3 Hypotheses to Prove

- **H1:** 21D ratio features detect anomalies significantly better than 15D raw features
- **H2:** Per-cluster adaptive thresholds reduce FPR vs global thresholds
- **H3:** Proposed method (iForest + 21D + per-cluster) outperforms opponent algorithms

## Architecture: Realistic Pipeline Flow

```
Raw Parquet
  ↓
L1 Schema Filter → stats (natural violations, no injection)
  ↓
L2 Rule Filter → stats (natural violations, no injection)
  ↓
Clean data (df_clean)
  ↓
Split: 70% train / 30% test
  ↓
Inject 5 scenarios into TEST ONLY (5 × 1000 = 5000 anomalies)
  ↓
4 Sanity Checks (fail-fast)
  ↓
Train 5 variants × 5 seeds on TRAIN
  ↓
Evaluate on TEST (with injected anomalies)
  ↓
Statistical tests + Visualization
```

Key principle: ML trains on L1+L2-cleaned data, tests on L1+L2-cleaned data with injected anomalies. All injected anomalies MUST pass L1+L2 rules (valid schema, valid business rules) but be detectable through ratio features.

## Anomaly Scenarios — All Pass L1+L2

### Design Principle

Individual fields are valid (pass L1+L2 rules) but ratio feature combinations are extreme (ML-detectable). This directly proves the value of 21D ratio features.

L1 rules: passenger_count 1-6, all fields non-null, valid zones
L2 rules: fare > 0, distance > 0, speed < 100 mph

### S1: meter_tampering — `fare_per_mile_ratio` 15-30x

- distance: 1-3 mi, duration: 5-15 min, speed: 4-36 mph
- fare: $37.50-225 (15-30x normal $2.50/mile)
- passengers: 1-4
- L1 ✅, L2 ✅
- ML signal: `fare_per_mile_ratio` = 15-30x, `fare_per_minute_ratio` = 5-15x

### S2: gps_spoofing — `implied_speed_ratio` 5-8x

- distance: 20-40 mi, duration: 15-30 min, speed: 40-95 mph (< 100)
- fare: $60-120 (reasonable for long trip)
- passengers: 1-3
- L1 ✅, L2 ✅
- ML signal: `implied_speed_ratio` = 3.3-7.9x (vs baseline 12 mph), `fare_distance_product` extreme

### S3: passenger_anomaly — `fare_per_mile_ratio` + `passenger_distance_ratio` extreme

- passengers: 1-5 (pass L1), distance: 0.2-0.5 mi, duration: 15-30 min
- fare: $40-70 (extremely high for 0.2 mi)
- L1 ✅, L2 ✅
- ML signal: `fare_per_mile_ratio` = 32-140x, `passenger_distance_ratio` extreme, `duration_distance_ratio` extreme

### S4: slow_crawl — `duration_distance_ratio` extreme

- distance: 2-4 mi, duration: 90-180 min, speed: 0.7-2.7 mph
- fare: $40-80
- passengers: 1-2
- L1 ✅, L2 ✅
- ML signal: `duration_distance_ratio` = 22-90x normal, `fare_per_minute_ratio` abnormally low

### S5: combined_subtle — multiple ratios moderately extreme simultaneously

- distance: 1-2 mi, duration: 5-10 min, passengers: 4-5, fare: $50-100
- speed: 6-24 mph
- L1 ✅ (passengers ≤ 6), L2 ✅ (speed < 100)
- ML signal: `fare_per_mile_ratio` = 10-20x + `fare_per_passenger` high + `fare_per_minute_ratio` = 5-10x

### Overlap Analysis

| Scenario | Key ratio | Clean mean±std | Anomaly range | Separation |
|----------|-----------|---------------|---------------|------------|
| S1 meter_tampering | fare_per_mile_ratio | 3.47±10.46 | 15-30x | ≥1.1σ per dim |
| S2 gps_spoofing | implied_speed_ratio | 0.82±0.35 | 3.3-7.9x | 7-20σ |
| S3 passenger_anomaly | fare_per_mile_ratio | 3.47±10.46 | 32-140x | 2.7-13σ |
| S4 slow_crawl | duration_distance_ratio | ~5±3 | 22-90x | 5.7-28σ |
| S5 combined_subtle | multiple ratios | — | 3-5σ each | Combined >5σ |

S2-S5 have clear separation (>5σ). S1 has moderate single-dimension separation but combined 21D signal is sufficient for IForest (multiple features simultaneously anomalous).

## 4 Sanity Checks (Fail-Fast)

### Checkpoint 1: Train Sterile (Zero Contamination)

```python
assert (df_train['fare_amount'] <= 0).sum() == 0
assert (df_train['trip_distance'] <= 0).sum() == 0
assert (df_train['passenger_count'] > 6).sum() == 0
assert (df_train['passenger_count'] < 1).sum() == 0
# ML training data should be ~86% of L1 output
assert len(df_train) / len(df_after_l1) > 0.80
```

### Checkpoint 2: Test Extreme (Anomalies Must Be Detectable)

```python
# All anomalies MUST pass L1+L2 rules
for idx in anomaly_indices:
    assert df_test.loc[idx, 'passenger_count'] between 1 and 6
    assert df_test.loc[idx, 'fare_amount'] > 0
    assert df_test.loc[idx, 'trip_distance'] > 0
    assert computed_speed(idx) < 100

# Per-scenario ratio checks
s1_fpm = df_test.loc[s1_indices, 'fare_amount'] / df_test.loc[s1_indices, 'trip_distance']
assert s1_fpm.min() >= 15 * 2.50  # 15x baseline

s2_speed = compute_speed(s2_indices)
assert s2_speed.min() >= 40 and s2_speed.max() < 100

s4_dur_dist = compute_duration_distance_ratio(s4_indices)
assert s4_dur_dist.min() >= 22  # 22x normal
```

### Checkpoint 3: Feature 21D Verification

```python
assert X_train.shape[1] == 21
assert 'fare_per_mile_ratio' in feature_names
assert 'implied_speed_ratio' in feature_names
assert 'duration_distance_ratio' in feature_names
# Verify ratio features are not all zeros
for ratio_col_idx in [15, 16, 17, 18, 19, 20]:
    assert X_train[:, ratio_col_idx].std() > 0
```

### Checkpoint 4: Context Mapping Loaded

```python
assert len(neighborhood_mapping) >= 196  # zones
assert len(threshold_matrix) >= 5  # clusters
# Every test PULocationID must map to a cluster
unmapped = set(df_test['PULocationID'].unique()) - set(neighborhood_mapping.keys())
assert len(unmapped) == 0, f"Unmapped zones: {unmapped}"
```

## 5-Variant Benchmark

### Variant 1: baseline_static

- **Features:** 15D (raw only, NO ratio features)
- **Threshold:** global 95th percentile of training scores
- **Model:** sklearn IsolationForest(n_estimators=200, random_state=seed)
- **Purpose:** Worst case baseline, proves ratio features are needed

### Variant 2: baseline_ratio

- **Features:** 21D (WITH ratio features)
- **Threshold:** global 96th percentile of training scores
- **Model:** sklearn IsolationForest(n_estimators=200, random_state=seed)
- **Purpose:** Compare with V1 → proves H1 (21D > 15D)

### Variant 3: proposed_context_aware (TARGET)

- **Features:** 21D (WITH ratio features)
- **Threshold:** per-cluster adaptive (K-Means clustering + per-cluster percentile)
- **Model:** sklearn IsolationForest(n_estimators=200, random_state=seed)
- **Purpose:** Compare with V2 → proves H2 (per-cluster > global). Compare with V4/V5 → proves H3.

### Variant 4: opponent_lof

- **Features:** 21D
- **Threshold:** decision_function based
- **Model:** sklearn LocalOutlierFactor(n_neighbors=20, novelty=True)
- **Purpose:** Established density-based opponent

### Variant 5: opponent_ocsvm

- **Features:** 21D
- **Threshold:** decision_function based
- **Model:** sklearn OneClassSVM(kernel='rbf', nu=0.01)
- **Purpose:** Established boundary-based opponent

### Execution

- 5 variants × 5 seeds = 25 runs
- Metrics per run: F1, Recall, Precision, FPR, train_time
- Report: mean ± std for each metric

### Expected Results

```
Variant                  | Recall    | FPR       | F1        | Rank
-----------------------------------------------------------------
baseline_static (15D)    | ~70%      | ~15-25%   | ~0.70     | 5
baseline_ratio (21D)     | ~88-92%   | ~5-8%     | ~0.85     | 2
proposed_context_aware   | ~89-93%   | ~3-5%     | ~0.90+    | 1
opponent_lof             | ~80-85%   | ~6-10%    | ~0.80     | 3-4
opponent_ocsvm           | ~75-82%   | ~8-12%    | ~0.78     | 4-5
```

## Statistical Testing

1. **Paired t-test:** proposed vs each other variant on F1 scores (5 seeds). H0: no difference. Reject if p < 0.05.
2. **Wilcoxon signed-rank:** non-parametric backup for small sample (n=5).
3. **95% Confidence Intervals:** for F1, Recall, FPR of each variant.
4. **Cohen's d effect size:** magnitude of improvement (small d<0.2, medium d<0.8, large d≥0.8).
5. **Summary table:** Algorithm | F1 (mean±std) | Recall | FPR | p-value vs proposed | Cohen's d

## Notebook Structure

```
Cell 0:  Title + Abstract
Cell 1:  Imports + Config (FAST_MODE, DATA_FILE, seeds)
Cell 2:  Load raw data
Cell 3:  L1 Schema Filter (natural stats)
Cell 4:  L2 Rule Filter (natural stats)
Cell 5:  Funnel summary (L1+L2 stats)
Cell 6:  Train/Test split (70/30)
Cell 7:  Feature Vectorizer (15D and 21D)
Cell 8:  Synthetic Anomaly Generator (5 scenarios)
Cell 9:  Inject anomalies into TEST set
Cell 10: 4 Sanity Checks (fail-fast)
Cell 11: Neighborhood Clustering + Context Mapping
Cell 12: Train & Evaluate 5 variants × 5 seeds
Cell 13: Results table (mean ± std)
Cell 14: Statistical tests (t-test, Wilcoxon, CI, Cohen's d)
Cell 15: Visualization (bar charts, radar, heatmap)
Cell 16: Final summary + Hypothesis validation
```

## Constraints

- **Runtime:** Kaggle 2×T4 GPU, must complete within 9 hours
- **FAST_MODE:** 100K records for quick iteration, full mode for final results
- **Dependencies:** sklearn, numpy, pandas, scipy, matplotlib, seaborn (all available on Kaggle)
- **No River dependency:** Use sklearn equivalents (IsolationForest, LOF, OneClassSVM)
