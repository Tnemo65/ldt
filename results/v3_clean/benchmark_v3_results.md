# Streaming Anomaly Detection Benchmark Protocol

> **Document Type:** Benchmark Protocol Specification
> **Version:** 3.3
> **Date:** 2026-05-11
> **Status:** Active
> **Changelog v3.3:** Dual-mode evaluation: Table A batch train-once, Table B prequential; added CA-DIF-EIA streaming variant to Table B; BAR Score scoped to Table B only; adaptive sliding-window threshold for Table B; fixed CD diagram parameters for both tables.
> **Note:** Exploratory empirical study. Rankings emerge from data. No algorithm is claimed superior a priori.

---

## 1. Introduction

### 1.1 Production Context

This benchmark evaluates **seven anomaly detection algorithms** as candidates for the ML branch of the CA-DQStream pipeline. The pipeline has two components:

1. **Canary Rules** (always-on): 7 deterministic checks filter obvious violations.
2. **ML Branch** (proposed): Anomaly detector scores records that pass Canary Rules.

ML's role in production is strictly **anomaly detection** — given a record that passes Canary Rules, does it look anomalous? Concept drift and adaptation are handled by separate infrastructure (IEC circuit breaker, ADWIN monitors) that is **outside the scope of ML evaluation**.

### 1.2 Scope

```
Dataset:       NYC Yellow Taxi, Jan-Dec 2024 (12 monthly parquet files)
Domain:        Taxi fare anomaly detection
Algorithms:    10 total — Table A (Batch): 6, Table B (Streaming): 4
Evaluation:    Table A: batch train-once; Table B: prequential
Seeds:         10 per configuration
Difficulties:  3 (easy, medium, hard)
Total runs:    Table A: 12 folds × 6 algorithms × 10 seeds × 3 difficulties = 2,160
               Table B: 12 folds × 4 algorithms × 10 seeds × 3 difficulties = 1,440
```

### 1.3 Research Questions

```
Q1: Which algorithm achieves the best anomaly detection accuracy
    (AUC-PR, F1, Precision, Recall) on NYC taxi fare data?
Q2: Are performance differences statistically significant across
    12 monthly folds?
Q3: Are there statistically indistinguishable performance groups?
Q4: What is the practical effect size of observed differences?
Q5: Does the Context-aware Grid provide measurable improvement over
    raw features? (Ablation Study)
Q6: How much labeled data does each algorithm require to maintain
    high AUC-PR? (BAR Score)
```

### 1.4 Evaluation Philosophy

- **Batch evaluation**: Each algorithm trains once on accumulated historical data, scores a test set. This mirrors how ML operates in production (one-shot training on accumulated clean data).
- **Fair comparison**: All algorithms receive identical features, identical data, identical anomaly injection.
- **Multi-dimensional ranking**: No single metric determines the winner.
- **No a priori claim**: Rankings are determined by data.

---

## 2. Dataset

### 2.1 Source

```
Source:        NYC Taxi and Limousine Commission (TLC)
URL:           https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
Format:        Parquet, monthly partitions
Year:          2024 (January through December)
Total records: ~43 million across 12 months
```

### 2.2 Monthly Folds

Twelve calendar months serve as twelve evaluation folds. Fold structure mimics production: train on accumulated clean data from previous months, evaluate on the current month.

```
Fold  1: Train Jan  → Test Feb   (inject anomalies into Feb test set)
Fold  2: Train Jan+Feb → Test Mar
...
Fold 11: Train Jan-Oct → Test Nov
Fold 12: Train Jan-Nov → Test Dec
```

**Note:** Fold k accumulates more training data than Fold 1. This reflects production reality: the system improves as it accumulates clean data. It is NOT a flaw — it is the intended behavior.

### 2.3 Data Cleaning

```
L1 — Schema Validation:
    - Required columns present and non-null
    - passenger_count ∈ [1, 6]
    - PULocationID, DOLocationID ∈ [1, 263]

L2 — Canary Rules (domain knowledge):
    - fare_amount > 0
    - trip_distance > 0
    - duration_seconds > 0
    - 0 < speed_mph < 100

L3 — Statistical Outlier Removal (IQR 3×):
    - Applied to: fare_amount, trip_distance, duration_seconds
```

Canary Rules are **not** a benchmark algorithm. They are always-on production filters applied before ML scoring.

---

## 3. Feature Engineering

### 3.1 25-Dimensional Feature Vector

```
Block 1 — Raw (15D):
    distance, duration_min, fare, passengers, total_amount,
    speed, fare_per_mile, fare_per_minute, fare_per_passenger,
    hour_of_day, day_of_week, is_weekend, is_rush_hour, is_night, month

Block 2 — Ratio to Baseline (6D):
    fare_per_mile / 2.5
    fare_per_minute / 0.67
    speed / 12.0
    passenger_count / distance
    fare × distance
    duration_min / distance

Block 3 — Cyclical Encoding (4D):
    sin(2π × hour / 24), cos(2π × hour / 24)
    sin(2π × day_of_week / 7), cos(2π × day_of_week / 7)
```

---

## 4. Anomaly Injection Protocol

### 4.1 Five Fraud Scenarios

Each scenario simulates a distinct fraud pattern, derived from known NYC taxi fraud typologies:

| # | Scenario | Mechanism | Fraud Description |
|---|----------|-----------|-----------------|
| 1 | meter_tampering | fare = distance × $2.50 × m | Overcharging via fare meter manipulation |
| 2 | gps_spoofing | high speed + normal fare | Phantom trips inflated by distance |
| 3 | passenger_anomaly | short distance + high fare | Overcharging short rides |
| 4 | slow_crawl | long duration + high fare | Traffic avoidance fraud |
| 5 | combined_subtle | multiple mild distortions | Realistic multi-factor fraud |

### 4.2 Three Difficulty Levels

```
EASY:   meter_mult=(10-20),  speed=(50-95mph),   pax_fare=($40-70),  crawl=(90-180min)
MEDIUM: meter_mult=(4-8),    speed=(30-60mph),   pax_fare=($15-30),  crawl=(40-80min)
HARD:   meter_mult=(1.5-3),  speed=(20-40mph),   pax_fare=($8-15),   crawl=(20-35min)
```

The HARD level produces anomalies that are challenging to distinguish from normal records, requiring a sophisticated detector.

### 4.3 Injection Parameters

```
Per scenario:     1,000 anomalies
Per fold:         5,000 anomalies (5 scenarios × 1,000)
Per difficulty:   5,000 anomalies
Contamination:   ~0.83% (5,000 anomalies / ~600,000 normal test records)
Anomaly seed:     varies by (fold_index, difficulty_index) to ensure
                  different positions across folds AND difficulties
```

---

## 5. Algorithms

Nine algorithms are evaluated, split into two architectural groups. Grouping reflects different adaptation capabilities — comparing them fairly requires separate evaluation tables.

### 5.1 Benchmark Table A — Batch-Trained (Static)

These algorithms train once on accumulated historical data and do not update during scoring. Their adaptation capability comes from full retraining (triggered externally by the IEC circuit breaker).

| # | Algorithm | Type | Justification | Source |
|---|-----------|------|---------------|--------|
| 1 | sklearn_IF | Tree-based isolation | Standard one-class baseline | sklearn |
| 2 | sklearn_OCSVM | Kernel-based | Classic one-class method | sklearn |
| 3 | sklearn_LOF | Density-based | Neighborhood-based method | sklearn |
| 4 | LSTM-Autoencoder | Deep learning | DL reconstruction-based baseline | Keras/TF |
| 5 | CA-DIF-EIA | Context-aware isolation | Proposed batch method | This work |
| 6 | METER-SCD | Hypernetwork adaptation | SCD module only; SOTA baseline | Zhu et al., VLDB 2024 |

### 5.2 Benchmark Table B — Streaming (Online Update)

These algorithms continuously update their model parameters as new data arrives. No external retraining trigger is needed.

| # | Algorithm | Type | Justification | Source |
|---|-----------|------|---------------|--------|
| 1 | sHST-River | Streaming tree | Half-Space Trees, online | Ting et al. |
| 2 | MemStream | Memory-based DL | Memory augmented for streaming | Bhatia et al., WWW 2022 |
| 3 | IForestASD | Sliding-window iForest | iForest adapted for streams via sliding window | Ding & Fei 2013, scikit-multiflow |
| 4 | CA-DIF-EIA (streaming) | Context-aware isolation + ADWIN-U | Proposed streaming method | This work |

### 5.3 CA-DIF-EIA (Proposed Method)

CA-DIF-EIA is a **context-aware deep isolation framework** with two deployment modes:

**Batch variant (Table A):**
```
1. Train on clean accumulated data (same as sklearn_IF)
2. Score each test record (anomaly score from isolation forest)
3. Context features weight isolation scores
4. No online update; no drift detection
```

**Streaming variant (Table B):**
```
1. Initialize on first 20% of accumulated train data
2. Score incoming record (isolation score + context weighting)
3. ADWIN-U monitors score distribution for drift
4. If drift detected AND label budget available: request label, update
5. Otherwise: continue scoring (no update)
```

The key design difference from sHST/MemStream/IForestASD: ADWIN-U uses the anomaly score stream itself (not labels) to detect drift, achieving near-zero label consumption for drift detection while maintaining competitive accuracy. This is what the BAR Score should reveal.

### 5.4 Hyperparameters

```
Batch Group:
  sklearn_IF:      n_estimators=200, contamination=0.05, random_state=seed
  sklearn_OCSVM:   kernel='rbf', gamma='scale', nu=0.05
  sklearn_LOF:     n_neighbors=20, contamination=0.05, novelty=True
  LSTM-AE:         hidden_dim=64, seq_len=1, epochs=10, batch_size=256, GPU-accelerated
  CA-DIF-EIA:      n_estimators=200, contamination=0.05, threshold=97th percentile
  METER-SCD:       base_detector=IsolationForest, n_estimators=200, gamma='scale'

Streaming Group:
  sHST-River:      depth=10, n_trees=25, window_size=250
  MemStream:       buffer_size=500, memory_size=200
  IForestASD:      n_estimators=200, max_samples=256, random_state=seed
  CA-DIF-EIA (streaming): n_estimators=200, adwin_delta=0.002, window_size=1000, label_budget=varies
```

Grid Search tuning is applied to ALL algorithms (including CA-DIF-EIA) using the Initial Training partition (months 1–2) to find the best configuration before the main evaluation phase. For each algorithm, a coarse grid of hyperparameter combinations is evaluated via 5-fold cross-validation on the Initial Training partition using AUC-PR as the selection criterion. The winning configuration is then frozen and used for all 12 monthly folds. This ensures all algorithms operate at their best-known setting for NYC taxi data — neither advantaged nor disadvantaged — prior to evaluation. No further tuning occurs during evaluation.

---

## 6. Evaluation Protocol

The dual-table design (Section 5) requires two distinct evaluation modes. Each table operates in its native environment to ensure fair, meaningful comparison.

### 6.1 Table A — Batch Evaluation (Static Train-Once)

For each fold, algorithms train once on accumulated historical data, then score the entire test month:

```
For each fold:
    1. Load and clean train months (accumulated from Jan to month-1)
    2. Load and clean test month
    3. Inject synthetic anomalies into test month
    4. Extract features from train (fit StandardScaler)
    5. Fit algorithm on scaled train features (train once; freeze)
    6. Score all test records (batch scoring, no per-record update)
    7. Compute metrics against injected ground truth
```

No retrain triggers, no online update. This directly measures: given clean accumulated training data, how well does this algorithm detect injected anomalies? This is the exact production mode: ML scores records after Canary Rules, and the IEC system (outside ML scope) triggers any retraining.

### 6.2 Table B — Prequential Evaluation (Test-Then-Train)

Streaming algorithms and CA-DIF-EIA (streaming variant) are evaluated in their native online environment using **Prequential Evaluation** (test-then-train):

```
For each fold:
    1. Warm-up: initialize model on first 20% of accumulated train data
    2. For each record in test month (chronological order):
        a. Score record → produce anomaly score (no ground truth used)
        b. Update running ground truth buffer (ADWIN for drift detection)
        c. If ADWIN signals drift AND label budget remains:
             - Request ground truth label for this record
             - Update model with (features, label)
             - Decrement label budget counter
        d. If no drift or no label budget: discard record
    3. Compute metrics against full injected ground truth at end of fold
```

**Label budget:** Each streaming algorithm receives a fixed label budget (Section 7.3). Labels are consumed only when ADWIN detects distributional shift. This reproduces the production scenario where labeling is expensive and must be prioritized.

**Rationale:** Streaming algorithms (sHST, MemStream, IForestASD) and CA-DIF-EIA's streaming variant are designed for continuous adaptation via sliding windows or memory modules. Forcing them into batch mode destroys their core mechanism and produces artificially poor results (lower bound, not true capability). Prequential evaluation restores their natural environment.

**Note on comparability:** Table A and Table B cannot be compared head-to-head because they measure different things. Table A measures static detection accuracy. Table B measures adaptive accuracy under concept drift with constrained labeling. Cross-group comparison uses only AUC-PR and BAR Score.

### 6.3 Scope Numbers

```
Table A (Batch):
  Folds: 12 × 6 algorithms × 10 seeds × 3 difficulties = 2,160 runs

Table B (Streaming):
  Folds: 12 × 4 algorithms × 10 seeds × 3 difficulties = 1,440 runs
  Label budgets: 5 (1%, 5%, 10%, 25%, 50%) × 3 difficulties = 15 label conditions
  Per-condition: 12 folds × 4 algorithms × 10 seeds = 480 runs
  Total Table B with BAR: 1,440 + (4 algorithms × 15 label conditions × 10 seeds) = 2,040 runs
```

---

## 7. Evaluation Metrics

### 7.1 Primary Metric

**AUC-PR** (Area Under Precision-Recall Curve) is the primary metric.

Rationale: Taxi fraud datasets are highly imbalanced (~1% contamination). AUC-ROC can reach 0.99 even with poor detection because TN dominates the calculation. AUC-PR directly measures the precision-recall tradeoff across all thresholds, making it the objective standard for anomaly detection in imbalanced settings.

### 7.2 Supplementary Metrics

| Metric | Role | Table |
|--------|------|-------|
| **AUC-ROC** | Threshold-independent, supplementary | Both |
| **F1 Score** | At optimal threshold | Table A only |
| **Precision** | Cost of false alarms | Table A only |
| **Recall** | Detection coverage | Table A only |
| **FPR** | False alarm rate | Table A only |
| **BAR Score** | Label efficiency at varying budgets | Table B only |

### 7.3 Threshold Selection

**Table A (Batch):** For threshold-dependent metrics (F1, Precision, Recall, FPR), the threshold is determined by maximizing F1 on the training set's anomaly scores (known labels). This threshold is then applied to test scores. This is reproducible and standardized.

**Table B (Streaming):** The threshold must be adaptive because the distribution of anomaly scores shifts during online evaluation. Each streaming algorithm uses a **sliding-window quantile threshold**: the threshold is set to the (1 − contamination)th percentile of anomaly scores within a rolling window of the most recent 1,000 records. This mimics production where operators use recent score distributions rather than historical ones.

**Primary ranking metric is AUC-PR.** It is threshold-independent and applies uniformly to both tables. F1, Precision, Recall are reported as supplementary context for Table A only.

### 7.4 BAR Score (Balanced Accuracy by Budget of Labeled Data)

BAR Score applies **only to Table B** (streaming algorithms and CA-DIF-EIA streaming variant). It measures how well each algorithm maintains accuracy as labeled data decreases — which is only meaningful when algorithms are actively consuming labels during evaluation.

```
For each streaming algorithm, for each (fold, difficulty, budget):
    budget ∈ {1%, 5%, 10%, 25%, 50%} of test-month labels available
    Label consumption priority: ADWIN drift signals only

    1. Initialize model with warm-up data
    2. Run prequential loop with label budget enforced
    3. Each time drift is detected and budget remains: consume one label
    4. Record AUC-PR at end of fold
    5. BAR(budget) = mean(AUC-PR) across folds at this budget

BAR Curve: AUC-PR_mean vs. labeled_data_budget
Interpretation: Higher AUC-PR at lower budgets = better label efficiency
```

The BAR Curve directly answers: "If we can only afford to label 5% of records, which algorithm maintains the best detection?" This is where CA-DIF-EIA's ADWIN-U module (Section 5.3) should demonstrate its advantage — requesting labels only on genuine drift rather than on every anomaly.

**Table A is excluded from BAR Score** because batch-trained algorithms consume zero labels during evaluation. BAR Score would be trivially 1.0 for all batch algorithms.

---

## 8. Statistical Analysis

### 8.1 Two-Stage Protocol

```
Stage 1: Post-hoc Pairwise Comparisons (Wilcoxon Signed-Rank)
    → Which specific pairs differ?

Stage 2: Effect Sizes (Cohen's d) + CD Diagram
    → How large is the difference, practically?
```

### 8.2 Stage 1: Wilcoxon Signed-Rank Test

**Primary pairwise test** for comparing two algorithms across N folds:

```
H0: median(score_alg_i) = median(score_alg_j)
H1: median(score_alg_i) ≠ median(score_alg_j)

Table A: 15 pairs (C(6,2)) × 4 metrics × 3 difficulties = 180 comparisons
Table B:  6 pairs (C(4,2)) × 1 metric (AUC-PR) × 3 difficulties = 18 comparisons
         (plus BAR Score analysis at 5 budgets × 3 difficulties = 15 label-condition comparisons)
```

**Rationale for omitting Friedman:** Adjacent monthly folds share seasonal structure, violating Friedman's assumption of independent folds. Using Friedman when its core assumption is violated yields invalid p-values. Wilcoxon Signed-Rank Test is the appropriate choice for paired temporal data — it is a non-parametric paired test that does not assume fold independence, only that paired differences are i.i.d. The test directly compares CA-DIF-EIA against each baseline on a fold-by-fold basis, which is the most precise comparison for time-series-structured evaluation.

**Multiple testing correction:**

```
Holm-Bonferroni (primary):
    p_adj_i = min(p_i × 21, 1.0) via step-down procedure

Benjamini-Hochberg (secondary, FDR-controlling):
    Less conservative, more powerful
```

### 8.3 Stage 2: Effect Sizes + CD Diagram

**Cohen's d (paired):**
```
d = mean(diff) / std(diff)

|d| < 0.2:  negligible
0.2 ≤ |d| < 0.5:  small
0.5 ≤ |d| < 0.8:  medium
|d| ≥ 0.8:       large (practically meaningful)
```

**Critical Difference Diagram** (Demšar 2006):
```
Table A (k=6): CD = q_α × √(k(k+1) / 6N) = 3.220 × √(42/72) ≈ 2.66
Table B (k=4): CD = q_α × √(k(k+1) / 6N) = 2.459 × √(20/72) ≈ 1.30

Algorithms connected by a bar are NOT significantly different (α=0.05)
```

---

## 9. Output Specification

### 9.1 Raw Results

```
results/v3/benchmark_results_batch.csv
  Group: Batch (6 algorithms)
  Columns: fold, month, difficulty, algorithm, seed,
           AUC_PR, AUC_ROC, F1, Precision, Recall, FPR,
           TP, FP, TN, FN, optimal_threshold,
           train_time_ms, score_time_ms
  Rows: 2,160

results/v3/benchmark_results_streaming.csv
  Group: Streaming (4 algorithms)
  Columns: fold, month, difficulty, algorithm, seed,
           AUC_PR, AUC_ROC,
           threshold_method (adaptive_quantile),
           labels_consumed, label_budget_used,
           train_time_ms, score_time_ms
  Rows: 1,440

results/v3/ablation_results.csv
  Columns: fold, algorithm, difficulty, seed,
           AUC_PR_A (control), AUC_PR_B (treatment), ΔAUC_PR
  Rows: ablation runs

results/v3/bar_score_results.csv
  Columns: algorithm, difficulty, seed,
           budget_1pct, budget_5pct, budget_10pct, budget_25pct, budget_50pct,
           AUC_PR_at_budget
  Rows: ~1,980
```

### 9.2 Statistical Results

```
results/v3/statistical_tests_batch.csv    — Wilcoxon + Holm/BH (Table A)
results/v3/statistical_tests_streaming.csv — Wilcoxon + Holm/BH (Table B)
results/v3/cd_ranks_batch.csv      — CD diagram ranks (Table A)
results/v3/cd_ranks_streaming.csv  — CD diagram ranks (Table B)
results/v3/ablation_stats.csv      — Ablation study: Wilcoxon Signed-Rank
results/v3/bar_score_summary.csv   — BAR score curves summary
```

### 9.3 Figures

```
Table A (Batch):
  figA1_aucpr_boxplot.png       — AUC-PR boxplot (6 algorithms)
  figA2_cd_easy.png            — CD diagram (easy)
  figA3_cd_medium.png          — CD diagram (medium)
  figA4_cd_hard.png            — CD diagram (hard)
  figA5_pvalue_heatmap.png      — Holm p-value heatmap

Table B (Streaming):
  figB1_aucpr_boxplot.png       — AUC-PR boxplot (3 algorithms)
  figB2_cd_all.png             — CD diagram (all difficulties)

Cross-group:
  figX1_bar_score.png          — BAR Score curves by labeled budget
  figX2_ablation.png           — Ablation ΔAUC_PR bar chart
  figX3_radar.png              — Radar: AUC-PR, F1, Recall, Precision
  figX4_per_fold_batch.png     — Per-fold AUC-PR (Table A)
  figX5_per_fold_stream.png    — Per-fold AUC-PR (Table B)
```

---

## 10. Ablation Study — Context-aware Grid

### 10.1 Research Question

Does the Context-aware Grid provide measurable improvement over raw features?

### 10.2 Protocol

For each algorithm in Batch Group, run two configurations:

```
Ablation Config A (Control):
    - Features: raw 15D (distance, duration, fare, speed, etc.)
    - No cyclical encoding
    - No context-aware weighting

Ablation Config B (Treatment):
    - Features: full 25D (with cyclical encoding + context ratios)
    - Same algorithm, same hyperparameters
```

### 10.3 Analysis

```
For each algorithm, difficulty, and fold:
    ΔAUC_PR = AUC_PR(Config_B) - AUC_PR(Config_A)

Summary: Mean ΔAUC_PR across folds with paired t-test and Wilcoxon test.
```

If ΔAUC_PR > 0 with statistical significance (p < 0.05) and effect size ≥ small, the Context-aware Grid is empirically justified. This is the standard for feature engineering claims in experimental CS papers (equivalent to Geohash / Voronoi grid encoding in spatio-temporal research).

---

## 11. Limitations

```
L1. Single domain: Results specific to NYC taxi fare data.
    Generalization requires evaluation on other domains.

L2. Single year: All folds from 2024. Seasonal patterns within one year
    may not reflect multi-year trends.

L3. Synthetic anomalies: Injected anomalies follow parameterized distributions.
    Real anomalies may have different characteristics. Parameter ranges
    are chosen to approximate realistic fraud scenarios.

L4. Temporal dependency: Adjacent folds share seasonal structure.
    Friedman test is conservative; Wilcoxon is the appropriate test.

L5. Tuning on Initial Training: All algorithms are hyperparameter-tuned on
    the Initial Training partition before evaluation. Rankings may shift
    if evaluated on a different domain without retuning.

L6. Adaptive threshold (Table B): Sliding-window quantile thresholds
    adapt to recent score distributions. Any threshold method is
    imperfect under concept drift; AUC-PR is the threshold-independent
    safeguard.

L7. Ablation scope: Context-aware Grid ablation measures feature impact
    only within the benchmark protocol. Production impact may differ.
```

---

## 12. Appropriate Claims

```
Permitted:
  - "On NYC taxi fare data, algorithm X achieved the highest mean AUC-PR
     across 12 monthly folds. Wilcoxon post-hoc tests with Holm
     correction confirm X outperforms Y (p_holm < 0.05)."
  - "Algorithms X, Y, Z form an indistinguishable group
     (CD > |rank_i - rank_j|)."
  - "The effect size for pair (X, Y) is large (d = 1.1), indicating
     practically meaningful differences."
  - "The Context-aware Grid ablation shows ΔAUC_PR = +0.04 (p < 0.01)
     via Wilcoxon Signed-Rank Test, providing empirical justification
     for the 25D feature vector."
  - "At 5% labeled data budget, CA-DIF-EIA (streaming) maintains AUC-PR = 0.71,
     outperforming MemStream (0.58) and sHST-River (0.42) — see BAR Score
     (Table B)."

NOT permitted:
  - "CA-DIF-EIA is the best anomaly detection algorithm."
  - Any claim not directly supported by empirical data.
  - Cross-table claims comparing Table A AUC-PR against Table B AUC-PR
    (they measure different things; Table B comparisons use BAR Score).
  - Throughput/latency claims without measurement on production hardware.
```

---

## 13. References

```
[1] Zhu et al. "METER: A Dynamic Concept Adaptation Framework for
    Online Anomaly Detection." VLDB 2024.
    arXiv:2312.16831

[2] Ding & Fei. "An Anomaly Detection Approach Based on Isolation
    Forest Algorithm for Streaming Data Using Sliding Window."
    BIBM 2013. (IForestASD)

[3] Bhatia et al. "MemStream: Memory-Based Streaming Anomaly Detection."
    WWW 2022.

[4] Ting et al. "Half-Space Trees: Fast Streaming Anomaly Detection."
    ICDM 2013. (sHST-River)

[5] Demšar. "Statistical Comparisons of Classifiers over Multiple Data Sets."
    JMLR 2006.
```

---

*Document Version: Benchmark Protocol v3.3*
*Last Updated: 2026-05-11*
*Status: Active*
