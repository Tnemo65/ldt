# Benchmark: Context-Aware Streaming Anomaly Detection

## 1. Problem

Taxi trip data streams continuously from Kafka. We need to detect fraudulent or anomalous trips in real-time while handling **concept drift** — the normal behavior of "valid trip" changes over time (seasonal patterns, pricing changes, zone popularity shifts).

**Requirements:**
- Process streaming taxi data with sub-second latency
- Detect anomalies without labeled data (unsupervised)
- Adapt to distribution shift without retraining from scratch
- Context-aware thresholds (fare in Manhattan differs from Queens)

## 2. Dataset Setup

### 2.1 Source Data

- **NYC Taxi Trip Record** (2024-2025, clean baseline, no anomalies)
  - Download from NYC TLC: trip records 2024-2025
  - Filter: remove trips with missing values, zero fares, impossible speeds
  - Approximate size: 500M+ records
  - Clean data first → inject anomalies second → evaluate
- Schema: `trip_time`, `trip_distance`, `fare_amount`, `tip_amount`, `pickup_zone`, `dropoff_zone`, tại sao claude luôn luôn hot patch lỗi mà không nhìn vào tổng quát flow dự án để sửa`passenger_count`, `trip_speed`
- Baseline: ~15K-50K clean records for offline evaluation

### 2.2 Training vs Test Split

```
Raw Data (2024-2025, ~500M records)
├── Step 1: Clean & Verify       → Remove invalid records
├── Step 2: Train/Test Split
│   ├── Warmup Set (500K)        → Train initial model, build context cells
│   └── Test Set (1M)            → Inject anomalies, evaluate
└── Streaming Window             → Online evaluation (future work)
```

### 2.3 Feature Engineering

21D feature vector per trip:

| Feature | Type | Description |
|---------|------|-------------|
| trip_time (sec) | Numeric | Total trip duration |
| trip_distance (mi) | Numeric | Distance traveled |
| fare_amount | Numeric | Base fare + extras |
| tip_amount | Numeric | Tip given |
| trip_speed | Numeric | distance / time |
| hour_of_day | Cyclic | Hour (0-23), sin/cos encoded |
| day_of_week | Cyclic | Day (0-6), sin/cos encoded |
| is_weekend | Binary | Saturday/Sunday flag |
| passenger_count | Numeric | Number of passengers |
| zone_pair | Categorical | pickup_zone × dropoff_zone |
| zone_cluster | Categorical | Neighborhood group |
| amount_per_mile | Numeric | fare / distance |
| amount_per_minute | Numeric | fare / time |
| tip_pct | Numeric | tip / fare |
| tip_pct_bucket | Categorical | Quantized tip percentage |
| zone_popularity | Numeric | Historical zone frequency |
| distance_ratio | Numeric | Actual vs expected distance |
| time_ratio | Numeric | Actual vs expected time |
| is_airport | Binary | JFK/LaGuardia pickup/dropoff |
| payment_type | Categorical | Cash vs card |

## 3. Data Injection (Ground Truth Creation)

Since real anomaly labels are unavailable, we inject synthetic anomalies with **known ground truth**.

### 3.1 Anomaly Types

| ID | Type | How It Works | Example |
|----|------|-------------|---------|
| A1 | **Fare Spike** | Multiply `fare_amount` by 3x-10x | $15 → $75 |
| A2 | **Zone Shift** | Swap `dropoff_zone` to geographically inconsistent location | Bronx → Midtown |
| A3 | **Speed Anomaly** | `trip_speed` outside realistic range | 80 mph in Manhattan |
| A4 | **Tip Manipulation** | `tip_amount` inflated or deflated by ±30% | $3 → $9 |
| A5 | **Short Trip Overcharge** | High fare for very short distance | $30 for 0.5 mi |
| A6 | **Long Trip Undercharge** | Low fare for very long distance | $5 for 20 mi |
| A7 | **Context Drift** | Gradual shift in feature distributions (e.g., avg fare increases 20%) | Seasonal pattern |

### 3.2 Injection Parameters

```python
anomaly_rate = 0.05        # 5% of test set are anomalies
severity_distribution = {
    "extreme": 0.30,        # A1, A2 — clearly wrong
    "moderate": 0.40,       # A3, A5, A6 — detectable
    "subtle": 0.30,        # A4, A7 — requires context awareness
}
```

### 3.3 Ground Truth Format

Each injected record gets a label:

```python
record = {
    "features": [...],      # 21D vector
    "is_anomaly": True,    # or False
    "anomaly_type": "A1",  # which injection type
    "ground_truth_score": 1.0,  # for PR curve computation
}
```

## 4. Difficulty Stratification

Anomalies are bucketed by how hard they are to detect:

### 4.1 Bucket Definitions

| Bucket | Criteria | Expected Anomaly Types |
|--------|----------|----------------------|
| **Easy** | Obvious outlier, deviates from all normal patterns | A1 (fare spike 10x), A2 (zone impossible), A3 (speed > 80 mph) |
| **Medium** | Deviates from global patterns but not context | A5 (short trip overcharge), A6 (long trip undercharge) |
| **Hard** | Only detectable with context-aware thresholds | A4 (tip manipulation), A7 (context drift) |

### 4.2 Context Definition

Context is defined by: `{hour_bucket} × {zone_cluster} × {day_type}`

Example contexts:
- "Friday 8PM, Manhattan, weekend" → fare $25, tip 18%
- "Tuesday 3PM, Queens, weekday" → fare $18, tip 12%

A tip of $3 might be normal in one context but suspicious in another.

### 4.3 Stratification Metric

```python
def classify_difficulty(record, context_threshold):
    score = anomaly_model.score(record)
    deviation = abs(score - context_threshold[record.context])

    if deviation > 0.8: return "easy"
    elif deviation > 0.4: return "medium"
    else: return "hard"
```

## 5. Model Baselines to Compare

### 5.1 Streaming Methods (Main Competitors)

| Model | Library | Description |
|-------|---------|-------------|
| **MemStream** | Custom (PyTorch) | Main target. Streams, context-aware, drift-adaptive. |
| **CADIFEia** | river | CUSUM + ADWIN drift detection + Isolation Forest |
| **IForestASD** | river | Streaming Isolation Forest with ADWIN |
| **sHST-River** | river | Self-Training Half-Space Trees |

### 5.2 Batch Methods (Reference Baselines)

| Model | Library | Description |
|-------|---------|-------------|
| **sklearn IF** | sklearn | Batch Isolation Forest, retrain periodically |
| **sklearn LOF** | sklearn | Local Outlier Factor (density-based) |
| **HBOS** | PyOD | Histogram-based Outlier Detection |
| **OCSVM** | sklearn | One-Class SVM |

### 5.3 Expected Rankings

```
AUC-ROC:  All methods should perform well (easy cases dominate)
AUC-PR:   Streaming methods (MemStream, CADIFEia) >> Batch methods
          (because AUC-PR penalizes false positives on imbalanced data)
Hard Cases: Only MemStream (context-aware) should significantly beat random
Latency:   Streaming methods (1-5ms) << Batch methods (50-500ms)
```

## 6. Questions to Answer

### Q1. Does MemStream beat batch methods on streaming data?
**Metric:** AUC-PR on streaming window, not just batch AUC-ROC.
**Hypothesis:** MemStream > sklearn IF > HBOS on AUC-PR due to lower false positive rate.

### Q2. How much does context-awareness help over global thresholds?
**Metric:** AUC-PR gap between MemStream (context) vs MemStream (global threshold).
**Hypothesis:** Context-aware thresholds improve AUC-PR by 20-50% on hard cases.

### Q3. How fast does MemStream adapt to concept drift?
**Metric:** Time-to-detect after drift injection.
**Hypothesis:** MemStream adapts within 100K records; batch methods require full retrain.

### Q4. What warmup size is needed?
**Metric:** Coverage of context cells (must reach >90% to be stable).
**Hypothesis:** 500K warmup covers 90%+ of cells.

### Q5. Which hyperparameters matter most?
**Metric:** Sensitivity analysis (ranked by impact on AUC-PR).
**Hyperparameters:** memory_len, k_neighbors, gamma, adwin_delta, warmup_epochs.

### Q6. What is the operational false positive rate in production?
**Metric:** FP rate on real (unlabeled) streaming data.
**Note:** Cannot compute without human review — estimate from labeled benchmark.

## 7. Experiment Plan for Best Hyperparameters

### 7.1 Phase 1: Individual Sweep (One Factor at a Time)

Run 14 experiments, each varying ONE hyperparameter:

| Exp | Parameter | Grid |
|-----|-----------|------|
| 1 | `memory_len` | 10K, 25K, 50K, 100K |
| 2 | `k_neighbors` | 5, 10, 20, 30, 50 |
| 3 | `gamma` (drift weight) | 0.0, 0.3, 0.5, 0.7, 0.9 |
| 4 | `default_beta` | 0.3, 0.5, 0.7 |
| 5 | `adwin_delta` | 0.001, 0.01, 0.05, 0.1 |
| 6 | `max_window` | 200, 500, 1000, 2000 |
| 7 | `warmup_epochs` | 100, 500, 1000 |
| 8 | `warmup_data_size` | 50K, 100K, 200K, 500K |
| 9 | `cell_minimum` | 5, 10, 20, 50 |
| 10 | `recent_scores_buffer` | 100, 500, 1000 |
| 11 | `quick_retrain_samples` | 1000, 5000, 10000 |
| 12 | `quick_retrain_epochs` | 10, 50, 100 |
| 13 | `alert_threshold_pct` | 90, 95, 99 |
| 14 | `kafka_partitions` | 1, 4, 8, 16 |

**Primary metric:** AUC-PR (not AUC-ROC — AUC-PR is more meaningful for imbalanced data).
**Secondary metrics:** AUC-ROC, F1, P@100, latency_p99.

### 7.2 Phase 2: Grid Search on Top Candidates

Take top-3 values from Phase 1 for top-5 parameters → full grid search.

```python
# Example: 3×3×3×3×3 = 243 configurations
grid = {
    "memory_len": [25K, 50K, 100K],
    "k_neighbors": [10, 20, 30],
    "gamma": [0.3, 0.5, 0.7],
    "adwin_delta": [0.01, 0.05],
    "warmup_data_size": [100K, 500K],
}
```

### 7.3 Phase 3: Ablation Study

Remove components one at a time to measure contribution:

| Ablation | Remove | Expected Impact |
|----------|--------|----------------|
| No context | Global thresholds | AUC-PR drops 20-50% on hard cases |
| No ADWIN | Static thresholds | Cannot adapt to drift |
| No retraining | Freeze model after warmup | Degrades over time |
| No neighborhood mapping | Use raw zones | Loses geographic generalization |

### 7.4 Phase 4: Rigorous Multi-Seed Evaluation

Run final model with 3 random seeds × 3 folds × 3 difficulty levels.

```
3 seeds (42, 123, 456)
× 3 folds (data split)
× 3 difficulties (easy, medium, hard)
× 3 budget % (0.1%, 0.5%, 1.0%)
= 81 runs per algorithm
```

Report: mean ± std across seeds/folds. Use statistical tests (Wilcoxon) to compare.

### 7.5 Phase 5: Production Benchmark

Deploy to Flink pipeline with real Kafka data:

| Test | Metric |
|------|--------|
| Throughput | Records/second (target: 10K+ rps) |
| Latency | P50, P95, P99 (target: <10ms P99) |
| Drift adaptation | Time-to-detect after injection |
| Resource usage | CPU, memory, GPU utilization |
| False positive rate | % flagged on clean data |
