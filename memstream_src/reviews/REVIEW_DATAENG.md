# Data Engineer Review — PLAN_v3.md

**Reviewer:** Principal Data Engineer (12 yrs ETL, Spark, dbt, data lakehouse)
**Date:** 2026-05-12
**Files Reviewed:** PLAN_v3.md (§2, §3, §7), `src/features/vectorizer.py`, `src/operators/schema_validator.py`, `src/operators/canary_rules.py`, `src/operators/if_scoring_operator.py`, `src/ml/train_iforest.py`, `scripts/inject_anomalies.py`, `scripts/compute_thresholds.py`, `scripts/fit_scaler.py`, `scripts/sanitize_baseline.py`, `scripts/download_data.py`

---

## 1. Data Quality Validation Pipeline

### 1.1 Schema Validator — Minor Issues

- **Good:** Checks required fields, zone range 1-263, null detection
- **Issue (MEDIUM):** No duration validation. `tpep_dropoff_datetime <= tpep_pickup_datetime` can produce negative durations, causing downstream NaN/Inf
- **Issue (LOW):** `total_amount < fare_amount` not flagged

### 1.2 Canary Rules — Rush Hour Mismatch — CRITICAL

Three different rush hour definitions across files:

| File | Morning Rush | Evening Rush |
|------|------------|-------------|
| `canary_rules.py` | 7–9 (3 hrs) | 16–19 (3 hrs) |
| `if_scoring_operator.py` | 6–10 (4 hrs) | 17–20 (4 hrs) |
| `zone_mapping.py` | 6–10 (4 hrs) | 17–21 (4 hrs) |

Records classified as "rush_hour" in Layer 1 (canary) won't match Layer 3 (scoring) → inconsistent threshold application.

### 1.3 Datetime Parsing — ISO-Only

`canary_rules.py` uses `datetime.fromisoformat()` which only supports ISO 8601. NYC taxi data uses multiple formats.

---

## 2. Feature Engineering — CRITICAL: 21D vs 25D Mismatch

### The Most Critical Issue

| Component | Dimensions | File |
|-----------|-----------|------|
| `FeatureVectorizer.transform()` | **21D** | `src/features/vectorizer.py` |
| `FeatureVectorizer.transform_batch()` | **21D** | `src/features/vectorizer.py` |
| `train_iforest.py` | **21D** | `src/ml/train_iforest.py` |
| **MemStream plan** | **25D** | `PLAN_v3.md` feature_extractor |
| **MemStream config** | `in_dim=25` | `PLAN_v3.md` config.py |
| **MemStream validation** | 25D shape | `PLAN_v3.md` line 686–690 |

MemStream will crash on first record with shape validation error.

### Current 21D Features

```
Raw (5): distance, duration_minutes, fare, passengers, total
Derived (4): speed, fare_per_mile, fare_per_minute, fare_per_passenger
Temporal (6): hour, day_of_week, is_weekend, is_rush_hour, is_night, month
Ratio (6): fare_per_mile_ratio, fare_per_minute_ratio, implied_speed_ratio, passenger_distance_ratio, fare_distance_product, duration_distance_ratio
```

### PLAN_v3 Canonical 25D Features

```
Raw (5): trip_distance, dur_min, fare_amount, passenger_count, total_amount
Derived (4): speed_mph, fare_per_mile, fare_per_min, fare_per_pax
Circular (4): hour_sin, hour_cos, dow_sin, dow_cos
Temporal (6): is_weekend, month, is_rush_hour, is_night, is_early_morning, is_late_night
Ratio (6): norm_fare_per_mile, norm_fare_per_min, norm_speed, pax_per_mile, fare_times_dist, dur_per_dist
```

**Migration requires:**
1. Replace `hour`, `day_of_week` (2 features) with `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos` (4 features)
2. Add `is_early_morning`, `is_late_night` (2 new features)
3. Retrain IsolationForest with 25D
4. Recompute all thresholds with 25D
5. Retrain MemStream with 25D

### Other Vectorizer Issues

- **HIGH:** Returns `np.float64` but PyTorch expects `np.float32`
- **MEDIUM:** eps value inconsistency between vectorizer and PLAN_v3
- **MEDIUM:** Speed calculation uses hours vs minutes mismatch

---

## 3. Baseline Data for Warmup

### 3.1 Sanitization Pipeline

**Good:** 3-step process, speed filter at 100 mph, 3×IQR

**Issues:**
- **HIGH:** No deduplication by trip_id
- **MEDIUM:** No pickup > dropoff validation
- **MEDIUM:** Sequential IQR filter can remove valid records

### 3.2 Scaler Fitting — CRITICAL: iterrows() Performance Bug

```python
# scripts/fit_scaler.py lines 40–45:
for start in range(0, n, batch_size):
    end = min(start + batch_size, n)
    batch = df.iloc[start:end]
    for _, row in batch.iterrows():  # ← SLOWEST POSSIBLE pandas operation
        features_list.append(vec.transform(row.to_dict()))
```

`iterrows()` is 100–1000x slower than vectorized. For 2.4M records, this takes **hours** instead of **minutes**.

**Fix:** Replace with single vectorized call:
```python
X = vectorizer.transform_batch(df)  # One vectorized call
```

---

## 4. Temporal Split Strategy

### 4.1 No Temporal Ordering — HIGH

```python
df = df.sample(frac=1, random_state=42)  # Shuffles all data!
```

Temporal split must use sorted data:
```python
df = df.sort_values('tpep_pickup_datetime').reset_index(drop=True)
train_end = int(len(df) * 0.6)
```

### 4.2 Threshold Computation — Undefined Variable CRITICAL

```python
# scripts/compute_thresholds.py line 138:
'sample_size': n,  # n is NOT defined — NameError at runtime!
```

Also: Time window mismatch vs. canary_rules definitions.

---

## 5. Anomaly Injection

### 5.1 inject_anomalies.py Issues

- **HIGH:** Random seed not propagated to `generate()`
- **HIGH:** `replace=True` sampling duplicates templates
- **MEDIUM:** Duration recalculation doesn't update derived features
- **MEDIUM:** Scenario imbalance for non-divisible `n_anomalies`

### 5.2 PLAN_v3 vs Source Mismatch

**PLAN_v3:** Ratio modification approach with difficulty levels
**Source:** Scenario-based generation

These are incompatible approaches. Tests written against one will fail on the other.

---

## 6. Data Quality Monitoring

### 6.1 No Prometheus Metrics in Source

`src/operators/*.py` have zero Prometheus instrumentation.

### 6.2 SLO Monitoring

`config.py` defines SLOs but no Prometheus recording rules exist to measure them.

---

## 7. CRITICAL Issues

| # | File | Line | Issue | Fix |
|---|------|------|-------|-----|
| 1 | `src/features/vectorizer.py` | 82 | 21D → must be 25D for MemStream | Add circular encoding + 2 new temporal flags |
| 2 | `scripts/fit_scaler.py` | 40–45 | `iterrows()` → hours of runtime | Replace with `transform_batch()` |
| 3 | `scripts/compute_thresholds.py` | 138 | `n` undefined → NameError | Change to `sample_size` |
| 4 | `src/operators/canary_rules.py` | 80, 143 | Rush hour mismatch (3 defs) | Standardize via `zone_mapping.py` |

---

## 8. HIGH Issues

| # | File | Issue | Fix |
|---|------|-------|-----|
| 1 | `src/features/vectorizer.py` | float64 vs float32 | Change to `np.float32` |
| 2 | `src/operators/canary_rules.py` | `fromisoformat()` only | Use robust multi-format `parse_datetime()` |
| 3 | `src/operators/if_scoring_operator.py` | No null handling | Add `np.nan_to_num()` before scoring |
| 4 | `scripts/inject_anomalies.py` | Seed not propagated | Add explicit `np.random.seed(42)` in `generate()` |
| 5 | `src/ml/train_iforest.py` | Validates on same data | Add held-out validation set |

---

## 9. MEDIUM/LOW Issues

| # | File | Issue |
|---|------|-------|
| 1 | `scripts/sanitize_baseline.py` | No dedup by trip_id |
| 2 | `scripts/sanitize_baseline.py` | No pickup > dropoff validation |
| 3 | `scripts/inject_anomalies.py` | `replace=True` sampling |
| 4 | `src/operators/schema_validator.py` | No `total_amount < fare_amount` check |
| 5 | `src/operators/canary_rules.py` | No speed > 100 mph check |

---

## 10. Priority Fixes

### Fix 1: Standardize Rush Hour (single source of truth)

```python
# src/zone_mapping.py — define once, import everywhere
RUSH_HOURS_MORNING = (6, 10)   # 06:00–09:59
RUSH_HOURS_EVENING = (17, 21)   # 17:00–20:59
```

### Fix 2: Robust Datetime Parsing

```python
# Shared utility in src/utils/datetime_parser.py
def parse_datetime(value):
    if pd.isna(value):
        return None
    if isinstance(value, str):
        for fmt in ['%Y-%m-%d %H:%M:%S', '%m/%d/%Y %I:%M:%S %p',
                     '%Y-%m-%dT%H:%M:%S', '%m/%d/%Y %H:%M']:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        return None
    return value
```

### Fix 3: Update FeatureVectorizer to 25D

```python
# src/features/vectorizer.py — replace transform() temporal section:

# Circular encoding (4 features) — replaces raw hour, day_of_week
hour = pickup.hour
dow = pickup.weekday()
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
dow_sin = np.sin(2 * np.pi * dow / 7)
dow_cos = np.cos(2 * np.pi * dow / 7)

is_weekend = 1 if dow >= 5 else 0
is_rush_hour = 1 if ((6 <= hour < 10) or (17 <= hour < 21)) else 0
is_night = 1 if (hour < 6 or hour > 22) else 0
is_early_morning = 1 if (5 <= hour < 7) else 0
is_late_night = 1 if (hour >= 23 or hour <= 4) else 0
month = pickup.month

# 25D vector — float32 for PyTorch
features = np.array([
    # Raw (5)
    distance, duration_minutes, fare, passengers, total,
    # Derived (4)
    speed, fare_per_mile, fare_per_minute, fare_per_passenger,
    # Circular encoding (4)
    hour_sin, hour_cos, dow_sin, dow_cos,
    # Temporal flags (6)
    is_weekend, month, is_rush_hour, is_night, is_early_morning, is_late_night,
    # Ratio features (6)
    fare_per_mile_ratio, fare_per_minute_ratio, implied_speed_ratio,
    passenger_distance_ratio, fare_distance_product, duration_distance_ratio,
], dtype=np.float32)
```

---

*Reviewed by: Principal Data Engineer | 2026-05-12*
