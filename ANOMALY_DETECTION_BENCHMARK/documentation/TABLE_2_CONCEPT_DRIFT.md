# Table 2: Concept Drift Detection Models

**Purpose**: Detect changes in the underlying data distribution over time

**Our Task**: ❌ These were INCORRECTLY used for anomaly detection

---

## Results Summary (2 Models)

| Model | AUC (Anomaly) | Category | Works On | Designed For |
|-------|---------------|----------|----------|--------------|
| EWMA (α=0.3) | 0.7954 ⚠️ | Window-based | **Raw Features** (trip_count) | Distribution shifts |
| CUSUM | 0.4727 ❌ | Window-based | **Raw Features** (trip_count) | Mean shifts |

---

## Why They Performed Poorly on Anomaly Detection

### ❌ CUSUM (0.4727 AUC) - Worse than Random!

**What it's designed for:**
```
Detect sustained mean shift:
Before: [100, 102, 98, 101, 99, 103] → mean ≈ 100
After:  [150, 152, 148, 151, 149, 153] → mean ≈ 150
         ↑ CUSUM detects this shift
```

**What we asked it to do (anomaly detection):**
```
Normal: [100, 102, 98, 101, 99, 103, 97, 100]
Anomaly: [100, 102, 98, 500, 99, 103, 97, 100]  ← Detect this spike
         ↑ CUSUM FAILS - it accumulates and expects sustained change
```

**Why it failed:**
- CUSUM accumulates deviations: `cusum_t = max(0, cusum_{t-1} + (x_t - target) - drift)`
- A single spike gets diluted in the cumulative sum
- It needs **multiple consecutive deviations** to trigger
- Point anomalies don't persist long enough

**Example:**
```python
# CUSUM on point anomaly
target = 100, drift = 0.5
x = [100, 500, 100, 100, 100]  # One spike

cusum = [0, 399.5, 499, 598.5, 698]  # Keeps growing!
# CUSUM never resets because it accumulates
# But the anomaly was only at t=1
```

---

### ⚠️ EWMA (0.7954 AUC) - Moderate Performance

**What it's designed for:**
```
Detect gradual trend:
Data: [100, 105, 110, 115, 120, 125, 130]
      ↑ Traffic is gradually increasing
```

**What we asked it to do:**
```
Normal: [100, 102, 98, 101, 99, 103]
Anomaly: [100, 102, 500, 101, 99, 103]  ← Detect spike
```

**Why it performed moderately (0.7954):**
- EWMA tracks exponentially weighted average
- It CAN detect large spikes because they deviate from the moving average
- BUT it's not optimal because:
  1. The moving average "chases" the anomaly
  2. It can't distinguish between drift and spike
  3. Tuning α is tricky (too small = slow, too large = unstable)

**Example:**
```python
# EWMA on point anomaly
α = 0.3
x = [100, 100, 500, 100, 100]

ewma = [100, 100, 220, 184, 158.8]
score = |x - ewma| = [0, 0, 280, -84, -58.8]
# Spike detected, but EWMA "polluted" for several windows after
```

---

## How Concept Drift Detection Actually Works

### 1. EWMA (Exponentially Weighted Moving Average)

**Formula:**
```python
ewma_t = α × x_t + (1-α) × ewma_{t-1}
anomaly_score = |x_t - ewma_t|
```

**Parameters:**
- `α ∈ [0,1]`: Smoothing factor
  - Small α (e.g., 0.1): Slow adaptation, detects gradual drift
  - Large α (e.g., 0.9): Fast adaptation, detects sudden shifts

**Works on**: Raw feature values (e.g., `trip_count`)

**Best for:**
- Detecting **gradual trends** (traffic increasing over weeks)
- Detecting **seasonal shifts** (summer vs winter patterns)
- **Process control** (manufacturing, SLA monitoring)

**Example Use Case:**
```python
# Monitor daily average trip count
daily_avg = [1000, 1020, 1050, 1080, 1120, 1150]  # Growing trend
ewma = [1000, 1006, 1019, 1037, 1062, 1088]
# EWMA detects the upward trend
```

---

### 2. CUSUM (Cumulative Sum)

**Formula:**
```python
cusum_pos_t = max(0, cusum_pos_{t-1} + (x_t - target) - drift)
cusum_neg_t = max(0, cusum_neg_{t-1} + (target - x_t) - drift)
alert if cusum_pos_t > threshold or cusum_neg_t > threshold
```

**Parameters:**
- `target`: Expected mean (e.g., 100 trips/hour)
- `drift`: Allowable drift before accumulating (e.g., 0.5)
- `threshold`: Alert threshold (e.g., 5.0)

**Works on**: Raw feature values (e.g., `trip_count`)

**Best for:**
- Detecting **mean shifts** (average trips changed from 100 to 110)
- **Quality control** (manufacturing defect rate increased)
- **SLA monitoring** (latency increased from 10ms to 15ms)

**Example Use Case:**
```python
# Manufacturing process - detect if mean shifts from 100 to 105
target = 100
x = [100, 101, 99, 105, 106, 104, 105, 107]  # Shift at t=3
#        ↑ normal    ↑ shifted mean

cusum_pos = [0, 0.5, 0, 4.5, 10, 13.5, 18, 24.5]
# Alert at t=4 when cusum > threshold (e.g., 5)
```

---

## Correct Use of EWMA/CUSUM

### ✅ For Concept Drift Detection

**Scenario 1: Detect Traffic Pattern Change**
```python
# Monitor daily average trip count
df_daily = df.groupby(df['window_start'].dt.date)['trip_count'].mean()

# Apply EWMA
ewma = df_daily.ewm(alpha=0.3).mean()
drift_score = abs(df_daily - ewma)

# Alert if drift_score > threshold for N consecutive days
```

**Scenario 2: Detect Mean Shift in Hourly Traffic**
```python
# Monitor hourly average for 9am slot
df_9am = df[df['hour'] == 9]['trip_count']

# Apply CUSUM
target = df_9am.mean()  # Historical mean
cusum = compute_cusum(df_9am, target, drift=0.5, threshold=5.0)

# Alert when cusum crosses threshold
```

### ❌ For Anomaly Detection (What We Did)

**What we did wrong:**
```python
# We computed EWMA/CUSUM on every 30-min window
ewma = df['trip_count'].ewm(alpha=0.3).mean()
df['ewma_score'] = abs(df['trip_count'] - ewma)

# Then evaluated as if it's anomaly detection
roc_auc_score(df['is_anomaly'], df['ewma_score'])  # Wrong!
```

**Why it's wrong:**
- EWMA/CUSUM are designed for **sequential monitoring**, not **retrospective scoring**
- They accumulate state over time (not independent per-window)
- They detect **distribution changes**, not **individual outliers**

---

## RRCF: The Hybrid Model (SKIPPED)

### Can Do BOTH Anomaly + Drift

**RRCF (Robust Random Cut Forest)**
- Works on: Raw features or feature vectors
- Method: CoDisp (Collusive Displacement) in random cut trees
- **Detects anomalies**: Single unusual points (high CoDisp)
- **Detects drift**: Sustained pattern changes (consistent CoDisp shift)

**Why we skipped it:**
- Too slow: ~3 hours for 17,529 samples
- Requires inserting/removing each test point into forest

**When to use:**
- Real-time streaming data (AWS Kinesis, Kafka)
- Need both anomaly and drift detection
- Can afford online computation cost

---

## Comparison: Anomaly vs Drift Detection

| Aspect | Anomaly Detection | Concept Drift Detection |
|--------|-------------------|-------------------------|
| **Detects** | Individual unusual points | Distribution changes |
| **Example** | "500 trips at 3am" (spike) | "Average shifted from 100 to 110" |
| **Timeframe** | Single data point | Multiple points / time window |
| **Methods** | Z-Score, IForest, VAE | EWMA, CUSUM, KS-test |
| **Works on** | Feature vectors (multi-dim) | Feature values (single metric) |
| **Output** | Anomaly score per point | Drift magnitude over time |
| **Evaluation** | AUC-ROC, Precision, Recall | Detection delay, false alarm rate |

---

## When to Use Each Type

### Use Anomaly Detection When:
- ✅ Detecting **individual outliers** (spikes, drops, unusual patterns)
- ✅ Real-time alerting on **single data points**
- ✅ Classification task: "Is this point normal or anomalous?"
- **Examples**: Fraud detection, sensor failure, network intrusion

### Use Concept Drift Detection When:
- ✅ Monitoring **distribution changes** over time
- ✅ Detecting **sustained shifts** in mean, variance, or patterns
- ✅ Process control: "Has the system behavior changed?"
- **Examples**: A/B test impact, seasonal changes, model degradation

### Use Both (Hybrid) When:
- ✅ Streaming data with both outliers and distribution shifts
- ✅ Need to distinguish "spike" vs "new normal"
- **Example**: Traffic monitoring (detect accidents + pattern changes)

---

## Recommendations for NYC Taxi Data

### ✅ For Anomaly Detection (Current Task):
Use **Table 1 models** (Z-Score, IQR, HBOS, VAE, IForest)

### ✅ For Concept Drift Detection (Future Work):
Apply EWMA/CUSUM to **aggregated metrics**:

```python
# 1. Compute daily statistics
df_daily = df.groupby(df['window_start'].dt.date).agg({
    'trip_count': ['mean', 'std', 'max']
})

# 2. Apply EWMA to detect daily trend changes
ewma_daily = df_daily['trip_count']['mean'].ewm(alpha=0.3).mean()

# 3. Apply CUSUM to detect mean shift
cusum_daily = compute_cusum(
    df_daily['trip_count']['mean'],
    target=historical_mean,
    drift=5.0,
    threshold=10.0
)
```

**Use cases:**
- "Traffic patterns changed after pandemic"
- "Seasonal shift from summer to winter"
- "New ride-sharing service affected demand"

---

## Key Takeaways

1. ❌ **EWMA/CUSUM are NOT for point anomaly detection**
2. ✅ **They work on raw features (trip_count), not ML model outputs**
3. ⚠️ **CUSUM's 0.47 AUC is EXPECTED - it's designed for drift, not anomalies**
4. 📊 **For drift detection, apply them to aggregated metrics (daily/weekly averages)**
5. 🔄 **RRCF can do both, but is too slow for batch evaluation**
6. 🎯 **Use Table 1 models (Z-Score, HBOS, VAE) for anomaly detection**
