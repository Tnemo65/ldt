# Data Engineer Review - CA-DQStream + MemStream Hybrid v4

**Reviewer:** Data Engineer (ETL, Stream Processing, Data Quality Specialist)  
**Date:** 2026-05-12  
**Plan Version:** v4  
**Focus:** ETL Pipeline Design, Schema & Data Model, Data Quality, Stream Processing

---

## Summary

The plan demonstrates solid data engineering fundamentals with well-structured ETL pipelines, proper schema definitions, and thoughtful data quality controls. However, there are **2 CRITICAL issues**, **4 HIGH issues**, and **5 MEDIUM issues** that require attention before production deployment. The most concerning issues are in the data ingestion layer (Kafka partitioning), feature engineering (normalization leakage), and anomaly injection methodology.

**Overall Assessment:** The architecture is sound, but data engineering specifics need refinement to ensure data integrity, scalability, and correctness in production.

---

## Data Pipeline Analysis

### 1. Ingestion Layer

#### Kafka Topics & Consumer Groups

| Component | Design | Assessment |
|-----------|--------|------------|
| **Topic Structure** | Implicit in plan | ⚠️ Not explicitly defined |
| **Partitioning Key** | `neighborhood` (from KeyedProcessFunction) | ✅ Correct for locality |
| **Consumer Groups** | One per operator | ⚠️ Need explicit definition |
| **Offset Management** | Flink managed | ✅ Standard |

**Issue (HIGH):** The plan does not define Kafka topic structure explicitly. Need to clarify:
- Number of partitions (recommend 6-12 for neighborhood-level parallelism)
- Replication factor (3 for production)
- Retention policy (at least 7 days for replay capability)
- Topic naming convention (e.g., `cadqstream.trips.raw`, `cadqstream.scores.output`)

#### Data Flow

```
Kafka Topic: cadqstream.trips.raw
    │
    ├── Partition 0 (Manhattan) ──► MemStreamScoringOperator (KeyedProcessFunction)
    ├── Partition 1 (Brooklyn)  ──► MemStreamScoringOperator
    ├── Partition 2 (Queens)     ──► MemStreamScoringOperator
    └── Partition N (other)     ──► MemStreamScoringOperator

Output: cadqstream.scores.output (anomaly scores + metadata)
```

**Issue (MEDIUM):** Event ordering guarantees not specified. For taxi trip data:
- **Within a neighborhood:** Order matters for memory updates
- **Recommendation:** Partition by `(neighborhood, hour_bucket)` composite key

### 2. Feature Engineering

#### 25D Feature Vector Correctness

| Feature | Name | Type | Range | Assessment |
|---------|------|------|-------|------------|
| 1-5 | Raw features | Continuous | Varied | ✅ |
| 6-9 | Derived features | Continuous | Varied | ✅ |
| 10-13 | Circular encoding | Periodic | [-1, 1] | ✅ |
| 14 | Weekend flag | Binary | {0, 1} | ✅ |
| 15 | Month | Discrete | [1, 12] | ⚠️ Consider cyclical |
| 16-21 | Ratio features | Continuous | Varied | ✅ |
| 22-25 | Temporal flags | Binary | {0, 1} | ✅ |

**Issue (MEDIUM):** `month` feature (index 14) is not circularly encoded like hour/day-of-week. Recommendation: Apply sin/cos encoding for month as well:

```python
month_sin = np.sin(2 * np.pi * month / 12).astype(self.dtype)
month_cos = np.cos(2 * np.pi * month / 12).astype(self.dtype)
```

#### Circular Encoding Validation

```python
# Hour encoding: period = 24
hour_sin = np.sin(2 * np.pi * hour / 24)  # ✅ Correct
hour_cos = np.cos(2 * np.pi * hour / 24)  # ✅ Correct

# Day-of-week encoding: period = 7
dow_sin = np.sin(2 * np.pi * dow / 7)  # ✅ Correct
dow_cos = np.cos(2 * np.pi * dow / 7)  # ✅ Correct
```

**Assessment:** Circular encoding is correctly implemented for hour and day-of-week. ✅

#### Normalization Strategy

**Issue (CRITICAL - Data Leakage):**

The plan shows normalization stats computed from `train_data` in `warmup()`:

```python
# Line 653-655 in memstream_core.py
self.mem_data = torch.from_numpy(train_data[:self.memory_len]).float().to(self.device)
self.mean, self.std = self.mem_data.mean(0), self.mem_data.std(0)
```

However, `train_data` includes data used for memory initialization AND autoencoder training. This creates a subtle leakage: the AE is trained on data whose statistics inform the normalization applied during its own training.

**Impact:** The AE sees "clean" normalized data during warmup, but during streaming, new data normalized with these stats may have different distribution characteristics.

**Recommendation:** Split warmup data into:
1. Normalization statistics computation (first 10%)
2. Autoencoder training (middle 80%)
3. Memory initialization (last 10%)

Or use running statistics (EMA) as noted in config:
```python
ema_alpha: float = 0.01  # Welford EMA for running stats
```

### 3. Data Quality Controls

#### Validation Splits (Train/Val/Test/Calibration)

The plan defines:
```
60% warmup (training)
20% calibration (beta tuning)
20% test (evaluation)
```

**Issue (HIGH):** The temporal order of data is critical for streaming systems. Using `df.sample(frac=1, random_state=42)` shuffles the data, destroying temporal structure.

**For streaming anomaly detection:**
- **Train/Calibrate:** Use earlier time periods
- **Test:** Use later time periods (mimics production)

**Recommended Split:**
```
Time-ordered data (NOT shuffled):
├── 60% warmup (Month 1-6)      → AE training + memory init
├── 20% calibration (Month 7-9) → Beta calibration
└── 20% test (Month 10-12)      → Evaluation
```

**Code Fix:**
```python
# WRONG (current plan)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# CORRECT
df = df.sort_values('tpep_pickup_datetime').reset_index(drop=True)
n = len(df)
train_end = int(n * 0.6)
calib_end = int(n * 0.8)
```

#### Calibration Data Handling

**Good Practices Observed:**
- ✅ Separate calibration set (20%) not used in training
- ✅ Beta calibrated on held-out data
- ✅ ECE/MCE metrics for calibration quality

**Missing:**
- ⚠️ No explicit handling of concept drift between calibration and test periods
- ⚠️ No recalibration strategy defined for production

**Recommendation:** Implement rolling calibration window:
```python
# Recalibrate beta every N records or T time window
# Use IEC-triggered recalibration for drift events
```

### 4. Stream Processing

#### Kafka Topic Structure (Not Explicitly Defined)

**Missing Definition:**
```yaml
Topics:
  - cadqstream.trips.raw:
      partitions: 12
      replication_factor: 3
      retention_ms: 604800000  # 7 days
      config:
        cleanup.policy: delete
        min.insync.replicas: 2
        
  - cadqstream.scores.output:
      partitions: 12
      replication_factor: 3
      retention_ms: 2592000000  # 30 days for analytics
      
  - cadqstream.iec.control:
      partitions: 6
      retention_ms: 86400000  # 1 day
```

#### Event Ordering Guarantees

**Issue (HIGH):** The plan uses `KeyedProcessFunction` keyed by `neighborhood`, but:

1. **Within-key ordering:** Flink guarantees processing order within a key partition
2. **Across-keys ordering:** NOT guaranteed (natural for parallel processing)

**For taxi trip anomaly detection:**
- Trips within same neighborhood should be processed in pickup time order
- Current design guarantees this ✅

**Edge Case:** Late arrivals (out-of-order events)
- **Recommendation:** Implement watermark strategy with allowed lateness
```python
# In Flink job configuration
env.get_config().set_auto_watermark_interval(1000)  # 1 second watermark
.allowed_lateness(Time.minutes(5))  # 5-minute grace period
```

#### Late Data Handling

**Issue (MEDIUM):** Late data handling not explicitly addressed.

**Recommendation:**
```python
class LateDataHandler:
    """
    Handle late arrivals for taxi trip scoring.
    
    NYC TLC data can have late submissions up to 24 hours.
    """
    
    LATE_THRESHOLD_HOURS = 6
    
    def is_late(self, record_timestamp: datetime, watermark: datetime) -> bool:
        return record_timestamp < watermark - timedelta(hours=self.LATE_THRESHOLD_HOURS)
    
    def handle_late(self, record, context):
        # Option 1: Side output to dead letter queue
        # Option 2: Process anyway with warning
        # Option 3: Skip (not recommended for billing data)
        context.output(LATE_TAG, record)
```

### 5. Data Quality Validation

#### Anomaly Injection Methodology

**Issue (HIGH):** The anomaly injection in `inject_anomalies()` modifies data in-place:

```python
df_anom['fare_amount'] = df_anom['trip_distance'] * 2.5 * rng.uniform(...)
```

This creates unrealistic patterns:
1. All injected anomalies have `fare_amount = trip_distance * 2.5 * factor`
2. This is deterministic and learnable
3. Real anomalies have more diverse patterns

**Recommendation:** Inject anomalies with multiple strategies:

```python
def inject_anomalies_realistic(df, n_anomalies, seed, difficulty):
    """More realistic anomaly injection."""
    strategies = ['swap_location', 'speed_extreme', 'fare_ratio', 
                  'duration_anomaly', 'passenger_count']
    
    for idx in anomaly_indices:
        strategy = rng.choice(strategies)
        if strategy == 'swap_location':
            # Swap pickup/dropoff locations
            pass
        elif strategy == 'speed_extreme':
            # Unrealistic speed
            pass
        # ...
```

#### Validation Checks

| Check | Status | Implementation |
|-------|--------|----------------|
| Schema validation | ⚠️ Partial | FeatureVectorizer handles None/NaN |
| Data type enforcement | ✅ Good | float32 throughout |
| Range validation | ⚠️ Basic | Clamping implemented |
| Uniqueness (trip IDs) | ❌ Missing | No duplicate detection |
| Temporal consistency | ⚠️ Partial | parse_datetime with fallbacks |
| Referential integrity | ❌ Missing | No LocationID validation |

**Missing Validation Tests:**
```python
def test_schema_validation():
    """Verify all required fields present."""
    required_fields = ['trip_distance', 'dur_min', 'fare_amount', 
                       'passenger_count', 'tpep_pickup_datetime', 'PULocationID']
    for field in required_fields:
        assert field in record

def test_location_id_valid():
    """NYC TLC zone IDs are 1-263."""
    assert 1 <= record['PULocationID'] <= 263
    assert 1 <= record['DOLocationID'] <= 263

def test_temporal_consistency():
    """Dropoff after pickup."""
    assert record['tpep_dropoff_datetime'] > record['tpep_pickup_datetime']
    assert record['dur_min'] > 0
```

---

## Issues Found

### CRITICAL Issues

| ID | Issue | Location | Impact | Fix Required |
|----|-------|----------|--------|--------------|
| C1 | **Temporal shuffle destroys order** | `train_warmup.py` line 1448 | Concept drift evaluation invalid | Use time-ordered splits, not random shuffle |
| C2 | **Normalization leakage in AE training** | `memstream_core.py` lines 653-655 | Overfitted normalization | Split warmup data: stats vs training |

### HIGH Issues

| ID | Issue | Location | Impact | Fix Required |
|----|-------|----------|--------|--------------|
| H1 | **Kafka topic structure undefined** | Ingestion layer | Deployment ambiguity | Define topic configs explicitly |
| H2 | **Event ordering semantics unclear** | Stream processing | Data consistency risk | Document partitioning strategy |
| H3 | **Late data handling missing** | Flink operator | Lost/invalid scores | Implement watermark + side outputs |
| H4 | **Anomaly injection unrealistic** | `inject_anomalies()` | Benchmark validity | Multi-strategy injection |

### MEDIUM Issues

| ID | Issue | Location | Impact | Fix Required |
|----|-------|----------|--------|--------------|
| M1 | Month not circularly encoded | `feature_extractor.py` line 322 | Inconsistent periodicity | Add month_sin/month_cos |
| M2 | No duplicate trip ID detection | Data validation | Data integrity | Add uniqueness check |
| M3 | Location ID range not validated | Data validation | Bad data propagation | Add zone ID range check |
| M4 | Recalibration strategy missing | Calibration | Stale thresholds | Rolling window recalibration |
| M5 | Checkpoint data retention undefined | Deployment | Recovery risk | Define retention policy |

### LOW Issues

| ID | Issue | Location | Impact | Fix Required |
|----|-------|----------|--------|--------------|
| L1 | No data lineage tracking | Pipeline observability | Debugging difficulty | Add trace IDs |
| L2 | Missing data freshness metrics | Monitoring | SLO compliance | Track ingestion lag |
| L3 | No backfill strategy | Deployment | Historical analysis gap | Document procedure |

---

## Recommendations

### Priority 1: Fix Data Split (CRITICAL)

```python
# In train_warmup.py - Replace lines 1447-1455
def prepare_data_splits(df: pd.DataFrame):
    """
    Prepare temporal splits for streaming anomaly detection.
    
    CRITICAL: Use time-ordered splits, NOT random shuffle.
    This mimics production where we train on past, predict future.
    """
    # Sort by time FIRST
    df = df.sort_values('tpep_pickup_datetime').reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * 0.6)
    calib_end = int(n * 0.8)
    
    return {
        'warmup': df.iloc[:train_end],
        'calibration': df.iloc[train_end:calib_end],
        'test': df.iloc[calib_end:],
    }
```

### Priority 2: Fix Normalization Leakage (CRITICAL)

```python
# In memstream_core.py - warmup() method
def warmup(self, normal_data: np.ndarray, epochs: int = 5000):
    n = len(normal_data)
    
    # 1. Compute stats from FIRST portion (not used in training)
    stats_end = int(n * 0.1)
    stats_data = normal_data[:stats_end]
    self.mean = torch.from_numpy(stats_data.mean(axis=0)).float()
    self.std = torch.from_numpy(stats_data.std(axis=0)).float()
    self.std = torch.clamp(self.std, min=1e-8)
    
    # 2. Training data: middle portion
    train_start = stats_end
    train_end = int(n * 0.9)
    train_data = normal_data[train_start:train_end]
    
    # 3. Memory init: last portion (normalized with computed stats)
    memory_data = normal_data[train_end:]
    memory_tensor = torch.from_numpy(memory_data).float()
    memory_normalized = self._normalize(memory_tensor)
    self.mem_data = memory_tensor
    self.memory = self.encoder(memory_normalized)
```

### Priority 3: Define Kafka Infrastructure

```yaml
# docker-compose.yml additions
kafka:
  image: confluentinc/cp-kafka:7.5.0
  environment:
    KAFKA_NUM_PARTITIONS: 12
    KAFKA_DEFAULT_REPLICATION_FACTOR: 3
    KAFKA_MIN_INSYNC_REPLICAS: 2
    KAFKA_RETENTION_MS: 604800000  # 7 days
    
topics:
  - name: cadqstream.trips.raw
    partitions: 12
    replication_factor: 3
    retention_hours: 168  # 7 days
    
  - name: cadqstream.scores.output
    partitions: 12
    replication_factor: 3
    retention_hours: 720  # 30 days
    
  - name: cadqstream.iec.control
    partitions: 6
    replication_factor: 3
    retention_hours: 24
```

### Priority 4: Implement Late Data Handling

```python
# In memstream_scoring_op.py
class LateDataHandler:
    LATE_THRESHOLD = timedelta(hours=6)
    
    @staticmethod
    def handle(record, context, late_tag):
        """Emit late data to side output."""
        context.output(
            late_tag,
            {
                **record,
                'late_reason': 'watermark_exceeded',
                'late_timestamp': datetime.now().isoformat(),
            }
        )

# In Flink job setup
late_records = stream \
    .filter(lambda r: is_late(r)) \
    .process(LateDataHandler()) \
    .get_side_output(late_tag)
    
late_records.add_sink(kafka_producer('cadqstream.trips.late'))
```

### Priority 5: Enhance Anomaly Injection

```python
def inject_anomalies_realistic(df: pd.DataFrame, n: int, 
                                 difficulty: str = 'medium') -> pd.DataFrame:
    """Multi-strategy anomaly injection."""
    df_anom = df.sample(n=n, random_state=42).copy()
    
    strategies = {
        'speed': inject_speed_anomaly,
        'fare': inject_fare_anomaly,
        'location': inject_location_anomaly,
        'temporal': inject_temporal_anomaly,
    }
    
    for idx in df_anom.index:
        strategy = np.random.choice(list(strategies.keys()))
        df_anom.loc[idx] = strategies[strategy](df_anom.loc[idx], difficulty)
    
    return pd.concat([df, df_anom])
```

---

## Verification Checklist

- [ ] **Schema validated:** All 25 features correctly defined and typed
- [ ] **Data splits verified:** Time-ordered splits implemented (not shuffled)
- [ ] **Feature consistency checked:** Canonical FeatureVectorizer used everywhere
- [ ] **Normalization leakage fixed:** Separate stats from training data
- [ ] **Kafka topics defined:** Partition count, retention, replication documented
- [ ] **Event ordering documented:** Partitioning key and guarantees specified
- [ ] **Late data handling implemented:** Watermarks and side outputs
- [ ] **Anomaly injection validated:** Multi-strategy injection replaces deterministic
- [ ] **Data quality tests added:** Schema, range, uniqueness validation
- [ ] **Calibration strategy defined:** Rolling window recalibration planned
- [ ] **Checkpoint retention policy:** Recovery time objective documented

---

## Appendix: Data Engineering Metrics to Monitor

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| Kafka Consumer Lag | < 1000 records | > 10,000 records |
| Ingestion Latency (p99) | < 500ms | > 2000ms |
| Data Freshness | < 1 minute | > 5 minutes |
| Null Rate per Feature | < 1% | > 5% |
| Duplicate Rate | < 0.1% | > 1% |
| Schema Validation Failures | < 0.01% | > 0.1% |
| Late Data Rate | < 5% | > 15% |

---

**Reviewer Sign-off:** Data Engineer  
**Date:** 2026-05-12  
**Next Review:** After fixes applied (target: REVISION_1)
