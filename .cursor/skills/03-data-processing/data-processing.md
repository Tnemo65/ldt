---
name: data-processing
description: >
  Design, build, and optimize data processing pipelines for research and production.
  Covers ETL/ELT workflows, streaming data processing, data validation, feature
  engineering, and data quality monitoring. Use when designing data pipelines,
  implementing ETL/ELT processes, handling data quality issues, optimizing
  data processing costs, or building streaming data workflows.
tools: Read Write Edit Bash Glob Grep Task AskQuestion WebSearch
model: opus
---

# Data Processing — Data Pipeline Engineering Agent

Designs, builds, and optimizes data processing pipelines for research and production. Covers ETL/ELT workflows, streaming data processing, data validation, feature engineering, and data quality monitoring.

## When to Use

- **Data pipeline design**: Architecting ETL/ELT workflows for research data
- **Streaming data processing**: Apache Kafka, Flink, Spark Streaming pipelines
- **Data quality monitoring**: DQ rule frameworks, anomaly detection, data validation
- **Feature engineering**: Transforming raw data into ML-ready features
- **Data quality issues**: Missing data, schema drift, duplicate detection
- **Pipeline optimization**: Throughput improvement, latency reduction, cost optimization

## Core Capabilities

### 1. Streaming Data Processing

Design Flink, Spark Streaming, or Kafka Streams pipelines:

```python
# Apache Flink streaming pipeline structure
PIPELINE_COMPONENTS = {
    "sources": ["KafkaSource", "FileSource", "PubSubSource"],
    "transforms": ["MapFunction", "FlatMapFunction", "FilterFunction",
                   "KeyedProcessFunction", "WindowFunction"],
    "sinks": ["KafkaSink", "JdbcSink", "PrintSink"],
    "state": ["ValueState", "ListState", "MapState", "BroadcastState"],
    "time": ["EventTime", "ProcessingTime", "IngestionTime"],
    "watermarks": ["FixedWatermarkStrategy", "BoundedOutOfOrderness"]
}
```

**For StreamDQ project** (from `final/` directory):
- Kafka → Flink pipeline with SYN/SEM (Python) and CRS (Java) rule evaluation
- Context-aware thresholds via BroadcastState
- TQS aggregation with tumbling windows
- See `final/05_ALGORITHM_DESIGN/FORMULATION.md` for algorithm specs
- See `final/05_ALGORITHM_DESIGN/DATA_STRUCTURES.md` for Flink state design

### 2. ETL/ELT Pipeline Design

```
ETL: Extract → Transform → Load
  └─► Batch processing (daily/hourly)
  └─► CDC (Change Data Capture)
  └─► Incremental loads

ELT: Extract → Load → Transform
  └─► Data lake patterns
  └─► Columnar storage (Parquet, ORC)
  └─► Modern dbt-based transformations
```

### 3. Data Quality Framework

Design DQ rule frameworks with layered validation:

```
SYN (Syntactic): Record-level checks
  ├─ Null/NaN detection
  ├─ Type validation
  └─ Range checks

SEM (Semantic): Plausibility checks
  ├─ Context-aware thresholds
  ├─ Distribution anomalies
  └─ Cross-field consistency

CRS (Cross-Record): Stateful checks
  ├─ Trajectory validation (GPS)
  ├─ Deduplication
  └─ Temporal consistency
```

### 4. Feature Engineering

Transform raw data into ML-ready features:

```python
FEATURE_TYPES = {
    "numerical": ["min-max scaling", "z-score normalization", "log transform"],
    "categorical": ["one-hot", "target encoding", "ordinal encoding"],
    "temporal": ["lag features", "rolling statistics", "time since event"],
    "text": ["TF-IDF", "embeddings", "token count"],
    "graph": ["node degree", "page rank", "community features"]
}
```

### 5. Data Validation & Monitoring

Implement streaming DQ monitoring:

```python
DQ_METRICS = {
    "completeness": "null_rate, missing_field_rate",
    "validity": "out_of_range_rate, type_mismatch_rate",
    "timeliness": "stale_data_rate, inter_event_variance",
    "consistency": "duplicate_rate, cross_record_violation_rate",
    "uniqueness": "hash_collision_rate, exact_duplicate_rate"
}
```

## Streaming Data Quality Patterns

### Pattern 1: Context-Aware Thresholds

```python
# L0-L5 hierarchical fallback for streaming data
def get_context_threshold(event, broadcast_state):
    for level in [L0, L1, L2, L3, L4]:
        key = compute_context_key(event, level)
        stats = broadcast_state.get(level, key)
        if stats.count >= MIN_SAMPLES[level]:
            return compute_threshold(stats, level)
    return PHYSICS_PRIOR  # L5 fallback
```

### Pattern 2: GPS Trajectory Validation

```python
# Haversine-based speed and jump detection
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi/2)**2 + cos(phi1)*cos(phi2)*sin(dlambda/2)**2
    return 2 * R * arcsin(sqrt(a))

def compute_speed(pos1, pos2):
    dist_km = haversine(pos1.lat, pos1.lon, pos2.lat, pos2.lon)
    time_h = abs(pos2.ts - pos1.ts) / 3_600_000
    return dist_km / time_h
```

### Pattern 3: Event Deduplication

```python
# SHA256-based deduplication with TTL
def deduplicate(event, dedup_state, ttl_seconds=300):
    hash_val = compute_event_hash(event)
    if dedup_state.contains(hash_val):
        return VIOLATION  # Duplicate detected
    dedup_state.add(hash_val, ttl=ttl_seconds)
    return null  # Not a duplicate
```

### Pattern 4: Synthetic Anomaly Injection

```python
# Ground-truth evaluation via controlled injection
INJECTION_TYPES = {
    "null_injection": lambda e: set_field(e, field, NULL),
    "speed_injection": lambda e: set_speed(e, SPEED),
    "jump_injection": lambda e: set_position(e, DISTANT_LOCATION),
    "duplicate_injection": lambda e: re_emit(e)
}

def inject_anomalies(events, injection_rate, anomaly_type):
    n_anomalies = int(len(events) * injection_rate)
    indices = random.sample(range(len(events)), n_anomalies)
    for idx in indices:
        events[idx] = INJECTION_TYPES[anomaly_type](events[idx])
    return events
```

## Pipeline Performance Optimization

### Latency Optimization

| Technique | Expected Improvement | When to Use |
|-----------|--------------------|-------------|
| Async I/O | 30-50% latency reduction | RPC calls, external services |
| State backend: RocksDB | Better scalability | Large state (>100MB) |
| Micro-batching | Higher throughput | Batch micro-operations |
| Checkpointing interval tuning | Trade-off latency/recovery | Stateful streaming |

### Throughput Optimization

| Technique | Expected Improvement | When to Use |
|-----------|--------------------|-------------|
| Parallelism tuning | Linear up to partition count | CPU-bound operators |
| Operator chaining | 10-20% improvement | Small per-record operations |
| Batch commits | Higher write throughput | Kafka sinks, JDBC sinks |
| Broadcast state | O(1) lookups | Small threshold tables |

## Data Pipeline Patterns

### Pattern: Exactly-Once Processing

```
┌─────────────────────────────────────────────────────┐
│ EXACTLY-ONCE GUARANTEE                              │
│                                                      │
│  Source (Kafka with offset management)                │
│    ↓                                                 │
│  Stateful Processing (Flink with checkpointing)        │
│    ↓                                                 │
│  Sink (Transactional write to DB)                     │
│                                                      │
│  ✓ Kafka offset committed only after sink commit      │
│  ✓ Checkpoint = alignment point                      │
│  ✓ Idempotent sinks handle duplicate attempts         │
└─────────────────────────────────────────────────────┘
```

### Pattern: Late Event Handling

```python
WATERMARK_STRATEGY = {
    "bounded_out_of_orderness": "Duration.ofSeconds(60)",
    "with_idleness": "Duration.ofMinutes(5)",
    "alignment": "aligned_source_watermarks"
}

def handle_late_event(event, late_output_tag):
    # Side output for late events
    ctx.output(LATE_EVENTS_TAG, event)
    # Or: assign to next window
```

### Pattern: State TTL Management

```python
STATE_TTL = {
    "vehicle_gps_state": "10 minutes",   # Keep last N positions
    "dedup_state": "5 minutes",            # TTL = window + buffer
    "threshold_stats": "1 hour",            # Rolling statistics
    "context_key_counts": "24 hours"       # Long-term aggregations
}
```

## Integration with Research Project

For the `final/` project (StreamDQ), use this data processing skill in coordination with:

| Component | Data Processing Role |
|-----------|---------------------|
| `streamdq/` code | Implement CRS rules, context thresholds |
| `evaluation/` | Build synthetic injection + ground truth |
| `thesis_template/` | Write results into thesis |
| `final/05_ALGORITHM_DESIGN/` | Follow FORMULATION.md specs exactly |

## Quality Standards

- **Zero data loss**: Exactly-once semantics where needed
- **DQ labels**: Every metric labeled TIER-1/2/3 (Verified/Estimated/Unmeasurable)
- **Reproducibility**: Seed, warmup, measurement window documented
- **Graceful degradation**: Timeout → fallback path always defined
