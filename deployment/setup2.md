# CA-DQStream Setup 2: Production Pipeline Integration
## Gap Analysis & Implementation Plan

---

## 1. Root Cause Analysis: Why Output Topics & Tables Are Empty

### 1.1 The Fundamental Problem

The Flink pipeline in `src/flink_job_complete.py` only has **`.print()` sinks** — all output goes to stdout (container logs), NOT to Kafka output topics or PostgreSQL tables.

From `flink_job_complete.py` lines 302-323:
```python
# Debug output (print to console for now)
valid_stream.print()           # -> stdout only
violation_stream.print()      # -> stdout only
canary_stream.filter(...).print()  # -> stdout only
voting_stream.print()          # -> stdout only
meta_stream.print()            # -> stdout only
iec_stream.print()            # -> stdout only

# TODO: Add PostgreSQL sinks when testing with real data
# TODO: Add Kafka sinks for dq-meta-stream, iec-action-replay
```

The sinks **already exist** in `src/sinks/postgres_sink.py` and Kafka sinks can be created via `FlinkKafkaProducer`, but **they are not wired into the pipeline**.

### 1.2 What Exists vs What's Missing

| Component | Status | File |
|-----------|--------|------|
| Kafka Input Source (taxi-nyc-raw) | WORKING | `flink_job_complete.py:160` |
| Layer 1 operators | WORKING | `ParseJsonFunction`, watermark, dedup |
| Layer 2 operators | WORKING | CanaryRules, IFScoring |
| Layer 3 operators | WORKING | VotingEnsemble, MetaAggregate |
| Layer 4 operators | WORKING | IECOperator |
| JDBC sink code | EXISTS (unused) | `sinks/postgres_sink.py` |
| Kafka output sinks | MISSING | Need to create |
| Sink wiring | MISSING | `flink_job_complete.py` |
| ML model training | MISSING | Phase 0 not executed |
| ML model upload | MISSING | No artifact registered |
| Schema Registry integration | MISSING | JSON only, no Avro |
| Custom Prometheus metrics | MISSING | Pipeline emits no custom metrics |
| Grafana dashboards | PARTIAL | Reference non-existent metrics |
| Anomaly simulation | MISSING | No injection scripts |

---

## 2. Architecture: What We Have vs What Phase 0/1 Requires

### 2.1 Current Pipeline Flow

```
Kafka (taxi-nyc-raw)
    └── Flink Job (flink_job_complete.py)
            ├── Layer 1: Parse -> Dedupe -> Validate -> valid_stream + violation_stream
            ├── Layer 2: Canary Rules -> canary_stream + IFScoring -> complex_stream
            ├── Layer 3: Rendezvous -> Voting -> meta_stream
            ├── Layer 4: IEC -> iec_stream
            └── ALL .print() -> container stdout (LOST)
```

### 2.2 Target Pipeline Flow (After Setup 2)

```
Kafka (taxi-nyc-raw)
    └── Flink Job
            ├── Layer 1: Parse -> Dedupe -> Validate
            │       ├── valid_stream -> JDBC sink -> taxi_trips_raw, schema_violations
            │       └── Kafka sink -> dq-stream-processed
            ├── Layer 2: Canary -> canary_stream -> JDBC -> canary_violations
            │            Complex -> anomaly_scores -> JDBC
            │            Kafka sink -> dq-stream-anomalies
            ├── Layer 3: Voting -> meta_stream
            │       └── Kafka sink -> dq-meta-stream
            ├── Layer 4: IEC -> iec_stream
            │       └── Kafka sink -> iec-action-replay
            └── Custom metrics -> Prometheus (cadqstream_pipeline_*)

Kafka (dq-hard-rule-violations) <- Producer injects anomalies
                                              │
                                              └── Flink reads + flags

MLflow (artifact storage)
    └── MinIO (artifact bucket)
            ├── anomaly_detector_v1.pkl
            ├── anomaly_detector_v2.pkl
            └── drift_report_latest.json

Prometheus
    └── scrape targets: kafka-exporter, postgres-exporter, node-exporter, flink-jobmanager:9248
            └── recording rules + custom cadqstream_pipeline_* metrics

Grafana
    └── 5 dashboards with live Prometheus queries
```

---

## 3. Implementation Plan

### 3.1 Priority Matrix

| Priority | Task | Impact | Effort | Blocks |
|----------|------|--------|--------|--------|
| P0 | Wire JDBC + Kafka sinks into pipeline | Fills PostgreSQL + Kafka output topics | Medium | All downstream |
| P1 | Create anomaly simulation producer | Fires Prometheus alerts, populates tables | Medium | Alerts, ML |
| P2 | Add custom Prometheus metrics to pipeline | Powers Grafana dashboards | Medium | Dashboards |
| P3 | Train & register ML models in MLflow | Provides anomaly scores, enables IEC | High | Layer 2 ML |
| P4 | Create professional Grafana dashboards | Live monitoring for council demo | Medium | Demo readiness |
| P5 | Create Prometheus recording rules & alerts | Automated anomaly detection | Low | Operations |
| P6 | Schema Registry integration | Standardized data contracts | Low | Future use |

---

## 4. Detailed Tasks

### 4.1 P0: Wire Sinks into Flink Pipeline

**Modify:** `src/flink_job_complete.py`

**Current output section (broken):**
```python
# ALL .print() - no actual data sinks
valid_stream.print()
violation_stream.print()
canary_stream.filter(ViolationFilter()).print()
voting_stream.print()
meta_stream.print()
iec_stream.print()
```

**Required changes:**

1. **Add Kafka sink import:**
```python
from pyflink.datastream.connectors.kafka import FlinkKafkaProducer
from pyflink.common.serialization import SimpleStringSchema
```

2. **Create Kafka sink factory:**
```python
def create_kafka_sink(topic, bootstrap_servers):
    props = {'bootstrap.servers': bootstrap_servers}
    return FlinkKafkaProducer(
        topic=topic,
        serialization_schema=SimpleStringSchema(),
        producer_config=props
    )
```

3. **Wire sinks to each stream:**
```python
# Layer 1: Valid records -> PostgreSQL + Kafka
BOOTSTRAP = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
valid_stream.add_sink(create_jdbc_sink('taxi_trips_raw', ...))
valid_stream.add_sink(create_kafka_sink('dq-stream-processed', BOOTSTRAP))

violation_stream.add_sink(create_jdbc_sink('schema_violations', ...))
violation_stream.add_sink(create_kafka_sink('dq-hard-rule-violations', BOOTSTRAP))

# Layer 2: Canary violations -> PostgreSQL + Kafka
canary_stream.add_sink(create_jdbc_sink('canary_violations', ...))
canary_stream.add_sink(create_kafka_sink('dq-stream-anomalies', BOOTSTRAP))

# Complex ML scores -> PostgreSQL
complex_stream.add_sink(create_jdbc_sink('anomaly_scores', ...))

# Layer 3: Meta-metrics -> Kafka
meta_stream.add_sink(create_kafka_sink('dq-meta-stream', BOOTSTRAP))
meta_stream.add_sink(create_jdbc_sink('meta_metrics', ...))

# Layer 4: IEC decisions -> Kafka + PostgreSQL
iec_stream.add_sink(create_kafka_sink('iec-action-replay', BOOTSTRAP))
iec_stream.add_sink(create_jdbc_sink('drift_events', ...))
```

4. **Fix `record_to_*_row` functions** - they need to extract proper fields from the pipeline's internal record format (which uses NYC taxi column names like `VendorID`, `tpep_pickup_datetime`, etc.)

---

### 4.2 P1: Anomaly Simulation Producer

**Create:** `deployment/kafka/anomaly_producer.py`

This Python script injects synthetic anomalies into `dq-hard-rule-violations` topic to simulate real-world data quality issues. This populates output topics and triggers Prometheus alerts.

**Anomaly types to simulate:**

| Type | Description | Frequency |
|------|-------------|-----------|
| `INVALID_ZONE` | PULocationID or DOLocationID outside 1-263 | 5% |
| `NEGATIVE_FARE` | fare_amount < 0 | 2% |
| `IMPOSSIBLE_SPEED` | fare_amount/trip_distance > 200 mph implied | 3% |
| `ZERO_PASSENGER` | passenger_count = 0 | 1% |
| `MISSING_FIELD` | Required field null | 4% |
| `DRIFT_SPIKE` | 10x normal fare for 5 min window | Event-triggered |
| `SEASONAL_SHIFT` | Gradual fare increase over 30 min | Event-triggered |

**Kafka topics to write to:**
- `dq-hard-rule-violations` — for Flink to read and process
- `taxi-nyc-raw` — fresh anomaly-injected raw data

**Producer behavior:**
- Runs continuously (daemon mode)
- Produces batch of normal records, then injects anomaly burst
- Logs anomaly type and count for Prometheus scraping

---

### 4.3 P2: Custom Prometheus Metrics

**Modify:** `src/flink_job_complete.py` and operator files

**Add metrics registry:**
```python
from pyflink.common.metrics import Counter, Gauge, Histogram

# Per-operator metrics
class MetricsCollector:
    def __init__(self, env, name):
        self.records_processed = Counter(
            env.get_metrics_registry(),
            f"cadqstream_{name}_records_processed"
        )
        self.anomalies_detected = Counter(
            env.get_metrics_registry(),
            f"cadqstream_{name}_anomalies"
        )
        self.processing_latency_ms = Histogram(
            env.get_metrics_registry(),
            f"cadqstream_{name}_latency_ms"
        )
```

**Metrics to expose:**

| Metric Name | Type | Labels | Source |
|-----------|------|--------|--------|
| `cadqstream_records_input_total` | Counter | topic | Kafka source |
| `cadqstream_records_valid_total` | Counter | layer | After validation |
| `cadqstream_records_violation_total` | Counter | type | Schema validator |
| `cadqstream_anomalies_canary_total` | Counter | rule | Canary rules |
| `cadqstream_anomalies_ml_total` | Counter | neighborhood | IFScoring |
| `cadqstream_meta_volume` | Gauge | neighborhood | MetaAggregator |
| `cadqstream_meta_anomaly_rate` | Gauge | neighborhood | MetaAggregator |
| `cadqstream_iec_decisions_total` | Counter | strategy | IECOperator |
| `cadqstream_iec_drift_detected_total` | Counter | neighborhood | IECOperator |
| `cadqstream_pipeline_lag_seconds` | Gauge | topic | Kafka consumer |
| `cadqstream_checkpoint_size_bytes` | Gauge | job_id | Checkpointing |

---

### 4.4 P3: ML Model Training & Registration

**Create:** `deployment/scripts/train_model.py`

This script trains the anomaly detection model using historical NYC taxi data (Phase 0 from plan), registers it in MLflow, and uploads to MinIO.

**Pipeline:**
```
1. Load historical data from MinIO (or synthetic)
2. Layer 1 filter (schema + physical impossibility) -> clean baseline
3. Feature engineering: 21D ratio features
4. Train IsolationForest on clean data
5. Compute per-cluster thresholds (baseline stats)
6. Register in MLflow:
   - Model: anomaly_detector_v1.pkl
   - Parameters: contamination, n_estimators, features
   - Metrics: precision, recall, f1
   - Artifacts: model.pkl, threshold.json
7. Upload to MinIO: ml-models/anomaly_detector_v1.pkl
8. Update Flink pipeline env var: MODEL_VERSION=v1
```

**Model storage:**
- MinIO: `ml-models/anomaly_detector_v1.pkl`
- MLflow run with params, metrics, and artifacts

---

### 4.5 P4: Professional Grafana Dashboards

**Modify/Create:** `deployment/grafana/dashboards/*.json`

**Dashboard 1: Pipeline Overview (CONSEIL DEMO)**
- End-to-end flow: Kafka in -> Flink -> PostgreSQL out
- Real-time throughput (records/sec) per layer
- Latency histogram (p50, p95, p99)
- Active alerts panel
- System status grid

**Dashboard 2: Kafka Deep Dive**
- Message rate per topic (input vs output)
- Consumer lag (Flink vs exporters)
- Partition balance
- Retention bytes

**Dashboard 3: Flink Jobs**
- Job status (running/failed/restarting)
- Checkpoint health (size, duration, #failed)
- Task manager slots (used/total)
- Custom pipeline metrics (records processed, anomalies detected)
- Watermark progress

**Dashboard 4: Data Quality**
- Violation rate over time (by type)
- Anomaly rate by neighborhood
- Meta-metrics heatmap
- Drift detection events timeline

**Dashboard 5: Infrastructure**
- Container CPU/memory across all services
- Disk usage
- Network I/O
- PostgreSQL connection pool
- Kafka broker health

---

### 4.6 P5: Prometheus Alert Rules

**Modify:** `deployment/prometheus/alert-rules/cadqstream-alerts.yml`

**Add pipeline-specific alerts:**

```yaml
groups:
  - name: cadqstream_pipeline
    rules:
      - alert: HighAnomalyRate
        expr: cadqstream_anomalies_canary_total / cadqstream_records_valid_total > 0.15
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Anomaly rate exceeds 15% threshold"

      - alert: DriftDetected
        expr: increase(cadqstream_iec_drift_detected_total[5m]) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Drift detected in {{ $labels.neighborhood }}"

      - alert: PipelineStalled
        expr: rate(cadqstream_records_input_total[5m]) == 0
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "No records processed for 10 minutes"

      - alert: HighConsumerLag
        expr: kafka_consumer_group_lag_sum > 10000
        for: 5m
        labels:
          severity: warning

      - alert: CheckpointFailure
        expr: flink_jobmanager_job_last_checkpoint_duration > 300000
        for: 1m
        labels:
          severity: critical
```

---

### 4.7 P6: Schema Registry Integration

**Option A (Recommended):** Keep JSON, register schemas via REST API
```bash
# In kafka-init script
curl -X POST http://schema-registry:8081/subjects/taxi-nyc-raw-value/versions \
  -H "Content-Type: application/vnd.schemaregistry.v1+json" \
  --data '{"schema": "{\"type\": \"record\", ...}"}'
```

**Option B:** Register Avro schemas for all pipeline topics
- Requires Avro serialization in Flink (via `FlinkKafkaProducer` with Avro schema)
- More production-grade but more complex

---

## 5. Files to Create/Modify

### 5.1 New Files

| File | Purpose | Priority |
|------|---------|----------|
| `deployment/kafka/anomaly_producer.py` | Inject synthetic anomalies | P1 |
| `deployment/scripts/train_model.py` | Train + register ML model | P3 |
| `deployment/grafana/dashboards/pipeline-dq.json` | Data Quality dashboard | P4 |
| `deployment/grafana/dashboards/infrastructure.json` | Infra monitoring | P4 |
| `deployment/grafana/dashboards/conseil-demo.json` | Council demo dashboard | P4 |

### 5.2 Files to Modify

| File | Changes | Priority |
|------|---------|----------|
| `src/flink_job_complete.py` | Wire JDBC + Kafka sinks, add metrics | P0 |
| `src/sinks/postgres_sink.py` | Fix record mapping, add missing sinks | P0 |
| `deployment/prometheus/alert-rules/cadqstream-alerts.yml` | Add pipeline alerts | P5 |
| `deployment/grafana/dashboards/*.json` | Fix datasource, add real queries | P4 |
| `deployment/kafka/init-scripts/01-create-topics.sh` | Add schema registry registration | P6 |
| `deployment/scripts/start-producer.ps1` | Add anomaly injection option | P1 |

---

## 6. Execution Order

```
Phase A: Quick Wins (30 min)
├─ A1. Wire JDBC sinks into flink_job_complete.py
├─ A2. Wire Kafka output sinks into flink_job_complete.py
├─ A3. Create anomaly_producer.py
└─ A4. Restart pipeline + run anomaly producer
    -> PostgreSQL tables start filling
    -> Kafka output topics populate

Phase B: Monitoring (30 min)
├─ B1. Add custom Prometheus metrics to pipeline
├─ B2. Update Prometheus scrape config (add flink:9248)
├─ B3. Create council-demo dashboard
├─ B4. Create data-quality dashboard
└─ B5. Update alert rules

Phase C: ML Integration (60 min)
├─ C1. Create train_model.py script
├─ C2. Run training on synthetic data
├─ C3. Register model in MLflow + upload to MinIO
├─ C4. Update Flink pipeline to load model from MinIO
└─ C5. Verify anomaly scores in PostgreSQL

Phase D: Polish (30 min)
├─ D1. Create infrastructure dashboard
├─ D2. Fix Kafka deep-dive dashboard
├─ D3. Schema Registry integration
├─ D4. Write HOW_TO_RUN.txt
└─ D5. End-to-end verification
```

---

## 7. Dependencies & Prerequisites

| Item | Status | Action |
|------|--------|--------|
| Flink pipeline running | OK | Keep |
| Kafka topics created | OK | Keep |
| PostgreSQL schema | OK | Keep |
| MinIO buckets | OK | Keep |
| Prometheus/Grafana | OK | Update configs |
| ML models | MISSING | Create training script |
| Anomaly simulation | MISSING | Create producer |
| Sink wiring | MISSING | Modify flink_job_complete.py |
| Custom metrics | MISSING | Add to operators |

---

## 8. Verification Checklist

After Setup 2, ALL of these must return non-zero/non-empty:

```bash
# PostgreSQL - tables have data
docker exec ldt-postgres psql -U cadqstream -d dq_pipeline -c "SELECT COUNT(*) FROM taxi_trips_raw; SELECT COUNT(*) FROM schema_violations; SELECT COUNT(*) FROM canary_violations;"

# Kafka - output topics have messages
docker exec ldt-kafka kafka-console-consumer --topic dq-stream-processed --from-beginning --max-messages 10 --bootstrap-server localhost:9092

# Kafka - anomaly producer writes to dq-hard-rule-violations
docker exec ldt-kafka kafka-console-consumer --topic dq-hard-rule-violations --from-beginning --max-messages 5 --bootstrap-server localhost:9092

# MinIO - ML model artifact exists
docker exec ldt-minio mc ls local/ml-models/

# MLflow - model registered
curl -s http://localhost:5000/api/2.0/preview/mlflow/runs/search -X POST -H "Content-Type: application/json" -d '{"experiment_ids":[]}'

# Prometheus - custom metrics present
curl -s http://localhost:9090/api/v1/query?query=cadqstream_records_input_total

# Grafana - dashboards load with data
# Open http://localhost:3000 -> Check each dashboard

# Prometheus - alerts can fire (simulate anomaly)
# Run anomaly producer with HIGH rate -> check Prometheus alerts page
```

---

## 9. Summary: Setup 1 vs Setup 2

| What | Setup 1 (Done) | Setup 2 (Needed) |
|------|-----------------|-------------------|
| Kafka topics | Created | **Wire output sinks** |
| PostgreSQL schema | Created | **Wire JDBC sinks** |
| Flink job | RUNNING | **Add sink calls** |
| MinIO buckets | Created | **Upload ML models** |
| MLflow | Empty | **Train + register models** |
| Prometheus | Rules exist | **Add pipeline metrics** |
| Grafana | Dashboards exist | **Fix queries, add live data** |
| Anomaly data | Producer sends | **Add simulation** |
| Schema Registry | Empty | **Register schemas** |
