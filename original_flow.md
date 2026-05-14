# CA-DQStream — Original Flow: Full Architecture & Complete Data Flow

> **Ngày:** 2026-05-13
> **Hệ thống:** CA-DQStream (Context-Aware Data Quality Stream) — Streaming Anomaly Detection Pipeline
> **Framework:** Apache Flink 1.17.1 (Python), Apache Kafka 7.5.0, MinIO, Prometheus, Grafana
> **ML Backend:** sklearn IsolationForest (Complex Branch)

---

## Mục lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Full Service Architecture — Tất cả các service và port](#2-full-service-architecture--tất-cả-các-service-và-port)
3. [Complete Data Flow — Luồng dữ liệu chi tiết từ đầu đến cuối](#3-complete-data-flow--luồng-dữ-liệu-chi-tiết-từ-đầu-đến-cuối)
4. [Layer-by-Layer Deep Dive](#4-layer-by-layer-deep-dive)
5. [ML Thresholds & Concept Drift — Chi tiết đầy đủ](#5-ml-thresholds--concept-drift--chi-tiết-đầy-đủ)
6. [All Scenarios & Edge Cases — Tất cả trường hợp](#6-all-scenarios--edge-cases--tất-cả-trường-hợp)
7. [Checkpoint & Recovery — Khôi phục lỗi](#7-checkpoint--recovery--khôi-phục-lỗi)
8. [Configuration Matrix — Bảng tham số đầy đủ](#8-configuration-matrix--bảng-tham-số-đầy-đủ)

---

## 1. Tổng quan kiến trúc

CA-DQStream là hệ thống **streaming data quality** 4 tầng xử lý bản tin taxi NYC trong thời gian thực:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL DATA SOURCES                                │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────────┐    │
│  │ anomaly_     │    │  fast_producer   │    │  produce_taxi_data    │    │
│  │ producer.py  │    │  (deployment/    │    │  (real parquet files) │    │
│  │ (7 anomaly   │    │   kafka/)        │    │                       │    │
│  │  types)      │    │  100 msg/sec     │    │  up to 1000 events/s  │    │
│  └──────┬───────┘    └────────┬─────────┘    └──────────┬─────────────┘    │
└─────────┼─────────────────────┼─────────────────────────┼───────────────────┘
          │                     │                         │
          ▼                     ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    KAFKA CLUSTER (kafka:9092)                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐  │
│  │  Topic: taxi-nyc-raw (4 partitions, replication=1, retention=7 days)    │  │
│  └─────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1: BASELINE VALIDATION                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐   │
│  │ ParseJsonFunction → WatermarkAssigner → AddTripId (MurmurHash3)       │   │
│  │ → key_by(trip_id) → DeduplicatorFunction (7-day TTL, RocksDB)         │   │
│  │ → SchemaValidator (required fields, zone 1-263)                        │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
│                           ↓ valid      ↓ invalid                            │
└─────────────────────────────────────────────────────────────────────────────┘
          │                                          │
          ▼                                          ▼
┌────────────────────────────┐       ┌────────────────────────────────────────┐
│  Kafka: dq-hard-rule-     │       │  MinIO: cadqstream-violations/            │
│  violations (L1 schema     │       │  MinIO: cadqstream-raw/                  │
│  violations)               │       └────────────────────────────────────────┘
└────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 2: DUAL-BRANCH PROCESSING                                            │
│                                                                              │
│  ┌──────────────────────────────────────┐  ┌──────────────────────────────┐  │
│  │  CANARY BRANCH (7 static rules)      │  │  COMPLEX BRANCH (ML)        │  │
│  │  CanaryRulesValidator                │  │  IFScoringOperator           │  │
│  │  ────────────────                    │  │  ─────────────────────────  │  │
│  │  • negative_fare (fare ≤ 0)          │  │  FeatureVectorizer          │  │
│  │  • zero_distance_with_fare           │  │  (21 dimensions)            │  │
│  │  • invalid_passengers (0 or >6)      │  │      ↓                      │  │
│  │  • invalid_payment (not 1-6)        │  │  sklearn IsolationForest     │  │
│  │  • extreme_fare (> $1000)           │  │  (200 trees, max_samples=256│  │
│  │  • extreme_duration (>24h)          │  │   contamination=0.001)      │  │
│  │  • negative_duration                 │  │      ↓                      │  │
│  │  ────────────────                    │  │  anomaly_score =            │  │
│  │  Pass ALL records through with        │  │    -score_samples(x)        │  │
│  │  violation flags (no filtering)      │  │  is_anomaly = score > ctx_thr │  │
│  └──────────────────────────────────────┘  └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
          │                                          │
          ▼                                          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  Kafka: dq-stream-anomalies           │  MinIO: cadqstream-raw/                  │
│  (canary violations)                   │  MinIO: cadqstream-anomalies/           │
└──────────────────────────────────────────────────────────────────────────────┘
          │                                          │
          └──────────────┬───────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 3: RENDEZVOUS + VOTING ENSEMBLE                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐   │
│  │  canary_stream.union(complex_stream)                                   │   │
│  │      ↓                                                                 │   │
│  │  VotingEnsembleFunction                                               │   │
│  │  ─────────────────────                                                │   │
│  │  Priority: Canary overrides ML                                        │   │
│  │  • Canary has violations → ANOMALY, source=canary_rule, confidence=1.0 │   │
│  │  • ML flags anomaly → ANOMALY, source=complex_ml, confidence=score/thr │   │
│  │  • Both clean → CLEAN, source=both_agree, confidence=1-(score/thr)    │   │
│  │      ↓                                                                 │   │
│  │  MetaAggregateFunction (1-min tumbling window, per neighborhood)       │   │
│  │  → 6 meta-metrics: volume, null_rate, violation_rate, anomaly_rate,    │   │
│  │    avg_anomaly_score, delta_score                                     │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
│                           ↓                                                  │
│  Kafka: dq-meta-stream (voting results)                                      │
│  MinIO: cadqstream-metrics/ (Parquet)                                       │
└─────────────────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 4: IEC (Intelligent Evolution Controller)                             │
│  ┌────────────────────────────────────────────────────────────────────────┐   │
│  │  IECOperator                                                          │   │
│  │  ───────────                                                          │   │
│  │  ① MultiInstanceADWIN — 36 instances (6 neighborhoods × 6 metrics)     │   │
│  │     Metrics: volume, null_rate, violation_rate, anomaly_rate,          │   │
│  │              avg_anomaly_score, delta_score                           │   │
│  │      ↓                                                                │   │
│  │  ② DriftAggregator — severity assessment (none/low/moderate/high)     │   │
│  │      ↓                                                                │   │
│  │  ③ METER Hypernetwork (sklearn, pickle) — strategy prediction         │   │
│  │     OR Fallback rules (if METER unavailable)                          │   │
│  │      ↓                                                                │   │
│  │  ④ Strategy Execution                                                 │   │
│  │     do_nothing / adjust_threshold / retrain_model / switch_model      │   │
│  └────────────────────────────────────────────────────────────────────────┘   │
│                           ↓                                                  │
│  Kafka: iec-action-replay                                                   │
│  MinIO: cadqstream-drift/ (Parquet)                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Full Service Architecture — Tất cả các service và port

### 2.1 Network Topology

```
Docker Network: cadqstream-net (bridge)

Tất cả container nằm trong cùng một bridge network, giao tiếp qua service name (`kafka`, `flink-jobmanager`, etc.)

### 2.2 Tất cả 18 service trong docker-compose.yml

```
USER
  │ port 9092
  ▼
┌─────────────────────────────────────────────────────────────────┐
│ KAFKA CLUSTER                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │   zookeeper     │  │     kafka       │  │ schema-registry │  │
│  │  port 2181      │  │ 9092, 29092    │  │    port 8082    │  │
│  └─────────────────┘  └────────┬────────┘  └─────────────────┘  │
│                                │                                 │
│                         ┌──────┴──────────┐                     │
│                         │   kafka-init     │ (one-shot)          │
│                         │  (create topics) │                     │
│                         └─────────────────┘                     │
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐  │
│  │   kafka-ui       │  │ kafka-exporter   │  │kafka-producer  │  │
│  │   port 8080     │  │   port 9308      │  │ (data source)  │  │
│  └─────────────────┘  └─────────────────┘  └────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                           │ Kafka:9092
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ FLINK CLUSTER                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │flink-jobmanager │  │flink-taskmanager │  │   flink-init    │  │
│  │   port 8081     │  │  (4 slots/TM)    │  │   (one-shot)    │  │
│  │  Prometheus:9248│  │  Prometheus:9249│  │ submits job     │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ OBJECT STORAGE                                                    │
│  ┌─────────────────┐  ┌─────────────────┐                     │
│  │     minio       │  │   minio-init    │ (one-shot)          │
│  │ 9000, 9001      │  │  (create bucket)│                     │
│  └─────────────────┘  └─────────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ ML PLATFORM                                                       │
│  ┌─────────────────┐  ┌─────────────────┐                       │
│  │     mlflow      │  │mlflow-exporter  │                       │
│  │   port 5000     │  │   port 9251     │                       │
│  └─────────────────┘  └─────────────────┘                       │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│ OBSERVABILITY                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐  │
│  │   prometheus    │  │     grafana      │  │node-exporter   │  │
│  │   port 9090     │  │   port 3000      │  │  port 9100     │  │
│  └─────────────────┘  └─────────────────┘  └────────────────┘  │
│                                                                  │
│  ┌─────────────────┐                                             │
│  │cadqstream-      │                                             │
│  │metrics (9250)   │ ◄── Flink job gửi metrics về đây qua HTTP   │
│  └─────────────────┘                                             │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Port Mapping Tổng hợp

| Host Port | Service | Container Port | Protocol |
|---|---|---|---|
| 2181 | Zookeeper | 2181 | TCP |
| 8080 | Kafka UI | 8080 | HTTP |
| 8081 | Flink JobManager REST | 8081 | HTTP |
| 8082 | Schema Registry | 8081 | HTTP |
| 9000 | MinIO API | 9000 | HTTP/S3 |
| 9001 | MinIO Console | 9001 | HTTP |
| 9100 | Node Exporter | 9100 | HTTP |
| 9187 | MinIO Exporter | 9187 | HTTP |
| 9250 | Cadqstream Metrics | 9250 | HTTP |
| 9251 | MLflow Exporter | 9251 | HTTP |
| 9308 | Kafka Exporter | 9308 | HTTP |
| 3000 | Grafana | 3000 | HTTP |
| 5000 | MLflow | 5000 | HTTP |
| 9090 | Prometheus | 9090 | HTTP |
| 9092 | Kafka PLAINTEXT | 9092 | TCP |
| 29092 | Kafka Internal | 29092 | TCP |

### 2.4 Named Volumes

| Volume | Size Estimate | Purpose |
|---|---|---|
| `ldt-zookeeper-data` | ~10 MB | Zookeeper transaction logs |
| `ldt-kafka-data` | ~500 MB+ | Kafka log segments (7-day retention) |
| `ldt-minio-data` | ~500 MB+ | MinIO object storage |
| `ldt-prometheus-data` | ~200 MB | Prometheus TSDB (15-day retention) |
| `ldt-grafana-data` | ~50 MB | Grafana dashboards/plugins |
| `ldt-mlflow-data` | ~100 MB+ | MLflow tracking server |
| `ldt-flink-jobmanager-data` | ~100 MB | Flink JM state/checkpoints |
| `ldt-flink-taskmanager-data` | ~500 MB | Flink TM state (RocksDB) |

### 2.5 Resource Limits per Service

| Service | CPU | Memory | Shared Memory |
|---|---|---|---|
| zookeeper | 0.5 | 512 MB | - |
| kafka | 2 | 4 GB | 256 MB |
| schema-registry | 0.5 | 1 GB | - |
| kafka-ui | 0.5 | 512 MB | - |
| kafka-exporter | 0.25 | 256 MB | - |
| minio | 1.5 | 2 GB | 128 MB |
| minio-init | 1 | 2 GB | - |
| flink-jobmanager | 1.5 | 2 GB | - |
| flink-taskmanager | 4 | 8 GB | 1 GB |
| mlflow | 0.5 | 2 GB | - |
| mlflow-exporter | 0.1 | 128 MB | - |
| prometheus | 0.5 | 2 GB | - |
| grafana | 0.5 | 1 GB | - |
| node-exporter | 0.1 | 128 MB | - |
| cadqstream-metrics | 0.1 | 128 MB | - |

---

## 3. Complete Data Flow — Luồng dữ liệu chi tiết từ đầu đến cuối

### 3.1 Kafka Topic Definitions

**File:** `deployment/kafka/init-scripts/01-create-topics.sh`

| Topic | Partitions | Replication | Retention | Cleanup Policy | Schema |
|---|---|---|---|---|---|
| `taxi-nyc-raw` | 4 | 1 | 7 ngày | delete | NYC taxi JSON |
| `dq-stream-processed` | 4 | 1 | 7 ngày | delete | Enriched JSON |
| `dq-stream-anomalies` | 4 | 1 | 30 ngày | compact | Anomaly JSON |
| `dq-meta-stream` | 4 | 1 | 7 ngày | delete | Meta-metric JSON |
| `iec-action-replay` | 1 | 1 | 1 ngày | delete | IEC decision JSON |
| `dq-metrics` | 1 | 1 | 30 ngày | compact | Metric JSON |
| `dq-hard-rule-violations` | 4 | 1 | 30 ngày | compact | Violation JSON |
| `if-model-updates` | 1 | 1 | 7 ngày | compact | Model update |

### 3.2 Record Schema — Tất cả các field

#### Input: `taxi-nyc-raw` (NYC Yellow Taxi JSON)

```json
{
  "VendorID": 1,
  "tpep_pickup_datetime": "2024-01-15 08:30:00",
  "tpep_dropoff_datetime": "2024-01-15 08:55:00",
  "passenger_count": 2,
  "trip_distance": 3.5,
  "RatecodeID": 1,
  "store_and_fwd_flag": "N",
  "PULocationID": 161,
  "DOLocationID": 237,
  "payment_type": 1,
  "fare_amount": 12.50,
  "extra": 0.50,
  "mta_tax": 0.50,
  "tip_amount": 2.50,
  "tolls_amount": 0,
  "improvement_surcharge": 1.00,
  "total_amount": 17.00,
  "congestion_surcharge": 2.50,
  "Airport_fee": 0
}
```

#### Enriched Record (sau Layer 2 — Can contain all fields)

```json
{
  // Layer 1 fields
  "VendorID": 1,
  "tpep_pickup_datetime": "2024-01-15 08:30:00",
  "tpep_dropoff_datetime": "2024-01-15 08:55:00",
  "passenger_count": 2,
  "trip_distance": 3.5,
  "PULocationID": 161,
  "DOLocationID": 237,
  "payment_type": 1,
  "fare_amount": 12.50,
  "total_amount": 17.00,
  "trip_id": "a3f5b8c1d2e4...",
  
  // Layer 2 — Canary
  "canary_violations": [],
  "has_violation": false,
  "violation_count": 0,
  
  // Layer 2 — Complex/ML (sklearn IsolationForest)
  "anomaly_score": 0.23,
  "threshold": 0.50,
  "is_anomaly": false,
  "context_key": "medium_evening_rush_weekday_manhattan",
  "ml_model": "sklearn_iforest_v1",
  
  // Layer 3 — Voting
  "final_decision": "CLEAN",
  "decision_source": "both_agree",
  "confidence": 0.77,
  
  // Layer 4 — IEC
  "drifts_detected": [],
  "drift_assessment": {"severity": "none"},
  "iec_strategy": "do_nothing",
  "iec_confidence": 1.0,
  "action_result": {"action": "none", "message": "No adaptation needed"}
}
```

---

## 4. Layer-by-Layer Deep Dive

### 4.1 Layer 1: Baseline Validation — Chi tiết từng operator

#### ParseJsonFunction
```
Input:  byte[] (raw JSON string)
Output: dict hoặc None (nếu json.loads thất bại)
Logic:  json.loads(value) → drop record nếu exception
Error:  Silently drop (None record bị filter ở bước tiếp theo)
```

#### WatermarkAssigner (`create_watermark_strategy()`)
```
Event time: tpep_pickup_datetime (Unix ms)
Bounded out-of-orderness: 10 giây
Idleness timeout: 30 giây (fix PyFlink v1.9 bug)
Watermark: max(timestamp) - 10s
Late data: vẫn được xử lý (không drop)
```

#### AddTripIdFunction
```
Composite key = "VendorID|tpep_pickup_datetime|PULocationID|DOLocationID|fare_amount"
Hash: MurmurHash3.hash_bytes(composite.encode()).hex()
Output: 64-character hex string
Collision: Có thể xảy ra (MurmurHash3 không perfect) → Deduplicator lọc
```

#### DeduplicatorFunction
```
State: ValueState("seen", BOOLEAN) — keyed by trip_id
TTL: 7 ngày (Time.days(7))
Update type: OnCreateAndWrite
Visibility: NeverReturnExpired
Backend: RocksDB (production default)
Logic:
  if seen.value() == True → return None (drop duplicate)
  else → seen.update(True) → pass through
Fallback: Nếu state không init → local dict _local_seen
```

#### SchemaValidator
```
Required fields: trip_distance, fare_amount, PULocationID, DOLocationID, passenger_count
Zone range: PULocationID, DOLocationID phải trong 1-263
Logic: Kiểm tra tất cả required fields có tồn tại và zone hợp lệ
  → Valid: pass to valid_stream
  → Invalid: pass to violation_stream
Sharing: Một instance validator được chia sẻ qua ValidFilter/InvalidFilter class
```

### 4.2 Layer 2A: Canary Branch — Chi tiết 7 rules

**File:** `src/operators/canary_rules.py`

Mỗi rule kiểm tra một điều kiện business cố định:

| Rule ID | Tên rule | Điều kiện | Severity |
|---|---|---|---|
| R1 | `negative_fare` | `fare_amount <= 0` | critical |
| R2 | `zero_distance_with_fare` | `trip_distance == 0 AND fare_amount > 0` | critical |
| R3 | `invalid_passengers` | `passenger_count == 0 OR passenger_count > 6` | warning |
| R4 | `invalid_payment` | `payment_type NOT IN {1, 2, 3, 4, 5, 6}` | warning |
| R5 | `extreme_fare` | `fare_amount > 1000` | warning |
| R6 | `extreme_duration` | `trip_duration > 24 * 3600` giây | critical |
| R7 | `negative_duration` | `trip_duration < 0` | critical |
| R8 | `total_less_than_fare` | `total_amount < fare_amount` | warning |

**Output fields sau CanaryRulesValidator:**
```python
{
    'canary_violations': ['extreme_fare', 'invalid_passengers'],  # list of rule names
    'has_violation': True,       # bool: any violation?
    'violation_count': 2,         # int: total violation count
}
```

**Key design:** Tất cả record được pass through (không drop). Violations được flag nhưng vẫn đi tiếp vào Layer 3 để đồng bộ với Complex branch qua Rendezvous.

**Sinks sau Canary:**
- `ViolationFilter()` → `dq-stream-anomalies` (Kafka) + metrics
- `CleanRecordFilter()` → `dq-stream-processed-clean` (Kafka)

### 4.3 Layer 2B: Complex Branch — Chi tiết sklearn IsolationForest

**Files:**
- `src/operators/if_scoring_operator.py` — IFScoringOperator (Broadcast State pattern)
- `src/operators/mock_if_scoring_operator.py` — MockIFScoringOperator (heuristic fallback)
- `src/ml/train_iforest.py` — Training pipeline
- `src/ml/train_ocsvm.py` — Alternative: OneClassSVM (River)
- `src/ml/train_gaussian.py` — Alternative: Gaussian density estimator
- `src/operators/broadcast_state_loader.py` — Broadcast state model loader

#### 4.3.1 Feature Extraction (21D)

**File:** `src/features/vectorizer.py` → `FeatureVectorizer`

**CRITICAL INNOVATION:** 6 ratio features normalize by baseline values, reducing variance 10-100x vs raw features. Prototype: 92.2% Recall, 5.0% FPR (vs 81.5%/63.6% with 15D raw).

| Index | Feature | Type | Range/Values | Source |
|---|---|---|---|---|
| 0 | `trip_distance` | float | [0, ~100+] | Direct field |
| 1 | `trip_duration_minutes` | float | [0, +x] | (dropoff - pickup).seconds / 60 |
| 2 | `fare_amount` | float | [0, ~500+] | Direct field |
| 3 | `passenger_count` | float | [1, 6] | Direct field |
| 4 | `total_amount` | float | [0, +x] | Direct field |
| 5 | `speed_mph` | float | [0, ~200+] | trip_distance / hours |
| 6 | `fare_per_mile` | float | [0, +x] | fare_amount / distance |
| 7 | `fare_per_minute` | float | [0, +x] | fare_amount / minutes |
| 8 | `fare_per_passenger` | float | [0, +x] | fare_amount / passengers |
| 9 | `hour` | float | [0, 23] | pickup.hour |
| 10 | `day_of_week` | float | [0, 6] | pickup.weekday() |
| 11 | `is_weekend` | float | {0, 1} | weekday >= 5 |
| 12 | `is_rush_hour` | float | {0, 1} | 7-9am or 4-7pm |
| 13 | `is_night` | float | {0, 1} | hour < 6 or hour > 22 |
| 14 | `month` | float | [1, 12] | pickup.month |
| 15 | `fare_per_mile_ratio` | float | [0, +x] | fare_per_mile / 2.5 (baseline) |
| 16 | `fare_per_minute_ratio` | float | [0, +x] | fare_per_minute / 0.67 (baseline) |
| 17 | `implied_speed_ratio` | float | [0, +x] | speed / 12.0 (baseline) |
| 18 | `passenger_distance_ratio` | float | [0, +x] | passengers / distance |
| 19 | `fare_distance_product` | float | [0, +x] | fare × distance |
| 20 | `duration_distance_ratio` | float | [0, +x] | duration_minutes / distance |

**BATCH vectorization:** `vectorizer.transform_batch(df)` — vectorized pandas, 100-500x faster than row-by-row.

#### 4.3.2 sklearn IsolationForest

**File:** `src/ml/train_iforest.py`

**Architecture:**
```
Input (21D)
  ↓
sklearn.ensemble.IsolationForest
  n_estimators=200
  max_samples=256
  contamination=0.001
  max_features=1.0 (all features)
  n_jobs=-1 (all CPU cores)
  random_state=42
  ↓
score_samples(x) → raw_score (negative, lower = more anomalous)
  ↓
anomaly_score = -raw_score  (higher = more anomalous)
  ↓
is_anomaly = anomaly_score > context_threshold
```

**Scoring per record:**
```
1. features = FeatureVectorizer.transform(record)  # 21D
2. features_scaled = scaler.transform([features])[0]  # StandardScaler
3. raw_score = iforest.score_samples(features_scaled.reshape(1,-1))[0]
4. anomaly_score = -raw_score  # Negate: higher = more anomalous
5. context_key = get_context_key(record)  # 4D key
6. threshold = thresholds[context_key] or global_threshold
7. is_anomaly = anomaly_score > threshold
```

#### 4.3.3 Context-Aware Thresholds (4D Clustering)

**File:** `src/config/threshold_matrix.json`

**Context key format:**
```
{trip_type}_{time_window}_{day_type}_{neighborhood}
```

- `trip_type`: short (<2mi) / medium (2-10mi) / long (>10mi)
- `time_window`: morning_rush (6-10am) / evening_rush (5-8pm) / night (10pm-6am) / midday (else)
- `day_type`: weekday / weekend
- `neighborhood`: manhattan / brooklyn / queens / bronx / staten_island / airport

**Ví dụ:** `medium_evening_rush_weekday_manhattan`

**Nguồn gốc:** Thresholds được tính từ benchmark offline, dùng 95th percentile của anomaly scores trong mỗi context partition trên clean baseline data.

#### 4.3.4 Model Loading — Broadcast State Pattern

**File:** `src/operators/broadcast_state_loader.py` + `src/operators/if_scoring_operator.py`

**V1.9 Critical Bug Fix:** Phải gọi `broadcast_state.clear()` trước `put()` để tránh stale state accumulation và memory leak.

```
Kafka Topic: if-model-updates (compacted, single partition)
  ↓
BroadcastStateLoaderFunction.process_broadcast_element()
  → broadcast_state.clear()  ← V1.9 BUG FIX
  → broadcast_state.put("current_model", model_bytes)
  → broadcast_state.put("scaler", scaler_bytes)
  → broadcast_state.put("thresholds", thresholds_json)
  ↓
IFScoringOperator.open()
  → runtime_context.get_broadcast_state()
  → pickle.loads(model_bytes)
  → pickle.loads(scaler_bytes)
  → json.loads(thresholds)
```

**Initial model setup:**
```python
# Trước khi chạy Flink job:
python src/operators/broadcast_state_loader.py
# Tạo models/initial_model_update.json
# Produce lên Kafka: kafka-console-producer < models/initial_model_update.json
```

#### 4.3.5 Alternative: OneClassSVM (River)

**File:** `src/ml/train_ocsvm.py` — River anomaly.OneClassSVM

- Online learning: `model.learn_one(feature_dict)`
- Streaming-compatible, không cần full batch training
- `nu=0.1` — upper bound on outlier fraction

#### 4.3.6 Output fields sau IFScoringOperator

```python
{
    'anomaly_score': 0.23,           # float: -score_samples (higher = more anomalous)
    'threshold': 0.50,               # float: context-aware threshold
    'is_anomaly': False,             # bool: score > threshold
    'context_key': 'medium_evening_rush_weekday_manhattan',  # str: 4D context
    'ml_model': 'sklearn_iforest_v1', # str: model identifier
    'scoring_error': None,           # str or None
}
```

### 4.4 Layer 3: Voting Ensemble — Priority-based fusion

**File:** `src/operators/meta_aggregator.py` → `VotingEnsembleFunction`

**Logic ưu tiên:**

```
IF has_violation == True (Canary):
    → final_decision = 'ANOMALY'
    → decision_source = 'canary_rule'
    → confidence = 1.0

ELIF is_anomaly == True (ML):
    → final_decision = 'ANOMALY'
    → decision_source = 'complex_ml'
    → confidence = min(anomaly_score / threshold, 1.0)

ELSE (both clean):
    → final_decision = 'CLEAN'
    → decision_source = 'both_agree'
    → confidence = 1.0 - (anomaly_score / threshold)
```

**Meta-Metrics Windowing (1-minute TumblingEventTimeWindow, per neighborhood):**

| Metric | Formula | Source |
|---|---|---|
| `volume` | count | All records in window |
| `null_rate` | nulls / volume | Records with null fare/distance/passengers |
| `violation_rate` | violations / volume | Canary violations |
| `anomaly_rate` | anomalies / volume | ML anomalies (is_anomaly=True) |
| `avg_anomaly_score` | sum(scores) / count | ML anomaly_score mean |
| `delta_score` | \|violation_rate - anomaly_rate\| / (violation_rate + anomaly_rate + ε) | Cross-detection divergence |

**Critical insight:** `union()` được dùng thay vì `connect()` + `CoProcessFunction`, nên **mỗi record chỉ xuất hiện một lần trong merged stream** — record đi qua đúng một trong hai nhánh (canary HOẶC complex), không phải cả hai. Điều này có nghĩa là:

- Một record đi qua **Canary** → trong voting có `has_violation` nhưng KHÔNG có `is_anomaly`
- Một record đi qua **Complex** → trong voting có `is_anomaly` nhưng KHÔNG có `canary_violations`
- **RendezvousOperator** (CoProcessFunction) **KHÔNG được sử dụng** — được skip vì timer issues

### 4.5 Layer 4: IEC — Intelligent Evolution Controller

**File:** `src/operators/iec_operator.py` + `src/iec/adwin_multi_instance.py`

---

## 5. ML Thresholds & Concept Drift — Chi tiết đầy đủ

### 5.1 Threshold Initialization — Offline

#### 5.1.1 Global Threshold (default)

```python
# src/config/threshold_matrix.json
{
  "version": "1.0",
  "percentile": 95,
  "global_threshold": 0.950190435880176,  # 95th percentile on clean baseline
  "thresholds": {
    "short_night_weekday_zone_high_volume_1": 0.94385363399863,
    "medium_night_weekday_zone_airports": 0.9411799590721223,
    // ... 100 context keys total
  }
}
```

**Cách xác định:** 95th percentile của `anomaly_score` trên clean baseline data trong mỗi context partition. Giá trị này phản ánh ngưỡng mà chỉ 5% normal records vượt qua.

#### 5.1.2 Context-Aware Thresholds (100 contexts)

**File:** `src/config/threshold_matrix.json`

**Cấu trúc context key:**
```
{trip_type}_{time_window}_{day_type}_{neighborhood}
```

- `trip_type`: short (<2mi) / medium (2-10mi) / long (>10mi)
- `time_window`: morning_rush / evening_rush / night / midday
- `day_type`: weekday / weekend
- `neighborhood`: manhattan / brooklyn / queens / bronx / staten_island / airport / unknown

**Neighborhood mapping:**

| Neighborhood | PULocationID Ranges |
|---|---|
| manhattan | 1-50 |
| brooklyn | 51-100 |
| queens | 101-150 |
| bronx | 151-200 |
| airport | 132 (JFK), 138 (LGA) |
| staten_island | 201-263 |
| unknown | Ngoài phạm vi 1-263 |

#### 5.1.3 IForest Threshold Specifics

**sklearn IsolationForest scoring:**
- `score_samples(x)` trả về negative scores: lower = more anomalous
- Để so sánh trực tiếp, ta negate: `anomaly_score = -raw_score` (higher = more anomalous)
- Score distribution: normal records cluster near 0, anomalies have more negative raw scores → higher anomaly_score

**Threshold comparison:**
```
is_anomaly = anomaly_score > threshold
           = (-raw_score) > threshold
           = raw_score < -threshold
```

### 5.2 Threshold Adjustment — Online (Concept Drift Response)

Threshold được điều chỉnh **online** khi concept drift được phát hiện. Có **2 cơ chế**:

#### 5.2.1 Cơ chế 1: IEC `_compute_adjusted_threshold()`

**Trigger:** Khi IEC chọn strategy `adjust_threshold`

**Logic:**
```python
# src/operators/iec_operator.py

anomaly_rate = meta_metrics.get('anomaly_rate', 0.05)

if anomaly_rate > 0.15:   # High FPR → too many false positives
    new_threshold = 0.55  # Increase → fewer anomalies
elif anomaly_rate < 0.03: # Low TPR → missing real anomalies
    new_threshold = 0.45  # Decrease → more sensitive
else:
    new_threshold = 0.50  # Normal → default
```

**Khi nào trigger:**
- `DriftAggregator.assess_drift_severity()` trả về `severity = 'low'`
- Điều kiện: `0 < n_recent_drifts < 3` (dưới threshold)

**Khi nào thực sự áp dụng:**
- Hiện tại **chỉ logged**, KHÔNG được apply vào `IFScoringOperator` (production stub: `"# In production: Adjust anomaly detection thresholds"`)

#### 5.2.2 Cơ chế 2: Model Retrain/Switch (IEC)

Khi `anomaly_rate` drift nặng, IEC strategy `retrain_model` hoặc `switch_model` được trigger → model mới với thresholds mới được load qua `if-model-updates` Kafka topic (Broadcast State).

### 5.3 Concept Drift Detection — ADWIN-U Chi tiết

#### 5.3.1 ADWIN-U: MultiInstanceADWIN (36 instances)

**File:** `src/iec/adwin_multi_instance.py`

**Cấu trúc:** 6 neighborhoods × 6 metrics = 36 instances

| Metric | Delta (Sensitivity) | Giải thích |
|---|---|---|
| `volume` | 0.005 | Low sensitivity — high variance, dễ false positive |
| `null_rate` | 0.001 | **Highest sensitivity** — critical indicator |
| `violation_rate` | 0.002 | Medium sensitivity |
| `anomaly_rate` | 0.002 | Medium sensitivity |
| `avg_anomaly_score` | 0.003 | Low sensitivity |
| `delta_score` | 0.002 | Medium sensitivity |

**Neighborhoods:**
```
manhattan, brooklyn, queens, bronx, staten_island, airport
```

**ADWIN Detection Algorithm:**

```python
# src/iec/adwin_multi_instance.py → class ADWIN

ADWIN(window_size=1000):
    for each new value:
        1. Append value to sliding window
        2. If n > 100: check drift
           a. Try all split points from n/4 to 3n/4
           b. For each split:
              - left = window[:split], right = window[split:]
              - n1 = len(left), n2 = len(right)
              - if n1 < 20 or n2 < 20: skip
              - mean1 = avg(left), mean2 = avg(right)
              - m = 1 / (1/n1 + 1/n2)  # harmonic mean
              - epsilon_cut = (2/m) * sqrt(delta)  # delta = sensitivity param
              - if |mean1 - mean2| > epsilon_cut:
                  drift_detected = True
        3. If len(window) > 1000: evict oldest
        4. Return drift_detected
```

**Drift detected khi nào:**
- Hai nửa của sliding window có means khác nhau nhiều hơn threshold `epsilon_cut`
- `epsilon_cut` nhỏ khi `delta` lớn → dễ detect drift (sensitive)
- `epsilon_cut` lớn khi `delta` nhỏ → khó detect drift (robust)

**Min samples before checking:** 100
**Min samples per sub-window:** 20
**Max window size:** 1000

#### 5.3.2 Drift Detection trong IECOperator

**File:** `src/operators/iec_operator.py`

36 instances trong `MultiInstanceADWIN`:
- Mỗi instance update mỗi khi meta-metric mới đến
- Drift events được gửi qua Kafka topic `iec-action-replay`

```python
# Trong IECOperator.map(meta_metrics)
drifts = self.adwin_u.update_meta_metrics(meta_metrics)
# → Gọi update() cho tất cả 36 instances (6 neighborhoods × 6 metrics)
# → Trả về list các drift events
```

### 5.4 Drift Response Flow — Full Sequence

```
1. MetaAggregator: 1-minute window emits meta-metrics for "manhattan"
   │
2. IECOperator.map(meta_metrics):
   │
3. → MultiInstanceADWIN.update_meta_metrics() → drift event for manhattan×anomaly_rate
   │
4. → DriftAggregator.add_drift() → thêm vào recent_drifts list
   │
5. → DriftAggregator.assess_drift_severity() → severity
   │
6. → METER.predict() hoặc fallback rules → strategy
   │
7. → _execute_strategy(strategy):
    ├─ do_nothing: log, no action
    ├─ adjust_threshold: log new threshold (stub trong production)
    ├─ retrain_model: emit retrain signal
    └─ switch_model: emit model switch signal
   │
8. → Kafka: iec-action-replay với IEC decision
   │
9. → BroadcastStateLoaderFunction nhận signal → load new model
   │
10. → IFScoringOperator nhận broadcast state update → hot-swap model
```

### 5.5 METER Hypernetwork — Strategy Prediction

**File:** `src/operators/iec_operator.py` → `_predict_strategy()`

#### Training Pipeline

METER model được train offline qua 3 bước:

**Bước 1: Real drift data** — `scripts/train_meter_from_benchmark.py`
- Chạy CA-DQStream với concept drift injection trên NYC Taxi data
- 24 cấu hình: 4 drift types × 3 seeds × 2 algorithms
- Extract meta-features từ post-drift windows: volume, null_rate, violation_rate, anomaly_rate, avg_anomaly_score, delta_score
- Label strategy dựa trên ADWIN drift_count + recovery_ratio

**Bước 2: Synthetic data** — `scripts/generate_meter_training_data.py`
- 7 drift scenarios: normal_operation, threshold_drift, distribution_shift, concept_drift, transient_spike, gradual_degradation, abrupt_severe_drift
- Ranges được calibrate từ benchmark results
- delta_score computed theo thesis equation: `|violation_rate - anomaly_rate| / (violation_rate + anomaly_rate + epsilon)`

**Bước 3: Train MLP** — `src/ml/train_meter.py`
- Data: 90 real records + 1200 synthetic = 1290 total
- Architecture selection: (64,32,16), (128,64,32), (32,16) — pick best via 5-fold CV
- Best: (128, 64, 32) — CV F1 = 0.957
- Sample weights (inverse frequency) cho class imbalance
- Test accuracy: **97.3%**, Test F1: **97.3%**

#### Input features (6D):
```python
features = [
    meta_metrics.get('volume', 1000),
    meta_metrics.get('null_rate', 0.0),
    meta_metrics.get('violation_rate', 0.0),
    meta_metrics.get('anomaly_rate', 0.0),
    meta_metrics.get('avg_anomaly_score', 0.0),
    meta_metrics.get('delta_score', 0.0),
]
```

#### Model files:
```
models/meter_hypernetwork.pkl   # sklearn MLPClassifier, architecture (128, 64, 32)
models/meter_scaler.pkl          # StandardScaler (fitted)
models/meter_metadata.json       # metadata: accuracy, features, CV results
```

#### Verification results (scripts/verify_meter.py):
- All 4 strategy classes correctly triggered
- 88% agreement với fallback rule-based
- All edge cases handled

#### Output classes:
```python
strategy_names = {
    0: 'do_nothing',      # No drift → continue normal
    1: 'adjust_threshold', # Minor drift → adjust beta
    2: 'retrain_model',   # Moderate drift → retrain IForest
    3: 'switch_model'     # Severe drift → switch to alternative
}
```

#### Prediction logic:
```python
features_scaled = scaler.transform(features)
strategy_id = model.predict(features_scaled)[0]
probs = model.predict_proba(features_scaled)[0]
confidence = probs[strategy_id]
```

#### Fallback (nếu METER model không load được):
```python
severity = drift_assessment['severity']
if severity == 'none':       → do_nothing, confidence=1.0
elif severity == 'low':      → adjust_threshold, confidence=0.7
elif severity == 'moderate': → retrain_model, confidence=0.8
else:                        → switch_model, confidence=0.9
```

#### Severity thresholds:
```python
drift_threshold = 3
n_recent = len(recent_drifts)
if n_recent == 0:       → severity = 'none'
elif n_recent < 3:      → severity = 'low'
elif n_recent < 6:      → severity = 'moderate'
else:                    → severity = 'high'
```

---

## 6. All Scenarios & Edge Cases — Tất cả trường hợp

### 6.1 Scenario Matrix

| # | Scenario | Data Path | Operators Affected | Output Topic | Notes |
|---|---|---|---|---|---|
| S1 | Normal record | Kafka → Parse → Dedup → Valid → Canary(clean) → Complex(normal) → Voting(CLEAN) → IEC | All | `dq-stream-processed`, `dq-stream-processed-clean` | Optimal path |
| S2 | Schema violation (L1) | Kafka → Parse → Dedup → Invalid → violation_stream | ParseJson, SchemaValidator | `dq-hard-rule-violations` | Dropped from pipeline |
| S3 | Canary rule violation | Kafka → Parse → Dedup → Valid → Canary(violations) → Voting(ANOMALY, canary) → IEC | CanaryRulesValidator | `dq-stream-anomalies` | Canary overrides ML |
| S4 | ML anomaly (canary clean) | Kafka → Parse → Dedup → Valid → Canary(clean) → Complex(anomaly) → Voting(ANOMALY, ml) → IEC | IFScoringOperator | `dq-stream-anomalies` | ML caught something |
| S5 | Both canary AND ML anomaly | Kafka → Parse → Dedup → Valid → Canary(violations) → Complex(anomaly) → Voting(ANOMALY, canary) | Both branches | `dq-stream-anomalies` | Canary wins (priority) |
| S6 | Duplicate trip_id | Kafka → Parse → Dedup → (seen=True) → None | Deduplicator | — | Dropped silently |
| S7 | JSON parse failure | Kafka → ParseJson → None → filter(None) | ParseJsonFunction | — | Dropped silently |
| S8 | Feature extraction failure | Kafka → ... → IFScoring → features=None → scoring_error='feature_extraction_failed' | FeatureVectorizer | `dq-stream-processed` | Score = -1.0, is_anomaly = False |
| S9 | Model load from broadcast state | IFScoringOperator.open() → get broadcast state | BroadcastStateLoader | All | Model hot-swapped |
| S10 | No model in broadcast state | IFScoringOperator.open() → RuntimeError | IFScoringOperator | — | Job fails fast |
| S11 | IEC do_nothing | Normal flow + IEC evaluates drift | IECOperator | `iec-action-replay` | Log action only |
| S12 | IEC adjust_threshold | Normal flow + IEC strategy=adjust | IECOperator | `iec-action-replay` | Log new threshold (stub) |
| S13 | IEC retrain_model | Normal flow + IEC strategy=retrain | IECOperator | `iec-action-replay` | Emit retrain signal |
| S14 | IEC switch_model | Normal flow + IEC strategy=switch | IECOperator | `iec-action-replay` | Emit switch signal |
| S15 | Concept drift (manhattan) | ADWIN detects manhattan drift | MultiInstanceADWIN, IEC | All | Per-neighborhood isolation |
| S16 | Concept drift (global) | Multiple ADWIN instances trigger | DriftAggregator, IEC | All | Severity=moderate/high |
| S17 | METER model unavailable | IECOperator.open() | IECOperator | All | Fallback to rule-based strategy |
| S18 | Late data (>10s out-of-order) | Kafka → WatermarkAssigner | WatermarkAssigner | All | Vẫn được xử lý (watermark bound) |
| S19 | Idle partition (>30s no data) | Kafka → WatermarkAssigner | WatermarkAssigner | — | Partition marked idle (PyFlink bug fix) |
| S20 | anomaly_rate > 15% | IEC decision | IECOperator | — | Threshold tăng lên 0.55 |
| S21 | anomaly_rate < 3% | IEC decision | IECOperator | — | Threshold giảm xuống 0.45 |
| S22 | Zone outside 1-263 | Kafka → ... → SchemaValidator → Invalid | SchemaValidator | `dq-hard-rule-violations` | Invalid zone |
| S23 | Unknown context key | IFScoring → get_context_key() → unknown context | IFScoringOperator | — | Falls back to global_threshold |
| S24 | OCSVM alternative | Normal flow | IFScoringOperator | All | Switch model via BroadcastState |

### 6.2 Edge Cases Chi tiết

#### Edge Case 1: JSON Parse Failure
```
Input: "not valid json{"
→ ParseJsonFunction.map() returns None
→ filter(lambda x: x is not None) drops it
→ Metrics: KHÔNG được emit (vì không đi qua L1ValidSinkFunction)
→ Không ghi vào Kafka topic nào
→ Không ghi vào MinIO nào
→ Recovery: KHÔNG thể recover — data đã mất
```

#### Edge Case 2: Deduplication Collision
```
Record A: composite="1|2024-01-15 08:30:00|161|237|12.50" → trip_id="a3f5..."
Record B: composite="1|2024-01-15 08:30:00|161|237|12.50" → trip_id="a3f5..." (SAME!)
→ MurmurHash3 collision possible but rare
→ Deduplicator drops B (seen=True từ A)
→ B KHÔNG xuất hiện trong output
→ Violations từ B bị miss
→ Frequency: Rất thấp (64-bit hash)
```

#### Edge Case 3: Model Broadcast State Not Available
```
IFScoringOperator.open():
→ runtime_context.get_broadcast_state(model_state_desc)
→ broadcast_state.get("current_model") returns None
→ Raises RuntimeError("Model not found in Broadcast State - run model loader first")
→ Flink job FAILS FAST at startup
→ Fix: Produce initial model to Kafka if-model-updates topic before starting job
```

#### Edge Case 4: Parallelism Partitioning
```
Parallelism = 4 (4 TaskManager slots)

Deduplicator (keyed by trip_id):
  → Records với cùng trip_id LUÔN đi vào cùng slot
  → Deduplication state được chia theo key
  → 7-day TTL state tồn tại trong RocksDB

IFScoringOperator (BroadcastProcessFunction pattern):
  → Mỗi slot load MỘT copy của IFScoringOperator
  → Mỗi slot có IForest model riêng (từ broadcast state)
  → KHÔNG có cross-slot coordination cho scoring
  → KHÔNG có cross-slot coordination cho drift detection

IECOperator (NOT keyed):
  → Mỗi slot có IEC instance riêng
  → Mỗi slot có MultiInstanceADWIN riêng cho 6 neighborhoods
  → KHÔNG có cross-slot coordination cho IEC decisions
  → Các slot có thể đưa ra strategy khác nhau cho cùng neighborhood
```

#### Edge Case 5: Watermark Stalling
```
Kafka partition có nguồn gốc từ partition X nhưng không có record mới trong 30s+:
→ Watermark không advance cho partition X
→ Global watermark bị giữ lại bởi partition "chậm nhất"
→ Records từ các partition khác có thể bị delay vì watermark
→ PyFlink 1.9 fix: with_idleness(Duration.ofSeconds(30))
  → Partition X được đánh dấu idle sau 30s
  → Watermark advance bình thường bất chấp partition X
```

#### Edge Case 6: Zone Unknown
```
PULocationID = 999 (không trong 1-263):
→ SchemaValidator: valid (zone range check chỉ 1-263, không reject)
→ FeatureVectorizer: get_borough_from_zone(999) → 'unknown'
→ IFScoringOperator: context_key = 'unknown_unknown_unknown_unknown'
→ Threshold: fallback to global_threshold (vì context không có trong threshold_matrix)
→ IFScoringOperator.is_anomaly dựa trên global threshold
→ IECOperator: neighborhood 'unknown' không trong MultiInstanceADWIN default list
  → DriftAggregator.add_drift() có thể nhận neighborhood='unknown'
```

#### Edge Case 7: V1.9 Broadcast State Memory Leak
```
Without clear():
→ broadcast_state.put() được gọi nhiều lần với model updates
→ Mỗi put() thêm entry vào MapState
→ MapState không overwrite mà accumulate
→ Long-running job → memory leak → OOM crash

With clear() (V1.9 fix):
→ broadcast_state.clear() được gọi TRƯỚC mỗi put()
→ MapState reset về empty trước khi thêm model mới
→ Không có memory leak
→ CRITICAL: Thứ tự phải là clear() → put() → put(), không ngược lại
```

---

## 7. Checkpoint & Recovery — Khôi phục lỗi

### 7.1 Flink Checkpoint Configuration

```python
checkpoint_config.set_checkpointing_mode(CheckpointingMode.EXACTLY_ONCE)
checkpoint_config.set_checkpoint_interval(600000)    # 10 phút (600s)
checkpoint_config.set_min_pause_between_checkpoints(300000)  # 5 phút
checkpoint_config.set_checkpoint_timeout(1200000)   # 20 phút
checkpoint_config.set_max_concurrent_checkpoints(1)
checkpoint_config.enable_externalized_checkpoints(
    ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION
)
```

### 7.2 What is Checkpointed vs Not Checkpointed

| Component | Type | Checkpointed | Survives Flink Restart | Survives Flink Cancel |
|---|---|---|---|---|
| Kafka consumer offsets | Managed by Flink | YES (to ZK/KRaft) | YES | NO |
| Deduplicator state (trip_id→bool) | KeyedValueState | YES (RocksDB) | YES | YES (RETAIN_ON_CANCELLATION) |
| Rendezvous inbox state | OperatorMapState | YES (RocksDB) | YES | YES |
| IF model weights | sklearn pickle | **NO** (via Broadcast State) | YES (Kafka compacted) | YES (Kafka) |
| IForest scaler | sklearn pickle | **NO** (via Broadcast State) | YES (Kafka) | YES (Kafka) |
| Threshold matrix | JSON | **NO** (via Broadcast State) | YES (Kafka) | YES (Kafka) |
| METER model | sklearn pickle | NO (file-based) | NO | NO |
| ADWIN-U instances (36) | river.ADWIN objects | NO (in-memory only) | NO | NO |
| IECOperator counters | In-memory | NO | NO | NO |

### 7.3 Recovery Scenarios

#### Scenario A: TaskManager Crash (auto-restart)
```
1. Flink detects TM failure
2. Restart strategy: exponential-delay (10s → 300s max, 2x multiplier)
3. Kafka consumer seek to last committed offset (EXACTLY_ONCE)
4. Deduplicator state: replayed from RocksDB checkpoint
5. IFScoringOperator: reloads model từ Broadcast State (Kafka compacted topic)
   → Kafka consumer group 'cadqstream-complete-pipeline' replay
   → BroadcastStateLoader nhận lại message từ if-model-updates
   → Model được load lại
6. IECOperator: Counters reset (in-memory, không checkpointed)
7. MultiInstanceADWIN: Reset (vì không checkpointed)
8. Processing tiếp tục từ last committed offset
```

#### Scenario B: JobManager Crash
```
1. ZooKeeper/KRaft elect new leader
2. Flink HA picks up checkpoint metadata
3. All operators replay from last successful checkpoint
4. IFScoringOperator: same as Scenario A (model từ Kafka)
5. Kafka consumer: replay từ committed offset
```

#### Scenario C: Cancel Job (external kill)
```
1. RETAIN_ON_CANCELLATION: checkpoint preserved on disk
2. Restart: replay from checkpoint → state intact
3. Kafka consumer: replay từ committed offset
4. Model: vẫn available trong Kafka compacted topic
```

### 7.4 State Backend

```yaml
state.backend: rocksdb
state.backend.incremental: true
state.checkpoints.dir: s3://cadqstream-checkpoints/flink/checkpoints
state.savepoints.dir: s3://cadqstream-checkpoints/flink/savepoints
```

**RocksDB settings:**
```
rocksdb.memory.managed: true
rocksdb.memory.managed.memory-size: 40% of taskmanager managed memory
rocksdb.block.cache-size: 512MB
rocksdb.writebuffer.size: 64MB
rocksdb.writebuffer.count: 3
```

---

## 8. Configuration Matrix — Bảng tham số đầy đủ

### 8.1 Kafka Producer/Consumer

| Parameter | Producer | Consumer |
|---|---|---|
| `bootstrap.servers` | `kafka:9092` | `kafka:9092` |
| `linger.ms` | 5 | - |
| `batch.size` | 32768 | - |
| `buffer.memory` | 67108864 | - |
| `compression.type` | lz4 | - |
| `acks` | 1 | - |
| `group.id` | - | `cadqstream-complete-pipeline` |
| `auto.offset.reset` | - | `earliest` |
| `request.timeout.ms` | - | 120000 |
| `session.timeout.ms` | - | 60000 |
| `heartbeat.interval.ms` | - | 10000 |
| `max.poll.interval.ms` | - | 300000 |
| `consumer.timeout.ms` | - | 60000 |
| `metadata.max.age.ms` | - | 30000 |

### 8.2 Flink Environment

| Parameter | Value |
|---|---|
| Parallelism | 4 |
| `taskmanager.network.memory.fraction` | 0.15 |
| `taskmanager.network.memory.min` | 128 MB |
| Checkpoint mode | EXACTLY_ONCE |
| Checkpoint interval | 600,000 ms (10 min) |
| Min pause | 300,000 ms (5 min) |
| Checkpoint timeout | 1,200,000 ms (20 min) |
| Max concurrent | 1 |
| Externalized cleanup | RETAIN_ON_CANCELLATION |
| Restart strategy | exponential-delay (10s → 300s, 2× multiplier) |
| State backend | rocksdb (incremental) |
| Checkpoint storage | S3 (`s3://cadqstream-checkpoints/flink/checkpoints`) |

### 8.3 sklearn IsolationForest

| Parameter | Default | Production | Unit |
|---|---|---|---|
| `n_estimators` | 200 | 200 | trees |
| `max_samples` | 256 | 256 | records per tree |
| `contamination` | 0.001 | 0.001 | fraction |
| `max_features` | 1.0 | 1.0 | fraction of features |
| `n_jobs` | -1 | -1 | all CPU cores |
| `random_state` | 42 | 42 | - |
| `bootstrap` | False | False | - |
| Feature dimensions | 21D | 21D | - |
| Threshold source | Context-aware | Context-aware | - |
| Model format | pickle | pickle | - |

### 8.4 FeatureVectorizer

| Parameter | Value | Description |
|---|---|---|
| Total dimensions | 21D | 5 raw + 4 derived + 6 temporal + 6 ratio |
| Batch method | `transform_batch(df)` | Vectorized pandas |
| Single-record method | `transform(record)` | Row-by-row |
| Baseline fare_per_mile | 2.5 | $/mile |
| Baseline fare_per_minute | 0.67 | $/min |
| Baseline speed | 12.0 | mph |

### 8.5 IForest Scoring Operator (Broadcast State)

| Parameter | Value | Notes |
|---|---|---|
| Model source | Broadcast State (Kafka) | Hot-swappable |
| Scaler source | Broadcast State (Kafka) | Hot-swappable |
| Threshold source | Broadcast State (Kafka) | Hot-swappable |
| Neighborhood mapping | Broadcast State (Kafka) | Optional |
| Model update topic | `if-model-updates` | Compacted, 1 partition |
| V1.9 broadcast clear | YES | `broadcast_state.clear()` before `put()` |

### 8.6 ADWIN-U (IEC)

| Metric | Delta | Sensitivity |
|---|---|---|
| `volume` | 0.005 | Low |
| `null_rate` | 0.001 | **Highest** |
| `violation_rate` | 0.002 | Medium |
| `anomaly_rate` | 0.002 | Medium |
| `avg_anomaly_score` | 0.003 | Low |
| `delta_score` | 0.002 | Medium |

**Total instances:** 6 neighborhoods × 6 metrics = 36

### 8.7 IEC Strategy Thresholds

| Condition | Threshold | New Threshold | Purpose |
|---|---|---|---|
| `anomaly_rate > 0.15` | 0.15 | 0.55 | Giảm FPR |
| `anomaly_rate < 0.03` | 0.03 | 0.45 | Tăng TPR |
| `0.03 ≤ anomaly_rate ≤ 0.15` | - | 0.50 | Default |

### 8.8 METER Hypernetwork

| Parameter | Value |
|---|---|
| Architecture | MLP (128, 64, 32) |
| Input features | 6D (volume, null_rate, violation_rate, anomaly_rate, avg_anomaly_score, delta_score) |
| Output classes | 4 (do_nothing, adjust_threshold, retrain_model, switch_model) |
| Training data | 90 real + 1200 synthetic = 1290 |
| CV F1 | 0.957 |
| Test accuracy | 97.3% |

### 8.9 MinIO Storage

All pipeline outputs are persisted to MinIO via Flink StreamingFileSink in Parquet format.

**Bucket layout:**
```
raw-zone/         → taxi_trips_raw         (valid records from Layer 1)
quarantine-zone/  → schema_violations       (Layer 1 parse/schema failures)
                   → canary_violations     (Layer 2 canary rule failures)
clean-zone/       → anomaly_scores          (Layer 2 ML scoring results)
                   → meta_metrics           (Layer 3 windowed meta-metrics)
                   → drift_events           (Layer 4 IEC decisions)
                   → alerts                (IEC + pipeline alerts)
```

### 8.10 Prometheus Metrics

**Metrics emitted from Flink job** (via HTTP POST to `cadqstream-metrics:9250`):

| Metric Name | Type | Labels | Trigger |
|---|---|---|---|
| `cadqstream_records_input_total` | counter | topic | mỗi Kafka message nhận |
| `cadqstream_records_valid_total` | counter | layer (L1/L2) | mỗi record pass schema |
| `cadqstream_records_violation_total` | counter | layer, type | mỗi L1 violation |
| `cadqstream_violation_records_total` | counter | layer, type | mỗi record có canary violation |
| `cadqstream_anomalies_canary_total` | counter | layer, rule | mỗi rule violation |
| `cadqstream_anomalies_ml_total` | counter | layer, neighborhood | mỗi ML anomaly |
| `cadqstream_iec_decisions_total` | counter | strategy | mỗi IEC decision |
| `cadqstream_iec_drift_detected_total` | counter | neighborhood | mỗi drift event |
| `cadqstream_meta_volume` | gauge | neighborhood | meta-metric window |
| `cadqstream_meta_anomaly_rate` | gauge | neighborhood | meta-metric window |

**Additional IEC metrics (direct POST):**
| Metric Name | Type | Labels |
|---|---|---|
| `iec_confidence` | gauge | neighborhood |
| `meta_anomaly_rate` | gauge | neighborhood |
| `meta_null_rate` | gauge | neighborhood |
| `meta_delta_score` | gauge | neighborhood |

### 8.11 Kafka Sink Rate Limits

| Topic | Interval | Max Batch | Compression |
|---|---|---|---|
| `dq-hard-rule-violations` | 3s | 100 records | lz4 |
| `dq-stream-processed` | 3s | 200 records | lz4 |
| `dq-stream-anomalies` | 3s | 100 records | lz4 |
| `dq-stream-processed-clean` | 3s | 100 records | lz4 |
| `dq-meta-stream` | 3s | 100 records | lz4 |
| `iec-action-replay` | 60s | 10 records | lz4 |

---

## Appendix A: Kafka Topic Data Flow Summary

```
                                    ┌─ dq-hard-rule-violations (L1 violations)
Kafka: taxi-nyc-raw ──L1──→ valid_stream ──┼─ dq-stream-processed (L1 valid)
  (JSON string)           │                └─ L1ValidSinkFunction metrics
                         │
                         ├─── L2A: Canary ──┬─ dq-stream-anomalies (violations)
                         │   (7 rules)      ├─ dq-stream-processed-clean (clean)
                         │                  └─ CanaryViolationSinkFunction metrics
                         │
                         └─── L2B: Complex ─── dq-stream-anomalies (ML anomalies)
                             (IForest)        └─ MLAnomalySinkFunction metrics
                                                  │
                                                  ▼
                                    L3: Voting (union)
                                    MetaAggregateFunction (1-min window)
                                                  │
                                    dq-meta-stream (voting results + meta-metrics)
                                                  │
                                                  ▼
                                    L4: IEC
                                     │
                                     └─ iec-action-replay (IEC decisions)
```

## Appendix B: MinIO Storage Flow Summary

```
                    ┌─ cadqstream-raw/taxi_trips_raw/ (Parquet)
valid_stream ───────┼─ cadqstream-violations/schema_violations/ (Parquet)
                    │
                    ├─ cadqstream-violations/canary_violations/ (Parquet)
                    ├─ cadqstream-anomalies/anomaly_scores/ (Parquet)
                    │
                    ├─ cadqstream-metrics/meta_metrics/ (Parquet)
                    └─ cadqstream-drift/drift_events/ (Parquet)

All outputs written via Flink StreamingFileSink with Parquet format.
```

## Appendix C: Context Key Reference

```
{trip_type}_{time_window}_{day_type}_{neighborhood}

trip_type:
  short  → trip_distance < 2 miles
  medium → 2 miles <= trip_distance < 10 miles
  long   → trip_distance >= 10 miles

time_window:
  morning_rush → 6am ≤ hour < 10am
  evening_rush → 5pm <= hour < 8pm
  night       → 10pm ≤ hour < 6am
  midday      → all other hours

day_type:
  weekday  → Monday-Friday
  weekend  → Saturday-Sunday

neighborhood:
  manhattan      → PULocationID 1-50
  brooklyn      → PULocationID 51-100
  queens        → PULocationID 101-150
  bronx         → PULocationID 151-200
  airport       → PULocationID 132 (JFK), 138 (LGA)
  staten_island → PULocationID 201-263
  unknown       → Outside 1-263
```

---

*Document version: 2.0 — Updated 2026-05-13 (sklearn IsolationForest Complex Branch)*
