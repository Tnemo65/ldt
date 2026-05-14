# CA-DQStream with Full MemStream - Complete Architecture & Flow

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Infrastructure Architecture](#2-infrastructure-architecture)
3. [Service Responsibilities](#3-service-responsibilities)
4. [Complete Data Flow](#4-complete-data-flow)
5. [Layer-by-Layer Deep Dive](#5-layer-by-layer-deep-dive)
6. [Kafka Topics](#6-kafka-topics)
7. [MemStream Deep Dive](#7-memstream-deep-dive)
8. [Storage & Checkpointing](#8-storage--checkpointing)
9. [Observability Stack](#9-observability-stack)
10. [Startup Sequence](#10-startup-sequence)
11. [Data Models](#11-data-models)

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         CA-DQSTREAM WITH FULL MEMSTREAM                               │
│                    4-Layer Streaming Anomaly Detection Pipeline                       │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   NYC Taxi Data                                                                      
│        │                                                                            
│        ▼                                                                            
│   ┌─────────────┐                                                                    
│   │    Kafka    │  taxi-nyc-raw topic                                               
│   │   Source    │                                                                   
│   └─────────────┘                                                                    
│        │                                                                            
│        ▼                                                                            
│   ┌─────────────────────────────────────────────────────────────────────────┐       
│   │                    FLINK STREAMING PIPELINE                              │       
│   │  ┌───────────────────────────────────────────────────────────────────┐  │       
│   │  │ LAYER 1: Baseline Validation                                      │  │       
│   │  │   Parse JSON → Watermark → KeyGen → Dedup → Schema Validation    │  │       
│   │  └───────────────────────────────┬───────────────────────────────────┘  │       
│   │                                  │ (valid records)                       │       
│   │                                  ▼                                       │       
│   │  ┌───────────────────────────────────────────────────────────────────┐  │       
│   │  │ LAYER 2: Dual-Branch Processing                                    │  │       
│   │  │                                                                   │  │       
│   │  │   ┌─────────────────────┐         ┌────────────────────────────┐  │  │       
│   │  │   │ Canary Branch      │         │ Complex Branch (MEMSTREAM) │  │  │       
│   │  │   │ (7 Rules)         │         │                            │  │  │       
│   │  │   │                   │         │  ┌──────────────────────┐  │  │  │       
│   │  │   │ • fare/dist      │         │  │ 25D Feature Extract │  │  │  │       
│   │  │   │ • speed rules    │         │  └──────────────────────┘  │  │  │       
│   │  │   │ • passenger cnt  │         │  ┌──────────────────────┐  │  │  │       
│   │  │   │                   │         │  │ Denoising AE        │  │  │  │       
│   │  │   │                   │         │  │ 25→50→25            │  │  │  │       
│   │  │   │                   │         │  └──────────────────────┘  │  │  │       
│   │  │   │                   │         │  ┌──────────────────────┐  │  │  │       
│   │  │   │                   │         │  │ Memory Module (FIFO)│  │  │  │       
│   │  │   │                   │         │  └──────────────────────┘  │  │  │       
│   │  │   │                   │         │  ┌──────────────────────┐  │  │  │       
│   │  │   │                   │         │  │ ADWIN Drift Detect  │  │  │  │       
│   │  │   │                   │         │  └──────────────────────┘  │  │  │       
│   │  │   │                   │         │  ┌──────────────────────┐  │  │  │       
│   │  │   │                   │         │  │ BAR Controller      │  │  │  │       
│   │  │   │                   │         │  │ (1-5% update rate) │  │  │  │       
│   │  │   └─────────────────────┘         └────────────────────────────┘  │  │       
│   │  └───────────────────────────────┬───────────────────────────────────┘  │       
│   │                                  │ (both branches)                      │       
│   │                                  ▼                                       │       
│   │  ┌───────────────────────────────────────────────────────────────────┐  │       
│   │  │ LAYER 3: Rendezvous + Voting Ensemble                            │  │       
│   │  │   Merge Canary + Complex → Voting (Canary overrides ML)           │  │       
│   │  │   → 1-min Windowed Meta-Metrics per neighborhood                │  │       
│   │  └───────────────────────────────┬───────────────────────────────────┘  │       
│   │                                  │ (meta-metrics)                       │       
│   │                                  ▼                                       │       
│   │  ┌───────────────────────────────────────────────────────────────────┐  │       
│   │  │ LAYER 4: IEC (Intelligent Evolution Controller)                   │  │       
│   │  │   ADWIN-U Drift Detection (36 instances, 1 per neighborhood)      │  │       
│   │  │   METER Strategy Prediction (adjust/retrain/switch)              │  │       
│   │  └───────────────────────────────────────────────────────────────────┘  │       
│   └─────────────────────────────────────────────────────────────────────────┘       
│        │                                                                            
│        ▼                                                                            
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            
│   │    MinIO    │  │    Kafka    │  │    Kafka    │  │ Prometheus  │            
│   │  (sinks)   │  │ (meta-stream)│ │ (IEC action)│  │ (metrics)   │            
│   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘            
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Infrastructure Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              DOCKER COMPOSE SERVICES                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │                        LAYER 1: KAFKA INFRASTRUCTURE                          │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │  │
│  │  │  zookeeper  │→ │    kafka    │→ │schema-reg   │  │  kafka-ui   │        │  │
│  │  │  :2181      │  │  :9092      │  │  :8081      │  │  :8080      │        │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │  │
│  │                                  │                    │                      │  │
│  │                          ┌─────────────┐      ┌─────────────┐               │  │
│  │                          │kafka-init   │      │kafka-export │               │  │
│  │                          │(creates     │      │ :9308       │               │  │
│  │                          │ topics)     │      │             │               │  │
│  │                          └─────────────┘      └─────────────┘               │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │                           LAYER 2: DATABASE                                  │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │  │
│  │  │   minio     │→ │ minio-metr │  │  mlflow     │        │  │
│  │  │  :9000/9001 │  │   :9096    │  │   :5000    │        │  │
│  │  │             │  │  (pooling)   │  │             │  │             │        │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │  │
│  │                                                                              │  │
│  │  Tables: taxi_trips_raw, schema_violations, canary_violations,              │  │
│  │          anomaly_scores, iec_decisions                                       │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │                           LAYER 3: STORAGE                                    │  │
│  │  ┌─────────────┐  ┌─────────────┐                                            │  │
│  │  │    minio    │→ │ minio-init   │                                            │  │
│  │  │ :9000/:9001 │  │ (creates     │                                            │  │
│  │  │             │  │  buckets)    │                                            │  │
│  │  └─────────────┘  └─────────────┘                                            │  │
│  │                                                                              │  │
│  │  Buckets: cadqstream-checkpoints                                             │  │
│  │  Paths: flink/checkpoints, flink/savepoints, ml-models, mlflow-artifacts    │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │                        LAYER 4: FLINK STREAMING                               │  │
│  │  ┌──────────────────┐                                                       │  │
│  │  │flink-jobmanager  │                                                       │  │
│  │  │     :8081        │                                                       │  │
│  │  └────────┬─────────┘                                                       │  │
│  │           │                                                                 │  │
│  │           ▼                                                                 │  │
│  │  ┌──────────────────┐  ┌────────────────┐  ┌────────────────┐              │  │
│  │  │flink-taskmanager │  │  flink-init    │  │ kafka-producer │              │  │
│  │  │                  │  │(submits job)   │  │  (demo data)   │              │  │
│  │  └──────────────────┘  └────────────────┘  └────────────────┘              │  │
│  │                                                                              │  │
│  │  Checkpoint Storage: RocksDB → MinIO (S3)                                  │  │
│  │  State Backend: rocksdb (incremental checkpoints)                           │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │                        LAYER 5: ML PLATFORM                                  │  │
│  │  ┌─────────────┐  ┌─────────────┐                                            │  │
│  │  │    mlflow   │→ │mlflow-export│                                            │  │
│  │  │   :5000     │  │   :9251     │                                            │  │
│  │  │             │  │             │                                            │  │
│  │  └─────────────┘  └─────────────┘                                            │  │
│  │                                                                              │  │
│  │  Stores: Trained models, experiment runs, metrics                            │  │
│  │  Artifact Backend: MinIO S3                                                  │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐  │
│  │                      LAYER 6: OBSERVABILITY                                  │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │  │
│  │  │ prometheus  │← │   grafana    │  │node-export  │  │cadqstream-   │      │  │
│  │  │   :9090     │  │   :3000     │  │   :9100     │  │  metrics     │      │  │
│  │  │             │  │             │  │             │  │   :9250      │      │  │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘      │  │
│  │                                                                              │  │
│  │  Scrapes: Flink :9248/9249, Kafka :9308, MinIO :9096, MLflow :9251       │  │
│  └──────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Service Responsibilities

### 3.1 Kafka Infrastructure

| Service | Port | Role |
|---------|------|------|
| `zookeeper` | 2181 | Kafka metadata coordination |
| `kafka` | 9092 | Message broker - source and sink |
| `schema-registry` | 8081 | Avro/JSON schema validation |
| `kafka-ui` | 8080 | Web UI for Kafka management |
| `kafka-exporter` | 9308 | Prometheus metrics for Kafka |
| `kafka-init` | - | Creates 8 Kafka topics on startup |

### 3.2 Database Services

| Service | Port | Role |
|---------|------|------|
| `minio` | 9000/9001 | Primary S3-compatible object storage |
| `minio-exporter` | 9096 | Prometheus metrics for MinIO |
| `mlflow` | 5000 | MLflow tracking server |

### 3.3 Storage Services

| Service | Port | Role |
|---------|------|------|
| `minio` | 9000/9001 | S3-compatible object storage |
| `minio-init` | - | Creates buckets on startup |

**MinIO Buckets & Paths:**
```
cadqstream-checkpoints/
├── flink/
│   ├── checkpoints/    # Flink RocksDB checkpoints
│   └── savepoints/     # Manual savepoints
├── ml-models/          # Trained MemStream models
└── mlflow-artifacts/  # MLflow experiment artifacts
```

### 3.4 Flink Services

| Service | Role |
|---------|------|
| `flink-jobmanager` | Job submission, REST API, checkpoint coordination |
| `flink-taskmanager` | Parallel task execution (4 slots) |
| `flink-init` | Submits `flink_job_complete.py` on startup |
| `kafka-producer` | Generates synthetic NYC taxi data for testing |

### 3.5 ML Platform

| Service | Port | Role |
|---------|------|------|
| `mlflow` | 5000 | Experiment tracking, model registry |
| `mlflow-exporter` | 9251 | Prometheus metrics for MLflow |

### 3.6 Observability

| Service | Port | Role |
|---------|------|------|
| `prometheus` | 9090 | Metrics collection & storage |
| `grafana` | 3000 | Dashboards & alerting |
| `node-exporter` | 9100 | Host-level metrics |
| `cadqstream-metrics` | 9250 | Custom Flink metrics endpoint |

---

## 4. Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          COMPLETE DATA FLOW                                          │
│                    From Raw Data to Anomaly Detection to Storage                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  [0] DATA SOURCE                                                                      │
│  ─────────────                                                                       │
│  kafka-producer                                                                      │
│       │                                                                              
│       │ Generates synthetic NYC taxi trips                                           
│       │ (VendorID, tpep_pickup_datetime, tpep_dropoff_datetime,                    
│       │  passenger_count, trip_distance, PULocationID, DOLocationID,                 
│       │  fare_amount, extra, mta_tax, tip_amount, tolls_amount,                    
│       │  improvement_surcharge, total_amount, payment_type)                         
│       ▼                                                                              
│  [1] KAFKA INPUT                                                                     │
│  ────────────                                                                       
│  Topic: taxi-nyc-raw                                                                
│       │                                                                              
│       │ Raw JSON messages                                                           
│       │ {"VendorID":1,"tpep_pickup_datetime":"2024-01-15 08:30:00",...}            
│       ▼                                                                              
│  [2] FLINK PIPELINE                                                                 
│  ───────────────                                                                    
│       │                                                                              
│       ├──────────────────────────────────────────────────────────────┐              
│       │                                                              │              
│       ▼                                                              ▼              
│  ┌─────────────┐                                              ┌─────────────┐       
│  │   LAYER 1   │                                              │  LAYER 1    │       
│  │   (Parse)   │                                              │ (Watermark) │       
│  └─────────────┘                                              └─────────────┘       
│       │                                                              │              
│       ▼                                                              ▼              
│  ┌─────────────┐                                              ┌─────────────┐       
│  │   LAYER 1   │                                              │   LAYER 1    │       
│  │   (KeyGen)  │                                              │  (Dedup)    │       
│  └─────────────┘                                              └─────────────┘       
│       │                                                              │              
│       ▼                                                              ▼              
│  ┌─────────────────────────────────────────────────────────────────────────┐       
│  │                         LAYER 1: Schema Validation                          │       
│  │                                                                          │       
│  │  ┌──────────────────────────────────────────────────────────────────┐   │       
│  │  │ Valid Records ─────────────────────────────────────────────────── │   │       
│  │  │   ↓                                                              │   │       
│  │  └──────────────────────────────────────────────────────────────────┘   │       
│  │                                                                          │       
│  │  ┌──────────────────────────────────────────────────────────────────┐   │       
│  │  │ Invalid Records ──────────────────────────────────────────────── │   │       
│  │  │   ↓ → Kafka: dq-hard-rule-violations                            │   │       
│  │  └──────────────────────────────────────────────────────────────────┘   │       
│  └─────────────────────────────────────────────────────────────────────────┘       
│       │ (valid)                                                                    
│       ▼                                                                              
│  ┌─────────────────────────────────────────────────────────────────────────┐       
│  │                     LAYER 2: DUAL-BRANCH PROCESSING                         │       
│  │                                                                          │       
│  │  ┌─────────────────────────────┐    ┌─────────────────────────────────┐  │       
│  │  │      CANARY BRANCH          │    │       COMPLEX BRANCH             │  │       
│  │  │      (7 Rules Engine)       │    │      (FULL MEMSTREAM)            │  │       
│  │  │                             │    │                                  │  │       
│  │  │  Rule 1: fare/dist ratio   │    │  ┌────────────────────────┐    │  │       
│  │  │  Rule 2: fare/minute ratio│    │  │ 25D Feature Extraction │    │  │       
│  │  │  Rule 3: implied speed     │    │  │ • Temporal: 8D        │    │  │       
│  │  │  Rule 4: passenger count   │    │  │ • Monetary: 7D        │    │  │       
│  │  │  Rule 5: trip distance    │    │  │ • Spatial: 2D        │    │  │       
│  │  │  Rule 6: duration bounds  │    │  │ • Trip: 8D            │    │  │       
│  │  │  Rule 7: payment type     │    │  └────────────────────────┘    │  │       
│  │  │                             │    │              ↓                │  │       
│  │  │ Output:                     │    │  ┌────────────────────────┐    │  │       
│  │  │ • canary_violations: []     │    │  │ Denoising Autoencoder  │    │  │       
│  │  │ • is_clean: bool            │    │  │                        │    │  │       
│  │  │ • confidence: float         │    │  │   25 → 50 → 25        │    │  │       
│  │  └─────────────────────────────┘    │  │                        │    │  │       
│  │                                    │  │   z = encoder(x)        │    │  │       
│  │                                    │  │   x' = decoder(z)      │    │  │       
│  │                                    │  └────────────────────────┘    │  │       
│  │                                    │              ↓                │  │       
│  │                                    │  ┌────────────────────────┐    │  │       
│  │                                    │  │    Memory Module       │    │  │       
│  │                                    │  │                        │    │  │       
│  │                                    │  │   FIFO: 50K samples     │    │  │       
│  │                                    │  │   Gradient detached    │    │  │       
│  │                                    │  └────────────────────────┘    │  │       
│  │                                    │              ↓                │  │       
│  │                                    │  ┌────────────────────────┐    │  │       
│  │                                    │  │      Scoring           │    │  │       
│  │                                    │  │                        │    │  │       
│  │                                    │  │  recon_error = ||x-x'|| │  │  │       
│  │                                    │  │  mem_dist = kNN(z,M)   │    │  │       
│  │                                    │  │  score = max(re, md)  │    │  │       
│  │                                    │  │  is_anomaly = s > β   │    │  │       
│  │                                    │  └────────────────────────┘    │  │       
│  │                                    │              ↓                │  │       
│  │                                    │  ┌────────────────────────┐    │  │       
│  │                                    │  │   BAR Controller       │    │  │       
│  │                                    │  │                        │    │  │       
│  │                                    │  │  should_update_memory  │    │  │       
│  │                                    │  │  → drift: update      │    │  │       
│  │                                    │  │  → budget: update     │    │  │       
│  │                                    │  │  → else: skip (1-5%)  │    │  │       
│  │                                    │  └────────────────────────┘    │  │       
│  │                                    │                                  │  │       
│  │                                    │  Output:                         │  │       
│  │                                    │  • anomaly_score: float         │  │       
│  │                                    │  • is_anomaly: bool             │  │       
│  │                                    │  • neighborhood: str           │  │       
│  │                                    │  • drift_detected: bool         │  │       
│  │                                    │  • bar_rate: float              │  │       
│  │                                    └─────────────────────────────────┘  │       
│  └─────────────────────────────────────────────────────────────────────────┘       
│       │ (canary)           │ (complex)                                             
│       ▼                    ▼                                                       
│  ┌─────────────────────────────────────────────────────────────────────────┐       
│  │                         LAYER 3: VOTING ENSEMBLE                          │       
│  │                                                                          │       
│  │  ┌──────────────────────┐     ┌──────────────────────┐                  │       
│  │  │ Canary Result        │     │ Complex Result        │                  │       
│  │  │ • has_violations     │     │ • anomaly_score       │                  │       
│  │  │ • is_clean           │     │ • is_anomaly          │                  │       
│  │  └──────────────────────┘     └──────────────────────┘                  │       
│  │                                │                                             │       
│  │                                │ (same record, both enriched)               
│  │                                ▼                                            
│  │                    ┌──────────────────────┐                                  
│  │                    │  Voting Ensemble     │                                  
│  │                    │                      │                                  
│  │                    │  Rule:               │                                  
│  │                    │  CANARY OVERRIDES ML │                                  
│  │                    │                      │                                  
│  │                    │  if canary_violate:  │                                  
│  │                    │    decision = ANOMALY │                                  
│  │                    │  elif ML_anomaly:    │                                  
│  │                    │    decision = ANOMALY │                                  
│  │                    │  else:               │                                  
│  │                    │    decision = NORMAL  │                                  
│  │                    └──────────────────────┘                                  
│  │                                │                                            
│  │                                ▼                                            
│  │                    ┌──────────────────────┐                                  
│  │                    │  1-Min Window        │                                  
│  │                    │  (per neighborhood)  │                                  
│  │                    │                      │                                  
│  │                    │  Aggregate:         │                                  
│  │                    │  • volume            │                                  
│  │                    │  • anomaly_rate      │                                  
│  │                    │  • canary_rate       │                                  
│  │                    │  • avg_anomaly_score │                                  
│  │                    └──────────────────────┘                                  
│  │                                │                                            
│  │                                ▼                                            
│  │  ┌──────────────────────────────────────────────────────────────────┐   │       
│  │  │                       OUTPUTS                                        │   │       
│  │  │  • Kafka: dq-meta-stream (per-record + windowed)                  │   │       
│  │  │  • MinIO: anomaly_scores/ (via S3-compatible sink)                      │   │       
│  │  └──────────────────────────────────────────────────────────────────┘   │       
│  └─────────────────────────────────────────────────────────────────────────┘       
│       │ (voting results)                                                             
│       ▼                                                                              
│  ┌─────────────────────────────────────────────────────────────────────────┐       
│  │                     LAYER 4: IEC (INTELLIGENT EVOLUTION CONTROLLER)        │       
│  │                                                                          │       
│  │  Input: Windowed meta-metrics per neighborhood                            │       
│  │                                                                          │       
│  │  ┌──────────────────────────────────────────────────────────────────┐   │       
│  │  │                    ADWIN-U Drift Detection                          │   │       
│  │  │                     (36 instances)                                 │   │       
│  │  │                                                                      │   │       
│  │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │   │       
│  │  │  │ Manhattan   │ │  Brooklyn    │ │   Queens    │ │   Bronx     │ │   │       
│  │  │  │  ADWIN      │ │   ADWIN     │ │   ADWIN     │ │   ADWIN     │ │   │       
│  │  │  │ (anom_rate) │ │  (anom_rate)│ │  (anom_rate)│ │  (anom_rate)│ │   │       
│  │  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │   │       
│  │  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │   │       
│  │  │  │   Bronx     │ │  Airport    │ │ Staten Isl. │ │  ...        │ │   │       
│  │  │  │   ADWIN     │ │   ADWIN     │ │   ADWIN     │ │   ADWIN     │ │   │       
│  │  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘ │   │       
│  │  └──────────────────────────────────────────────────────────────────┘   │       
│  │                                │                                         │       
│  │                                ▼                                         │       
│  │  ┌──────────────────────────────────────────────────────────────────┐   │       
│  │  │                   METER Strategy Prediction                        │   │       
│  │  │                                                                      │   │       
│  │  │  IF drift_detected(neighborhood):                                   │   │       
│  │  │      strategy = predict_strategy(history)                            │   │       
│  │  │      → 'adjust':   Thay doi threshold beta                         │   │       
│  │  │      → 'retrain':  Warmup lai MemStream AE                         │   │       
│  │  │      → 'switch':   Chuyen sang model备                           │   │       
│  │  │                                                                      │   │       
│  │  │  ELSE:                                                              │   │       
│  │  │      strategy = 'monitor' (tiếp tục theo dõi)                        │   │       
│  │  └──────────────────────────────────────────────────────────────────┘   │       
│  │                                                                          │       
│  │  Output:                                                                │       
│  │  • Kafka: iec-action-replay                                            │       
│  │  • MinIO: iec_decisions/                                            │       
│  └─────────────────────────────────────────────────────────────────────────┘       
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Layer-by-Layer Deep Dive

### 5.1 Layer 1: Baseline Validation

```
INPUT: Raw Kafka message (JSON)
       {"VendorID":1,"tpep_pickup_datetime":"2024-01-15 08:30:00",...}

PROCESSING STEPS:
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Step 1: Parse JSON                                                                  │
│ ─────────────────────                                                                  │
│ raw_bytes → json.loads() → dict                                                     │
│                                                                                      │
│ Step 2: Watermark Assigner                                                          │
│ ───────────────────────────                                                          │
│ Extract event time from tpep_pickup_datetime                                        
│ Set watermark = event_time - 30s (handling delay)                                    │
│                                                                                      │
│ Step 3: Key Generation                                                              │
│ ───────────────────────                                                             │
│ Generate trip_id = SHA256(VendorID + pickup_datetime + ...)                         │
│                                                                                      │
│ Step 4: Deduplication                                                               │
│ ─────────────────────                                                               │
│ State: keyed by trip_id                                                             │
│ If trip_id exists → DROP (duplicate)                                                │
│ If trip_id new → PASS & store                                                       │
│                                                                                      │
│ Step 5: Schema Validation                                                           │
│ ───────────────────────                                                             │
│ Check required fields:                                                              │
│   • trip_distance >= 0                                                             │
│   • fare_amount >= 0                                                                │
│   • passenger_count 1-9                                                             │
│   • pickup_datetime valid                                                           │
│   • dropoff_datetime valid                                                         │
│                                                                                      │
│ OUTPUT SPLIT:                                                                       │
│   • Valid → Layer 2                                                                 │
│   • Invalid → Kafka: dq-hard-rule-violations                                        
│              → MinIO: cadqstream-violations/                                        
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Layer 2: Dual-Branch Processing

```
LAYER 2a: CANARY BRANCH (7 Rules)
═══════════════════════════════════════════════════════════════════════════════════════

Rule 1: Fare/Distance Ratio
────────────────────────────
fare_per_mile = fare_amount / trip_distance
IF fare_per_mile > 50 OR fare_per_mile < 0.5:
    VIOLATION

Rule 2: Fare/Minute Ratio  
────────────────────────────
fare_per_minute = fare_amount / duration_minutes
IF fare_per_minute > 10 OR fare_per_minute < 0.1:
    VIOLATION

Rule 3: Implied Speed
───────────────────────
speed = trip_distance / duration_hours
IF speed > 100 OR speed < 0:
    VIOLATION

Rule 4: Passenger Count
────────────────────────
IF passenger_count == 0 OR passenger_count > 9:
    VIOLATION

Rule 5: Trip Distance
───────────────────────
IF trip_distance > 500:
    VIOLATION

Rule 6: Duration Bounds
────────────────────────
IF duration_minutes > 1440 (24h) OR duration_minutes < 1:
    VIOLATION

Rule 7: Payment Type
─────────────────────
IF payment_type NOT IN [1,2,3,4,5,6]:
    VIOLATION

Output Fields:
  • canary_violations: List[str]  # ['rule_1', 'rule_3']
  • is_clean: bool                  # len(violations) == 0
  • confidence: float              # 1.0 - (len(violations) * 0.1)


LAYER 2b: COMPLEX BRANCH (FULL MEMSTREAM)
═══════════════════════════════════════════════════════════════════════════════════════

Step 1: 25D Feature Extraction
───────────────────────────────
Raw Record → FeatureVectorizer25D → 25D Feature Vector

Feature Breakdown (25D):
┌─────────────────────────────────────────────────────────────────┐
│ Temporal (8D)                                                  │
│   f[0-1]: pickup_hour (sin, cos)                               │
│   f[2-3]: dropoff_hour (sin, cos)                             │
│   f[4-5]: pickup_day_of_week (sin, cos)                        │
│   f[6-7]: dropoff_day_of_week (sin, cos)                       │
├─────────────────────────────────────────────────────────────────┤
│ Monetary (7D)                                                  │
│   f[8]:  fare_amount                                           │
│   f[9]:  extra                                                 │
│   f[10]: mta_tax                                               │
│   f[11]: tip_amount                                            │
│   f[12]: tolls_amount                                          │
│   f[13]: improvement_surcharge                                  │
│   f[14]: total_amount                                          │
├─────────────────────────────────────────────────────────────────┤
│ Spatial (2D)                                                   │
│   f[15]: pickup_borough_encoded (0-6)                          │
│   f[16]: dropoff_borough_encoded (0-6)                         │
├─────────────────────────────────────────────────────────────────┤
│ Trip Characteristics (8D)                                      │
│   f[17]: trip_distance                                         │
│   f[18]: passenger_count                                       │
│   f[19]: trip_duration_minutes                                 │
│   f[20]: speed_mph                                             │
│   f[21]: fare_per_mile                                         │
│   f[22]: is_airport_trip (0/1)                                 │
│   f[23]: is_rush_hour (0/1)                                    │
│   f[24]: is_weekend (0/1)                                      │
└─────────────────────────────────────────────────────────────────┘

Step 2: Denoising Autoencoder Scoring
───────────────────────────────────────
25D Features → ENCODER → 50D Latent → DECODER → 25D Reconstruction

Loss = MSE(x, x')

Step 3: Memory Distance
────────────────────────
z = encoder(x)  # 50D latent representation
mem_dist = kNN(z, memory, k=10)  # Mean distance to 10 nearest neighbors

Step 4: Final Score
────────────────────
recon_error = ||x - decoder(z)||²
score = max(recon_error, mem_dist)
is_anomaly = score > beta (default: 0.5)

Step 5: BAR Controller (Memory Update Decision)
───────────────────────────────────────────────
ADWIN monitors anomaly_score stream per neighborhood

Decision Logic:
  IF ADWIN detects drift in neighborhood:
      UPDATE memory with z (detached)
      reason = "drift_detected"
  ELIF IEC grants budget:
      UPDATE memory with z
      reason = "iec_budget_granted"
  ELIF current_bar_rate < min_budget (1%):
      UPDATE memory (prevent starvation)
      reason = "minimum_budget_guarantee"
  ELSE:
      SKIP update
      reason = "no_budget"

Output Fields:
  • anomaly_score: float          # Higher = more anomalous
  • is_anomaly: bool              # score > beta
  • threshold: float               # Beta threshold
  • neighborhood: str              # For per-neighborhood tracking
  • drift_detected: bool          # ADWIN detected drift
  • bar_rate: float               # Current update rate (1-5%)
  • bar_update_reason: str        # Why memory was/wasn't updated
  • memory_updates: int            # Total memory updates
  • ml_model: str                 # "memstream_v1"
```

### 5.3 Layer 3: Voting Ensemble

```
VOTING LOGIC:
═════════════

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         VOTING ENSEMBLE DECISION TREE                               │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│                          ┌─────────────────┐                                        │
│                          │ Record arrives  │                                        │
│                          └────────┬────────┘                                        │
│                                   │                                                 │
│                    ┌──────────────┴──────────────┐                                 │
│                    │                              │                                 │
│                    ▼                              ▼                                 │
│            ┌──────────────┐              ┌──────────────┐                          │
│            │ Canary says  │              │ Canary says  │                          │
│            │ VIOLATION    │              │ CLEAN        │                          │
│            └──────┬───────┘              └──────┬───────┘                          │
│                   │                              │                                  │
│                   │                              │                                  │
│                   ▼                              ▼                                  │
│           ┌──────────────┐              ┌──────────────┐                          │
│           │ Canary       │              │ ML says      │                          │
│           │ OVERRIDES    │              │ ANOMALY?     │                          │
│           │              │              └──────┬───────┘                          │
│           └──────┬───────┘                     │                                  │
│                  │                        ┌───┴────┐                             │
│                  │                    YES  │        │ NO                         │
│                  ▼                        ▼        ▼                              │
│          ┌──────────────┐          ┌────────┐ ┌──────────────┐                   │
│          │ ANOMALY      │          │ANOMALY │ │ NORMAL       │                   │
│          │ (canary)     │          │  (ML)  │ │              │                   │
│          └──────────────┘          └────────┘ └──────────────┘                   │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘

WINDOWED META-METRICS (1-minute windows per neighborhood):
═══════════════════════════════════════════════════════════════════

For each neighborhood (manhattan, brooklyn, queens, bronx, airport, ...):

  volume = COUNT(records in window)
  anomaly_rate = COUNT(is_anomaly=True) / volume
  canary_rate = COUNT(has_violations=True) / volume
  avg_anomaly_score = MEAN(anomaly_scores)
  avg_bar_rate = MEAN(bar_rates)
  drift_events = COUNT(drift_detected=True)
```

### 5.4 Layer 4: IEC

```
IEC FLOW:
═══════════

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    IEC (INTELLIGENT EVOLUTION CONTROLLER)                            │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  INPUT: 1-minute windowed meta-metrics per neighborhood                             │
│                                                                                      │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                    ADWIN-U DRIFT DETECTION (36 instances)                     │   │
│  │                                                                              │   │
│  │  Each neighborhood has its own ADWIN instance:                                │   │
│  │                                                                              │   │
│  │    Manhattan_ADWIN    → monitors Manhattan anomaly_rate stream                │   │
│  │    Brooklyn_ADWIN     → monitors Brooklyn anomaly_rate stream                 │   │
│  │    Queens_ADWIN       → monitors Queens anomaly_rate stream                   │   │
│  │    Bronx_ADWIN        → monitors Bronx anomaly_rate stream                    │   │
│  │    Airport_ADWIN      → monitors Airport anomaly_rate stream                  │   │
│  │    ... (30 more neighborhoods)                                               │   │
│  │                                                                              │   │
│  │  ADWIN Algorithm:                                                            │   │
│  │    1. Maintain sliding window of anomaly_rate values                        │   │
│  │    2. Compare recent window vs older window                                  │   │
│  │    3. If |mean_recent - mean_old| > epsilon:                                 │   │
│  │         → DRIFT DETECTED                                                      │   │
│  │         → Shrink window, continue monitoring                                 │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                                │
│                                    ▼                                                │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                   METER STRATEGY PREDICTION                                    │   │
│  │                                                                              │   │
│  │  When drift detected, predict best response strategy:                        │   │
│  │                                                                              │   │
│  │  Strategy Options:                                                           │   │
│  │                                                                              │   │
│  │  ┌─────────────────┬─────────────────────────────────────────────────┐     │   │
│  │  │ Strategy        │ Action                                          │     │   │
│  │  ├─────────────────┼─────────────────────────────────────────────────┤     │   │
│  │  │ 'adjust'        │ Thay đổi threshold beta                        │     │   │
│  │  │                 │ Ví dụ: 0.5 → 0.6 (tăng threshold)              │     │   │
│  │  ├─────────────────┼─────────────────────────────────────────────────┤     │   │
│  │  │ 'retrain'       │ Warmup lại MemStream AE                        │     │   │
│  │  │                 │ Với dữ liệu mới từ memory                      │     │   │
│  │  ├─────────────────┼─────────────────────────────────────────────────┤     │   │
│  │  │ 'switch'        │ Chuyển sang model khác                         │     │   │
│  │  │                 │ Ví dụ: IsolationForest, OCSVM                  │     │   │
│  │  └─────────────────┴─────────────────────────────────────────────────┘     │   │
│  │                                                                              │   │
│  │  Prediction based on:                                                         │   │
│  │    • Drift magnitude (small/medium/large)                                   │   │
│  │    • Drift duration (temporary/permanent)                                    │   │
│  │    • Historical strategy effectiveness                                       │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
│                                    │                                                │
│                                    ▼                                                │
│  ┌──────────────────────────────────────────────────────────────────────────────┐   │
│  │                         IEC ACTION REPLAY                                     │   │
│  │                                                                              │   │
│  │  When strategy decided:                                                       │   │
│  │    1. Log iec_decision to MinIO                                         │   │
│  │    2. Publish to Kafka: iec-action-replay                                    │   │
│  │    3. Action worker executes the strategy                                    │   │
│  └──────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Kafka Topics

| Topic | Partitions | Role | Producers | Consumers |
|-------|------------|------|-----------|-----------|
| `taxi-nyc-raw` | 4 | Raw taxi trip data | kafka-producer | Flink (L1) |
| `dq-hard-rule-violations` | 4 | L1 schema violations | Flink (L1) | - |
| `dq-stream-processed` | 4 | Valid L1 records | Flink (L1) | - |
| `dq-stream-anomalies` | 4 | Anomaly flagged records | Flink (L3) | - |
| `dq-stream-processed-clean` | 4 | Clean records (no violations) | Flink (L3) | - |
| `dq-meta-stream` | 4 | Meta-metrics + voting results | Flink (L3) | IEC |
| `iec-action-replay` | 4 | IEC decisions & actions | Flink (L4) | action-worker |
| `if-model-updates` | 1 | ML model broadcast | mlflow | Flink (Complex) |

---

## 7. MemStream Deep Dive

### 7.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              MEMSTREAM CORE ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  INPUT: 25D Feature Vector x                                                        │
│                                                                                      │
│                    ┌─────────────────────────────────────────┐                      │
│                    │           DENOISING AUTOENCODER          │                      │
│                    │                                          │                      │
│                    │  ┌─────────────────────────────────┐   │                      │
│  x (25D) ────────►│  │          ENCODER                  │   │                      │
│                    │  │   Linear(25, 50) → Tanh          │   │                      │
│                    │  │   Linear(50, 25) → Tanh          │   │                      │
│                    │  └──────────────┬──────────────────┘   │                      │
│                    │                 │                      │                      │
│                    │                 z (25D)                │                      │
│                    │                 │                      │                      │
│                    │                 ├──────────────────────┤                      │
│                    │                 │                      │                      │
│                    │  ┌──────────────▼──────────────────┐   │                      │
│                    │  │          DECODER                 │   │                      │
│                    │  │   Linear(25, 50) → Tanh          │   │                      │
│                    │  │   Linear(50, 25) → Tanh          │   │                      │
│                    │  └──────────────┬──────────────────┘   │                      │
│                    │                 │                      │                      │
│                    └─────────────────┼──────────────────────┘                      │
│                                      │ x' (25D)                                    │
│                                      │                                             │
│                                      ▼                                             │
│                         ┌────────────────────────┐                                 │
│                         │  RECONSTRUCTION ERROR  │                                 │
│                         │   MSE = ||x - x'||²    │                                 │
│                         └────────────┬───────────┘                                 │
│                                      │                                             │
│                                      ▼                                             │
│                         ┌────────────────────────┐                                 │
│                         │      MEMORY MODULE      │                                 │
│                         │                         │                                 │
│                         │   ┌─────────────────┐   │                                 │
│                         │   │ Memory Buffer   │   │                                 │
│                         │   │                 │   │                                 │
│                         │   │  z₁            │   │                                 │
│                         │   │  z₂            │   │                                 │
│                         │   │  z₃            │   │                                 │
│                         │   │  ...           │   │                                 │
│                         │   │  z₅₀₀₀₀        │   │                                 │
│                         │   │                 │   │                                 │
│                         │   │  (FIFO, 50K)   │   │                                 │
│                         │   └────────┬────────┘   │                                 │
│                         │            │            │                                 │
│                         │            ▼ kNN        │                                 │
│                         │   mem_dist = mean(k=10   │                                 │
│                         │             nearest)     │                                 │
│                         └────────────┬────────────┘                                 │
│                                      │                                             │
│                                      ▼                                             │
│                         ┌────────────────────────┐                                 │
│                         │     FINAL SCORE       │                                 │
│                         │  score = max(re, md)   │                                 │
│                         │                         │                                 │
│                         │  IF score > beta:       │                                 │
│                         │      → ANOMALY         │                                 │
│                         │  ELSE:                 │                                 │
│                         │      → NORMAL          │                                 │
│                         └────────────────────────┘                                 │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Warmup Phase (Before Production)

```
WARMUP PHASE:
═════════════

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                          MEMSTREAM WARMUP WORKFLOW                                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  1. COLLECT NORMAL DATA                                                             │
│     ───────────────────                                                             │
│     ~50,000 clean NYC taxi trips (no anomalies)                                    
│                                                                                      │
│  2. FEATURE EXTRACTION                                                              │
│     ───────────────────                                                             │
│     50,000 records → 50,000 × 25D feature vectors                                  
│                                                                                      │
│  3. SPLIT DATA                                                                      │
│     ────────────                                                                    │
│     • 10% → Stats (compute mean, std for normalization)                            
│     • 80% → Training (shuffled, mini-batches)                                       
│     • 10% → Memory init (last 10% of normal data)                                  
│                                                                                      │
│  4. TRAIN AUTOENCODER                                                              │
│     ───────────────────                                                             │
│     for epoch in range(500):                                                       
│         for batch in shuffled_training_data:                                         
│             x_noisy = batch + noise(0.1)                                           
│             x_recon = AE(x_noisy)                                                  
│             loss = MSE(x_recon, batch)                                             
│             optimizer.step()                                                        
│                                                                                      │
│     Early stopping if loss doesn't improve for 20 epochs                            
│                                                                                      │
│  5. INITIALIZE MEMORY                                                               │
│     ───────────────────                                                             │
│     Encode last 10% of data: z = AE.encoder(x)                                     
│     Store 50K z vectors in memory (FIFO initialized)                                
│                                                                                      │
│  6. SET THRESHOLD                                                                  │
│     ───────────────                                                                 │
│     beta = 0.5 (default)                                                           
│     Can be tuned based on validation set                                            
│                                                                                      │
│  7. SAVE CHECKPOINT                                                                 │
│     ────────────────                                                                 │
│     to MinIO: ml-models/memstream_warmup_v1.pt                                      │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Streaming Scoring (In Production)

```
STREAMING SCORING:
═══════════════════

Per-record processing:

┌─────────────────────────────────────────────────────────────────────────────────────┐
│  T = T + 1: New record arrives                                                     │
│                                                                                      │
│  1. FEATURE EXTRACTION                                                              │
│     x = extract_features(record)  # 25D                                            │
│                                                                                      │
│  2. SCORE WITH AUTOENCODER                                                          │
│     z = encoder(x)                    # 25D latent                                 
│     x_recon = decoder(z)              # 25D reconstruction                        
│     recon_error = MSE(x, x_recon)     # scalar                                     
│                                                                                      │
│  3. COMPUTE MEMORY DISTANCE                                                         │
│     mem_dist = kNN(z, memory, k=10)  # mean dist to 10 nearest                      
│                                                                                      │
│  4. FINAL SCORE                                                                     │
│     score = max(recon_error, mem_dist)                                             
│     is_anomaly = (score > beta)                                                    
│                                                                                      │
│  5. BAR CONTROLLER DECISION                                                        │
│     should_update, reason = BAR.should_update_memory(neighborhood, score)           
│                                                                                      │
│     IF should_update:                                                               │
│         encoder.eval()                                                              │
│         with torch.no_grad():                                                       
│             z_detached = encoder(x).detach()                                        │
│         memory.update(z_detached)  # FIFO replacement                              
│                                                                                      │
│  6. OUTPUT                                                                          │
│     return {record, score, is_anomaly, neighborhood, bar_rate, ...}                
│                                                                                      │
│  Total latency target: < 50ms per record                                            │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Storage & Checkpointing

### 8.1 Flink Checkpoint Storage

```
CHECKPOINT FLOW:
═════════════════

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                      │
│   Flink TaskManager                                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────────┐   │
│   │  RocksDB State Backend                                                        │   │
│   │                                                                              │   │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │   │
│   │  │ Dedup State │  │ Canary State│  │ Mem Memory  │  │  ADWIN      │        │   │
│   │  │ (trip_ids)  │  │(violations) │  │ (50K zvecs) │  │  windows    │        │   │
│   │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │   │
│   │         │                │                │                │                  │   │
│   └─────────┼────────────────┼────────────────┼────────────────┼──────────────────┘   │
│             │                │                │                │                      │
│             │    Incremental Checkpoint (every 30s)                               │
│             │                │                │                │                      │
│             ▼                ▼                ▼                ▼                      │
│   ┌─────────────────────────────────────────────────────────────────────────────┐  │
│   │                           MINIO (S3)                                         │  │
│   │                                                                              │  │
│   │   s3://cadqstream-checkpoints/flink/checkpoints/                              │  │
│   │   ├── job-xxx/                                                               │  │
│   │   │   ├── ckpt-1/                                                            │  │
│   │   │   │   ├── metadata                                                        │  │
│   │   │   │   ├── 00001.sst                                                      │  │
│   │   │   │   ├── 00002.sst                                                      │  │
│   │   │   │   └── ...                                                            │  │
│   │   │   └── ckpt-2/                                                            │  │
│   │   └── job-yyy/                                                               │  │
│   │                                                                              │  │
│   └─────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 ML Model Storage

```
ML MODEL STORAGE:
══════════════════

MinIO: s3://cadqstream-checkpoints/ml-models/

memstream_warmup_v1/
├── model.pt              # AE weights + normalization stats
├── memory_initial.pt     # Initial memory (50K z vectors)
├── config.json           # Hyperparameters
└── hmac                  # HMAC signature for integrity

memstream_checkpoints/
├── checkpoint_001.pt     # AE + memory state (updated every 10K records)
├── checkpoint_002.pt
└── ...

MLflow: http://mlflow:5000
├── Experiment: cadqstream-memstream
│   ├── Run: warmup_v1
│   │   ├── Metrics: loss, val_auc, val_pr
│   │   ├── Params: hidden_dim, memory_len, beta
│   │   └── Artifacts: model.pt, memory_initial.pt
│   └── Run: checkpoint_001
│       └── ...
```

---

## 9. Observability Stack

### 9.1 Metrics Flow

```
METRICS FLOW:
═════════════

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           METRICS COLLECTION FLOW                                    │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │ Flink JM    │     │ Flink TM    │     │   Kafka     │     │    MinIO    │       │
│  │  :9248      │     │  :9249      │     │  :9308      │     │ :9096/:9096 │       │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘     └──────┬──────┘       │
│         │                   │                   │                   │              │
│         │    Prometheus Scrapes (every 15s)     │                   │              │
│         │                   │                   │                   │              │
│         └───────────────────┼───────────────────┼───────────────────┘              │
│                             │                   │                                  │
│                             ▼                   ▼                                  │
│                   ┌─────────────────────────────────────┐                          │
│                   │            PROMETHEUS                │                          │
│                   │              :9090                  │                          │
│                   │                                      │                          │
│                   │  Time-series database:               │                          │
│                   │  • flink_jobmanager_*               │                          │
│                   │  • flink_taskmanager_*              │                          │
│                   │  • kafka_*                          │                          │
│                   │  • cadqstream_*                     │                          │
│                   └──────────────────┬──────────────────┘                          │
│                                      │                                            │
│                                      │ Grafana queries                             │
│                                      ▼                                            │
│                   ┌─────────────────────────────────────┐                          │
│                   │            GRAFANA                   │                          │
│                   │              :3000                   │                          │
│                   │                                      │                          │
│                   │  Dashboards:                         │                          │
│                   │  • Pipeline Overview                 │                          │
│                   │  • Layer-by-Layer Metrics           │                          │
│                   │  • Anomaly Detection                │                          │
│                   │  • Drift Detection                   │                          │
│                   │  • MemStream Memory                  │                          │
│                   │  • IEC Decisions                     │                          │
│                   └─────────────────────────────────────┘                          │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Custom Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `cadqstream_records_input_total` | Counter | topic | Total records ingested |
| `cadqstream_records_valid_total` | Counter | layer | Valid records per layer |
| `cadqstream_records_violation_total` | Counter | layer, type | Violations per layer |
| `cadqstream_violation_records_total` | Counter | layer, type | Records with violations |
| `cadqstream_anomalies_canary_total` | Counter | layer, rule | Canary rule violations |
| `cadqstream_anomalies_ml_total` | Counter | layer, neighborhood | ML detected anomalies |
| `cadqstream_iec_decisions_total` | Counter | strategy | IEC strategy decisions |
| `cadqstream_iec_drift_detected_total` | Counter | neighborhood | Drift events |
| `cadqstream_meta_volume` | Gauge | neighborhood | Records per minute |
| `cadqstream_meta_anomaly_rate` | Gauge | neighborhood | Anomaly rate |

---

## 10. Startup Sequence

```
STARTUP SEQUENCE:
══════════════════

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           STARTUP TIMELINE                                          │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  T+0s     ┌─────────────────────────────────────────────────────────────────────┐ │
│           │ Docker Compose Up                                                       │ │
│           │ Starts all 20+ containers                                              │ │
│           └─────────────────────────────────────────────────────────────────────┘ │
│                │                                                                     │
│  T+30s    ┌────▼───────────────────────────────────────────────────────────────┐  │
│           │ MinIO, Kafka become healthy                               │  │
│           │ kafka-init creates 8 topics                                            │  │
│           │ minio-init creates buckets                                             │  │
│           └─────────────────────────────────────────────────────────────────────┘  │
│                │                                                                     │
│  T+60s    ┌────▼───────────────────────────────────────────────────────────────┐  │
│           │ Flink JobManager & TaskManager start                                 │  │
│           │ flink-init submits job: flink_job_complete.py                       │  │
│           │ Job status: CREATED → RUNNING                                         │  │
│           └─────────────────────────────────────────────────────────────────────┘  │
│                │                                                                     │
│  T+90s    ┌────▼───────────────────────────────────────────────────────────────┐  │
│           │ (OPTIONAL) Warmup Phase                                               │  │
│           │ IF -TrainModel flag:                                                  │  │
│           │   1. Generate 50K synthetic taxi trips                               │  │
│           │   2. Extract 25D features                                            │  │
│           │   3. Train MemStream AE (500 epochs)                                │  │
│           │   4. Initialize memory with last 10%                                  │  │
│           │   5. Save to MinIO: ml-models/memstream_warmup_v1/                   │  │
│           │   6. Log to MLflow: cadqstream-memstream                             │  │
│           └─────────────────────────────────────────────────────────────────────┘  │
│                │                                                                     │
│  T+120s   ┌────▼───────────────────────────────────────────────────────────────┐  │
│           │ Model Broadcast (if using IFScoringOperator)                         │  │
│           │   1. Download model from MinIO                                        │  │
│           │   2. Publish to Kafka: if-model-updates                              │  │
│           │   3. Flink BroadcastState receives model                              │  │
│           │ NOTE: MemStreamScoringOperator loads directly from MinIO/checkpoint   │  │
│           └─────────────────────────────────────────────────────────────────────┘  │
│                │                                                                     │
│  T+150s   ┌────▼───────────────────────────────────────────────────────────────┐  │
│           │ Kafka Producer starts                                                 │  │
│           │   • Sends 100 records/second to taxi-nyc-raw                         │  │
│           │   • Flink pipeline processes automatically                           │  │
│           └─────────────────────────────────────────────────────────────────────┘  │
│                │                                                                     │
│  T+180s   ┌────▼───────────────────────────────────────────────────────────────┐  │
│           │ Verification                                                          │  │
│           │   • Check Flink job: curl http://localhost:8081/jobs               │  │
│           │   • Check MinIO: mc ls local/cadqstream-anomalies/        │  │
│           │   • Check Kafka: kafka-ui (localhost:8080)                         │  │
│           │   • Check Grafana: dashboards (localhost:3000)                      │  │
│           └─────────────────────────────────────────────────────────────────────┘  │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 11. Data Models

### 11.1 MinIO Data Paths

```sql
-- MinIO S3 Paths (cadqstream-checkpoints bucket)
-- Instead of tables, data is stored as Parquet/JSON files by timestamp

-- Layer 1: Valid raw trips
-- Path: cadqstream-anomalies/trips/taxi_trips_raw/YYYY/MM/DD/HH/trips.parquet
-- Schema equivalent:
/*
trip_id: VARCHAR(64) PRIMARY KEY
vendor_id: INT
pickup_datetime: TIMESTAMP
dropoff_datetime: TIMESTAMP
passenger_count: INT
trip_distance: FLOAT
pickup_location_id: INT
dropoff_location_id: INT
payment_type: INT
fare_amount: FLOAT
total_amount: FLOAT
created_at: TIMESTAMP
*/

-- Layer 1: Schema violations
-- Path: cadqstream-violations/schema/YYYY/MM/DD/HH/violations.parquet
/*
id: BIGINT (auto-increment simulated by file order)
trip_id: VARCHAR(64)
violation_type: VARCHAR(50)
violation_reason: TEXT
kafka_offset: BIGINT
kafka_partition: INT
created_at: TIMESTAMP
*/

-- Layer 2: Canary violations
-- Path: cadqstream-anomalies/canary/YYYY/MM/DD/HH/canary.parquet
/*
id: BIGINT
trip_id: VARCHAR(64)
violation_types: JSON
violation_count: INT
fare_amount: FLOAT
trip_distance: FLOAT
passenger_count: INT
payment_type: INT
pickup_datetime: TIMESTAMP
final_decision: VARCHAR(20)
decision_source: VARCHAR(20)
confidence: FLOAT
created_at: TIMESTAMP
*/

-- Layer 2: ML anomaly scores
-- Path: cadqstream-anomalies/ml/YYYY/MM/DD/HH/anomaly_scores.parquet
/*
id: BIGINT
trip_id: VARCHAR(64)
anomaly_score: FLOAT
threshold: FLOAT
is_anomaly: BOOLEAN
neighborhood: VARCHAR(20)
drift_detected: BOOLEAN
bar_rate: FLOAT
bar_update_reason: VARCHAR(30)
memory_updates: INT
ml_model: VARCHAR(20)
created_at: TIMESTAMP
*/

-- Layer 3: Meta-metrics
-- Path: cadqstream-metrics/meta/YYYY/MM/DD/HH/meta_metrics.parquet
/*
id: BIGINT
window_start: TIMESTAMP
neighborhood: VARCHAR(20)
volume: INT
anomaly_rate: FLOAT
canary_rate: FLOAT
avg_anomaly_score: FLOAT
avg_bar_rate: FLOAT
drift_events: INT
created_at: TIMESTAMP
*/

-- Layer 4: IEC decisions
-- Path: cadqstream-metrics/iec/YYYY/MM/DD/HH/iec_decisions.parquet
/*
id: BIGINT
neighborhood: VARCHAR(20)
strategy: VARCHAR(20)
trigger_drift: BOOLEAN
drift_magnitude: FLOAT
anomaly_rate_before: FLOAT
anomaly_rate_after: FLOAT
executed_at: TIMESTAMP
*/
```

### 11.2 Kafka Message Schemas

```json
// taxi-nyc-raw (input)
{
  "VendorID": 1,
  "tpep_pickup_datetime": "2024-01-15 08:30:00",
  "tpep_dropoff_datetime": "2024-01-15 08:45:00",
  "passenger_count": 2,
  "trip_distance": 2.5,
  "PULocationID": 79,
  "DOLocationID": 170,
  "fare_amount": 8.5,
  "extra": 0.5,
  "mta_tax": 0.5,
  "tip_amount": 2.0,
  "tolls_amount": 0.0,
  "improvement_surcharge": 0.3,
  "total_amount": 11.3,
  "payment_type": 1
}

// dq-meta-stream (output from Layer 3)
{
  "trip_id": "abc123...",
  "timestamp": "2024-01-15T08:30:00Z",
  "neighborhood": "manhattan",
  "canary_violations": [],
  "is_clean": true,
  "anomaly_score": 0.35,
  "is_anomaly": false,
  "threshold": 0.5,
  "drift_detected": false,
  "bar_rate": 0.03,
  "bar_update_reason": "no_budget",
  "memory_updates": 1500,
  "final_decision": "NORMAL",
  "decision_source": "canary"
}

// iec-action-replay (output from Layer 4)
{
  "timestamp": "2024-01-15T08:31:00Z",
  "neighborhood": "manhattan",
  "strategy": "adjust",
  "trigger_drift": true,
  "drift_magnitude": 0.15,
  "anomaly_rate_before": 0.05,
  "anomaly_rate_after": null,
  "action_params": {
    "beta_old": 0.5,
    "beta_new": 0.55
  }
}
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           COMPONENT SUMMARY                                         │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  Layer 1:  Schema Validation → Valid/Invalid split                                 │
│  Layer 2a: Canary Rules (7 rules) → Fast, interpretable decisions                 │
│  Layer 2b: MemStream (AE + Memory) → Deep anomaly detection                       │
│  Layer 3:  Voting Ensemble → Canary overrides ML                                    │
│  Layer 4:  IEC (ADWIN + METER) → Drift detection + strategy                        │
│                                                                                      │
│  Storage:     MinIO (all sinks, checkpoints, models)                             │
│  Messaging:   Kafka (8 topics)                                                       │
│  Processing:  Flink (4 layers, RocksDB state)                                       │
│  ML Platform: MemStream + MLflow                                                     │
│  Observability: Prometheus + Grafana                                                 │
│                                                                                      │
│  Key Metrics:                                                                      │
│    • Anomaly Detection AUC-PR: 0.9996 (MemStream benchmark)                         │
│    • Throughput: 100+ records/second (demo), scalable to 10K+                       │
│    • Latency: < 50ms per record                                                    │
│    • BAR Rate: 1-5% (label efficiency)                                              │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 12. ML Deep Dive: MemStream - Complete Technical Explanation

### 12.1 Problem Context: Why Streaming Anomaly Detection?

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                    WHY STREAMING ANOMALY DETECTION?                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  Traditional Batch Anomaly Detection:                                               │
│  ───────────────────────────────────────                                            │
│                                                                                      │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │ Day 1-30   │ ──► │   Train     │ ──► │   Deploy    │ ──► │   Output    │       │
│  │   Data     │     │  Model     │     │   Model     │     │  (static)   │       │
│  └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘       │
│                                                                                      │
│  Problem: Model becomes stale over time as data distribution changes                 │
│                                                                                      │
│  ─────────────────────────────────────────────────────────────────────────────────   │
│                                                                                      │
│  Streaming Anomaly Detection:                                                        │
│  ───────────────────────────────                                                     │
│                                                                                      │
│  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐               │
│  │  t=1  │─►│  t=2  │─►│  t=3  │─►│  t=4  │─►│  t=5  │─►│  t=6  │  → ...     │
│  │  ⬤   │  │  ⬤   │  │  ⬤   │  │  ⬤   │  │  ⬤   │  │  ⬤   │               │
│  └───────┘  └───────┘  └───────┘  └───────┘  └───────┘  └───────┘               │
│       │         │         │         │         │         │                        │
│       ▼         ▼         ▼         ▼         ▼         ▼                        │
│  ┌─────────────────────────────────────────────────────────────────────┐           │
│  │                     CONTINUOUS MODEL UPDATES                            │           │
│  │                                                                     │           │
│  │  Model adapts to:                                                   │           │
│  │  • New patterns (e.g., new taxi fare structure)                    │           │
│  │  • Concept drift (e.g., COVID lockdown → no traffic)               │           │
│  │  • Seasonal changes (e.g., NYC marathon → road closures)           │           │
│  │  • Data quality changes (e.g., GPS sensor degradation)            │           │
│  └─────────────────────────────────────────────────────────────────────┘           │
│                                                                                      │
│  ─────────────────────────────────────────────────────────────────────────────────   │
│                                                                                      │
│  Why NYC Taxi Anomaly Detection is Hard:                                            │
│  ─────────────────────────────────────                                              │
│                                                                                      │
│  1. HIGH VOLUME: ~500K trips/day × 365 days = 180M+ records/year                   │
│     → Cannot store all data for batch processing                                    │
│                                                                                      │
│  2. CONCEPT DRIFT: NYC taxi patterns change over time                               │
│     → COVID (2020): Traffic dropped 90%                                             │
│     → Post-COVID: Rush hour shifted                                                │
│     → Weather events: Surge pricing patterns                                        │
│                                                                                      │
│  3. LABELS ARE EXPENSIVE:                                                          │
│     → Manual anomaly labeling costs $1-10/record                                    │
│     → 1% of 180M records = $1.8M - $18M/year                                       │
│     → Need label-efficient approach                                                 │
│                                                                                      │
│  4. LOW LATENCY REQUIREMENT:                                                        │
│     → Real-time fraud detection needs < 100ms response                              │
│     → Batch processing unsuitable                                                   │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 12.2 MemStream Algorithm Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         MEMSTREAM ALGORITHM OVERVIEW                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  Reference: Bhatia, S., Liu, R., &amp; Muthiah, S. (2022). "MemStream: Memory-Based    │
│  Streaming Anomaly Detection". ACM SIGKDD Conference.                                 │
│                                                                                      │
│  ─────────────────────────────────────────────────────────────────────────────────   │
│                                                                                      │
│  CORE INTUITION:                                                                    │
│  ────────────────                                                                    │
│                                                                                      │
│  "Normal data points cluster together in feature space.                              │
│   Anomalies are far from normal clusters AND far from memory."                       │
│                                                                                      │
│  ─────────────────────────────────────────────────────────────────────────────────   │
│                                                                                      │
│  TWO PHASES:                                                                         │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                        PHASE 1: WARMUP (OFFLINE)                             │    │
│  │                                                                              │    │
│  │  ┌───────────────────────────────────────────────────────────────────────┐  │    │
│  │  │  Input:  ~50,000 NORMAL taxi trips (clean data)                       │  │    │
│  │  │                                                                       │  │    │
│  │  │  Process:                                                             │  │    │
│  │  │  1. Extract 25D features from each record                             │  │    │
│  │  │  2. Train Denoising Autoencoder (AE) to reconstruct normal data      │  │    │
│  │  │  3. Initialize Memory Module with encoded normal patterns             │  │    │
│  │  │  4. Set anomaly threshold (beta)                                     │  │    │
│  │  │                                                                       │  │    │
│  │  │  Output: Pre-trained AE + Initial Memory + Beta threshold             │  │    │
│  │  └───────────────────────────────────────────────────────────────────────┘  │    │
│  │                                                                              │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                    │                                                 │
│                                    ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                     PHASE 2: STREAMING (ONLINE)                            │    │
│  │                                                                              │    │
│  │  ┌───────────────────────────────────────────────────────────────────────┐  │    │
│  │  │  Input:  Single taxi trip (streaming, one at a time)                 │  │    │
│  │  │                                                                       │  │    │
│  │  │  Process (per record):                                                │  │    │
│  │  │  1. Extract 25D features                                             │  │    │
│  │  │  2. Encode to latent space                                          │  │    │
│  │  │  3. Compute reconstruction error                                    │  │    │
│  │  │  4. Compute memory distance (kNN)                                    │  │    │
│  │  │  5. Combine scores → anomaly score                                   │  │    │
│  │  │  6. BAR Controller decides: update memory or not?                    │  │    │
│  │  │  7. Update memory if allowed                                         │  │    │
│  │  │                                                                       │  │    │
│  │  │  Output: Anomaly score + Decision (ANOMALY/NORMAL)                   │  │    │
│  │  └───────────────────────────────────────────────────────────────────────┘  │    │
│  │                                                                              │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 12.3 Input: 25D Feature Vector

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         INPUT: 25D FEATURE VECTOR                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  Raw NYC Taxi Record:                                                                │
│  ────────────────────                                                                │
│  {                                                                                  │
│    "VendorID": 1,                                                                  │
│    "tpep_pickup_datetime": "2024-01-15 08:30:00",                                   │
│    "tpep_dropoff_datetime": "2024-01-15 08:45:00",                                  │
│    "passenger_count": 2,                                                            │
│    "trip_distance": 2.5,            ← miles                                        │
│    "PULocationID": 79,               ← Manhattan zone                               │
│    "DOLocationID": 170,              ← Upper East Side                              │
│    "fare_amount": 8.50,                                                             │
│    "extra": 0.50,                                                                   │
│    "mta_tax": 0.50,                                                                 │
│    "tip_amount": 2.00,                                                              │
│    "tolls_amount": 0.00,                                                            │
│    "improvement_surcharge": 0.30,                                                   │
│    "total_amount": 11.30,                                                          │
│    "payment_type": 1                                                               │
│  }                                                                                  │
│                                                                                      │
│  ─────────────────────────────────────────────────────────────────────────────────   │
│                                                                                      │
│  Feature Extraction Process:                                                         │
│  ───────────────────────────────                                                    │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 1: Parse datetime → compute derived values                            │    │
│  │ ────────────────────────────────────────────────────────────────────────    │    │
│  │                                                                              │    │
│  │  pickup_hour = 8                                                            │    │
│  │  dropoff_hour = 8                                                           │    │
│  │  pickup_dow = 1 (Monday)                                                   │    │
│  │  dropoff_dow = 1 (Monday)                                                  │    │
│  │  duration_minutes = 15                                                      │    │
│  │  speed_mph = 2.5 / (15/60) = 10 mph                                       │    │
│  │  fare_per_mile = 8.50 / 2.5 = 3.40 $/mile                                  │    │
│  │  is_airport = False (zones not airport)                                    │    │
│  │  is_rush_hour = True (8:30 AM weekday)                                     │    │
│  │  is_weekend = False                                                         │    │
│  │  pickup_borough = 'manhattan' (zone 79)                                    │    │
│  │  dropoff_borough = 'manhattan' (zone 170)                                 │    │
│  │                                                                              │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                    │                                                 │
│                                    ▼                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │ STEP 2: Build 25D feature vector                                          │    │
│  │ ────────────────────────────────────────────────────────────────────────    │    │
│  │                                                                              │    │
│  │  Index  │  Feature Name                  │  Value      │  Encoding           │    │
│  │  ───────┼───────────────────────────────┼─────────────┼────────────────    │    │
│  │   0     │  pickup_hour_sin              │  sin(8)     │  [-1, 1]           │    │
│  │   1     │  pickup_hour_cos              │  cos(8)     │  [-1, 1]           │    │
│  │   2     │  dropoff_hour_sin             │  sin(8)     │  [-1, 1]           │    │
│  │   3     │  dropoff_hour_cos             │  cos(8)     │  [-1, 1]           │    │
│  │   4     │  pickup_dow_sin               │  sin(1)     │  [-1, 1]           │    │
│  │   5     │  pickup_dow_cos               │  cos(1)     │  [-1, 1]           │    │
│  │   6     │  dropoff_dow_sin              │  sin(1)     │  [-1, 1]           │    │
│  │   7     │  dropoff_dow_cos              │  cos(1)     │  [-1, 1]           │    │
│  │   8     │  trip_distance                │  2.5        │  [0, 500]          │    │
│  │   9     │  fare_amount                  │  8.50       │  [0, 1000]         │    │
│  │   10    │  extra                        │  0.50       │  [0, 100]          │    │
│  │   11    │  mta_tax                      │  0.50       │  [0, 10]          │    │
│  │   12    │  tip_amount                   │  2.00       │  [0, 500]          │    │
│  │   13    │  tolls_amount                 │  0.00       │  [0, 100]          │    │
│  │   14    │  improvement_surcharge        │  0.30       │  [0, 10]          │    │
│  │   15    │  total_amount                 │  11.30      │  [0, 2000]         │    │
│  │   16    │  passenger_count              │  2          │  [0, 9]            │    │
│  │   17    │  trip_duration_minutes        │  15.0       │  [0, 1440]         │    │
│  │   18    │  speed_mph                    │  10.0       │  [0, 100]          │    │
│  │   19    │  fare_per_mile                │  3.40       │  [0, 50]           │    │
│  │   20    │  is_airport_trip              │  0          │  {0, 1}            │    │
│  │   21    │  is_rush_hour                 │  1          │  {0, 1}            │    │
│  │   22    │  is_weekend                   │  0          │  {0, 1}            │    │
│  │   23    │  pickup_borough_encoded        │  0 (manh.) │  [0, 6]            │    │
│  │   24    │  dropoff_borough_encoded      │  0 (manh.) │  [0, 6]            │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
│  ─────────────────────────────────────────────────────────────────────────────────   │
│                                                                                      │
│  WHY THESE 25 FEATURES?                                                              │
│  ───────────────────────                                                             │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │ Category       │ # │ Rationale                                           │    │
│  │ ───────────────┼────┼─────────────────────────────────────────────────     │    │
│  │ Temporal       │ 8  │ Time patterns affect normal behavior               │    │
│  │                │    │ (rush hour ≠ night, weekday ≠ weekend)              │    │
│  │ ───────────────┼────┼─────────────────────────────────────────────────     │    │
│  │ Monetary       │ 7  │ Fare structure is primary anomaly signal            │    │
│  │                │    │ Anomalies often involve inflated fares              │    │
│  │ ───────────────┼────┼─────────────────────────────────────────────────     │    │
│  │ Spatial        │ 2  │ Different neighborhoods have different patterns     │    │
│  │                │    │ Airport trips differ from Manhattan trips           │    │
│  │ ───────────────┼────┼─────────────────────────────────────────────────     │    │
│  │ Trip Char.     │ 8  │ Physical constraints catch impossible trips         │    │
│  │                │    │ (negative distance, 0 duration, 500mph speed)      │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
│  ─────────────────────────────────────────────────────────────────────────────────   │
│                                                                                      │
│  WHY SIN/COS ENCODING FOR TIME?                                                     │
│  ─────────────────────────────────                                                  │
│                                                                                      │
│  Problem: Hour = 23 and Hour = 1 are only 2 hours apart numerically                 │
│          but mathematically: |23 - 1| = 22 (WRONG!)                                  │
│                                                                                      │
│  Solution: Circular encoding with sin/cos                                            │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │  Hour  │  hour_sin          │  hour_cos          │  Euclidean distance     │    │
│  │  ──────┼───────────────────┼───────────────────┼──────────────────────    │    │
│  │   0    │  sin(0) = 0.000   │  cos(0) = 1.000   │                        │    │
│  │   1    │  sin(15) = 0.259  │  cos(15) = 0.966  │  dist(0,1) = 0.068    │    │
│  │   8    │  sin(120) = 0.866  │  cos(120) = -0.5  │                        │    │
│  │  23    │  sin(345) = -0.259 │  cos(345) = 0.966 │  dist(23,1) = 0.068  │    │
│  │                                                                              │    │
│  │  Result: Hour 23 and 1 are correctly recognized as "close"                  │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 12.4 Processing: Denoising Autoencoder

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                  PROCESSING: DENOISING AUTOENCODER (DAE)                              │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ─────────────────────────────────────────────────────────────────────────────────   │
│                                                                                      │
│  WHAT IS AN AUTOENCODER?                                                            │
│  ─────────────────────────                                                            │
│                                                                                      │
│  An Autoencoder is a neural network trained to copy its input to its output.         │
│  It learns to compress data into a lower-dimensional representation (encoding)       │
│  and then reconstruct the original data from that encoding.                            │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                          │    │
│  │      INPUT            ENCODER           DECODER           OUTPUT          │    │
│  │       (25D)           (25→50→25)        (25→50→25)          (25D)        │    │
│  │                                                                          │    │
│  │    ┌──────┐     ┌──────────┐    ┌──────────┐     ┌──────┐              │    │
│  │    │      │     │ Linear   │    │          │     │      │              │    │
│  │  ─►│  x   │───►│ (25,50)  │───►│  LATENT  │───►│      │──►           │    │
│  │    │      │     │ Tanh     │    │    z    │     │      │   x'         │    │
│  │    │      │     │ Linear   │    │  (25D)  │     │      │              │    │
│  │    └──────┘     │ (50,25)  │    └──────────┘     └──────┘              │    │
│  │                  │ Tanh     │                                           │    │
│  │                  └──────────┘                                           │    │
│  │                                                                          │    │
│  │                         Training: Minimize ||x - x'||²                   │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
│  ─────────────────────────────────────────────────────────────────────────────────   │
│                                                                                      │
│  WHY DENOISING AUTOENCODER?                                                         │
│  ───────────────────────────────                                                     │
│                                                                                      │
│  Regular Autoencoder: learns to compress normal data well                           │
│                                                                                      │
│  Denoising Autoencoder: adds noise to input during training,                         │
│                         forces the network to learn robust features                  │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │  Training Step:                                                            │    │
│  │  ───────────────                                                            │    │
│  │                                                                           │    │
│  │  1. Take clean input x                                                    │    │
│  │  2. Add Gaussian noise: x_noisy = x + N(0, σ²)                          │    │
│  │  3. Train: minimize ||x - decoder(encoder(x_noisy))||²                  │    │
│  │                                                                           │    │
│  │  Result: Network learns to "denoise" and capture essential structure       │    │
│  │          of normal data, ignoring random fluctuations                      │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
│  ─────────────────────────────────────────────────────────────────────────────────   │
│                                                                                      │
│  HOW THE DENOISING AUTOENCODER DETECTS ANOMALIES:                                    │
│  ────────────────────────────────────────────────────                               │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                           │    │
│  │  NORMAL CASE:                                                             │    │
│  │  ─────────────                                                            │    │
│  │                                                                           │    │
│  │    x_normal = [2.5, 8.5, 15, 10, 3.4, ...]  (typical Manhattan trip)     │    │
│  │                                                                           │    │
│  │    z = encoder(x_normal)           →  z = [0.2, -0.1, 0.5, ...]          │    │
│  │    x' = decoder(z)                →  x' ≈ [2.5, 8.5, 15, 10, 3.4, ...]  │    │
│  │                                                                           │    │
│  │    reconstruction_error = ||x - x'|| = LOW (e.g., 0.05)                  │    │
│  │    → DECISION: NORMAL                                                    │    │
│  │                                                                           │    │
│  │  ──────────────────────────────────────────────────────────────────────   │    │
│  │                                                                           │    │
│  │  ANOMALY CASE:                                                           │    │
│  │  ─────────────                                                            │    │
│  │                                                                           │    │
│  │    x_anomaly = [2.5, 5000, 15, 10, 2000, ...]  ($5000 fare?!?)           │    │
│  │                                                                           │    │
│  │    z = encoder(x_anomaly)        →  z = [0.9, 0.8, -0.7, ...]           │    │
│  │    x' = decoder(z)               →  x' ≈ [2.5, 12, 15, 10, 8, ...]     │    │
│  │                              (decoder tries to reconstruct normal values)  │    │
│  │                                                                           │    │
│  │    reconstruction_error = ||x - x'|| = HIGH (e.g., 45.2)                 │    │
│  │    → DECISION: ANOMALY                                                   │    │
│  │                                                                           │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
│  ─────────────────────────────────────────────────────────────────────────────────   │
│                                                                                      │
│  ARCHITECTURE DETAILS:                                                              │
│  ───────────────────────                                                            │
│                                                                                      │
│  Input Layer:     25 neurons (one per feature)                                      │
│  Hidden Layer 1: 50 neurons + Tanh activation                                       │
│  Bottleneck:      25 neurons (same as input - symmetric architecture)               │
│  Hidden Layer 2:  50 neurons + Tanh activation                                       │
│  Output Layer:    25 neurons                                                        │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                           │    │
│  │  Layer           │ Shape      │ Parameters │ Activation                    │    │
│  │  ───────────────┼────────────┼────────────┼───────────────                │    │
│  │  Input          │ (batch,25) │      0     │ -                              │    │
│  │  Encoder Linear1│ (batch,50) │ 25×50 + 50│ -                              │    │
│  │  Encoder Tanh1  │ (batch,50) │      0     │ tanh                           │    │
│  │  Encoder Linear2│ (batch,25) │ 50×25 + 25│ -                              │    │
│  │  Encoder Tanh2  │ (batch,25) │      0     │ tanh                           │    │
│  │  ───────────────────────────────────────────────                            │    │
│  │  Decoder Linear1│ (batch,50) │ 25×50 + 50│ -                              │    │
│  │  Decoder Tanh1  │ (batch,50) │      0     │ tanh                           │    │
│  │  Decoder Linear2│ (batch,25) │ 50×25 + 25│ -                              │    │
│  │  Decoder Tanh2  │ (batch,25) │      0     │ tanh                           │    │
│  │  ───────────────────────────────────────────────                            │    │
│  │  OUTPUT         │ (batch,25) │            │                                │    │
│  │                                                                           │    │
│  │  Total Parameters: 2 × (25×50 + 50 + 50×25 + 25) = 2 × 2,650 = 5,300     │    │
│  │                                                                           │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
│  WHY TANGH ACTIVATION?                                                              │
│  ─────────────────────                                                               │
│                                                                                      │
│  1. Bounded: output in [-1, 1] — stable gradients                                   │
│  2. Zero-centered: better for backpropagation                                        │
│  3. Sparse: only some neurons activate — better generalization                      │
│  4. Proven effective in autoencoders for anomaly detection                         │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 12.5 Processing: Memory Module & kNN

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                       PROCESSING: MEMORY MODULE + KNN                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ─────────────────────────────────────────────────────────────────────────────────   │
│                                                                                      │
│  WHY A MEMORY MODULE?                                                               │
│  ─────────────────────                                                               │
│                                                                                      │
│  Problem with Autoencoder alone:                                                    │
│  ─────────────────────────────────                                                  │
│  An AE trained on normal data will try to reconstruct ANY input,                     │
│  even anomalies. The reconstruction error will be high for anomalies,                │
│  BUT there's a risk:                                                               │
│                                                                                      │
│  1. Anomalies similar to training data → low reconstruction error                  │
│     (e.g., a $50 fare in a neighborhood where $50 fares are common)                │
│                                                                                      │
│  2. Normal but rare data → high reconstruction error                               │
│     (e.g., a $200 airport trip — expensive but legitimate)                         │
│                                                                                      │
│  Solution: Memory Module                                                            │
│  ───────────────────                                                                │
│  Store representative examples of normal patterns in memory.                          │
│  Compare incoming data to memory to detect if it's far from ALL normal patterns.     │
│                                                                                      │
│  ─────────────────────────────────────────────────────────────────────────────────   │
│                                                                                      │
│  MEMORY MODULE STRUCTURE:                                                           │
│  ───────────────────────────                                                        │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                           │    │
│  │   Memory = Array of encoded normal patterns (z vectors)                   │    │
│  │   Shape: [50,000 × 25]  (50K samples, each 25D)                          │    │
│  │                                                                           │    │
│  │   ┌────────────────────────────────────────────────────────────────┐     │    │
│  │   │  Memory                                                         │     │    │
│  │   │                                                                 │     │    │
│  │   │  z₁   [ 0.12, -0.34,  0.56, ..., -0.78 ]  ← Encoded normal    │     │    │
│  │   │  z₂   [-0.23,  0.45, -0.67, ...,  0.89 ]                      │     │    │
│  │   │  z₃   [ 0.34, -0.56,  0.78, ..., -0.91 ]                      │     │    │
│  │   │  ...                                                         │     │    │
│  │   │  ...                                                         │     │    │
│  │   │  ...                                                         │     │    │
│  │   │  z₅₀₀₀₀ [ 0.45, -0.67,  0.89, ..., -0.12 ]                    │     │    │
│  │   │                                                                 │     │    │
│  │   └────────────────────────────────────────────────────────────────┘     │    │
│  │                                                                           │    │
│  │   FIFO (First-In-First-Out) circular buffer                               │    │
│  │   New patterns replace oldest patterns                                   │    │
│  │                                                                           │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
│  ─────────────────────────────────────────────────────────────────────────────────   │
│                                                                                      │
│  K-NEAREST NEIGHBORS (KNN) DISTANCE:                                               │
│  ───────────────────────────────────────                                             │
│                                                                                      │
│  For each incoming record, compute distance to k nearest neighbors in memory:        │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                           │    │
│  │  Algorithm:                                                                │    │
│  │  ─────────────────────────────────────────────────────────────────────    │    │
│  │                                                                           │    │
│  │  1. Encode incoming record: z_query = encoder(x)                         │    │
│  │  2. Compute Euclidean distance to ALL memory vectors:                    │    │
│  │                                                                           │    │
│  │     dist_i = ||z_query - z_i||₂  for i = 1 to 50,000                   │    │
│  │                                                                           │    │
│  │  3. Find k smallest distances (k=10 by default)                         │    │
│  │                                                                           │    │
│  │     kNN_distances = [d₁, d₂, d₃, ..., d₁₀]  (sorted ascending)        │    │
│  │                                                                           │    │
│  │  4. memory_distance = mean(kNN_distances)                                │    │
│  │                                                                           │    │
│  │  Visualization:                                                            │    │
│  │  ───────────────                                                            │    │
│  │                                                                           │    │
│  │              ●   Memory points (normal patterns)                         │    │
│  │            ● ● ●                                                          │    │
│  │          ● ● ● ● ●                                                        │    │
│  │         ● ● ● ● ● ● ●                                                     │    │
│  │        ● ● ● ● ● ● ● ● ●                                                   │    │
│  │       ● ● ● ● ● ● ● ● ● ●                                                  │    │
│  │              │                                                            │    │
│  │              │  z_query (incoming)                                         │    │
│  │              │                                                            │    │
│  │              │  k=3 nearest neighbors                                      │    │
│  │              ◉  ← distance = mean(d₁, d₂, d₃)                             │    │
│  │                                                                           │    │
│  │  Case A: z_query in dense normal region → small kNN distance              │    │
│  │  Case B: z_query far from all memory → large kNN distance                │    │
│  │                                                                           │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
│  ─────────────────────────────────────────────────────────────────────────────────   │
│                                                                                      │
│  WHY k=10 NEIGHBORS (NOT k=1)?                                                     │
│  ─────────────────────────────────                                                  │
│                                                                                      │
│  k=1:  Sensitive to noise, single nearest neighbor might be outlier                 │
│  k=10: Robust to noise, smooth local density estimate                              │
│  k=50: Too large, might include points from different neighborhoods                │
│                                                                                      │
│  With k=10:                                                                              │
│  • Robust to individual memory corruption                                            │
│  • Captures local density rather than single point                                  │
│  • Common choice in anomaly detection literature                                    │
│                                                                                      │
│  ─────────────────────────────────────────────────────────────────────────────────   │
│                                                                                      │
│  GRADIENT DETACHMENT:                                                               │
│  ─────────────────────                                                              │
│                                                                                      │
│  When updating memory, the encoded vector z is DETACHED from the computation graph:  │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                           │    │
│  │  # WITHOUT DETACHMENT (WRONG):                                           │    │
│  │  z = encoder(x)                                                          │    │
│  │  memory.update(z)    ← z.grad will be computed!                          │    │
│  │                      ← Memory update affects encoder training!            │    │
│  │                      ← CAN LEARN ANOMALY PATTERNS!                        │    │
│  │                                                                           │    │
│  │  # WITH DETACHMENT (CORRECT):                                            │    │
│  │  z = encoder(x)                                                          │    │
│  │  z_detached = z.detach()  ← Gradient flow cut!                           │    │
│  │  memory.update(z_detached) ← No gradient, no learning effect              │    │
│  │                      ← Memory only stores, never trains                  │    │
│  │                                                                           │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 12.6 Scoring: Combining Reconstruction Error & Memory Distance

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                      SCORING: FINAL ANOMALY SCORE                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ─────────────────────────────────────────────────────────────────────────────────   │
│                                                                                      │
│  THE COMBINED SCORE FORMULA:                                                        │
│  ─────────────────────────────                                                      │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                           │    │
│  │                                                                           │    │
│  │    score = max(reconstruction_error, memory_distance)                      │    │
│  │                                                                           │    │
│  │                      ┌──────────────────────────────────────────────┐     │    │
│  │                      │                                              │     │    │
│  │    recon_error =     │  || x - decoder(encoder(x)) ||²            │     │    │
│  │                      │                                              │     │    │
│  │    mem_distance =    │  mean(kNN(x, Memory, k=10))                │     │    │
│  │                      │                                              │     │    │
│  │    score =           │  max(recon_error, mem_distance)             │     │    │
│  │                      │                                              │     │    │
│  │    is_anomaly =      │  score > beta                               │     │    │
│  │                      └──────────────────────────────────────────────┘     │    │
│  │                                                                           │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
│  ─────────────────────────────────────────────────────────────────────────────────   │
│                                                                                      │
│  WHY MAX() INSTEAD OF ADDITION?                                                     │
│  ───────────────────────────────                                                    │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                           │    │
│  │  If we used: score = recon_error + mem_distance                           │    │
│  │                                                                           │    │
│  │  Problem: A small recon_error + small mem_distance = moderate score       │    │
│  │           Might miss anomalies that are "somewhat unusual" in both ways   │    │
│  │                                                                           │    │
│  │  With max(): score = max(recon_error, mem_distance)                      │    │
│  │                                                                           │    │
│  │  A record is flagged if EITHER:                                           │    │
│  │  • Reconstruction error is high (AE can't reconstruct it)                 │    │
│  │  • OR Memory distance is high (no similar patterns in memory)              │    │
│  │                                                                           │    │
│  │  This is a UNION condition: anomaly if (A OR B), not (A AND B)           │    │
│  │                                                                           │    │
│  │  Result: More sensitive to diverse types of anomalies                      │    │
│  │                                                                           │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
│  ─────────────────────────────────────────────────────────────────────────────────   │
│                                                                                      │
│  SCORING DECISION EXAMPLES:                                                         │
│  ───────────────────────────────                                                    │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │  Example 1: Obvious Anomaly (high recon + high memory dist)              │    │
│  │  ─────────────────────────────────────────────────────────────────────    │    │
│  │                                                                           │    │
│  │  Trip: $5,000 fare, 1 mile, 1 minute (fraud)                             │    │
│  │                                                                           │    │
│  │  recon_error = 45.2   (AE: "I can't reconstruct $5000 fare!")            │    │
│  │  mem_distance = 12.8  (No memory of $5000 fare trips)                    │    │
│  │  score = max(45.2, 12.8) = 45.2                                         │    │
│  │  is_anomaly = (45.2 > 0.5) = TRUE                                       │    │
│  │                                                                           │    │
│  ├─────────────────────────────────────────────────────────────────────────────┤    │
│  │  Example 2: Novel Pattern (low recon + high memory dist)                 │    │
│  │  ─────────────────────────────────────────────────────────────────────    │    │
│  │                                                                           │    │
│  │  Trip: $45 fare, 15 miles, 30 min (unusual long airport trip)            │    │
│  │                                                                           │    │
│  │  recon_error = 0.3    (AE: "Okay, I can reconstruct this...")           │    │
│  │  mem_distance = 8.5   (AE: "...but I've never seen THIS pattern")       │    │
│  │  score = max(0.3, 8.5) = 8.5                                            │    │
│  │  is_anomaly = (8.5 > 0.5) = TRUE                                        │    │
│  │                                                                           │    │
│  │  Key insight: Memory distance catches NOVEL patterns AE misses!           │    │
│  │                                                                           │    │
│  ├─────────────────────────────────────────────────────────────────────────────┤    │
│  │  Example 3: Normal Trip (low recon + low memory dist)                   │    │
│  │  ─────────────────────────────────────────────────────────────────────    │    │
│  │                                                                           │    │
│  │  Trip: $8.50 fare, 2.5 miles, 15 min (typical Manhattan midday)          │    │
│  │                                                                           │    │
│  │  recon_error = 0.05  (AE: "Perfect, normal pattern!")                   │    │
│  │  mem_distance = 0.12  (Many similar trips in memory)                    │    │
│  │  score = max(0.05, 0.12) = 0.12                                         │    │
│  │  is_anomaly = (0.12 > 0.5) = FALSE                                      │    │
│  │                                                                           │    │
│  ├─────────────────────────────────────────────────────────────────────────────┤    │
│  │  Example 4: Edge Case (moderate recon + moderate memory dist)           │    │
│  │  ─────────────────────────────────────────────────────────────────────    │    │
│  │                                                                           │    │
│  │  Trip: $25 fare, 3 miles (possibly expensive but possible)             │    │
│  │                                                                           │    │
│  │  recon_error = 0.4    (AE: "Somewhat unusual but okay...")               │    │
│  │  mem_distance = 0.45   (Memory: "Seen similar trips rarely")            │    │
│  │  score = max(0.4, 0.45) = 0.45                                          │    │
│  │  is_anomaly = (0.45 > 0.5) = FALSE                                     │    │
│  │                                                                           │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
│  ─────────────────────────────────────────────────────────────────────────────────   │
│                                                                                      │
│  THRESHOLD (BETA) SELECTION:                                                        │
│  ─────────────────────────────                                                      │
│                                                                                      │
│  Default: beta = 0.5                                                                │
│                                                                                      │
│  How to tune:                                                                      │
│  1. Use validation set with known labels                                           │
│  2. Sweep beta from 0.1 to 2.0                                                    │
│  3. Plot Precision-Recall curve                                                    │
│  4. Choose beta that maximizes F1 or meets precision/recall target                   │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                           │    │
│  │    Precision                                                             │    │
│  │        ↑                                                                │    │
│  │        │     ╭─────────────                                              │    │
│  │        │    ╱              ╲                                             │    │
│  │        │   ╱                ╲                                            │    │
│  │        │  ╱                  ╲                                           │    │
│  │        │ ╱                    ╲                                          │    │
│  │        │╱                      ╲                                         │    │
│  │        └─────────────────────────────→ Recall                              │    │
│  │                                 ↑                                          │    │
│  │                          F1-optimal point                                 │    │
│  │                          (beta = 0.5)                                    │    │
│  │                                                                           │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 12.7 Online Learning: BAR Controller

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                      ONLINE LEARNING: BAR CONTROLLER                                   │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ─────────────────────────────────────────────────────────────────────────────────   │
│                                                                                      │
│  THE LABEL EFFICIENCY PROBLEM:                                                      │
│  ───────────────────────────────                                                     │
│                                                                                      │
│  Traditional Online Learning:                                                        │
│  ────────────────────────────                                                        │
│  • Every record updates the model                                                    │
│  • Problem: Learning from ALL records (including anomalies!)                        │
│  • Result: Model gradually learns to accept anomalies as "normal"                   │
│  • Cost: 100% label cost (need labels for every update)                           │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                           │    │
│  │  Anomaly arrives: $5000 fare                                             │    │
│  │       │                                                                   │    │
│  │       ▼                                                                   │    │
│  │  Traditional: memory.update(encoder($5000_fare))  ← BAD!                 │    │
│  │              Memory now includes fraudulent pattern                        │    │
│  │              Future $5000 fares might seem "normal"                       │    │
│  │                                                                           │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
│  BAR (Budget Allocation Rate) Controller:                                           │
│  ─────────────────────────────────────────                                           │
│  • Only update memory when it's SAFE to do so                                      │
│  • Uses ADWIN to detect concept drift                                               │
│  • Target: 1-5% of records update memory (vs 100% traditional)                       │
│  • Result: ~95-99% label cost reduction                                             │
│                                                                                      │
│  ─────────────────────────────────────────────────────────────────────────────────   │
│                                                                                      │
│  BAR CONTROLLER ALGORITHM:                                                          │
│  ─────────────────────────────                                                      │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                           │    │
│  │  def should_update_memory(neighborhood, score):                            │    │
│  │      """                                                                 │    │
│  │      Decision: Should we add this record's pattern to memory?             │    │
│  │      Returns: (should_update: bool, reason: str)                          │    │
│  │      """                                                                 │    │
│  │                                                                           │    │
│  │      # ──────────────────────────────────────────────────────────────   │    │
│  │      # RULE 1: ADWIN Drift Detection                                    │    │
│  │      # ──────────────────────────────────────────────────────────────   │    │
│  │      adwin = get_adwin_for(neighborhood)                                 │    │
│  │      drift_detected = adwin.update(score)                                │    │
│  │      if drift_detected:                                                  │    │
│  │          return True, "drift_detected"                                   │    │
│  │                                                                           │    │
│  │      # ──────────────────────────────────────────────────────────────   │    │
│  │      # RULE 2: IEC Budget Grant                                         │    │
│  │      # ──────────────────────────────────────────────────────────────   │    │
│  │      if iec_grants_budget:                                                │    │
│  │          return True, "iec_budget_granted"                              │    │
│  │                                                                           │    │
│  │      # ──────────────────────────────────────────────────────────────   │    │
│  │      # RULE 3: Minimum Budget Guarantee                                 │    │
│  │      # ──────────────────────────────────────────────────────────────   │    │
│  │      if current_bar_rate < min_budget_fraction:                           │    │
│  │          return True, "minimum_budget_guarantee"                        │    │
│  │                                                                           │    │
│  │      # ──────────────────────────────────────────────────────────────   │    │
│  │      # RULE 4: No Update                                                 │    │
│  │      # ──────────────────────────────────────────────────────────────   │    │
│  │      return False, "no_budget"                                           │    │
│  │                                                                           │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
│  ─────────────────────────────────────────────────────────────────────────────────   │
│                                                                                      │
│  ADWIN DRIFT DETECTION INSIDE BAR:                                                   │
│  ─────────────────────────────────────                                              │
│                                                                                      │
│  ADWIN = ADaptive WINdowing                                                          │
│                                                                                      │
│  Purpose: Detect when the statistical properties of the score stream change         │
│                                                                                      │
│  Algorithm:                                                                            │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                           │    │
│  │  Window: [w₁, w₂, w₃, ..., wₙ]  (sliding, up to 1000 values)            │    │
│  │                                                                           │    │
│  │  For each new score s:                                                    │    │
│  │  1. Add s to window                                                      │    │
│  │  2. Try all possible split points                                        │    │
│  │  3. Compare mean(window_left) vs mean(window_right)                      │    │
│  │  4. If |mean_left - mean_right| > ε:                                     │    │
│  │         → DRIFT DETECTED                                                  │    │
│  │         → Shrink window (keep recent half)                               │    │
│  │                                                                           │    │
│  │  Example:                                                                │    │
│  │  ────────                                                                │    │
│  │                                                                           │    │
│  │  Before drift: scores = [0.1, 0.2, 0.1, 0.2, 0.1, ...]                  │    │
│  │                 mean ≈ 0.15                                              │    │
│  │                                                                           │    │
│  │  After drift: scores = [0.5, 0.6, 0.5, 0.6, ...]                       │    │
│  │               mean ≈ 0.55                                                │    │
│  │               |0.55 - 0.15| = 0.40 > ε (drift!)                        │    │
│  │                                                                           │    │
│  │  Action: Update memory with recent normal patterns                        │    │
│  │          This adapts the model to the new concept                        │    │
│  │                                                                           │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
│  ─────────────────────────────────────────────────────────────────────────────────   │
│                                                                                      │
│  BAR RATE STATISTICS:                                                               │
│  ───────────────────────                                                             │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                           │    │
│  │  Tracking:                                                                │    │
│  │  • total_records: How many records processed                              │    │
│  │  • memory_updates: How many times memory was updated                      │    │
│  │  • bar_rate = memory_updates / total_records                             │    │
│  │                                                                           │    │
│  │  Typical values:                                                          │    │
│  │  • bar_rate = 0.03 (3%) in stable period                                │    │
│  │  • bar_rate = 0.08 (8%) after drift detected                           │    │
│  │  • bar_rate = 0.01 (1%) minimum guarantee                              │    │
│  │                                                                           │    │
│  │  Visualization:                                                           │    │
│  │                                                                           │    │
│  │    Updates │                                                             │    │
│  │        ▲   │░░░░░░░░░                                                    │    │
│  │        │   │░░░░▓▓▓▓░░░░░                                               │    │
│  │        │   │░░░▓▓░░░░░░░░░                                               │    │
│  │        │   │░░▓░░░░░░░░░░░                                               │    │
│  │        │   │▓▓░░░░░░░░░░░░                                               │    │
│  │        └───┴──────────────────────────────► Time                          │    │
│  │             │░│ = No update  │▓▓│ = Update                              │    │
│  │                   97%            3%                                     │    │
│  │                                                                           │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
│  ─────────────────────────────────────────────────────────────────────────────────   │
│                                                                                      │
│  MINIMUM BUDGET GUARANTEE:                                                          │
│  ───────────────────────────                                                        │
│                                                                                      │
│  Problem: If bar_rate drops below 1%, model becomes stale                           │
│                                                                                      │
│  Solution: Force at least 1% update rate to prevent starvation                      │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                           │    │
│  │  if bar_rate < 0.01:    # 1% minimum                                    │    │
│  │      # Force an update to prevent starvation                             │    │
│  │      return True, "minimum_budget_guarantee"                             │    │
│  │                                                                           │    │
│  │  This ensures:                                                           │    │
│  │  • Model stays fresh even in stable periods                              │    │
│  │  • Memory doesn't become completely static                               │    │
│  │  • Handles gradual concept drift                                         │    │
│  │                                                                           │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 12.8 Algorithm Summary: Why MemStream?

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│              WHY MEMSTREAM IS THE RIGHT ALGORITHM FOR THIS PROBLEM                    │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ─────────────────────────────────────────────────────────────────────────────────   │
│                                                                                      │
│  REQUIREMENT 1: Streaming (one record at a time)                                    │
│  ──────────────────────────────────────────────────────────────                     │
│  ✓ MemStream is designed for streaming from the ground up                          │
│  ✓ Processes one record at a time                                                  │
│  ✓ O(1) memory updates (FIFO circular buffer)                                      │
│  ✓ No need to store all historical data                                            │
│                                                                                      │
│  REQUIREMENT 2: Label Efficiency (labels are expensive)                            │
│  ────────────────────────────────────────────────────────────────                   │
│  ✓ BAR Controller achieves 1-5% update rate                                       │
│  ✓ ADWIN ensures updates only when statistically safe                             │
│  ✓ ~95-99% cost reduction vs traditional approaches                                │
│  ✓ Only labels needed: drift detection + occasional validation                     │
│                                                                                      │
│  REQUIREMENT 3: Concept Drift Adaptation                                           │
│  ──────────────────────────────────────────────                                     │
│  ✓ ADWIN detects distributional changes in real-time                              │
│  ✓ Memory module naturally adapts to new normal patterns                           │
│  ✓ FIFO ensures old patterns don't dominate                                        │
│  ✓ IEC provides strategic responses (adjust/retrain/switch)                        │
│                                                                                      │
│  REQUIREMENT 4: Interpretability                                                    │
│  ───────────────────────────────                                                    │
│  ✓ Anomaly score = max(recon_error, memory_distance) — explainable                 │
│  ✓ Canary rules provide rule-based explanations                                    │
│  ✓ Voting ensemble: "Canary OR ML flagged this"                                     │
│  ✓ Memory patterns can be visualized (t-SNE, PCA)                                 │
│                                                                                      │
│  REQUIREMENT 5: Low Latency (< 50ms)                                               │
│  ──────────────────────────────────────                                            │
│  ✓ Lightweight architecture (5,300 parameters vs millions in deep learning)        │
│  ✓ Single forward pass through small network                                       │
│  ✓ kNN uses efficient PyTorch operations                                          │
│  ✓ Memory lookup is O(1) with circular buffer                                      │
│                                                                                      │
│  REQUIREMENT 6: High Accuracy                                                      │
│  ───────────────────────────                                                        │
│  ✓ Benchmark AUC-PR: 0.9996 (near-perfect)                                        │
│  ✓ Combines two complementary signals:                                             │
│    - Reconstruction error: detects "can't reconstruct" anomalies                   │
│    - Memory distance: detects "never seen this pattern" anomalies                  │
│  ✓ kNN provides local density estimation                                          │
│  ✓ Denoising training improves robustness                                          │
│                                                                                      │
│  ─────────────────────────────────────────────────────────────────────────────────   │
│                                                                                      │
│  COMPARISON WITH ALTERNATIVES:                                                      │
│  ───────────────────────────────                                                    │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │  Algorithm          │ Streaming │ Label Eff. │ Drift Adap. │ Latency │    │    │
│  │  ─────────────────────────────────────────────────────────────────────     │    │
│  │  IsolationForest   │ ✗ (batch) │ ✗ (100%)  │ ✗          │ Medium  │    │    │
│  │  OCSVM             │ ✗ (batch) │ ✗ (100%)  │ ✗          │ Medium  │    │    │
│  │  Autoencoder only  │ ✓         │ ✗ (100%)  │ ✗          │ Low     │    │    │
│  │  LSTM/Transformer  │ ✓         │ ✗ (100%)  │ Partial    │ High    │    │    │
│  │  MemStream         │ ✓         │ ✓ (1-5%)  │ ✓          │ Low     │    │    │
│  │  (ours)            │           │            │            │          │    │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
│  ─────────────────────────────────────────────────────────────────────────────────   │
│                                                                                      │
│  FINAL SCORING ALGORITHM PSEUDOCODE:                                                │
│  ───────────────────────────────────────────                                         │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                                                                           │    │
│  │  def score_record(x):                                                     │    │
│  │      """Score a single record for anomaly detection."""                   │    │
│  │                                                                           │    │
│  │      # ──────────────────────────────────────────────────────────────   │    │
│  │      # STEP 1: Extract 25D features                                      │    │
│  │      # ──────────────────────────────────────────────────────────────   │    │
│  │      features = extract_25d_features(x)                                 │    │
│  │                                                                           │    │
│  │      # ──────────────────────────────────────────────────────────────   │    │
│  │      # STEP 2: Encode to latent space                                   │    │
│  │      # ──────────────────────────────────────────────────────────────   │    │
│  │      z = encoder(features)  # (25D → 25D)                              │    │
│  │                                                                           │    │
│  │      # ──────────────────────────────────────────────────────────────   │    │
│  │      # STEP 3: Reconstruct                                               │    │
│  │      # ──────────────────────────────────────────────────────────────   │    │
│  │      x_recon = decoder(z)  # (25D → 25D)                               │    │
│  │                                                                           │    │
│  │      # ──────────────────────────────────────────────────────────────   │    │
│  │      # STEP 4: Compute reconstruction error                             │    │
│  │      # ──────────────────────────────────────────────────────────────   │    │
│  │      recon_error = mean((features - x_recon)²)                         │    │
│  │                                                                           │    │
│  │      # ──────────────────────────────────────────────────────────────   │    │
│  │      # STEP 5: Compute kNN memory distance                             │    │
│  │      # ──────────────────────────────────────────────────────────────   │    │
│  │      distances = cdist(z, memory, p=2)  # Eucl. dist to all memory    │    │
│  │      kNN_dist = mean(sorted(distances)[:k])                            │    │
│  │                                                                           │    │
│  │      # ──────────────────────────────────────────────────────────────   │    │
│  │      # STEP 6: Combined score                                          │    │
│  │      # ──────────────────────────────────────────────────────────────   │    │
│  │      score = max(recon_error, kNN_dist)                                │    │
│  │                                                                           │    │
│  │      # ──────────────────────────────────────────────────────────────   │    │
│  │      # STEP 7: Decision                                                │    │
│  │      # ──────────────────────────────────────────────────────────────   │    │
│  │      is_anomaly = (score > beta)                                        │    │
│  │                                                                           │    │
│  │      # ──────────────────────────────────────────────────────────────   │    │
│  │      # STEP 8: BAR controller decides memory update                    │    │
│  │      # ──────────────────────────────────────────────────────────────   │    │
│  │      neighborhood = get_neighborhood(x)                                 │    │
│  │      update, reason = bar.should_update_memory(neighborhood, score)     │    │
│  │                                                                           │    │
│  │      if update:                                                          │    │
│  │          z_detached = z.detach()                                        │    │
│  │          memory.update(z_detached)                                       │    │
│  │                                                                           │    │
│  │      return {                                                           │    │
│  │          'anomaly_score': score,                                        │    │
│  │          'is_anomaly': is_anomaly,                                      │    │
│  │          'recon_error': recon_error,                                    │    │
│  │          'mem_distance': kNN_dist,                                      │    │
│  │          'bar_update': update,                                          │    │
│  │          'bar_reason': reason                                           │    │
│  │      }                                                                  │    │
│  │                                                                           │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### 12.9 Complete ML Pipeline Summary

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                         COMPLETE ML PIPELINE SUMMARY                                  │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                     WARMUP PHASE (Before Production)                          │    │
│  │                                                                              │    │
│  │  Input: 50K clean taxi trips                                                 │    │
│  │                                                                              │    │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │    │
│  │  │   Parse    │───►│  Extract    │───►│    Train    │───►│   Save     │   │    │
│  │  │   JSON     │    │   25D Feat  │    │   AE (500e) │    │   Model    │   │    │
│  │  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘   │    │
│  │                                                                              │    │
│  │  Output: memstream_warmup_v1/ (AE + Memory + Beta)                           │    │
│  │          ↓                                                                   │    │
│  │          Upload to MinIO                                                      │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                          │                                          │
│                                          ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                    STREAMING PHASE (In Production)                             │    │
│  │                                                                              │    │
│  │  Input: One taxi trip at a time (from Kafka)                                  │    │
│  │                                                                              │    │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐   │    │
│  │  │   Parse    │───►│  Extract    │───►│    Score    │───►│    BAR      │   │    │
│  │  │   JSON     │    │   25D Feat  │    │   w/ AE     │    │  Controller │   │    │
│  │  └─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘   │    │
│  │                                                                │          │    │
│  │  ┌──────────────────────────────────────────────────────────┐  │          │    │
│  │  │                                                           │  │          │    │
│  │  │   1. Encode: z = encoder(x)                             │  │          │    │
│  │  │   2. Reconstruct: x' = decoder(z)                      │  │          │    │
│  │  │   3. Recon error: ||x - x'||²                         │  │          │    │
│  │  │   4. kNN distance: kNN(z, Memory, k=10)                │  │          │    │
│  │  │   5. Score: max(recon_error, mem_distance)             │  │          │    │
│  │  │   6. Decision: is_anomaly = (score > beta)              │  │          │    │
│  │  │                                                           │  │          │    │
│  │  └───────────────────────────────────────────────────────────┼───────────┘    │    │
│  │                                                              │                │    │
│  │  ┌──────────────────────────────────────────────────────────┐ │                │    │
│  │  │  BAR Controller:                                        │ │                │    │
│  │  │    • ADWIN monitors score per neighborhood              │ │                │    │
│  │  │    • If drift: update memory                           │ │                │    │
│  │  │    • If stable: skip (1-5% update rate)                │ │                │    │
│  │  └──────────────────────────────────────────────────────────┘                │    │
│  │                                                              │                │    │
│  │  Output: Anomaly score + Decision + Memory update (if allowed)               │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                          │                                          │
│                                          ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                       IEC DRIFT MANAGEMENT                                   │    │
│  │                                                                              │    │
│  │  • ADWIN monitors aggregated anomaly_rate per neighborhood                 │    │
│  │  • If drift detected in neighborhood:                                       │    │
│  │      → Strategy = 'adjust' (change beta)                                    │    │
│  │      → Strategy = 'retrain' (warmup AE again)                              │    │
│  │      → Strategy = 'switch' (use different model)                           │    │
│  │  • Execute strategy automatically                                            │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
│  ─────────────────────────────────────────────────────────────────────────────────   │
│                                                                                      │
│  KEY ALGORITHMS SUMMARY:                                                            │
│  ───────────────────────────                                                        │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │  Component           │  Algorithm                  │  Purpose                │    │
│  │  ────────────────────┼────────────────────────────┼─────────────────────     │    │
│  │  Feature Extraction  │  Domain-specific rules      │  25D feature vector     │    │
│  │  Anomaly Scoring     │  Denoising Autoencoder     │  Reconstruction error   │    │
│  │  Novelty Detection   │  k-Nearest Neighbors (k=10)│  Memory distance        │    │
│  │  Drift Detection    │  ADWIN                     │  Detect concept drift    │    │
│  │  Update Control     │  BAR Controller            │  Label-efficient updates │    │
│  │  Strategy Selection │  METER                     │  Choose response action │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 13. Key Metrics Reference

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            KEY ML METRICS                                            │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  Anomaly Detection Performance:                                                      │
│  ──────────────────────────────────                                                 │
│  • AUC-PR: 0.9996 (MemStream benchmark on NYC Taxi)                                 │
│  • AUC-ROC: 0.9998                                                                  │
│  • Precision: 0.95 @ 5% FPR                                                          │
│  • Recall: 0.98                                                                      │
│                                                                                      │
│  System Performance:                                                                 │
│  ─────────────────────                                                                │
│  • Throughput: 100+ records/second (demo), scalable to 10K+                        │
│  • Latency: < 50ms per record (p99)                                                 │
│  • Memory: 50K × 25D × 4 bytes ≈ 5MB                                               │
│  • Model Size: 5,300 parameters ≈ 21KB                                             │
│                                                                                      │
│  BAR Controller Performance:                                                         │
│  ───────────────────────────                                                         │
│  • Update Rate: 1-5% (target: 3%)                                                  │
│  • Drift Detection: < 5 minutes (ADWIN window)                                     │
│  • False Positive Rate: < 1%                                                       │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

