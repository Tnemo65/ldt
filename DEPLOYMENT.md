# CA-DQStream Deployment Guide

Production deployment guide for the **Context-Aware Data Quality Stream Processing System** (CA-DQStream). This guide walks a new engineer through a complete deployment from a clean environment to a fully verified, running system.

**Estimated time:** 15-25 minutes (first run, includes image builds; ~5 min on subsequent runs)

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Prerequisites](#2-prerequisites)
3. [Environment Setup (.env)](#3-environment-setup-env)
4. [Docker Network & Data Directories](#4-docker-network--data-directories)
5. [Build Custom Docker Images](#5-build-custom-docker-images)
6. [Startup: One-Command Full Deployment](#6-startup-one-command-full-deployment)
7. [Service Verification: Layer by Layer](#7-service-verification-layer-by-layer)
8. [End-to-End Flow Test (Inject Data)](#8-end-to-end-flow-test-inject-data)
9. [Verify All Grafana Dashboards](#9-verify-all-grafana-dashboards)
10. [Troubleshooting](#10-troubleshooting)
11. [Stopping & Restarting](#11-stopping--restarting)
12. [Service URLs & Credentials](#12-service-urls--credentials)
13. [Quick Reference Commands](#13-quick-reference-commands)

---

## 1. Architecture Overview

> **Pipeline type:** Sequential (Phase 3). Records flow L1 → L2 → L3 → L4. No parallel branches.

| Layer | Description |
|-------|-------------|
| **L1** | Kafka → Flink: SafeParseJson → Watermark(30s) → Dedup → SchemaValidator → violations → MinIO cadqstream-violations/ |
| **L2** | Sequential: CanaryRulesValidator (7 rules) → ExtractNeighborhood → MemStreamScoringOperator (34D→AE→kNN→ContextBeta) |
| **L3** | MetaAggregator: 1-min tumbling window, 6 meta-metrics per neighborhood → MinIO cadqstream-metrics/ + Kafka dq-stream-unified |
| **L4** | IEC: MultiInstanceADWIN (60 instances) → DriftAggregator → Strategy: do_nothing / quick_retrain → Kafka iec-action-replay |

> **Note:** ADWIN instances = 10 neighborhoods × 6 metrics = 60 (not 36). Kafka topic `dq-meta-stream` is unused in Phase 3 (meta written to MinIO `cadqstream-metrics/`).

### Kafka Topics (Active)

| Topic | Partitions | Purpose |
|-------|-----------|---------|
| `taxi-nyc-raw-v2` | 8 | Raw taxi event input (canonical topic) |
| `taxi-nyc-raw` | 8 | Legacy input (baseline job only) |
| `dq-stream-unified` | 4 | Unified pipeline output (all event types) |
| `iec-action-replay` | 1 | IEC drift action buffer |
| `iec-action-dlq` | 1 | IEC dead letter queue |
| `memstream-model-updates` | 4 | Retrained model broadcast (compact) |

### MinIO Buckets (8 total)

| Bucket | Purpose |
|--------|---------|
| `cadqstream-raw` | Valid taxi trips |
| `cadqstream-violations` | Schema + canary violations |
| `cadqstream-anomalies` | MemStream anomaly scores |
| `cadqstream-metrics` | MetaAggregator windowed metrics |
| `cadqstream-drift` | IEC decisions and alerts |
| `cadqstream-dlq` | Dead letter queue records |
| `cadqstream-checkpoints` | Flink RocksDB checkpoints |
| `ml-models` | Model artifacts and configs |

---

## 2. Prerequisites

### 2.1 Hardware

- **RAM:** 16 GB minimum, 32 GB recommended
- **CPU:** 4+ cores
- **Disk:** 50 GB free
- **OS:** Windows 10/11 (with Docker Desktop + WSL2), Linux, or macOS

### 2.2 Software

Verify each component is installed and running:

```powershell
# Check Docker
docker --version
# Expected: Docker version 24.x.x or higher

# Check Docker Compose v2 (plugin)
docker compose version
# Expected: Docker Compose version v2.x.x or higher

# Check WSL2 (Windows only)
wsl --list --verbose
# Expected: Ubuntu or similar listed and running
```

### 2.3 Ports Required

The following ports must be free before deployment:

| Port | Service |
|------|---------|
| 2181 | Zookeeper |
| 9092 | Kafka (internal) |
| 29092 | Kafka (host) |
| 8080 | Kafka UI |
| 8081 | Flink JobManager UI |
| 8082 | Schema Registry |
| 3000 | Grafana |
| 9000 | MinIO API |
| 9001 | MinIO Console |
| 9090 | Prometheus |
| 9100 | Node Exporter |
| 9250 | cadqstream-metrics |
| 9308 | Kafka Exporter |
| 6379 | Redis |

---

## 3. Environment Setup (.env)

### 3.1 Create the .env File

Copy the `.env` file from the project root — it is already pre-configured with all values:

The `.env` file at `C:\proj\ldt\.env` is already pre-configured. All passwords and keys are set. Do NOT copy from an `.env.example` — the real file is in place.

### 3.2 Required Variables

The `.env` file is already at `C:\proj\ldt\.env` with all values configured. View it directly:

```powershell
Get-Content .env
```

Key values already set:

| Variable | Value | Service |
|----------|-------|---------|
| `MINIO_ROOT_USER` | `cadqstream` | MinIO |
| `MINIO_ROOT_PASSWORD` | `CADQStream2026!` | MinIO |
| `CADQSTREAM_PASS` | `CADQStream2026!` | Shared (Redis, Grafana, MinIO) |
| `REDIS_PASSWORD` | `CADQStream2026!` | Redis |
| `GRAFANA_PASSWORD` | `CADQStream2026!` | Grafana |
| `INTERNAL_API_KEY` | `cadqstream_internal_key_v1_...` | ML Service |
| `METRICS_API_KEY` | `cadqstream_metrics_key_v1_...` | ML Service |
| `MEMSTREAM_MODEL_SIGNING_KEY` | `cadqstream_hmac_memstream_...` | Model HMAC |
| `IEC_SIGNING_KEY` | `cadqstream_hmac_iec_...` | IEC HMAC |

### 3.3 Generate New Credentials (Optional)

If you want to regenerate all security keys, run:

---

## 4. Docker Network & Data Directories

Docker Compose automatically creates the `cadqstream-net` bridge network on first `up`.

Named volumes are created automatically:

- `ldt-zookeeper-data` — Zookeeper data
- `ldt-kafka-data` — Kafka broker data
- `ldt-redis-data` — Redis persistence
- `ldt-minio-data` — MinIO object storage
- `ldt-flink-checkpoints` — Flink RocksDB checkpoints
- `ldt-flink-jobmanager-log` — JobManager logs
- `ldt-flink-taskmanager-log` — TaskManager logs
- `ldt-grafana-data` — Grafana dashboards and config
- `ldt-prometheus-data` — Prometheus time-series data
- `ldt-ml-models` — Persisted ML model artifacts

---

## 5. Build Custom Docker Images

**IMPORTANT:** On first run, you MUST build the custom images. Docker Compose can build them automatically, but explicit build ensures proper layer caching.

### 5.1 Build All Images (First Run / After Code Changes)

From the project root (`C:\proj\ldt`). Note the `-f Dockerfile` flag — it is required for services whose Dockerfile lives in a subdirectory (build context differs from the Dockerfile location):

```powershell
# Flink image: project-root context
docker build -t ldt-flink:1.18.1-py -f deployment/flink/Dockerfile .

# cadqstream-metrics: subdirectory context
docker build -t ldt-cadqstream-metrics:latest -f deployment/cadqstream-metrics/Dockerfile deployment/cadqstream-metrics/

# ml-service: project-root context (requires src/api/ml_service.py and deployment/ml-service/requirements.txt)
docker build -t ldt-ml-service:latest -f deployment/ml-service/Dockerfile .

# stats-writer: subdirectory context
docker build -t ldt-stats-writer:latest -f deployment/stats-writer/Dockerfile deployment/stats-writer/

# action-replay-worker: subdirectory context
docker build -t ldt-action-replay-worker:latest -f deployment/action-replay-worker/Dockerfile deployment/action-replay-worker/

# kafka-producer: subdirectory context
docker build -t ldt-kafka-producer:latest -f deployment/kafka/Dockerfile.producer deployment/kafka/

# e2e-test-injector: subdirectory context
docker build -t ldt-e2e-test-injector:latest -f deployment/kafka/Dockerfile.e2e-test deployment/kafka/
```

> **Note:** The `ldt-flink:1.18.1-py` image takes 5-15 minutes on first build (downloads PyTorch CPU-only, Apache Beam, PyFlink, Kafka connector JARs). This is cached on subsequent builds.

### 5.2 Build Tips

```powershell
# To force a clean rebuild of a specific service (e.g., after updating ml_service.py):
# ml-service uses project-root context
docker build --no-cache -t ldt-ml-service:latest -f deployment/ml-service/Dockerfile .

# cadqstream-metrics uses subdirectory context
docker build --no-cache -t ldt-cadqstream-metrics:latest -f deployment/cadqstream-metrics/Dockerfile deployment/cadqstream-metrics/

# To rebuild only the Flink image (fastest way to pick up code changes):
docker build -t ldt-flink:1.18.1-py -f deployment/flink/Dockerfile .
```

---

## 6. Startup: One-Command Full Deployment

### 6.1 Clean Slate (Idempotent — Safe to Run Multiple Times)

```powershell
# Navigate to project root (all commands run from C:\proj\ldt)
cd C:\proj\ldt

# Option A: Use the PowerShell start script (RECOMMENDED — builds images, starts all services, verifies)
powershell -ExecutionPolicy Bypass -File deployment/scripts/start.ps1

# Option B: Skip image rebuild if images already exist (faster restart)
powershell -ExecutionPolicy Bypass -File deployment/scripts/start.ps1 -SkipBuild

# Option C: Rebuild Flink only, then restart all services
powershell -ExecutionPolicy Bypass -File deployment/scripts/start.ps1 -ForceRestartFlinkJob

# Option D: Direct docker compose (requires images to be pre-built)
docker compose -f deployment/docker-compose.yml down --remove-orphans
docker compose -f deployment/docker-compose.yml up -d
```

### 6.2 Wait for Services to Initialize

After `docker compose up -d`, wait for the critical services:

```powershell
# Wait 60 seconds for all containers to start
Start-Sleep -Seconds 60

# Check container status
docker ps --filter "name=ldt-" --format "{{.Names}} {{.Status}}"
```

Expected: 19 containers running, most should show `(healthy)` after 2-3 minutes. (kafka-init and minio-init exit after setup.)

### 6.3 Verify All Containers Started

```powershell
docker ps --filter "name=ldt-" --format "table {{.Names}}\t{{.Status}}"
```

Expected output:
```
NAMES                       STATUS
ldt-action-replay-worker    Up X minutes
ldt-cadqstream-metrics       Up X minutes (healthy)
ldt-flink-init              Up X minutes (running)    # Auto-recovery supervisor
ldt-flink-jobmanager        Up X minutes (healthy)
ldt-flink-taskmanager       Up X minutes (healthy)
ldt-grafana                  Up X minutes (healthy)
ldt-kafka                   Up X minutes (healthy)
ldt-kafka-init              Up X minutes (exited)    # One-time init — exits after setup
ldt-kafka-producer          Up X minutes
ldt-kafka-ui                Up X minutes (healthy)
ldt-kafka-exporter          Up X minutes (healthy)
ldt-minio                   Up X minutes (healthy)
ldt-minio-init              Up X minutes (exited)    # One-time init — exits after setup
ldt-node-exporter           Up X minutes (healthy)
ldt-prometheus              Up X minutes (healthy)
ldt-redis                   Up X minutes (healthy)
ldt-schema-registry          Up X minutes (healthy)
ldt-stats-writer            Up X minutes
ldt-ml-service              Up X minutes (healthy)
ldt-zookeeper               Up X minutes (healthy)
```

> **Note:** `ldt-flink-init` runs indefinitely as the auto-recovery supervisor. It monitors the Flink job and auto-resubmits if it crashes (up to 5 retries with exponential backoff). `kafka-init` and `minio-init` exit cleanly after creating topics and buckets.

---

## 7. Service Verification: Layer by Layer

**Verify every service in order.** This is the core principle: check flow from beginning to end, not just locally.

---

### Layer 1: Kafka Infrastructure

#### Step L1-1: Zookeeper — Is it running?

```powershell
docker exec ldt-zookeeper bash -c 'echo ruok | nc localhost 2181'
# Expected: imok
```

#### Step L1-2: Kafka Broker — Is it healthy?

```powershell
docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --list
# Expected: list of topics including taxi-nyc-raw-v2, dq-stream-unified, iec-action-replay
```

#### Step L1-3: Kafka Topics — Are all required topics present?

```powershell
docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --list
# Expected (at minimum):
#   taxi-nyc-raw-v2
#   dq-stream-unified
#   iec-action-replay
#   iec-action-dlq
#   memstream-model-updates
```

#### Step L1-4: Kafka Produce → Consume End-to-End Test

```powershell
# Produce a test message
$testMsg = '{"VendorID":1,"tpep_pickup_datetime":"2026-05-20T14:00:00","tpep_dropoff_datetime":"2026-05-20T14:15:00","passenger_count":2,"trip_distance":3.5,"PULocationID":79,"DOLocationID":170,"fare_amount":12.50,"total_amount":15.75,"payment_type":1}'
$testMsg | docker exec -i ldt-kafka kafka-console-producer --bootstrap-server localhost:9092 --topic taxi-nyc-raw-v2

# Wait 5 seconds
Start-Sleep -Seconds 5

# Consume the message back
docker exec ldt-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic taxi-nyc-raw-v2 --from-beginning --max-messages 1 --consumer-timeout-ms 5000
# Expected: the message appears (or a similar one)
```

#### Step L1-5: Kafka Consumer Group Lag

```powershell
docker exec ldt-kafka kafka-consumer-groups --bootstrap-server localhost:9092 --all-groups --describe
# Expected: consumer group for action-replay-worker shows LAG=0
# Note: Flink consumer group lag on taxi-nyc-raw-v2 may be non-zero if no data is flowing yet
```

#### Step L1-6: Kafka UI (Web Interface)

```
Open: http://localhost:8080
# View: Topics, partitions, consumer groups, message content
```

#### Step L1-7: Schema Registry

```powershell
docker exec ldt-kafka bash -c 'curl -sf http://localhost:8081/subjects'
# Expected: JSON array of registered schemas (at least 3: taxi-nyc-raw-value, iec-action-replay-value, dq-stream-unified-value)
```

---

### Layer 1b: Redis

#### Step L1b-1: Redis — Is it reachable and responding?

```powershell
# Get REDIS_PASSWORD from your .env (default: CADQStream2026!)
docker exec ldt-redis redis-cli -a CADQStream2026! ping
# Expected: PONG
```

#### Step L1b-2: Redis Info — Check connected clients

```powershell
docker exec ldt-redis redis-cli -a CADQStream2026! info clients | Select-String "connected_clients"
# Expected: connected_clients:1 or more (ML service and/or Flink tasks)
```

---

### Layer 2: MinIO Storage

#### Step L2-1: MinIO — Is it reachable?

```powershell
docker exec ldt-minio mc ready local
# Expected: exit code 0 (mc ready local)
```

#### Step L2-2: MinIO Buckets — Are all 8 buckets present?

```powershell
docker exec ldt-minio mc ls local/
# Expected (all 8):
#   cadqstream-anomalies/
#   cadqstream-checkpoints/
#   cadqstream-dlq/
#   cadqstream-drift/
#   cadqstream-metrics/
#   cadqstream-raw/
#   cadqstream-violations/
#   ml-models/
```

#### Step L2-3: MinIO Bucket Security — Are sensitive buckets private?

```powershell
# Check violations bucket
docker exec ldt-minio mc anonymous get local/cadqstream-violations
# Expected: Not "Enabled" (should be private)

# Check ml-models bucket
docker exec ldt-minio mc anonymous get local/ml-models
# Expected: Not "Enabled" (should be private)
```

#### Step L2-4: MinIO Console (Web Interface)

```
Open: http://localhost:9001
Login: cadqstream / CADQStream2026! (from your .env)
# Browse buckets, upload/download files, view lifecycle policies
```

---

### Layer 4: Flink Streaming

#### Step L4-1: Flink JobManager — Is the REST API responding?

```powershell
docker exec ldt-flink-jobmanager curl -sf http://localhost:8081/overview
# Expected: JSON with Flink cluster overview
```

#### Step L4-2: Flink Cluster Overview — Check TaskManagers and Slots

```powershell
docker exec ldt-flink-jobmanager curl -sf http://localhost:8081/overview | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'TaskManagers: {d[\"taskmanagers\"]}, Slots total: {d[\"taskmanager\"][\"totalTaskManagerSlotNumber\"]}, Slots free: {d[\"taskmanager\"][\"totalAvailableSlotNumber\"]}')"
# Expected: TaskManagers: 1, Slots total: 16, Slots free: 12 (or similar)
```

#### Step L4-3: Flink Jobs — Is the pipeline job RUNNING?

```powershell
docker exec ldt-flink-jobmanager curl -sf http://localhost:8081/jobs | python3 -c "import sys,json; d=json.load(sys.stdin); running=[j for j in d['jobs'] if j['state']=='RUNNING']; failed=[j for j in d['jobs'] if j['state']=='FAILED']; print(f'RUNNING: {len(running)}, FAILED: {len(failed)}'); [print(f'  {j[\"name\"]} [{j[\"state\"]}]') for j in d['jobs'] if j['state'] in ['RUNNING','FAILED']]"
# Expected: At least 1 RUNNING job named "CA-DQStream Sequential Pipeline - Phase 3"
# If FAILED jobs exist: check flink-init logs
```

#### Step L4-4: Flink Checkpointing — Is it active?

```powershell
# Get the running job ID
$jobId = docker exec ldt-flink-jobmanager curl -sf http://localhost:8081/jobs | python3 -c "import sys,json; d=json.load(sys.stdin); r=[j for j in d['jobs'] if j['state']=='RUNNING']; print(r[0]['id'] if r else '')"

# Get checkpoint info
docker exec ldt-flink-jobmanager curl -sf "http://localhost:8081/jobs/$jobId/info" | python3 -c "import sys,json; d=json.load(sys.stdin); chk=d.get('checkpointing',{}); print(f'Last checkpoint: {chk.get(\"last-checkpoint-timestamp\",\"none\")}') if chk else print('Checkpointing: not configured')"
# Expected: last-checkpoint-timestamp is a large number (Unix ms) if checkpoints are running
```

#### Step L4-5: Flink Init (Auto-Recovery Supervisor) — Is it running?

```powershell
docker inspect --format='{{.State.Status}}' ldt-flink-init
# Expected: running

docker logs ldt-flink-init --tail 5
# Expected: HEALTHY messages or CONTINUOUS HEALTH MONITOR active
```

#### Step L4-6: Flink JobManager Error Log — Any critical errors?

```powershell
docker logs ldt-flink-jobmanager --tail 50 2>&1 | Select-String -Pattern "ERROR","Exception","Traceback" -Context 0,2 | Select-Object -First 5
# Expected: No ERROR entries, or only transient/resolved errors
```

#### Step L4-7: Flink UI (Web Interface)

```
Open: http://localhost:8081
# View: Job graph, TaskManagers, running jobs, checkpoint status, metrics
```

#### Step L4-8: Kafka → Flink → Kafka Flow (End-to-End)

```powershell
# Produce a test message
$msg = '{"VendorID":2,"tpep_pickup_datetime":"2026-05-20T14:30:00","tpep_dropoff_datetime":"2026-05-20T14:45:00","passenger_count":1,"trip_distance":5.5,"PULocationID":100,"DOLocationID":200,"fare_amount":18.50,"total_amount":22.00,"payment_type":1}'
$msg | docker exec -i ldt-kafka kafka-console-producer --bootstrap-server localhost:9092 --topic taxi-nyc-raw-v2

# Wait 10 seconds for Flink to process
Start-Sleep -Seconds 10

# Check dq-stream-unified for processed output
docker exec ldt-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic dq-stream-unified --from-beginning --max-messages 5 --consumer-timeout-ms 5000
# Expected: Messages with _event_type field (PROCESSED_RECORD, CANARY_VIOLATION, etc.)
```

---

### Layer 4b: ML Service

#### Step L4b-1: ML Service Health

```powershell
docker exec ldt-ml-service curl -sf http://localhost:8000/health
# Expected: {"status":"healthy","model_loaded":true,"redis_connected":true,...}
```

Key fields to verify:
- `status`: `"healthy"`
- `model_loaded`: `true` (model artifact loaded successfully)
- `redis_connected`: `true` (Redis reachable)

#### Step L4b-2: ML /predict Inference Endpoint

```powershell
# IMPORTANT: The /predict endpoint requires features as a nested 2D array.
# PowerShell's ConvertTo-Json flattens nested arrays, so use a JSON file instead:
$body = '{"features":[[900,3.5,15.5,2.5,0.33,0.95,0.14,0,2,100,170,5,1.3,0.16,0.1,0.05,1,1,0,1,0.87,0.5,0.3,0.8,0.2,0.7,0.4,0.6,0.1,0.9,0.15,0.85,0.25,0.75]]}'
$body | docker exec -i ldt-ml-service curl -sf -X POST -H "Content-Type: application/json" -d @- http://localhost:8000/predict
# Expected: {"predictions":[1],"scores":[...],"anomaly_count":1,"threshold":0.5,...}
```

#### Step L4b-3: ML Service Logs — HMAC and Model Verification

```powershell
docker logs ldt-ml-service --tail 50
# Expected: No ERROR entries
# Check for: HMAC verification, model loading success messages
```

---

### Layer 5: Observability — Prometheus

#### Step L5-1: Prometheus — Is it healthy?

```powershell
docker exec ldt-prometheus wget -qO- http://localhost:9090/-/healthy
# Expected: Prometheus Server is Healthy.
```

#### Step L5-2: Prometheus Scrape Targets — Are all targets being scraped?

```powershell
docker exec ldt-prometheus wget -qO- http://localhost:9090/api/v1/targets | python3 -c "import sys,json; d=json.load(sys.stdin); targets=d['data']['targets']; up=[t for t in targets if t['health']=='up']; down=[t for t in targets if t['health']!='up']; print(f'Up: {len(up)}, Down: {len(down)}'); [print(f'  {\"OK\" if t[\"health\"]==\"up\" else \"DOWN\"} {t[\"labels\"].get(\"job\",\"?\")}') for t in targets]"
# Expected: All targets "up" (at minimum: prometheus, flink-jobmanager, kafka-exporter, minio, ml-service, cadqstream-app)
```

#### Step L5-3: Prometheus — CadQStream Metrics Present

```powershell
# Check L1 metrics
docker exec ldt-prometheus wget -qO- "http://localhost:9090/api/v1/query?query=cadqstream_records_valid_total"
# Expected: {"status":"success","data":{"result":[...]}}

# Check ML metrics
docker exec ldt-prometheus wget -qO- "http://localhost:9090/api/v1/query?query=memstream_warmup_progress"
# Expected: {"status":"success",...}

# Check Kafka metrics
docker exec ldt-prometheus wget -qO- "http://localhost:9090/api/v1/query?query=kafka_topic_partition_current_offset"
# Expected: {"status":"success",...}
```

Key metric groups to verify (all should return `status: "success"`):

| Group | Metrics |
|-------|---------|
| L1-Ingestion | `cadqstream_records_valid_total`, `cadqstream_records_violation_total` |
| L2-Canary | `cadqstream_canary_violation_total` |
| L3-MemStream | `cadqstream_anomaly_score`, `memstream_scoring_latency_seconds` |
| L3-MetaAgg | `cadqstream_meta_window_record_count` |
| L4-IEC | `cadqstream_drift_detected`, `cadqstream_iec_action_total`, `cadqstream_circuit_breaker_state` |
| ML-Warmup | `memstream_warmup_progress`, `memstream_redis_connected` |
| ML-HMAC | `memstream_hmac_verification_total` |
| ML-Stats | `memstream_knn_avg_distance`, `memstream_memory_fill_rate`, `memstream_beta_staleness_seconds` |

#### Step L5-4: Prometheus Alert Rules — Are they loaded?

```powershell
docker exec ldt-prometheus wget -qO- http://localhost:9090/api/v1/rules | python3 -c "import sys,json; d=json.load(sys.stdin); groups=d['data']['groups']; print(f'Rule groups: {len(groups)}'); total=sum(len(g['rules']) for g in groups); alerting=sum(len([r for r in g['rules'] if r['type']=='alerting']) for g in groups); print(f'Total rules: {total}, Alerting: {alerting}')"
# Expected: Rule groups > 0, alerting rules > 0
```

#### Step L5-5: Prometheus UI

```
Open: http://localhost:9090
# View: Graph explorer, alerts, targets, rules, TSDB status
```

---

### Layer 6: Grafana Dashboards

#### Step L6-1: Grafana — Is it healthy?

```powershell
docker exec ldt-grafana curl -sf http://localhost:3000/api/health
# Expected: {"commit":"...","database":"ok","version":"10.1.0"}
```

#### Step L6-2: Grafana — How many dashboards are provisioned?

```powershell
docker exec ldt-grafana curl -sf -u "admin:CADQStream2026!" http://localhost:3000/api/search?type=dash-db | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Dashboards: {len(d)}'); [print(f'  - {db[\"title\"]}') for db in d]"
# Expected: 7 dashboards including:
#   - CA-DQStream: Pipeline Overview
#   - CA-DQStream: Kafka Overview
#   - CA-DQStream: Flink Jobs
#   - CA-DQStream: Data Quality
#   - CA-DQStream: Infrastructure
#   - CA-DQStream: ML Service
#   - MemStream: Data Quality
```

#### Step L6-3: Grafana — Verify Dashboard Panes Have Data

For each dashboard, open in Grafana UI and verify:

1. **Pipeline Overview** (`cadqstream-pipeline-overview`)
   - Kafka input rate pane (L1): data flowing
   - Flink processing pane: records processed
   - Violation rate pane (L2): data or zero (no test data yet)
   - Anomaly score pane (L3): data or zero
   - Drift detection pane (L4): data or zero
   - **CHECK:** All layer panes (L1, L2, L3, L4) must be present — not just L1

2. **Kafka Overview** (`cadqstream-kafka-overview`)
   - Broker health pane
   - Consumer lag pane
   - Topic message rates pane

3. **Flink Jobs** (`cadqstream-flink-jobs`)
   - Job status pane (RUNNING)
   - Checkpointing pane (last checkpoint timestamp)
   - Task slots pane (used/free)

4. **Data Quality** (`cadqstream-data-quality`)
   - Violation rate over time
   - Anomaly rate by neighborhood
   - Meta-metrics heatmap
   - Drift detection timeline

5. **Infrastructure** (`cadqstream-infrastructure`)
   - Host CPU pane
   - Host memory pane
   - Container resource panes

6. **ML Service** (`cadqstream-ml-service`)
   - Model loaded pane
   - Inference latency pane
   - HMAC verification pane
   - KNN stats pane

#### Step L6-4: Grafana Login

```
URL: http://localhost:3000
Username: admin
Password: (your GRAFANA_PASSWORD from .env, default: CADQStream2026!)
```

---

### Layer 6b: Stats Writer & cadqstream-metrics

#### Step L6b-1: cadqstream-metrics Service

```powershell
docker exec ldt-cadqstream-metrics curl -sf http://localhost:9250/health
# Expected: {"metrics_count":11,"status":"healthy"}
```

#### Step L6b-2: Stats Writer

```powershell
docker ps --filter "name=ldt-stats-writer" --format "{{.Status}}"
# Expected: Up (running)
```

#### Step L6b-3: Stats Metrics in MinIO

```powershell
docker exec ldt-minio mc ls local/cadqstream-metrics/
# After a few minutes: Parquet files should appear here
# If empty: stats-writer may need warmup time (up to 60 seconds)
```

#### Step L6b-4: Stats Metrics in Prometheus

```powershell
$metrics = @("cadqstream_anomaly_rate","cadqstream_false_positive_rate","cadqstream_records_processed_total")
foreach ($m in $metrics) {
    $r = docker exec ldt-prometheus wget -qO- "http://localhost:9090/api/v1/query?query=$m"
    $status = echo $r | python3 -c "import sys,json; d=json.load(sys.stdin); print('PRESENT' if d['data']['result'] else 'MISSING')"
    Write-Host "$m -> $status"
}
# Expected: All "PRESENT" after warmup
```

---

### Layer 7: Action Replay Worker & Kafka Producer

#### Step L7-1: Action Replay Worker

```powershell
docker ps --filter "name=ldt-action-replay-worker" --format "{{.Status}}"
# Expected: Up

# Check logs for any errors
docker logs ldt-action-replay-worker --tail 10
# Expected: No ERROR entries
```

#### Step L7-2: Kafka Producer

```powershell
docker ps --filter "name=ldt-kafka-producer" --format "{{.Status}}"
# Expected: Up

# Check if it is producing messages
docker exec ldt-kafka kafka-consumer-groups --bootstrap-server localhost:9092 --group kafka-producer --describe 2>$null
# If no producer group: producer may not have started yet or uses a different group
```

---

## 8. End-to-End Flow Test (Inject Data)

Run the automated end-to-end verification script:

```powershell
powershell -ExecutionPolicy Bypass -File deployment/scripts/deploy-e2e.ps1 -SkipBuild -SkipDeploy -Verbose
```

This runs all injection tests automatically:

| Test | What It Does | What to Expect |
|------|-------------|----------------|
| Normal record | Inject valid taxi trip | PROCESSED_RECORD in dq-stream-unified |
| L1 Schema violation | Missing `trip_distance`, invalid `PULocationID=999` | Record in cadqstream-violations/ |
| L2 Canary violation | Negative fare, zero distance, passengers=0 | CANARY_VIOLATION in dq-stream-unified |
| L3 Extreme anomaly | Fare=$999.99, distance=99.9 | High anomaly score in cadqstream-anomalies/ |
| Concept drift | 10 records with +20% fare each | ADWIN drift detection after ~3 min |

### Manual Injection Commands

```powershell
# Normal record
$normal = '{"VendorID":1,"tpep_pickup_datetime":"2026-05-20T15:00:00","tpep_dropoff_datetime":"2026-05-20T15:20:00","passenger_count":2,"trip_distance":3.5,"PULocationID":79,"DOLocationID":170,"fare_amount":12.50,"total_amount":15.75,"payment_type":1}'
$normal | docker exec -i ldt-kafka kafka-console-producer --bootstrap-server localhost:9092 --topic taxi-nyc-raw-v2

# L1: Missing trip_distance
$l1 = '{"VendorID":1,"tpep_pickup_datetime":"2026-05-20T15:01:00","tpep_dropoff_datetime":"2026-05-20T15:21:00","passenger_count":1,"PULocationID":79,"DOLocationID":170,"fare_amount":10.00,"total_amount":13.50,"payment_type":1}'
$l1 | docker exec -i ldt-kafka kafka-console-producer --bootstrap-server localhost:9092 --topic taxi-nyc-raw-v2

# L2: Negative fare
$l2 = '{"VendorID":1,"tpep_pickup_datetime":"2026-05-20T15:02:00","tpep_dropoff_datetime":"2026-05-20T15:17:00","passenger_count":1,"trip_distance":5.0,"PULocationID":79,"DOLocationID":170,"fare_amount":-5.00,"total_amount":2.00,"payment_type":1}'
$l2 | docker exec -i ldt-kafka kafka-console-producer --bootstrap-server localhost:9092 --topic taxi-nyc-raw-v2

# L3: Extreme anomaly
$l3 = '{"VendorID":1,"tpep_pickup_datetime":"2026-05-20T15:03:00","tpep_dropoff_datetime":"2026-05-20T15:18:00","passenger_count":6,"trip_distance":99.9,"PULocationID":138,"DOLocationID":229,"fare_amount":999.99,"total_amount":1050.00,"payment_type":1}'
$l3 | docker exec -i ldt-kafka kafka-console-producer --bootstrap-server localhost:9092 --topic taxi-nyc-raw-v2

# Wait 15 seconds, then verify
Start-Sleep -Seconds 15

# Check dq-stream-unified for all event types
docker exec ldt-kafka kafka-console-consumer --bootstrap-server localhost:9092 --topic dq-stream-unified --from-beginning --max-messages 20 --consumer-timeout-ms 8000
```

### Verify MinIO Buckets After Injection

```powershell
# Schema/canary violations
docker exec ldt-minio mc ls local/cadqstream-violations/
# Expected: Parquet files (after L1/L2 test)

# Anomaly scores
docker exec ldt-minio mc ls local/cadqstream-anomalies/
# Expected: Parquet files (after L3 test)

# IEC drift events
docker exec ldt-minio mc ls local/cadqstream-drift/
# Expected: Files after concept drift injection (wait ~3 minutes)
```

---

## 9. Verify All Grafana Dashboards

Open each dashboard in Grafana and verify **all panes have data**:

| Dashboard | Required Panes |
|-----------|--------------|
| Pipeline Overview | L1 Ingestion rate, L2 Violation rate, L3 Anomaly rate, L4 Drift detection, ML warmup, Throughput |
| Kafka Overview | Broker status, Consumer lag, Topic rates, Partition offsets |
| Flink Jobs | Job status (RUNNING), Checkpoint info, Task slots, Processing throughput |
| Data Quality | Violation rate, Anomaly rate, Meta-metrics heatmap, Drift timeline |
| Infrastructure | CPU, Memory, Disk I/O, Network |
| ML Service | Model loaded, Inference latency, HMAC status, KNN stats, Memory fill |

**Common mistake:** A dashboard may show L1 data but be missing L2, L3, or L4 panes. This means those pipeline layers are not producing metrics — go back and check the Flink job and cadqstream-metrics service.

---

## 10. Troubleshooting

### Kafka Won't Start

```powershell
# Check Zookeeper first
docker logs ldt-zookeeper --tail 20

# Check Kafka logs
docker logs ldt-kafka --tail 30

# Clear Kafka data volume (fixes InconsistentClusterIdException)
docker volume rm ldt-kafka-data
docker compose -f deployment/docker-compose.yml up -d kafka

# Verify broker ID
docker exec ldt-kafka bash -c 'echo $KAFKA_BROKER_ID'
# Expected: 1
```

### Flink Job Not Running

```powershell
# Check flink-init logs (auto-recovery supervisor)
docker logs ldt-flink-init --tail 50

# Check if job exists but failed
docker exec ldt-flink-jobmanager curl -sf http://localhost:8081/jobs | python3 -c "import sys,json; d=json.load(sys.stdin); [print(f'{j[\"name\"]} [{j[\"state\"]}]') for j in d['jobs']]"

# Check Flink JobManager logs for errors
docker logs ldt-flink-jobmanager --tail 100 | Select-String -Pattern "ERROR","Exception","Traceback" -Context 0,2

# Cancel failed jobs manually
$jobId = "your-job-id"
docker exec ldt-flink-jobmanager curl -sf -X PATCH "http://localhost:8081/jobs/$jobId/cancel"

# Restart flink-init
docker compose -f deployment/docker-compose.yml restart flink-init
```

### MinIO Buckets Not Created

```powershell
# Run minio-init manually
docker compose -f deployment/docker-compose.yml up minio-init
docker logs ldt-minio-init --tail 20

# Check mc alias
docker exec ldt-minio mc alias list
# Expected: local -> http://localhost:9000

# Manually create buckets
docker exec ldt-minio mc mb local/cadqstream-raw --ignore-existing
docker exec ldt-minio mc mb local/cadqstream-violations --ignore-existing
docker exec ldt-minio mc mb local/cadqstream-anomalies --ignore-existing
docker exec ldt-minio mc mb local/cadqstream-metrics --ignore-existing
docker exec ldt-minio mc mb local/cadqstream-drift --ignore-existing
docker exec ldt-minio mc mb local/cadqstream-dlq --ignore-existing
docker exec ldt-minio mc mb local/cadqstream-checkpoints --ignore-existing
docker exec ldt-minio mc mb local/ml-models --ignore-existing
```

### Prometheus Scrape Targets Down

```powershell
# Check which targets are down
docker exec ldt-prometheus wget -qO- http://localhost:9090/api/v1/targets | python3 -c "import sys,json; d=json.load(sys.stdin); [print(f'DOWN: {t[\"labels\"].get(\"job\",\"?\")}: {t[\"lastError\"]}') for t in d['data']['targets'] if t['health']!='up']"

# Verify the target is reachable from inside the container
docker exec ldt-prometheus wget -qO- http://ldt-flink-jobmanager:9248/metrics
# If this fails: networking issue between prometheus and flink containers
```

### ML Service Model Not Loaded

```powershell
# Check ML service logs
docker logs ldt-ml-service --tail 50

# Check if model files exist in ml-models bucket
docker exec ldt-minio mc ls local/ml-models/

# Check Redis connectivity
docker exec ldt-ml-service curl -sf http://localhost:8000/health
# Expected: redis_connected: true
```

### Container Restart Loop

```powershell
# Inspect the crashing container
docker logs --tail 100 ldt-<service-name>

# Check resource limits (OOM = not enough memory)
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Check disk space
docker system df
```

### Full Clean Reset

```powershell
cd deployment

# Stop everything
docker compose down --remove-orphans

# Remove all volumes (full data wipe)
docker compose down -v --remove-orphans

# Remove networks
docker network rm cadqstream-net 2>$null

# Prune ldt-* images
docker image prune -f --filter "reference=ldt-*"

# Restart fresh
docker compose up -d
```

---

## 11. Stopping & Restarting

### Stop (Preserve Data)

```powershell
# Graceful stop — all volumes preserved
docker compose -f deployment/docker-compose.yml down

# Or with Makefile
make down
```

### Stop + Remove Volumes (Full Data Wipe)

```powershell
docker compose -f deployment/docker-compose.yml down -v --remove-orphans

# Or with Makefile
make clean
```

### Restart a Specific Service

```powershell
# Restart Flink only (preserves job state via checkpoints)
docker compose -f deployment/docker-compose.yml restart flink-jobmanager flink-taskmanager

# Restart ML service only
docker compose -f deployment/docker-compose.yml restart ml-service

# Restart a specific container
docker restart ldt-kafka
```

### Update Code Without Full Redeploy

```powershell
# 1. Rebuild the Flink image (picks up code changes)
docker build -t ldt-flink:1.18.1-py -f deployment/flink/Dockerfile .

# 2. Recreate Flink containers (keeps Kafka, MinIO, Redis, etc. running)
docker compose -f deployment/docker-compose.yml up -d --force-recreate flink-jobmanager flink-taskmanager

# 3. flink-init auto-recovers: it will cancel the old job and resubmit with the new image
```

---

## 12. Service URLs & Credentials

| Service | URL | Credentials | Notes |
|---------|-----|-------------|-------|
| **Kafka UI** | http://localhost:8080 | No auth | Browse topics, partitions, consumer groups |
| **Flink UI** | http://localhost:8081 | No auth | Job graph, TaskManagers, checkpoints |
| **Grafana** | http://localhost:3000 | `admin` / `CADQStream2026!` | Dashboards |
| **Prometheus** | http://localhost:9090 | No auth | Metrics, alerts, targets |
| **MinIO Console** | http://localhost:9001 | `cadqstream` / `CADQStream2026!` | Browse buckets, files |
| **ML Service API** | http://localhost:8000 | No auth (internal network) | FastAPI inference |
| **cadqstream-metrics** | http://localhost:9250/metrics | No auth | App-level metrics for Prometheus |
| **Kafka** | `localhost:9092` (host), `kafka:9092` (container) | No auth | PLAINTEXT |

### Kafka Internal Network

From inside any container on `cadqstream-net`:

```
Kafka:           kafka:9092
Schema Registry:  schema-registry:8081
MinIO API:       minio:9000
Redis:           redis:6379
Flink REST:      flink-jobmanager:8081
ML Service:      ml-service:8000
```

---

## 13. Quick Reference Commands

### Health Check (One Liner)

```powershell
# All-in-one health status
docker ps --filter "name=ldt-" --format "{{.Names}} {{.Status}}" | python3 -c "
import sys
lines = [l.strip() for l in sys.stdin if l.strip()]
total = len(lines)
healthy = sum(1 for l in lines if 'healthy' in l or 'Up' in l.split()[1:])
print(f'Containers: {total}, Healthy: {healthy}, Unhealthy: {total-healthy}')
"
```

### Run Full Deployment + Verification

```powershell
powershell -ExecutionPolicy Bypass -File deployment/scripts/deploy-e2e.ps1
# Exit code 0 = all checks passed
# Exit code 1 = critical failure
# Exit code 2 = warnings only (operational)
```

### View Logs

```powershell
# All services
docker compose -f deployment/docker-compose.yml logs --tail 50 -f

# Specific service
docker logs ldt-flink-jobmanager --tail 50 -f

# Flink init (auto-recovery)
docker logs ldt-flink-init --tail 50 -f
```

### List Kafka Topics with Partitions

```powershell
docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --list | ForEach-Object { docker exec ldt-kafka kafka-topics --bootstrap-server localhost:9092 --topic $_ --describe | Select-Object -First 3 }
```

### Query Prometheus Directly

```powershell
# All cadqstream metrics
docker exec ldt-prometheus wget -qO- "http://localhost:9090/api/v1/label/__name__/values" | python3 -c "import sys,json; d=json.load(sys.stdin); cadq=[v for v in d['data'] if v.startswith('cadqstream') or v.startswith('memstream')]; print(f'CA-DQStream metrics: {len(cadq)}'); [print(f'  {m}') for m in sorted(cadq)[:20]]"

# Specific metric over time
docker exec ldt-prometheus wget -qO- "http://localhost:9090/api/v1/query_range?query=cadqstream_records_valid_total&start=now-5m&end=now&step=15s"
```

### Grafana Dashboard UID Lookup

```powershell
docker exec ldt-grafana curl -sf -u "admin:CADQStream2026!" http://localhost:3000/api/search?type=dash-db | python3 -c "import sys,json; [print(f'{db[\"title\"]} -> uid={db[\"uid\"]}') for db in json.load(sys.stdin)]"
```

---

## File Structure Reference

```
deployment/
├── docker-compose.yml          # Full production compose (all 18 services)
├── docker-compose-minimal.yml  # Minimal compose (core flow only)
├── Makefile                   # Deterministic orchestration (make up/down/reset/health/logs)
│
├── flink/
│   ├── Dockerfile             # Custom Flink 1.18.1 image (PyFlink + Python 3.10)
│   ├── flink-init.sh          # Job submission + auto-recovery supervisor
│   ├── requirements.txt        # Python dependencies for Flink container
│   └── taskmanager.conf       # TM JVM configuration
│
├── kafka/
│   ├── init-scripts/
│   │   ├── init.sh            # Topic creation + schema registration
│   │   ├── 01-create-topics.sh  # Creates taxi-nyc-raw-v2, dq-stream-unified, etc.
│   │   └── 02-fix-compact-topics.sh
│   ├── Dockerfile.producer    # Demo data producer
│   ├── Dockerfile.e2e-test    # E2E test injector
│   └── anomaly_producer.py    # Synthetic anomaly injection
│
├── minio/
│   └── init-scripts/
│       └── 01-create-buckets.sh  # Creates 8 buckets with lifecycle policies
│
├── ml-service/
│   ├── Dockerfile             # FastAPI ML service
│   └── requirements.txt      # Python dependencies
│
├── cadqstream-metrics/
│   └── Dockerfile             # HTTP scraper for Flink → Prometheus metrics
│
├── stats-writer/
│   └── Dockerfile             # Aggregates metrics → MinIO cadqstream-metrics/
│
├── action-replay-worker/
│   └── Dockerfile            # IEC retrain signal handler
│
├── prometheus/
│   ├── prometheus.yml         # Scrape configs (8 targets)
│   ├── alert-rules/
│   │   └── cadqstream-alerts.yml
│   └── rules/
│       └── memstream-alerts.yaml
│
├── grafana/
│   ├── provisioning/
│   │   ├── dashboards/       # 7 auto-provisioned dashboards
│   │   ├── datasources/       # Prometheus datasource
│   │   └── alerting/
│   └── dashboards/           # JSON dashboard definitions
│
└── scripts/
    ├── start.sh              # Full deployment (Bash)
    ├── stop.sh               # Graceful stop
    ├── reset.sh              # Clean slate
    ├── healthcheck.sh         # Comprehensive health verification
    ├── deploy-e2e.ps1        # Deploy + verify (PowerShell, recommended)
    ├── deploy-and-verify.ps1  # Full 11-phase verification
    └── deploy-prod.sh        # Production deploy (Bash)
```

---

## Architecture Notes

- **Storage:** MinIO only (no PostgreSQL). All pipeline output written as Parquet via Flink StreamingFileSink.
- **State:** Flink RocksDB with incremental checkpoints stored in local Docker volume (not S3A).
- **ML Model:** MemStream autoencoder loaded from `ml-models` bucket. HMAC verification on model integrity.
- **IEC:** ADWIN-U drift detection with METER hypernetwork strategy prediction. Adaptive thresholds cached in Redis.
- **Checkpointing:** EXACTLY_ONCE mode, 60-second interval, 3 retained checkpoints. On failure: auto-resubmit via flink-init supervisor (up to 5 retries).
- **Idempotency:** All docker compose operations are idempotent. Safe to re-run `up` at any time.
- **TLS:** All inter-service communication is PLAINTEXT (no TLS). For production, enable TLS on Kafka, MinIO, and Redis.
