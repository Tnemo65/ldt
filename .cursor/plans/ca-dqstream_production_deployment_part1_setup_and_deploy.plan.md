---
name: CA-DQStream Production Deployment — Part 1: Setup & Deploy
overview: Deploy the full CA-DQStream pipeline (Kafka + Flink + MemStream + Prometheus + Grafana + MinIO). Part 1 covers environment setup, data preparation, infrastructure build, and the deploy script.
todos:
  - id: setup-env
    content: Confirm .env file exists with all required secrets filled in
    status: pending
  - id: prep-data
    content: "Build demo dataset: extend inject_anomalies_memstream.py to 10 types (3 existing + 7 new hard-rule), generate deployment/data/demo_trips.parquet (1-2 weeks only)"
    status: pending
  - id: update-compose
    content: Update docker-compose.yml kafka-producer INPUT_FILE to point to deployment/data/demo_trips.parquet
    status: pending
  - id: build-flink
    content: Build custom Flink Docker image (ldt-flink:1.18.1-py) -- takes ~5-15 minutes
    status: pending
  - id: deploy-infra
    content: Start all services via docker compose up -d
    status: pending
  - id: health-check
    content: Run healthcheck.sh to verify all 19 services are up
    status: pending
  - id: upload-model
    content: Upload memstream_checkpoint_v1.pt to MinIO ml-models/ bucket
    status: pending
  - id: create-deploy-script
    content: Create deploy-prod.sh single-command script (bash for Git Bash/WSL2)
    status: pending
isProject: false
---

# CA-DQStream Production Deployment — Part 1: Setup & Deploy

## Phase 0: Pre-Deployment Setup

### 0a. Confirm `.env` file exists
`c:\proj\ldt\.env` already exists with secrets filled in. Verify all required vars are present:
- `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`, `GRAFANA_PASSWORD`, `REDIS_PASSWORD`
- `MEMSTREAM_MODEL_SIGNING_KEY`, `IEC_SIGNING_KEY` (32+ char strings)
- `INTERNAL_API_KEY`, `METRICS_API_KEY`, `MC_MINIO_USER`, `MC_MINIO_PASSWORD`

### 0b. Confirm data files exist
Both parquet files confirmed present:
- `c:\proj\ldt\HP_benchmark_v5\data\clean_2024.parquet` (1.27 GB)
- `c:\proj\ldt\HP_benchmark_v5\data\clean_2025.parquet`

### 0c. Upload MemStream model to MinIO
**Critical step** — Flink's `MemStreamScoringOperator` loads the model from MinIO `ml-models/` bucket, NOT from the local `.pt` file.

1. Start MinIO first (see Phase 2)
2. Upload the model:
   ```bash
   docker exec ldt-minio mc cp models/memstream/memstream_checkpoint_v1.pt local/ml-models/
   docker exec ldt-minio mc cp models/memstream/memstream_checkpoint_v1.pt.hmac local/ml-models/
   ```
3. Verify upload:
   ```bash
   docker exec ldt-minio mc ls local/ml-models/
   ```
4. Verify the checkpoint contains ContextBeta thresholds (80 values: 10 neighborhoods x 8 context cells). If the `.pt` was trained without ContextBeta, scoring will fail — check `src/ml/memstream_core.py` for how thresholds are loaded.

---

## Phase 1: Data Preparation — Build Demo Dataset

### 1a. Modify `inject_anomalies_memstream.py` for 10 anomaly types
File: `c:\proj\ldt\HP_benchmark_v5\data\inject_anomalies_memstream.py`

Current: 3 types (short_expensive, tip_anomaly, combo_short_long).
**Add 7 new types** to match the `anomaly_producer.py` hard-rule style:

| # | Name | Signal | Detection Layer |
|---|---|---|---|
| 5 | `invalid_zone` | PULocationID/DOLocationID = 300-500 (outside 1-263) | L1 Schema |
| 6 | `negative_fare` | fare_amount = -$50 to -$0.01 | L2 Canary |
| 7 | `zero_passenger` | passenger_count = 0 | L2 Canary |
| 8 | `impossible_speed` | 0.5 mile + $200-500 fare (implies >200 mph) | L2 Canary |
| 9 | `missing_field` | Random required field set to null | L1 Schema |
| 10 | `extreme_fare` | fare_amount = $1001-5000 | L2 Canary |
| 11 | `zero_distance_with_fare` | trip_distance = 0 + fare = $10-50 | L2 Canary |

Types 5-11 **pass all hard rules** but are detectable by MemStream's 34D features (same principle as existing types). Total anomaly rate stays at 3% (~0.3% per type).

Also keep the `prod` slice using `clean_2025.parquet` (already confirmed to exist).

### 1b. Generate demo data (NEW path, no override)
**Do not override existing files.** Output to `deployment/data/demo_trips.parquet`:
- Generate from `clean_2025.parquet` using the modified script
- Slice: **1-2 weeks of data only** (not all 2024 or all 2025 — demo needs short window)
- Output: `deployment/data/demo_trips.parquet` (clean + polluted versions)
- Also generate ground truth mask for evaluation

### 1d. Pre-generate warmup data (optional but recommended)
MemStream needs 8,192 warmup records before scoring begins. Generate a clean warmup set:
```bash
python HP_benchmark_v5/data/inject_anomalies_memstream.py --warmup-only
```
Or use a subset of `clean_2025.parquet` (first 10,000 rows). This ensures MemStream has a good memory foundation before demo starts.

---

## Phase 2: Build & Deploy Infrastructure

**Environment: Windows 10 + Docker Desktop + Git Bash**

### 2a. Build custom Flink Docker image
First build takes ~5-15 minutes. Subsequent builds are faster (Docker layer cache).
```bash
cd /c/proj/ldt/deployment
docker compose -f docker-compose.yml build --no-cache flink-jobmanager
```
This builds `ldt-flink:1.18.1-py` from `deployment/flink/Dockerfile`.

### 2b. Run full deployment
```bash
cd /c/proj/ldt/deployment
docker compose -f docker-compose.yml up -d
```
This starts all 17 services. The `docker-compose.yml` has `restart: unless-stopped` for all services, making it idempotent. After `up -d`, Docker continues running in background.

### 2c. Wait for all services to become healthy (~2-3 minutes)
Services start sequentially based on `depends_on` + `condition: service_healthy`. The `flink-init` container runs continuously and auto-submits the Flink job.

### 2d. Verify startup via healthcheck script
```bash
cd /c/proj/ldt
bash deployment/scripts/healthcheck.sh
```
This verifies all 19 services and reports Kafka topics / MinIO buckets.

### 2c. Verify startup sequence completed correctly
- Kafka topics created: `taxi-nyc-raw-v2`, `dq-stream-unified`, `memstream-model-updates`, `iec-action-replay`, `iec-action-dlq`
- MinIO buckets created: all 9 buckets
- Flink job submitted: "CA-DQStream Sequential Pipeline - Phase 3" in RUNNING state
- Prometheus scraping 8 targets

---

## Phase 6: Single-Command Production Deploy Script

Create `deployment/scripts/deploy-prod.sh` that combines all steps:
```bash
#!/bin/bash
# 1. Build custom Flink image
# 2. Start all services via docker compose
# 3. Wait for all health checks
# 4. Generate demo data (if not exists)
# 5. Start data producer
# 6. Start anomaly injector
# 7. Run E2E verification
# 8. Report Grafana URLs
```

The script should be **idempotent** — safe to run multiple times.

---

## Key File Changes

| File | Change | Priority |
|------|--------|----------|
| `HP_benchmark_v5/data/inject_anomalies_memstream.py` | Add 7 new anomaly types (total 10: 3 existing + 7 new hard-rule types); keep prod slice using `clean_2025.parquet` | HIGH |
| `deployment/data/demo_trips.parquet` | New file: 1-2 week sample with 10 types of injected anomalies; output to `deployment/data/` NOT `HP_benchmark_v5/data/` | HIGH |
| `deployment/docker-compose.yml` | Update `kafka-producer` `INPUT_FILE` env to point to `deployment/data/demo_trips.parquet` | HIGH |
| `deployment/scripts/deploy-prod.sh` | New: single-command deploy script (bash for Git Bash / WSL2) | MEDIUM |
