# CA-DQStream - Complete Deployment Guide

## 🚀 Deployment Order: Chạy Toàn Bộ Hệ Thống

### Bước 1: Start Infrastructure Services

```bash
# 1. Start Docker services (Kafka, MinIO)
docker compose up -d

# 2. Wait for services to be ready (30-60 seconds)
docker compose ps

# 3. Verify services
docker compose logs kafka | tail -20
```

**Expected:** All services "Up" and healthy

---

### Bước 2: Initialize Infrastructure

```bash
# 1. Create Kafka topics
./scripts/create_topics.sh

# Expected output:
# ✅ Topics created:
#    - taxi-nyc-raw (12 partitions)
#    - dq-schema-violations
#    - dq-hard-rule-violations
#    - dq-anomaly-scores
#    - dq-meta-stream
#    - if-model-updates (compacted)
#    - iec-action-replay

# 2. Register Avro schema
python scripts/register_schema.py

# 3. Setup MinIO buckets
./scripts/setup_minio.sh
```

---

### Bước 3: Train & Package ML Models

```bash
# 1. Train iForestASD v2 (if not already trained)
python src/ml/train_iforest.py \
  --n-trees 200 \
  --height 10 \
  --window-size 512

# Expected: models/iforest_model_v2.pkl (11.3 MB)

# 2. Train METER hypernetwork
python src/ml/train_meter.py --n-samples 1000

# Expected: models/meter_hypernetwork.pkl
#           models/meter_scaler.pkl

# 3. Package with MLflow
python scripts/package_mlflow.py

# 4. Verify models exist
ls -lh models/*.pkl
```

---

### Bước 4: Start ML Service (FastAPI)

```bash
# Option A: Direct Python
uvicorn src.api.ml_service:app --host 0.0.0.0 --port 8000 --workers 4

# Option B: Docker
docker build -f Dockerfile.ml-service -t cadqstream-ml-service .
docker run -d -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  --name ml-service \
  cadqstream-ml-service

# Verify health
curl http://localhost:8000/health

# Expected:
# {
#   "status": "healthy",
#   "models_loaded": 1,
#   "uptime_seconds": 5.2
# }
```

---

### Bước 5: Start Action Replay Worker

```bash
# Terminal 2: Start worker
python src/workers/action_replay_worker.py \
  --kafka-bootstrap localhost:9092 \
  --ml-service-url http://localhost:8000

# Expected output:
# ============================================================
# Action Replay Worker Initialized
# ============================================================
# Kafka: localhost:9092
# ML Service: http://localhost:8000
# Max Retries: 10
# Backoff Base: 2^n seconds
# ============================================================
```

---

### Bước 6: Start Complete Flink Pipeline

```bash
# Terminal 3: Start Flink job
python src/flink_job_complete.py

# Expected output:
# ===============================================================================
# CA-DQStream Complete Pipeline - 4 Layers Integrated
# ===============================================================================
#
# ✓ Environment configured
#   Parallelism: 4
#   Checkpointing: EXACTLY_ONCE (45s interval)
#
# ===============================================================================
# LAYER 1: Baseline Validation
# ===============================================================================
# ✓ Layer 1 operators connected:
#   - JSON parsing
#   - Watermark assignment (30s idleness)
#   - Trip ID generation (MurmurHash3)
#   - Deduplication (7-day TTL)
#   - Schema validation
#
# ===============================================================================
# LAYER 2: Dual-Branch Processing (Canary + Complex)
# ===============================================================================
# ✓ Canary Branch connected (7 business rules)
# ✓ Complex Branch connected (ML scoring)
#
# ===============================================================================
# LAYER 3: Rendezvous Sync + MetaAggregator
# ===============================================================================
# ✓ Rendezvous sync
# ✓ Voting Ensemble connected (Canary overrides ML)
# ✓ MetaAggregator connected (1-min windows, 6 meta-metrics)
#
# ===============================================================================
# LAYER 4: IEC (METER + ADWIN-U)
# ===============================================================================
# ✓ IEC Operator connected
#   - ADWIN-U drift detection (36 instances)
#   - METER strategy prediction
#   - Multi-strategy execution
#
# ===============================================================================
# STARTING COMPLETE PIPELINE
# ===============================================================================
```

---

### Bước 7: Produce Test Data

```bash
# Terminal 4: Start data producer
python scripts/produce_taxi_data.py \
  --data data/yellow_tripdata_2024-01.parquet \
  --rate 1000

# Expected:
# [Producer] Producing at 1000 eps
# [Producer] Sent: 1000 records
# [Producer] Sent: 2000 records
# ...
```

---

## 🔍 Monitoring & Verification

### Check Pipeline Health

```bash
# 1. Check Flink job is processing
# Look for output in Terminal 3 (Flink job)

# 2. Check Kafka topics have data
docker exec cadqstream_kafka_1 kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic taxi-nyc-raw \
  --from-beginning \
  --max-messages 5

# 3. Check MinIO bucket data
docker exec ldt-minio mc ls local/

# 4. Check ML service metrics
curl http://localhost:8000/metrics | grep inference
```

### Check Each Layer Output

```bash
# Layer 1 output: Valid records printed to console (Terminal 3)
# Look for: Trip IDs, timestamps, zone IDs

# Layer 2 output: Canary violations
# Look for: negative_fare, invalid_passengers, etc.

# Layer 3 output: Meta-metrics
# Look for: volume, null_rate, violation_rate, anomaly_rate

# Layer 4 output: IEC decisions
# Look for: strategy (do_nothing, adjust_threshold, retrain_model)
```

---

## 🛠️ Troubleshooting

### Services not starting?

```bash
# Check Docker
docker compose ps
docker compose logs

# Restart if needed
docker compose down
docker compose up -d
```

### Kafka topics not created?

```bash
# Verify Kafka is ready
docker exec cadqstream_kafka_1 kafka-broker-api-versions \
  --bootstrap-server localhost:9092

# Recreate topics
./scripts/create_topics.sh
```

### Models not loading?

```bash
# Check models exist
ls -lh models/

# Required files:
# - iforest_model_v2.pkl (11.3 MB)
# - scaler.pkl
# - context_thresholds_v2.json
# - meter_hypernetwork.pkl
# - meter_scaler.pkl
```

### Flink job errors?

```bash
# Check Python dependencies
pip install -r requirements.txt

# Check PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

# Run with verbose logging
python src/flink_job_complete.py 2>&1 | tee flink.log
```

---

## 📊 Expected Flow

```
Producer → Kafka (taxi-nyc-raw)
           ↓
    Layer 1: Validation
           ↓
    Layer 2: Canary + Complex
           ↓
    Layer 3: Rendezvous + Meta
           ↓
    Layer 4: IEC
           ↓
    Outputs: MinIO + Kafka + Metrics
```

**Normal throughput:** 1,000-5,000 events/sec
**Latency:** <100ms per layer
**Memory:** ~2GB for Flink job, ~500MB for ML service

---

## ✅ Validation

Run all validation scripts:

```bash
# Phase 1
python scripts/validate_phase1.py

# Phase 2
python scripts/validate_phase2.py

# Phase 3
python scripts/validate_phase3.py

# Phase 4
python scripts/validate_phase4.py
```

**Expected:** All checks pass (100%)

---

## 🏁 Complete System Running

When all services are running, you should have:

1. ✅ Docker services (6 containers)
2. ✅ ML service (port 8000)
3. ✅ Action Replay Worker
4. ✅ Flink job (4-layer pipeline)
5. ✅ Data producer

**Check endpoints:**
- ML Service: http://localhost:8000/docs
- Health: http://localhost:8000/health
- Metrics: http://localhost:8000/metrics
- Kafka UI: (optional) http://localhost:8080

---

## 🎯 Next Steps

1. **Monitor metrics:** Check Prometheus endpoint
2. **View results:** Check MinIO buckets for processed records
3. **Test drift:** Inject anomalies and watch IEC adapt
4. **Benchmark:** Run performance validation scripts
5. **Tune:** Adjust thresholds, model parameters

**Chúc mừng! Hệ thống đã chạy hoàn chỉnh! 🎉**
