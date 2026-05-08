# CA-DQStream - Laptop Deployment Guide for AI Agents

**Context**: Hệ thống này được thiết kế cho server với tài nguyên lớn. File này hướng dẫn điều chỉnh để chạy trên laptop 16GB RAM (Windows sử dụng ~8GB, còn ~8GB cho application).

---

## 📊 System Requirements & Constraints

### Hardware Available
- **RAM**: 16GB total → 8GB available (Windows uses 50%)
- **CPU**: 4+ cores (typical laptop)
- **Disk**: 20GB free space minimum
- **OS**: Windows 10/11 with Docker Desktop (WSL2)

### Memory Budget
```
Total 16GB:
├─ Windows OS:          8GB    (50% - cannot reduce)
└─ Application:         8GB    (50% - must fit everything here)
   ├─ Docker Services:  2GB    (Kafka 1GB + Postgres 512MB + Zookeeper 512MB)
   ├─ Flink Pipeline:   2GB    (single parallelism)
   ├─ ML Models:        500MB  (small iForest)
   ├─ Python/JVM:       1GB    (runtime overhead)
   └─ System Buffer:    2.5GB  (stability margin)
```

---

## 🔧 Required Adjustments

### 1. Docker Services Configuration

**Use**: `docker-compose.laptop.yml` (NOT `docker-compose.yml`)

**Key changes**:
```yaml
# Memory limits per service:
kafka:
  mem_limit: 1g              # Was: unlimited (could use 2GB+)
  KAFKA_HEAP_OPTS: "-Xmx512M -Xms256M"
  
postgres:
  image: postgres:15-alpine  # Was: postgres:15 (Alpine = smaller)
  mem_limit: 512m
  command: postgres -c shared_buffers=128MB -c max_connections=50
  
zookeeper:
  mem_limit: 512m

# SKIP these services (not needed for testing):
# - schema-registry  (use JSON instead of Avro)
# - pgbouncer        (no connection pooling needed)
# - minio            (no object storage needed)
```

**Total Docker memory**: ~2GB (vs 6GB+ in full version)

---

### 2. Kafka Topics Configuration

**Use**: `scripts/create_topics_laptop.sh` (NOT `scripts/create_topics.sh`)

**Key changes**:
```bash
# Partitions per topic:
taxi-nyc-raw:              2 partitions  # Was: 12
dq-schema-violations:      1 partition   # Was: 3
dq-hard-rule-violations:   1 partition   # Was: 3
dq-anomaly-scores:         1 partition   # Was: 4
dq-meta-stream:            1 partition   # Was: 2
if-model-updates:          1 partition   # Was: 1 (same)
iec-action-replay:         1 partition   # Was: 2

# Retention:
retention.ms: 3600000  # 1 hour (was: 168 hours = 7 days)
```

**Why**: Fewer partitions = less memory overhead, less parallelism needed

---

### 3. Flink Pipeline Configuration

**Use**: `src/flink_job_laptop.py` (NOT `src/flink_job_complete.py`)

**Key changes**:
```python
# Parallelism
env.set_parallelism(1)  # Was: 4
# Memory impact: 1 thread uses ~2GB, 4 threads use ~6-8GB

# Checkpointing
checkpoint_interval = 120000  # 120s (was: 45s)
# Why: Less frequent = less I/O overhead

# ML Scoring
# Use MockMLScoringFunction (random scores)
# Don't load actual model in pipeline
# Why: Model loading in operator = 500MB * parallelism
```

**Critical**: Parallelism=1 means:
- Only 1 thread processing events
- Throughput: 100-200 eps (vs 1000-5000 eps)
- But memory: 2GB (vs 6-8GB)

---

### 4. ML Model Training

**Adjust parameters** in `src/ml/train_iforest.py`:

```bash
# Laptop version (smaller model):
python src/ml/train_iforest.py \
  --n-trees 100 \        # Was: 200
  --height 8 \           # Was: 10  
  --window-size 256      # Was: 512

# Model size: ~5MB (vs 11MB full version)
# Training time: ~5 min (vs 10 min)
# Memory during training: ~2GB (vs 4GB)
```

**Why smaller model?**
- Fewer trees = faster inference, less memory
- Smaller window = less state to maintain
- Still effective for testing/demo purposes

---

### 5. Data Producer Configuration

**Reduce event rate**:

```bash
# Laptop version:
python scripts/produce_taxi_data.py \
  --file data/yellow_tripdata_2024-01.parquet \
  --rate 100 \      # Was: 1000 (10x slower)
  --limit 1000      # Process in small batches

# Why: Pipeline with parallelism=1 can only handle ~100-200 eps
```

---

### 6. Services to SKIP on Laptop

**Do NOT start** these (not critical for testing):

```bash
# ❌ Skip:
- FastAPI ML Service       (can test pipeline without API)
- Action Replay Worker     (retry logic not critical for demo)
- Schema Registry          (use JSON instead of Avro)
- PgBouncer                (no connection pooling needed)
- MinIO                    (no object storage needed)

# ✅ Keep only:
- Kafka + Zookeeper        (message broker - essential)
- Postgres                 (data sink - can also skip and use print())
```

---

## 🚀 Step-by-Step Deployment

### Step 1: Start Docker Services (2GB)

```bash
cd /path/to/ldt

# Use laptop config
docker compose -f docker-compose.laptop.yml up -d

# Wait for services to start
sleep 30

# Verify memory usage (~2GB total)
docker stats --no-stream
```

**Expected output**:
```
NAME        MEM USAGE / LIMIT
kafka       800MB / 1GB
postgres    300MB / 512MB
zookeeper   400MB / 512MB
```

---

### Step 2: Create Topics (Reduced Partitions)

```bash
./scripts/create_topics_laptop.sh

# Verify
docker exec kafka kafka-topics --bootstrap-server localhost:9092 --list
```

**Expected**: 7 topics with 1-2 partitions each

---

### Step 3: Train Small Model (Optional)

```bash
# Train lightweight model
python src/ml/train_iforest.py \
  --n-trees 100 \
  --height 8 \
  --window-size 256

# Verify model size (~5MB)
ls -lh models/iforest_model_v2.pkl
```

**Note**: Can skip this step and use mock ML in pipeline

---

### Step 4: Run Laptop Pipeline

```bash
# Set PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

# Run single-threaded pipeline
python src/flink_job_laptop.py
```

**Expected output**:
```
CA-DQStream Laptop Version - Lightweight (1 parallelism)
✓ Laptop config:
  Parallelism: 1 (single-threaded)
  Checkpointing: 120s interval
  Memory: Optimized for ~2GB
...
```

**Memory usage**: ~2GB for Flink job

---

### Step 5: Produce Test Data (Low Rate)

In another terminal:

```bash
# Produce 1000 records at 100 eps
python scripts/produce_taxi_data.py \
  --file data/yellow_tripdata_2024-01.parquet \
  --rate 100 \
  --limit 1000

# Expected: Completes in ~10 seconds
```

---

## 📸 Taking Screenshots for Reports

### Option 1: Add Kafka UI (Lightweight)

Add to `docker-compose.laptop.yml`:

```yaml
  kafka-ui:
    image: provectuslabs/kafka-ui:latest
    depends_on:
      - kafka
    ports:
      - "8080:8080"
    environment:
      KAFKA_CLUSTERS_0_NAME: local
      KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS: kafka:9092
    mem_limit: 512m
```

Then access: **http://localhost:8080**

**Screenshots to take**:
- Topics list (show partitions, messages)
- Topic details (messages, configuration)
- Consumer groups (show lag)

---

### Option 2: Flink Web UI (If using Flink standalone)

If running Flink with Web UI enabled:

```bash
# Start Flink cluster (optional - uses more memory)
# Download: https://flink.apache.org/downloads.html
./bin/start-cluster.sh

# Access: http://localhost:8081
```

**Screenshots to take**:
- Job overview (running jobs)
- Task metrics (records processed, latency)
- Checkpoints (checkpoint statistics)

---

### Option 3: Command-Line Outputs (No UI needed)

```bash
# 1. Docker stats (resource usage)
docker stats --no-stream > docker-stats.txt
# Screenshot the output

# 2. Kafka topic details
docker exec kafka kafka-topics \
  --bootstrap-server localhost:9092 \
  --describe --topic taxi-nyc-raw > kafka-topics.txt
# Screenshot the output

# 3. Postgres data counts
docker exec -it cadqstream-postgres-1 psql -U cadqstream -d dq_pipeline -c \
  "SELECT 'taxi_trips_raw' as table, COUNT(*) as records FROM taxi_trips_raw
   UNION ALL
   SELECT 'schema_violations', COUNT(*) FROM schema_violations;"
# Screenshot the output

# 4. Flink job console output
# Just screenshot the terminal where Flink is running
# Should show processed records, Layer 1/2/3/4 outputs

# 5. Pipeline metrics
# Screenshot Flink output showing:
# - Records processed per layer
# - Violations detected
# - Anomaly scores
# - IEC decisions
```

---

### Option 4: Python Monitoring Script (For Screenshots)

Create `scripts/monitor_for_screenshots.py`:

```python
#!/usr/bin/env python3
"""Generate monitoring output suitable for screenshots."""

import psycopg2
from kafka import KafkaConsumer, TopicPartition
from datetime import datetime

print("="*80)
print(f"CA-DQStream Monitoring Report - {datetime.now()}")
print("="*80)

# 1. Kafka Topics Stats
print("\n📊 Kafka Topics:")
print("-"*80)
consumer = KafkaConsumer(bootstrap_servers='localhost:9092')
topics = ['taxi-nyc-raw', 'dq-schema-violations', 'dq-anomaly-scores', 'dq-meta-stream']

for topic in topics:
    partitions = consumer.partitions_for_topic(topic)
    if partitions:
        total_messages = 0
        for p in partitions:
            tp = TopicPartition(topic, p)
            consumer.assign([tp])
            consumer.seek_to_end(tp)
            end_offset = consumer.position(tp)
            total_messages += end_offset
        print(f"  {topic:.<40} {total_messages:>10} messages")

# 2. Postgres Stats
print("\n💾 Database Records:")
print("-"*80)
conn = psycopg2.connect(
    host="localhost",
    port=5432,
    database="dq_pipeline",
    user="cadqstream",
    password="cadqstream123"
)
cursor = conn.cursor()

tables = ['taxi_trips_raw', 'schema_violations', 'hard_rule_violations']
for table in tables:
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        print(f"  {table:.<40} {count:>10} records")
    except:
        print(f"  {table:.<40} {'(table not exists)':>10}")

cursor.close()
conn.close()

print("\n" + "="*80)
print("Screenshot this output for your report!")
print("="*80)
```

Run and screenshot:
```bash
python scripts/monitor_for_screenshots.py
```

---

## 🎯 Recommended Screenshots for Report

### 1. **System Overview** (show resource usage is reasonable)
- Windows Task Manager → Performance tab
- Show: CPU ~30%, Memory ~12GB/16GB used

### 2. **Docker Services** (show all running)
```bash
docker compose -f docker-compose.laptop.yml ps
docker stats --no-stream
```

### 3. **Kafka Topics** (show data flowing)
```bash
docker exec kafka kafka-topics --bootstrap-server localhost:9092 --list
# Screenshot showing all 7 topics created
```

### 4. **Pipeline Running** (show Flink processing)
- Terminal with `python src/flink_job_laptop.py` output
- Show Layer 1, 2, 3, 4 startup messages
- Show some processed records

### 5. **Data Producer** (show data being sent)
- Terminal with producer output
- Show "Sent X records, Y events/sec"

### 6. **Pipeline Output** (show processing results)
- Flink console showing processed records
- Sample outputs from each layer

### 7. **Database Results** (show data persisted)
```bash
docker exec postgres psql -U cadqstream -d dq_pipeline -c \
  "SELECT COUNT(*) FROM taxi_trips_raw"
```

### 8. **Final Metrics** (show completion)
- Run `scripts/monitor_for_screenshots.py`
- Screenshot the summary table

---

## ⚠️ Important Notes for Laptop Deployment

### Memory Management

**Watch memory usage continuously**:
```bash
# Monitor Docker
watch -n 5 'docker stats --no-stream'

# If memory >90%, stop producer:
# Ctrl+C on producer terminal
# Let pipeline catch up (process existing messages)
# Then restart producer with smaller --limit
```

### Performance Expectations

**This is NOT production deployment** - it's for testing/demo:

| Metric | Laptop | Server (Full) |
|--------|--------|---------------|
| Throughput | 100-200 eps | 1000-5000 eps |
| Latency | 200-500ms | <100ms |
| Parallelism | 1 | 4+ |
| Memory | ~6-8GB | ~20GB+ |
| Uptime | Hours | Days/Weeks |

### Windows Docker Desktop Settings

**Critical settings** (Settings → Resources):
- Memory: **4GB** (NOT more! Windows needs the rest)
- CPUs: 2-4 cores
- Disk: 20GB+
- Enable WSL 2 backend

### Troubleshooting

**Out of Memory?**
```bash
# 1. Stop everything
docker compose -f docker-compose.laptop.yml down

# 2. Clear Docker cache
docker system prune -a --volumes

# 3. Restart with limits
docker compose -f docker-compose.laptop.yml up -d

# 4. Monitor
docker stats
```

**Pipeline too slow?**
```bash
# Reduce data rate even more
python scripts/produce_taxi_data.py --rate 50 --limit 100

# Or process in tiny batches
for i in {1..10}; do
  python scripts/produce_taxi_data.py --rate 100 --limit 100
  sleep 10  # Let pipeline catch up
done
```

---

## 📝 Summary: Key Files for Laptop

| Purpose | File | vs Server Version |
|---------|------|-------------------|
| Docker services | `docker-compose.laptop.yml` | 3 services (vs 6), 2GB (vs 6GB+) |
| Kafka topics | `scripts/create_topics_laptop.sh` | 1-2 partitions (vs 12) |
| Flink pipeline | `src/flink_job_laptop.py` | parallelism=1 (vs 4) |
| ML model training | `src/ml/train_iforest.py --n-trees 100` | 100 trees (vs 200) |
| Data producer | `--rate 100 --limit 1000` | 100 eps (vs 1000) |
| Deployment guide | `DEPLOYMENT_LAPTOP.md` | Step-by-step for 16GB laptop |

---

## 🤖 For AI Agents Reading This

**If you are an AI agent helping deploy this system on a 16GB laptop**:

1. ✅ **DO** use all `*.laptop.*` files
2. ✅ **DO** set parallelism=1
3. ✅ **DO** limit Docker to 4GB memory
4. ✅ **DO** reduce data rate to 100 eps
5. ✅ **DO** monitor memory continuously
6. ❌ **DON'T** use full server configs
7. ❌ **DON'T** try to run all services
8. ❌ **DON'T** produce data faster than 100 eps
9. ❌ **DON'T** assume unlimited resources

**Remember**: This is a **resource-constrained environment**. Every adjustment above is necessary to fit within 8GB available RAM. Skipping any adjustment will likely cause OOM (Out of Memory) errors.

---

**Last updated**: 2026-05-08
**Tested on**: Windows 11, 16GB RAM, Docker Desktop 4.x, WSL2
