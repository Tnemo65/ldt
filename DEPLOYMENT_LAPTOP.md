# CA-DQStream - Laptop Deployment (16GB RAM)

Optimized for personal laptops with limited resources.

## System Requirements

- **RAM**: 16GB (Windows uses ~8GB, leaving ~8GB for application)
- **CPU**: 4+ cores recommended
- **Disk**: 20GB free space
- **OS**: Windows 10/11 with WSL2 or Docker Desktop

## Memory Budget

```
Windows OS:        ~8GB  (50% of 16GB)
Docker Services:   ~2GB  (Kafka 1GB, Postgres 512MB, Zookeeper 512MB)
Flink Pipeline:    ~2GB  (parallelism=1)
ML Models:         ~500MB (small iForest)
Buffer:            ~3.5GB (for system stability)
-------------------------------------------------
Total:             ~16GB
```

## Quick Start

### 1. Start Lightweight Docker Services

```bash
# Use laptop-optimized compose file
docker compose -f docker-compose.laptop.yml up -d

# Wait 30s
sleep 30

# Verify (should use ~2GB total)
docker stats --no-stream
```

### 2. Create Topics (Reduced Partitions)

```bash
./scripts/create_topics_laptop.sh

# Expected: 1-2 partitions per topic (vs 12 in full version)
```

### 3. Train Lightweight Model (Optional)

```bash
# Smaller model: 100 trees (vs 200), height 8 (vs 10), window 256 (vs 512)
python src/ml/train_iforest.py \
  --n-trees 100 \
  --height 8 \
  --window-size 256

# Expected: ~5MB model (vs 11MB full version)
```

### 4. Run Laptop Pipeline

```bash
# Single-threaded, 120s checkpointing, mock ML
python src/flink_job_laptop.py
```

### 5. Produce Test Data (Low Rate)

```bash
# 100 events/sec (vs 1000 in full version)
python scripts/produce_taxi_data.py \
  --file data/yellow_tripdata_2024-01.parquet \
  --rate 100 \
  --limit 1000
```

## Key Differences from Full Version

| Component | Full Server | Laptop | Reason |
|-----------|-------------|--------|--------|
| **Parallelism** | 4 | 1 | Reduce memory |
| **Kafka partitions** | 12 | 2 | Reduce overhead |
| **Kafka heap** | 2GB | 512MB | Memory limit |
| **Postgres** | Standard | Alpine | Smaller image |
| **Checkpointing** | 45s | 120s | Reduce I/O |
| **ML model** | 200 trees | 100 trees | Reduce size |
| **Data rate** | 1000 eps | 100 eps | Prevent overload |
| **Sinks** | Postgres+Kafka | Print only | Simplify |

## Troubleshooting

### Out of Memory?

```bash
# Check Docker memory
docker stats

# If containers using >2GB, restart with lower limits:
docker compose -f docker-compose.laptop.yml down
docker compose -f docker-compose.laptop.yml up -d
```

### Slow Performance?

```bash
# Reduce data rate further
python scripts/produce_taxi_data.py --rate 50 --limit 500

# Or process in batches
python scripts/produce_taxi_data.py --rate 100 --limit 100
# Wait for processing to finish
# Then produce next batch
```

### Windows Specific

If using Docker Desktop on Windows:
1. Go to Docker Desktop → Settings → Resources
2. Set Memory: 4GB (not more - Windows needs the rest)
3. Set CPU: 2-4 cores
4. Apply & Restart

## What to Skip on Laptop

❌ Skip these (not needed for testing):
- Schema Registry (can use JSON instead of Avro)
- PgBouncer (connection pooling not needed for single client)
- MinIO (object storage not needed for testing)
- ML Service API (just run pipeline directly)
- Action Replay Worker (not critical for testing)

✅ Keep only essentials:
- Kafka (message broker)
- Zookeeper (Kafka dependency)
- Postgres (data sink) - or skip and use print() only

## Expected Performance

- **Throughput**: 100-200 events/sec
- **Latency**: 200-500ms per layer
- **Memory**: ~4-6GB total (Docker + Flink)
- **Run time**: Can run continuously for hours

## Monitor Resources

```bash
# Memory usage
docker stats --format "table {{.Name}}\t{{.MemUsage}}"

# Laptop resource monitor (Windows)
# Open Task Manager → Performance tab
# Watch: Memory, CPU, Disk

# If memory > 90%, stop producer and let pipeline catch up
```

---

**Note**: This is for development/testing only. For production, use full server deployment.
