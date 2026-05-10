# CA-DQStream Production Deployment

Production-grade end-to-end streaming platform deployment for the Context-Aware Data Quality Stream Processing System (CA-DQStream).

## Architecture

```
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │                              CA-DQStream 4-Layer Pipeline                    │
# ├─────────────────────────────────────────────────────────────────────────────┤
# │                                                                             │
# │  ┌──────────────────────────────────────────────────────────────────────┐ │
# │  │  Layer 1: Baseline Validation                                          │ │
# │  │  Kafka Source (taxi-nyc-raw)                                          │ │
# │  │    → Parse JSON → Watermark → KeyGen → Dedup → Schema Validation       │ │
# │  │    → taxi_trips_raw (PostgreSQL) + schema_violations                  │ │
# │  └──────────────────────────────────────────────────────────────────────┘ │
# │                                    ↓ (valid records)                         │
# │  ┌──────────────────────────────────────────────────────────────────────┐ │
# │  │  Layer 2: Dual-Branch Processing                                       │ │
# │  │  ┌──────────────────────┐    ┌──────────────────────────────────┐    │ │
# │  │  │  Canary Branch       │    │  Complex Branch (ML Scoring)      │    │ │
# │  │  │  (Rule-based)       │    │  (Anomaly Detection)              │    │ │
# │  │  │  7 business rules   │    │  Isolation Forest / ML models     │    │ │
# │  │  │  → hard violations  │    │  → anomaly_scores                 │    │ │
# │  │  └──────────────────────┘    └──────────────────────────────────┘    │ │
# │  └──────────────────────────────────────────────────────────────────────┘ │
# │                                    ↓ (both branches)                         │
# │  ┌──────────────────────────────────────────────────────────────────────┐ │
# │  │  Layer 3: Rendezvous + MetaAggregator                                 │ │
# │  │  CoProcessFunction (synchronize Canary + Complex)                      │ │
# │  │    → Voting Ensemble (Canary overrides ML)                             │ │
# │  │    → 1-min windowed meta-metrics per neighborhood (6 signals)          │ │
# │  │    → meta_metrics (PostgreSQL) + dq-meta-stream (Kafka)               │ │
# │  └──────────────────────────────────────────────────────────────────────┘ │
# │                                    ↓ (meta-metrics)                         │
# │  ┌──────────────────────────────────────────────────────────────────────┐ │
# │  │  Layer 4: IEC (Intelligent Evolution Controller)                      │ │
# │  │  ADWIN-U drift detection (36 instances per neighborhood×metric)        │ │
# │  │  METER hypernetwork strategy prediction                               │ │
# │  │  Multi-strategy execution (do_nothing / adjust / retrain / switch)    │ │
# │  │    → drift_events (PostgreSQL) + iec-action-replay (Kafka)           │ │
# │  └──────────────────────────────────────────────────────────────────────┘ │
# │                                                                             │
# └─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Docker Engine 24.0+
- Docker Compose v2.20+
- 16+ GB RAM (32 GB recommended)
- 50+ GB free disk space
- Git Bash, WSL2, or Linux shell for running `.sh` scripts

### Deployment (3 Commands)

```bash
# 1. Navigate to deployment directory
cd deployment

# 2. Make scripts executable (Unix/Linux only)
chmod +x scripts/*.sh

# 3. Start the entire stack (full reset + deployment)
bash scripts/start.sh
```

### Stopping

```bash
# Graceful stop (preserves data volumes)
bash scripts/stop.sh

# Hard stop (deletes all data volumes)
bash scripts/stop.sh --remove-volumes
```

### Health Check

```bash
bash scripts/check-health.sh
```

## Service Inventory

| Service | Container | Port | Endpoint | Credentials | Health Check |
|---------|-----------|------|----------|-------------|---------------|
| Kafka | ldt-kafka | 9092 | localhost:9092 | - | kafka-topics list |
| Zookeeper | ldt-zookeeper | 2181 | localhost:2181 | - | `ruok` |
| Schema Registry | ldt-schema-registry | 8082 | localhost:8082 | - | HTTP /subjects |
| Kafka UI | ldt-kafka-ui | 8080 | localhost:8080 | - | HTTP / |
| Kafka Exporter | ldt-kafka-exporter | 9308 | localhost:9308 | - | Prometheus metrics |
| PostgreSQL | ldt-postgres | 5432 | localhost:5432 | cadqstream/cadqstream123 | pg_isready |
| PgBouncer | ldt-pgbouncer | 6432 | localhost:6432 | cadqstream/cadqstream123 | TCP connect |
| Postgres Exporter | ldt-postgres-exporter | 9187 | localhost:9187 | - | Prometheus metrics |
| MinIO | ldt-minio | 9000/9001 | localhost:9000 / localhost:9001 | minioadmin/minioadmin123 | mc ready |
| Flink JobManager | ldt-flink-jobmanager | 8081 | localhost:8081 | - | REST /overview |
| Flink TaskManager | ldt-flink-taskmanager | - | internal | - | - |
| MLflow | ldt-mlflow | 5000 | localhost:5000 | - | HTTP / |
| Prometheus | ldt-prometheus | 9090 | localhost:9090 | - | /-/healthy |
| Grafana | ldt-grafana | 3000 | localhost:3000 | admin/admin123 | /api/health |
| Node Exporter | ldt-node-exporter | 9100 | localhost:9100 | - | Prometheus metrics |

## Kafka Topics

The pipeline uses the following Kafka topics:

| Topic | Partitions | Purpose |
|-------|------------|---------|
| `taxi-nyc-raw` | 4 | Raw taxi event input from data producer |
| `dq-stream-processed` | 4 | Deduplicated, validated records |
| `dq-stream-anomalies` | 4 | Detected anomalies from Layer 2 |
| `dq-meta-stream` | 4 | Windowed meta-metrics from Layer 3 |
| `dq-metrics` | 1 | Pipeline operational metrics |
| `dq-hard-rule-violations` | 4 | Canary rule violations |
| `iec-action-replay` | 1 | IEC drift action replay buffer |

## Data Flow

### Event Ingestion
```
Producer → taxi-nyc-raw (Kafka) → Flink Kafka Source
```

### Layer 1 (Baseline Validation)
```
Raw Stream → Parse JSON → Watermark (30s idleness) → TripID (MurmurHash3)
→ Deduplicate (7-day TTL) → Schema Validation (NYC TLC zones 1-263)
→ [violations] → schema_violations (PostgreSQL)
→ [valid] → taxi_trips_raw (PostgreSQL) → next layer
```

### Layer 2 (Dual-Branch)
```
Valid Stream → Canary Rules (7 rules: negative_fare, zero_distance, invalid_passengers, etc.)
            → canary_violations (PostgreSQL)

Valid Stream → ML Scoring (Isolation Forest + context features)
            → anomaly_scores (PostgreSQL)
```

### Layer 3 (Voting + Meta)
```
[Canary + Complex] → CoProcessFunction (Rendezvous)
→ Voting Ensemble (Canary overrides ML)
→ 1-min Tumbling Windows (per neighborhood)
→ meta_metrics (PostgreSQL) + dq-meta-stream (Kafka)
```

### Layer 4 (IEC)
```
meta_metrics → ADWIN-U (36 instances: 6 neighborhoods × 6 metrics)
→ DriftAggregator → METER Strategy Prediction
→ Strategy Execution (adjust_threshold / retrain_model / switch_model)
→ drift_events (PostgreSQL)
```

## Flink Job Submission

The pipeline is submitted as a Python DataStream job via the `flink-init` container:

```bash
# Job entry point
python3 /opt/flink/e2e/src/flink_job_complete.py

# Connects to:
#   - Kafka: kafka:9092 (topic: taxi-nyc-raw)
#   - PgBouncer: pgbouncer:5432 (database: dq_pipeline)
#   - State backend: filesystem (MinIO checkpoints)
```

### Checkpointing
- Mode: EXACTLY_ONCE
- Interval: 45 seconds
- Externalized: RETAIN_ON_CANCELLATION
- Storage: `s3://checkpoints/flink-checkpoints`

### Fault Tolerance
- Restart strategy: exponential-delay (10s initial → 5min max)
- Max concurrent checkpoints: 1
- Min pause between checkpoints: 30s

## Monitoring Guide

### Grafana Dashboards (5 Pre-configured)

1. **CA-DQStream: Pipeline Overview** (`cadqstream-pipeline-overview`)
   - End-to-end flow: Kafka in → Flink processing → PostgreSQL out
   - Per-layer metrics: input rates, violation rates, anomaly rates
   - IEC drift detection and strategy execution

2. **CA-DQStream: Kafka Overview** (`cadqstream-kafka-overview`)
   - Broker health, consumer lag, under-replicated partitions
   - Topic message rates and throughput
   - Consumer group offsets

3. **CA-DQStream: Flink Jobs** (`cadqstream-flink-jobs`)
   - Job/TaskManager status, checkpointing metrics
   - Processing throughput (records/sec per task)
   - JVM heap memory and watermark latency

4. **CA-DQStream: PostgreSQL Health** (`cadqstream-postgres-health`)
   - Connection pool (PgBouncer) stats
   - Query performance, transaction rates
   - Cache hit rate, table row counts

5. **CA-DQStream: Infrastructure** (`cadqstream-infrastructure`)
   - Host CPU, memory, disk I/O, network
   - Service availability

### Alert Rules

Prometheus alert rules cover:
- **Kafka**: broker down, consumer lag, under-replicated partitions
- **Flink**: job failed, checkpoint failure, TaskManager down, high GC time
- **PostgreSQL**: connection exhaustion, long-running queries, low cache hit
- **Infrastructure**: disk usage >80%, memory >85%, container restart loop
- **Pipeline**: E2E latency, record drops, anomaly rate spikes, drift detected

### Prometheus Targets

```
node-exporter:9100          # Host metrics
postgres-exporter:9187      # DB metrics
kafka-exporter:9308         # Kafka JMX metrics
flink-jobmanager:9248       # Flink JobManager metrics
flink-taskmanager:9249      # Flink TaskManager metrics
minio:9096                  # MinIO metrics
mlflow:5000                 # MLflow metrics
prometheus:9090              # Self-monitoring
```

## Troubleshooting

### Kafka not starting
```bash
# Check Zookeeper
docker logs ldt-zookeeper
# Check Kafka
docker logs ldt-kafka
# Verify broker ID
docker exec ldt-kafka bash -c 'echo $KAFKA_BROKER_ID'
```

### Flink job fails to submit
```bash
# Check JobManager logs
docker logs ldt-flink-jobmanager
# Verify Kafka connectivity from Flink container
docker exec ldt-flink-jobmanager bash -c 'nc -zv kafka 9092'
# Check REST API
curl http://localhost:8081/overview
```

### PostgreSQL connection refused
```bash
# Check PgBouncer logs
docker logs ldt-pgbouncer
# Test direct PostgreSQL connection
docker exec ldt-postgres psql -U cadqstream -d dq_pipeline -c 'SELECT 1'
# Test PgBouncer connection
docker exec ldt-pgbouncer bash -c 'echo "show pools;" | nc localhost 6432'
```

### Prometheus not scraping targets
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets | python3 -m json.tool
# Verify scrape config
docker exec ldt-prometheus cat /etc/prometheus/prometheus.yml
```

### MinIO buckets not created
```bash
# Run minio-init manually
docker compose up minio-init
# Check mc alias
docker exec ldt-minio mc alias set local http://localhost:9000 minioadmin minioadmin123
# List buckets
docker exec ldt-minio mc ls local/
```

### Container restart loop
```bash
# Inspect container logs
docker logs --tail 100 ldt-<service-name>
# Check resource limits
docker stats --no-stream
# Verify health check timeout (increase if needed)
```

### Reset Everything
```bash
bash scripts/stop.sh --remove-volumes
bash scripts/start.sh
```

## Development Workflow

### Update Flink Job Without Downtime

```bash
# 1. Savepoint the current job
SAVEPOINT=$(curl -s http://localhost:8081/jobs/<job-id>/savepoints \
  -X POST -H "Content-Type: application/json" \
  -d '{"format":"canonical"}' | python3 -c "import sys,json; print(json.load(sys.stdin)['savepoint'])"
echo "Savepoint: $SAVEPOINT"

# 2. Cancel with savepoint
curl -X PATCH http://localhost:8081/jobs/<job-id> \
  -H "Content-Type: application/json" \
  -d "{\"drain\":false,\"savepointFormat\":\"canonical\",\"customizedSavepointDest\":\"$SAVEPOINT\"}"

# 3. Submit new version
docker compose -f docker-compose.yml up flink-init

# 4. Resume from savepoint
curl -X POST http://localhost:8081/jobs/<new-job-id>/savepoints/restore \
  -H "Content-Type: application/json" \
  -d "{\"savepointPath\":\"$SAVEPOINT\"}"
```

### Hot Reload Python Operators

The Flink containers mount the `../src` directory as read-only. Update source files on the host and restart the job to apply changes:

```bash
# Restart Flink job
docker compose -f docker-compose.yml up -d --force-recreate flink-init
```

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `KAFKA_BOOTSTRAP_SERVERS` | `kafka:9092` | Kafka broker connection |
| `SCHEMA_REGISTRY_URL` | `http://schema-registry:8081` | Confluent Schema Registry |
| `POSTGRES_DB` | `dq_pipeline` | PostgreSQL database name |
| `POSTGRES_USER` | `cadqstream` | PostgreSQL username |
| `POSTGRES_PASSWORD` | `cadqstream123` | PostgreSQL password |
| `PGBOUNCER_HOST` | `pgbouncer` | PgBouncer hostname |
| `PGBOUNCER_PORT` | `5432` | PgBouncer port |
| `MINIO_ENDPOINT` | `minio:9000` | MinIO S3 endpoint |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `MINIO_SECRET_KEY` | `minioadmin123` | MinIO secret key |
| `FLINK_JOB_MANAGER_RPC_ADDRESS` | `flink-jobmanager` | Flink JobManager RPC address |
| `FLINK_PARALLELISM_DEFAULT` | `4` | Default parallelism |
| `FLINK_STATE_BACKEND` | `filesystem` | State backend type |
| `FLINK_CHECKPOINT_INTERVAL` | `45000` | Checkpoint interval (ms) |
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server URI |
| `PIPELINE_WATERMARK_IDLENESS_SECONDS` | `30` | Watermark idleness timeout |
| `PIPELINE_DEDUP_TTL_DAYS` | `7` | Deduplication TTL (days) |
| `IEC_ADWIN_DELTA` | `0.002` | ADWIN-U sensitivity |

## File Structure

```
deployment/
├── docker-compose.yml          # Main orchestration
├── .env                       # Environment variables
├── README.md                  # This file
│
├── flink/
│   ├── Dockerfile             # Custom Flink image
│   ├── flink-conf.yml        # Flink configuration
│   ├── jobmanager.conf       # JobManager JVM opts
│   └── taskmanager.conf      # TaskManager JVM opts
│
├── kafka/
│   └── init-scripts/         # (handled inline in docker-compose)
│
├── postgres/
│   ├── init-scripts/
│   │   ├── 01-init-schema.sql    # Full schema (8 tables, views, indexes)
│   │   └── 02-performance-tuning.sql
│   ├── pgbouncer.ini         # PgBouncer config
│   └── queries.yml            # Postgres exporter custom queries
│
├── prometheus/
│   ├── prometheus.yml         # Scrape configs for all targets
│   └── alert-rules/
│       └── cadqstream-alerts.yml  # 30+ alert rules
│
├── grafana/
│   ├── datasources/
│   │   └── prometheus.yml    # Prometheus datasource
│   ├── provisioning/
│   │   └── dashboards.yml    # Dashboard provider
│   └── dashboards/
│       ├── kafka-overview.json
│       ├── flink-jobs.json
│       ├── postgres-health.json
│       ├── infrastructure.json
│       └── pipeline-overview.json
│
├── minio/
│   └── init-scripts/         # (handled inline in docker-compose)
│
├── mlflow/
│   └── mlflow.env            # MLflow environment variables
│
└── scripts/
    ├── start.sh              # Full deployment script
    ├── stop.sh               # Stop and cleanup
    ├── wait-for.sh           # Service dependency wait utility
    ├── init-all.sh           # Run all init scripts
    └── check-health.sh        # Health verification
```

## Resource Allocation

The docker-compose.yml allocates resources based on a 32-core / 64 GB machine. For smaller hardware, adjust:

```yaml
# In docker-compose.yml, per-service deploy.resources.limits:
kafka:        { memory: 2G, cpus: "1.0" }
flink-taskmanager: { memory: 2G, cpus: "1.0", replicas: 1 }
postgres:     { memory: 1G, cpus: "0.5" }
prometheus:  { memory: 1G, cpus: "0.25" }
```

## Production Considerations

This deployment is production-ready for **single-node** scenarios. For multi-node HA, consider:

1. **Kafka**: Increase `offsets.topic.replication.factor` to 3, add more brokers
2. **PostgreSQL**: Set up streaming replication with a primary and replicas
3. **Flink**: Run HA JobManager with ZooKeeper/HA, increase TaskManager replicas
4. **MinIO**: Set up erasure coding or use distributed MinIO
5. **Prometheus**: Set up Thanos or Cortex for long-term storage
6. **TLS**: Enable SSL/TLS for Kafka, PostgreSQL, and REST APIs
7. **Secrets**: Use Docker secrets or a secrets manager for credentials
8. **Logging**: Forward container logs to a centralized logging system (ELK/Loki)
