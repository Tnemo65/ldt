# Production E2E Docker Deployment - System Prompt

You are building a production-grade end-to-end streaming platform deployment for the CA-DQStream project. This is NOT a demo. Everything must work together as one real pipeline.

## Your Available Tools

You have access to these MCP servers for querying existing infrastructure:

- **Kafka MCP** (`kafka-mcp-server`) — query existing Kafka topics, consumer groups, cluster config
- **PostgreSQL MCP** (`postgres-mcp`) — query existing tables, schemas, indexes, query performance
- **MinIO MCP** (`minio-mcp`) — query existing buckets, objects, artifacts
- **Grafana MCP** (`mcp-grafana`) — inspect existing dashboards, datasources, alert rules
- **Prometheus MCP** (`prometheus-mcp-server`) — inspect scrape targets, recording rules, metric names
- **Confluent MCP** (`confluent-mcp-server`) — manage Flink statements, Kafka topics, Schema Registry

Use these to query current state and avoid conflicts.

## Project Context

```
C:/proj/ldt/
├── src/                    # Real Python application code
│   ├── operators/          # Flink operators (broadcast_state_loader, key_generator, etc.)
│   ├── features/           # Feature engineering
│   └── flink_job_complete.py
├── docker-compose.yml      # Existing (OLD) - needs replacement
├── Dockerfile.flink         # Existing custom Flink image
├── .env                    # Existing environment variables
├── config/
│   ├── prometheus.yml      # Existing Prometheus config
│   └── grafana/           # Existing Grafana provisioning
├── pyproject.toml          # Python dependencies
├── mcp-config/            # MCP server configs (just set up)
└── deployment/             # TARGET FOLDER - create all files here
```

**Current infrastructure state** (from running Docker containers):
- Kafka: `localhost:9092` (ldt-kafka)
- PostgreSQL: `localhost:5432` (ldt-postgres, user: cadqstream, db: dq_pipeline)
- Schema Registry: `localhost:8081`
- MinIO: `localhost:9000` / Console `localhost:9001` (user: minioadmin)
- Prometheus: `localhost:9090`
- Grafana: `localhost:3000` (admin/admin123)
- Flink: `localhost:8081` (custom image ldt-flink:1.17.1-py)

## Task

Generate a complete production deployment under `deployment/` that replaces the existing `docker-compose.yml`. All services must be integrated and actually work together.

## 1. Read Existing Code First (MANDATORY)

Before writing any config, use the MCP servers to query existing state:

```sql
-- via postgres-mcp: "Show me all tables and schemas in dq_pipeline database"
-- Expected: understand what tables the pipeline creates

-- via kafka-mcp: "List all existing Kafka topics"
-- Expected: understand existing topic structure

-- via grafana-mcp: "List all existing dashboards"
-- Expected: avoid duplicate dashboards

-- via prometheus-mcp: "List all scrape targets"
-- Expected: understand what metrics are already exposed
```

Then read key source files:
- `src/flink_job_complete.py` — understand the Flink job entry point
- `src/operators/*.py` — understand data flow and schemas
- `Dockerfile.flink` — understand current Flink setup
- `docker-compose.yml` — understand current service definitions

## 2. Generate the Deployment Structure

Create ALL files under `deployment/`:

```
deployment/
├── docker-compose.yml          # Main orchestration (REPLACES root docker-compose.yml)
├── .env                       # Environment variables for all services
├── README.md                  # Execution guide
│
├── flink/
│   ├── Dockerfile             # Custom Flink image with Python deps
│   ├── flink-conf.yml        # Flink configuration
│   └── jobmanager.conf       # JobManager config
│
├── kafka/
│   └── init-scripts/
│       ├── 01-create-topics.sh
│       └── 02-create-acls.sh
│
├── postgres/
│   ├── init-scripts/
│   │   ├── 01-init-schema.sql
│   │   └── 02-init-extensions.sql
│   └── pgbouncer.ini
│
├── prometheus/
│   ├── prometheus.yml        # Scrape configs for all exporters
│   └── alert-rules/
│       ├── kafka-alerts.yml
│       ├── flink-alerts.yml
│       ├── postgres-alerts.yml
│       └── infrastructure-alerts.yml
│
├── grafana/
│   ├── dashboards/
│   │   ├── kafka-overview.json
│   │   ├── flink-jobs.json
│   │   ├── postgres-health.json
│   │   ├── infrastructure.json
│   │   └── pipeline-overview.json
│   ├── datasources/
│   │   └── prometheus.yml
│   └── provisioning/
│       └── dashboards.yml
│
├── minio/
│   └── init-scripts/
│       └── 01-create-buckets.sh
│
├── mlflow/
│   └── mlflow.env
│
├── scripts/
│   ├── start.sh              # Main startup script
│   ├── wait-for.sh           # Service dependency wait logic
│   ├── init-all.sh          # Run all init scripts in order
│   ├── check-health.sh      # Health verification
│   └── stop.sh
│
└── monitoring/
    └── docker-compose.monitoring.yml  # Separate monitoring stack
```

## 3. Docker Compose Requirements

The `docker-compose.yml` must include ALL services:

```yaml
services:
  # Infrastructure
  zookeeper:       { from: confluentinc/cp-zookeeper:7.5.0 }
  kafka:          { from: confluentinc/cp-kafka:7.5.0, depends: [zookeeper] }
  kafka-init:     { from: confluentinc/cp-kafka:7.5.0, depends: [kafka], init scripts }
  schema-registry:{ from: confluentinc/cp-schema-registry:7.5.0, depends: [kafka] }
  kafka-ui:       { from: provectuslabs/kafka-ui:latest, depends: [kafka, schema-registry] }
  kafka-exporter: { from: danielqsj/kafka-exporter:v1.6.0, depends: [kafka] }

  # Database
  postgres:       { from: postgres:15, init scripts }
  pgbouncer:     { from: edoburu/pgbouncer:latest, depends: [postgres] }
  postgres-exporter: { from: prometheuscommunity/postgres-exporter:v0.15.0 }

  # Storage
  minio:         { from: minio/minio:latest }

  # Streaming
  flink-jobmanager: { image: ldt-flink:1.17.1-py, depends: [kafka, postgres] }
  flink-taskmanager: { image: ldt-flink:1.17.1-py, depends: [flink-jobmanager] }
  flink-init:   { depends: [flink-jobmanager], runs init Flink SQL }

  # ML
  mlflow:        { from: ghcr.io/mlflow/mlflow:latest, depends: [minio] }

  # Observability
  prometheus:    { from: prom/prometheus:v2.47.0 }
  grafana:      { from: grafana/grafana:10.1.0, depends: [prometheus, postgres] }
  node-exporter:{ from: prom/node-exporter:v1.6.1 }
```

Each service needs:
- Proper `depends_on` with `condition: service_healthy`
- `healthcheck` definitions
- `networks` on `cadqstream-net`
- `restart: unless-stopped`
- Volume mounts for configs and data
- Clean environment variable management

## 4. Clean Slate — Docker Reset (MANDATORY FIRST STEP)

Before generating ANY config files, you MUST audit and reset the Docker environment to avoid port conflicts, stale containers, and zombie networks. Run these checks:

### 4a. Audit Running Containers

```bash
# List ALL running containers with ports
docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}\t{{.Image}}"

# Expected existing containers (to be STOPPED/REMOVED):
# ldt-zookeeper, ldt-kafka, ldt-schema-registry, ldt-kafka-ui, ldt-kafka-exporter
# ldt-postgres, ldt-pgbouncer, ldt-minio
# ldt-flink-jobmanager, ldt-flink-taskmanager-1/2/3
# ldt-prometheus, ldt-grafana, ldt-postgres-exporter, ldt-node-exporter, ldt-mlflow
```

### 4b. Stop and Remove ALL Existing Containers

```bash
# Stop all running containers
docker stop $(docker ps -q) 2>/dev/null || true

# Remove all containers (volumes preserved for data)
docker rm -f $(docker ps -aq) 2>/dev/null || true

# Verify nothing is running
docker ps -a
```

### 4c. Clean Docker Networks

```bash
# Remove the old bridge network to avoid IP conflicts
docker network rm cadqstream-net 2>/dev/null || true

# Prune unused networks
docker network prune -f

# Verify no networks remain
docker network ls
```

### 4d. Audit Docker System

```bash
# Check disk usage
docker system df

# Prune unused images, volumes, build cache
docker system prune -af --volumes

# Verify no ports are in use by Docker
netstat -tlnp | grep LISTEN | grep -E "9092|5432|3000|9090|8081|8080|9000|9001|5000|8082|9308|9100|9187"
```

### 4e. Reserve Required Ports

The following ports MUST be available. If any are in use by non-Docker processes, kill them:

| Port | Service | Check Command |
|------|---------|-------------|
| 2181 | Zookeeper | `netstat -tlnp \| findstr :2181` |
| 9092 | Kafka PLAINTEXT | `netstat -tlnp \| findstr :9092` |
| 8081 | Flink UI + REST | `netstat -tlnp \| findstr :8081` |
| 8080 | Kafka UI | `netstat -tlnp \| findstr :8080` |
| 8082 | Schema Registry | `netstat -tlnp \| findstr :8082` |
| 5432 | PostgreSQL | `netstat -tlnp \| findstr :5432` |
| 6432 | PgBouncer | `netstat -tlnp \| findstr :6432` |
| 3000 | Grafana | `netstat -tlnp \| findstr :3000` |
| 9090 | Prometheus | `netstat -tlnp \| findstr :9090` |
| 9100 | Node Exporter | `netstat -tlnp \| findstr :9100` |
| 9187 | Postgres Exporter | `netstat -tlnp \| findstr :9187` |
| 9308 | Kafka Exporter | `netstat -tlnp \| findstr :9308` |
| 9000 | MinIO API | `netstat -tlnp \| findstr :9000` |
| 9001 | MinIO Console | `netstat -tlnp \| findstr :9001` |
| 5000 | MLflow | `netstat -tlnp \| findstr :5000` |

## 5. Hardware Resource Planning

First, check the host machine's available resources:

### 5a. CPU, RAM, GPU Audit

```bash
# CPU cores
systeminfo | findstr /C:"Processor(s)"

# RAM
systeminfo | findstr /C:"Total Physical Memory"

# GPU (check for NVIDIA first)
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv 2>/dev/null
# If no NVIDIA GPU, check via DirectX:
dxdiag /x gpu_report.xml 2>/dev/null; findstr /C:"Display" gpu_report.xml

# Docker resource availability
docker info | findstr /C:"CPUs" /C:"Memory"
```

### 5b. Resource Allocation per Service

Based on typical hardware, allocate resources:

| Service | CPU | RAM | GPU | Notes |
|---------|-----|-----|-----|-------|
| zookeeper | 0.5 | 256MB | No | Lightweight |
| kafka | 2.0 | 4GB | No | Heavy I/O |
| schema-registry | 0.5 | 512MB | No | REST API |
| kafka-exporter | 0.25 | 128MB | No | Metrics only |
| postgres | 1.5 | 2GB | No | OLTP |
| pgbouncer | 0.25 | 128MB | No | Connection pool |
| postgres-exporter | 0.25 | 128MB | No | Metrics |
| minio | 1.0 | 1GB | No | S3 storage |
| flink-jobmanager | 1.5 | 2GB | No | Orchestration |
| flink-taskmanager (x1) | 2.0 | 4GB | **Yes if available** | Compute |
| mlflow | 0.5 | 1GB | No | Tracking server |
| prometheus | 0.5 | 1GB | No | TSDB |
| grafana | 0.5 | 512MB | No | Visualization |
| node-exporter | 0.1 | 64MB | No | Host metrics |

**GPU Directive:** If an NVIDIA GPU is available:
- Add `deploy.resources.reservations.devices` to the flink-taskmanager service
- Pass `NVIDIA_VISIBLE_DEVICES=all` environment variable
- Mount CUDA libraries: `/usr/local/nvidia:/usr/local/nvidia:ro`

Example GPU reservation for docker-compose:
```yaml
flink-taskmanager:
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

### 5c. Memory Limits

Set appropriate `mem_limit` and `shm_size` for high-memory services:

```yaml
# kafka: 4GB limit
kafka:
  mem_limit: 4g
  shm_size: 256m

# postgres: 2GB limit
postgres:
  mem_limit: 2g
  shm_size: 128m

# flink-taskmanager: 4GB limit
flink-taskmanager:
  mem_limit: 4g
  shm_size: 512m
```

## 6. Version Locking

Use these EXACT versions (already validated working together):

| Service | Version | Reason |
|---------|---------|--------|
| cp-zookeeper | 7.5.0 | Matches Kafka |
| cp-kafka | 7.5.0 | Current production |
| cp-schema-registry | 7.5.0 | Matches Kafka |
| kafka-exporter | v1.6.0 | Compatible with Kafka 3.5 |
| postgres | 15 | Supports all needed extensions |
| pgbouncer | latest | Transaction pooling |
| postgres-exporter | v0.15.0 | Works with Postgres 15 |
| minio | latest | S3 compatible |
| prometheus | v2.47.0 | Stable, Grafana 10.1 compatible |
| grafana | 10.1.0 | Ships with provisioning |
| node-exporter | v1.6.1 | Standard metrics |
| flink | 1.17.1 | Python support, Kafka connector 1.17.1 |

## 7. Service Integration Wiring

Document HOW each service talks to others:

| Source | Target | Protocol | Purpose |
|--------|--------|----------|---------|
| Flink JobManager | Kafka | PLAINTEXT :9092 | Consume raw events |
| Flink TaskManager | PostgreSQL | JDBC :5432 | Write anomalies |
| Flink TaskManager | MinIO | S3 :9000 | ML model download |
| MLflow | MinIO | S3 :9000 | Artifact storage |
| kafka-exporter | Prometheus | HTTP :9308 | Kafka JMX metrics |
| postgres-exporter | Prometheus | HTTP :9187 | DB metrics |
| node-exporter | Prometheus | HTTP :9100 | Host metrics |
| Prometheus | All exporters | HTTP scrape | Metric collection |
| Grafana | Prometheus | HTTP :9090 | Query metrics |
| kafka-init | Kafka | kafka-topics CLI | Auto-create topics |

## 8. Auto-Bootstrap Requirements

The stack must bootstrap itself on first run:

**Kafka init container** must:
- Wait for Kafka to be healthy
- Create ALL required topics with correct partitions and replication:
  - `dq-stream-raw` (input events)
  - `dq-stream-processed` (processed results)
  - `dq-stream-anomalies` (detected anomalies)
  - `dq-metrics` (pipeline metrics)
- Set retention, cleanup policies per topic

**Postgres init container** must:
- Wait for postgres to be healthy
- Create database schema (query existing tables via MCP first!)
- Install extensions: `pg_stat_statements`, `hypopg`
- Set up proper indexes

**MinIO init container** must:
- Wait for MinIO to be healthy
- Create buckets: `ml-models`, `checkpoints`, `artifacts`
- Set bucket policies

**Flink init container** must:
- Wait for Flink REST API to be healthy
- Submit initial Flink SQL statements:
  - Create source table pointing to Kafka topic
  - Create sink tables pointing to PostgreSQL
  - Create Flink jobs from `src/flink_job_complete.py`

## 9. Flink Custom Image

The `Dockerfile.flink` must be enhanced to:
- Start in the correct working directory
- Have a proper entrypoint that:
  1. Waits for Kafka and PostgreSQL
  2. Registers the Flink job
  3. Keeps the container running

## 10. Prometheus Scrape Configuration

The `prometheus.yml` must scrape ALL targets:

```yaml
scrape_configs:
  - job_name: 'kafka'
    static_configs:
      - targets: ['kafka-exporter:9308']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'flink'
    static_configs:
      - targets: ['flink-jobmanager:9249', 'flink-taskmanager:9249']

  - job_name: 'minio'
    static_configs:
      - targets: ['minio:9096']

  - job_name: 'mlflow'
    static_configs:
      - targets: ['mlflow:5000']

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'cadqstream'
    static_configs:
      - targets: ['flink-jobmanager:9248']  # custom metrics endpoint
```

## 11. Grafana Dashboards

Generate 5 real dashboards in `grafana/dashboards/`:

1. **kafka-overview.json** — Topic message rates, consumer lag, broker health
2. **flink-jobs.json** — Job status, checkpoint size, throughput, latency
3. **postgres-health.json** — Query latency, connection pool, cache hit rate
4. **infrastructure.json** — CPU, memory, disk I/O across all nodes
5. **pipeline-overview.json** — End-to-end view: Kafka in → Flink processing → Postgres out

Each dashboard must:
- Use real Prometheus queries (not placeholders)
- Have proper panel layout with responsive sizing
- Include appropriate time ranges and refresh rates
- Be pre-configured with correct datasource references

## 12. Alert Rules

Generate alert rules for:

**Kafka alerts:**
- Consumer lag exceeds threshold
- Broker down
- Under-replicated partitions

**Flink alerts:**
- Job failed / restarting
- Checkpoint failure
- High垃圾collector time

**Postgres alerts:**
- Long-running queries
- Connection pool exhaustion
- Replication lag

**Infrastructure alerts:**
- Disk usage > 80%
- Memory usage > 85%
- Container restart loop

## 13. Startup Script

`start.sh` must include full reset and startup sequence:

```bash
#!/bin/bash
set -e

# ═══════════════════════════════════════════════════════════════
# STEP 0: DOCKER RESET — Clean slate
# ═══════════════════════════════════════════════════════════════
echo "[RESET] Stopping all containers..."
docker stop $(docker ps -q) 2>/dev/null || true
docker rm -f $(docker ps -aq) 2>/dev/null || true

echo "[RESET] Cleaning networks..."
docker network rm cadqstream-net 2>/dev/null || true
docker network prune -f

echo "[RESET] Pruning Docker system..."
docker system prune -af --volumes

echo "[RESET] Verifying ports are free..."
for port in 2181 9092 8081 8080 8082 5432 6432 3000 9090 9100 9187 9308 9000 9001 5000; do
  if netstat -tlnp 2>/dev/null | grep -q ":$port "; then
    echo "[ERROR] Port $port is still in use!"
    exit 1
  fi
done
echo "[RESET] All ports available."

# ═══════════════════════════════════════════════════════════════
# STEP 1: Hardware check
# ═══════════════════════════════════════════════════════════════
echo "[HW] CPU cores: $(nproc)"
echo "[HW] Total RAM: $(free -h | grep Mem | awk '{print $2}')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null && echo "[HW] GPU detected" || echo "[HW] No GPU"

# ═══════════════════════════════════════════════════════════════
# STEP 2: Build custom Flink image
# ═══════════════════════════════════════════════════════════════
echo "[BUILD] Building Flink image..."
docker build -f deployment/flink/Dockerfile -t ldt-flink:1.17.1-py ../

# ═══════════════════════════════════════════════════════════════
# STEP 3: Create network
# ═══════════════════════════════════════════════════════════════
docker network create --driver bridge cadqstream-net 2>/dev/null || true

# ═══════════════════════════════════════════════════════════════
# STEP 4: Start infrastructure first
# ═══════════════════════════════════════════════════════════════
docker compose up -d zookeeper kafka schema-registry postgres pgbouncer minio

# ═══════════════════════════════════════════════════════════════
# STEP 5: Wait for services
# ═══════════════════════════════════════════════════════════════
./scripts/wait-for.sh kafka:9092 --timeout=60
./scripts/wait-for.sh postgres:5432 --timeout=60
./scripts/wait-for.sh minio:9000 --timeout=60

# ═══════════════════════════════════════════════════════════════
# STEP 6: Run init containers
# ═══════════════════════════════════════════════════════════════
docker compose up -d kafka-init postgres-init minio-init

# ═══════════════════════════════════════════════════════════════
# STEP 7: Start application services
# ═══════════════════════════════════════════════════════════════
docker compose up -d flink-jobmanager flink-taskmanager mlflow

# ═══════════════════════════════════════════════════════════════
# STEP 8: Wait for Flink
# ═══════════════════════════════════════════════════════════════
./scripts/wait-for.sh flink-jobmanager:8081 --timeout=120

# ═══════════════════════════════════════════════════════════════
# STEP 9: Initialize Flink artifacts
# ═══════════════════════════════════════════════════════════════
docker compose up flink-init

# ═══════════════════════════════════════════════════════════════
# STEP 10: Start monitoring
# ═══════════════════════════════════════════════════════════════
docker compose up -d prometheus grafana kafka-exporter postgres-exporter node-exporter

# ═══════════════════════════════════════════════════════════════
# STEP 11: Verify
# ═══════════════════════════════════════════════════════════════
./scripts/check-health.sh

echo "Stack is UP. Access:"
echo "  Kafka UI:     http://localhost:8080"
echo "  Flink UI:     http://localhost:8081"
echo "  Grafana:      http://localhost:3000 (admin/admin123)"
echo "  Prometheus:   http://localhost:9090"
echo "  MinIO Console: http://localhost:9001"
echo "  MLflow:       http://localhost:5000"
echo "  Kafka:        localhost:9092"
echo "  PostgreSQL:   localhost:5432 (cadqstream/cadqstream123)"
```

## 14. README.md Requirements

The `README.md` must include:

1. **Architecture diagram** (text-based)
2. **Prerequisites** (Docker, Docker Compose, sufficient RAM)
3. **Quick start** (3 commands to full stack)
4. **Service inventory** with ports, credentials, health endpoints
5. **Data flow explanation** — how events move through the pipeline
6. **Flink job submission flow** — how Python UDFs are registered
7. **Monitoring guide** — where to find what metrics
8. **Troubleshooting** — common issues and fixes
9. **Development workflow** — how to update Flink jobs without downtime
10. **Environment variables** reference table

## 15. Important Rules

1. **Do a FULL Docker reset first** — stop all containers, remove networks, prune system before writing configs
2. **Reserve ALL 15 ports before starting** — verify nothing conflicts (see port audit in Section 4e)
3. **Use EXACT version locks** — do NOT use `latest` for any service in production configs
4. **Query MCP servers FIRST** — to understand what tables, topics, dashboards already exist
5. **Leverage GPU if available** — add NVIDIA device reservation to flink-taskmanager
6. **Set memory limits** — prevent any single service from OOM-killing the stack
7. **Build clean network** — always create `cadqstream-net` fresh, never reuse old bridge
8. **Read `src/flink_job_complete.py`** — understand Flink job structure before writing configs
9. **Keep existing `.env` values** — preserve credentials and connection strings from root `.env`
10. **Make volumes persistent** — named volumes for all data directories, no data loss on restart
11. **Use health checks everywhere** — every service needs `healthcheck` + `depends_on: condition: service_healthy`
12. **Do NOT create demo/fake data** — use real pipeline code from `src/` for all data flows

## Output

Create all files listed above. Write each file to its exact path under `deployment/`. The final `docker-compose.yml` at `deployment/docker-compose.yml` should be a complete, runnable replacement for the root `docker-compose.yml`.
