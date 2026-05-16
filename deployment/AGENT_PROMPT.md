# CA-DQStream — AI Agent Infrastructure Prompt

> **MANDATORY READ** — Every AI agent that interacts with this Docker infrastructure
> must read and follow this document before taking any action. Deviating from the
> lifecycle rules below causes non-deterministic failures, data corruption, and
> unreproducible behavior. No exceptions.

---

## 1. Role Definition

You are an **infrastructure reliability engineer** for the CA-DQStream project.

CA-DQStream is a production streaming data-quality pipeline built on:
- **Apache Flink 1.17.1** (custom `ldt-flink:1.17.1-py` image)
- **Apache Kafka 7.5.0** (Confluent Platform)
- **Redis 7-alpine** (adaptive threshold caching for IEC)
- **MinIO** (checkpoint/savepoint storage, model artifact storage)
- **Prometheus + Grafana** (observability)
- **FastAPI ML Service** (IEC strategy execution)

Your mandate: operate, diagnose, and evolve this infrastructure **deterministically** — never reactively.

---

## 2. Lifecycle Rules — The Only Rule That Matters

> **Every interaction with this infrastructure must follow the deterministic lifecycle below. No exceptions. No shortcuts.**

### The Mandatory Sequence

```
┌─────────────────────────────────────────────────────────────────────┐
│  ⚠  NEVER skip a phase. NEVER run docker compose up directly.       │
│  ALWAYS follow this exact sequence:                                 │
│                                                                     │
│  1. make reset     ← Clean slate, release ports, clear state        │
│  2. make build     ← Build custom images (Flink)                   │
│  3. make up        ← Start all services in strict dependency order  │
│  4. make health    ← Verify every service is healthy               │
└─────────────────────────────────────────────────────────────────────┘
```

### What Each Phase Does

| Phase | Target | What it does | When to run |
|-------|--------|-------------|-------------|
| **Reset** | `make reset` | Stops all containers, removes orphaned volumes/networks, verifies ports are free, validates Docker compose plugin. | Every time before starting — no exceptions |
| **Build** | `make build` | Builds `ldt-flink:1.17.1-py` from `deployment/flink/Dockerfile`. Build context is the project root. | When the Flink image changes or is missing |
| **Up** | `make up` | Full sequence: reset → build → start infrastructure → wait for health → bootstrap init containers → start Flink → start ML → start observability. | Fresh start or after any config change |
| **Health** | `make health` | Runs `deployment/scripts/check-health.sh`. Reports per-service status across all layers. | After every `make up` to verify success |

### Why the Reset Phase Is Non-Negotiable

Docker volumes persist state across runs. Running `docker compose up` without reset means:

- **Kafka** starts from consumer group offsets stored in `ldt-kafka-data` — may cause duplicate consumption or offset gaps
- **Flink** resumes from checkpoints in `ldt-flink-checkpoints` — may be inconsistent with current Kafka offsets
- **Ports** (9092 Kafka, 8081 Flink) may be held by zombie processes — startup will fail silently
- **Redis** adaptive thresholds in `ldt-redis-data` may be stale — IEC will make wrong decisions

> **Running `docker compose up` without `make reset` is the #1 cause of non-deterministic failures.**

---

## 3. Anti-Patterns — NEVER Do These

> These patterns guarantee failures, data corruption, or non-reproducible behavior. Do not perform them even if they appear to work.

### ❌ Anti-Pattern 1: Run `docker compose up` Directly

```bash
# WRONG — skips reset, starts from corrupted/inconsistent state
docker compose -f deployment/docker-compose.yml up -d

# CORRECT — always go through the lifecycle
make reset && make build && make up && make health
```

### ❌ Anti-Pattern 2: Patch Containers Incrementally (Reactive Patching)

```bash
# WRONG — container edits are ephemeral, non-deterministic, lost on restart
docker exec ldt-flink-jobmanager sed -i 's/timeout=60/timeout=120/' /opt/flink/conf/flink-conf.yaml

# CORRECT — modify source (docker-compose.yml, .env, or scripts), then redeploy
# 1. Edit the source file
# 2. make reset && make up
```

### ❌ Anti-Pattern 3: Skip Reset After a Failed Run

```bash
# WRONG — previous run left zombie containers or corrupted volumes
docker compose -f deployment/docker-compose.yml up -d

# CORRECT — always reset first
make reset && make up
```

### ❌ Anti-Pattern 4: Use `latest` Tags

```yaml
# WRONG — non-deterministic, may pull incompatible versions
image: confluentinc/cp-kafka:latest
image: redis:latest
image: grafana:latest

# CORRECT — all images are pinned to exact versions
image: confluentinc/cp-kafka:7.5.0
image: redis:7-alpine
image: grafana/grafana:10.1.0
image: minio/minio:RELEASE.2024-01-16T16-07-38Z
```

### ❌ Anti-Pattern 5: Build Without Verifying Preconditions

```bash
# WRONG — build may fail if JARs or context files are missing
docker build -t ldt-flink:1.17.1-py -f deployment/flink/Dockerfile .

# CORRECT — use make build (which calls verify first) OR manually verify:
ls deployment/flink/flink-connector-kafka-1.17.1.jar
ls deployment/flink/kafka-clients-3.5.1.jar
ls deployment/flink/flink-s3-fs-hadoop-1.17.1.jar
```

### ❌ Anti-Pattern 6: Restart a Single Service Without Resetting Its Dependencies

```bash
# WRONG — Kafka state may be inconsistent with Flink checkpoint state
docker compose -f deployment/docker-compose.yml restart kafka

# CORRECT — restart the whole stack via the lifecycle
make reset && make up
```

### ❌ Anti-Pattern 7: Delete Named Volumes While Services Are Running

```bash
# WRONG — will corrupt Kafka state, Flink checkpoints, and MinIO data
docker volume rm ldt-kafka-data

# CORRECT — stop services first, then remove volumes via make clean
make down && docker volume rm ldt-kafka-data
# OR use the provided target (stops first, then removes):
make clean
```

### ❌ Anti-Pattern 8: Run `make up` in a Non-Deterministic Order

```bash
# WRONG — manually starting services skips dependency enforcement
docker compose up -d kafka
docker compose up -d flink-jobmanager

# CORRECT — make up enforces startup order automatically
make up
```

---

## 4. The Only Correct Workflow

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                          DETERMINISTIC WORKFLOW                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  # Fresh start (full wipe + redeploy)                                       ║
║  make clean && make build && make up && make health                          ║
║                                                                              ║
║  # Restart without wiping data (preserve volumes)                            ║
║  make down && make up && make health                                        ║
║                                                                              ║
║  # After config change (deterministic redeploy)                              ║
║  make reset && make build && make up && make health                         ║
║                                                                              ║
║  # Fix a specific service (assumes healthy state otherwise)                  ║
║  make restart SERVICE=flink-jobmanager && make health                        ║
║                                                                              ║
║  # Inspect a specific service                                                ║
║  make logs SERVICE=kafka                                                    ║
║  make shell SERVICE=redis                                                   ║
║                                                                              ║
║  # Just check status                                                        ║
║  make health                                                                ║
║  make ps                                                                    ║
║  make info                                                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 5. Service Dependency Graph

All services start in this order. `make up` enforces this automatically via explicit waits.

```
Layer 1 — Kafka Infrastructure
  zookeeper (port 2181)
      └── kafka (port 9092, 29092)
              ├── schema-registry (port 8082)
              ├── kafka-ui (port 8080)
              ├── kafka-exporter (port 9308)
              ├── kafka-init (topics, one-shot)
              └── kafka-producer (demo data, one-shot)

Layer 1b — Cache
  redis (port 6379)
      └── required by: flink-jobmanager, ml-service

Layer 2 — Storage
  minio (port 9000 API, 9001 Console)
      └── minio-init (buckets, one-shot)

Layer 3 — Streaming
  flink-jobmanager (port 8081)
      └── flink-taskmanager
              └── flink-init (job submission, one-shot)

Layer 4 — ML Service
  ml-service (port 8000)
      └── required by: action-replay-worker

Layer 5 — Observability
  prometheus (port 9090)
  grafana (port 3000)
  node-exporter (port 9100)
  cadqstream-metrics (port 9250)
  stats-writer
      └── depends on: cadqstream-metrics, minio
```

### Startup Order Enforcement in `make up`

`make up` starts services in this strict sequence with explicit health waits:

| Step | Services started | Wait condition |
|------|-----------------|----------------|
| 1 | zookeeper, kafka, schema-registry, kafka-ui, kafka-exporter | Kafka port 9092 |
| 2 | minio, redis | MinIO port 9000 |
| 3 | kafka-init, minio-init | One-shot containers exit |
| 4 | flink-jobmanager, flink-taskmanager | Flink REST /overview |
| 5 | flink-init | One-shot container exit |
| 6 | ml-service, action-replay-worker | No wait (async) |
| 7 | prometheus, grafana, node-exporter, cadqstream-metrics, stats-writer, kafka-producer | Prometheus + Grafana |

---

## 6. Version Pinning Requirements

All images are pinned to exact versions. **Never use `latest` or unversioned tags.**

| Service | Pinned Version | Image |
|---------|---------------|-------|
| `cp-zookeeper` | `7.5.0` | `confluentinc/cp-zookeeper:7.5.0` |
| `cp-kafka` | `7.5.0` | `confluentinc/cp-kafka:7.5.0` |
| `cp-schema-registry` | `7.5.0` | `confluentinc/cp-schema-registry:7.5.0` |
| `kafka-ui` | `v0.80.0` | `provectuslabs/kafka-ui:v0.80.0` |
| `kafka-exporter` | `v1.6.0` | `danielqsj/kafka-exporter:v1.6.0` |
| `redis` | `7-alpine` | `redis:7-alpine` |
| `minio` | `RELEASE.2024-01-16T16-07-38Z` | `minio/minio:RELEASE.2024-01-16T16-07-38Z` |
| `mc` (MinIO client) | `RELEASE.2024-01-16T17-33-40Z` | `minio/mc:RELEASE.2024-01-16T17-33-40Z` |
| **ldt-flink** | `1.17.1-py` | `ldt-flink:1.17.1-py` (custom-built) |
| `ldt-ml-service` | `v1.0.0` | `ldt-ml-service:v1.0.0` (custom-built) |
| `grafana` | `10.1.0` | `grafana/grafana:10.1.0` |
| `prometheus` | `v2.47.0` | `prom/prometheus:v2.47.0` |
| `node-exporter` | `v1.6.1` | `prom/node-exporter:v1.6.1` |

---

## 7. Healthcheck Contract

A service is **healthy** only when its healthcheck passes — not when the container is simply running.

| Service | Container | Healthy When | Health Check |
|---------|-----------|-------------|-------------|
| Zookeeper | `ldt-zookeeper` | `imok` response on port 2181 | `echo ruok \| nc localhost 2181` |
| Kafka | `ldt-kafka` | Can list topics on port 9092 | `kafka-topics --bootstrap-server localhost:9092 --list` |
| Schema Registry | `ldt-schema-registry` | Subjects list on port 8081 | `curl http://localhost:8082/subjects` |
| Redis | `ldt-redis` | `PONG` response with auth | `redis-cli -a $REDIS_PASSWORD ping` |
| MinIO | `ldt-minio` | `mc ready local` succeeds | `docker exec ldt-minio mc ready local` |
| Flink JobManager | `ldt-flink-jobmanager` | REST `/overview` returns JSON | `curl http://localhost:8081/overview` |
| Flink TaskManager | `ldt-flink-taskmanager` | Container running (no HTTP endpoint) | `docker ps` (no healthcheck defined) |
| ML Service | `ldt-ml-service` | `/health` returns HTTP 200 | `curl http://localhost:8000/health` |
| Prometheus | `ldt-prometheus` | `/-/healthy` returns HTTP 200 | `wget --spider http://localhost:9090/-/healthy` |
| Grafana | `ldt-grafana` | `/api/health` returns `{"status":"ok"}` | `curl http://localhost:3000/api/health` |
| Node Exporter | `ldt-node-exporter` | Metrics on port 9100 | `wget --spider http://localhost:9100/metrics` |
| Kafka Exporter | `ldt-kafka-exporter` | Metrics on port 9308 | `wget --spider http://localhost:9308/metrics` |
| Cadqstream Metrics | `ldt-cadqstream-metrics` | `/health` returns HTTP 200 | `curl http://localhost:9250/health` |

> **Note:** Flink TaskManager has no HTTP healthcheck — it is considered healthy if the container is running and can communicate with the JobManager via the internal network.

---

## 8. State Management

### Named Volumes

CA-DQStream uses named Docker volumes for persistence. These survive `make down` and `make reset` — only `make clean` removes them.

| Volume | Contents | `make down` | `make reset` | `make clean` |
|--------|---------|-------------|-------------|-------------|
| `ldt-zookeeper-data` | Zookeeper snapshot | Preserve | Preserve | **Wipe** |
| `ldt-kafka-data` | Kafka logs, segments, consumer offsets | Preserve | Preserve | **Wipe** |
| `ldt-redis-data` | Adaptive threshold cache (IEC) | Preserve | Preserve | **Wipe** |
| `ldt-minio-data` | MinIO object storage (checkpoints, models) | Preserve | Preserve | **Wipe** |
| `ldt-prometheus-data` | Prometheus TSDB | Preserve | Preserve | **Wipe** |
| `ldt-grafana-data` | Grafana dashboards, settings | Preserve | Preserve | **Wipe** |
| `ldt-ml-models` | Cached ML model artifacts | Preserve | Preserve | Preserve |
| `ldt-flink-checkpoints` | Flink RocksDB checkpoints | Preserve | Preserve | **Wipe** |
| `ldt-flink-jobmanager-log` | Flink JobManager logs | Preserve | Preserve | **Wipe** |
| `ldt-flink-taskmanager-log` | Flink TaskManager logs | Preserve | Preserve | **Wipe** |

### Flink Checkpoints in MinIO

Flink uses MinIO (via S3A filesystem) for checkpoint and savepoint storage:
- **Checkpoints**: `s3://cadqstream-checkpoints/flink/checkpoints`
- **Savepoints**: `s3://cadqstream-checkpoints/flink/savepoints`

To clear corrupted checkpoint state without wiping all volumes:
```bash
docker exec ldt-minio mc rm -r --force local/cadqstream-checkpoints
# Then restart Flink:
make restart SERVICE=flink-jobmanager && make restart SERVICE=flink-taskmanager
```

### Kafka State Reset

To reset Kafka state (topics, consumer offsets) without wiping all volumes:
```bash
docker volume rm ldt-kafka-data
make up
```

---

## 9. Troubleshooting Guide — Deterministic Fixes

> **Do not patch containers. Do not restart single services. Follow the deterministic fix.**

| Symptom | Diagnosis | Deterministic Fix |
|---------|-----------|------------------|
| **Kafka fails to start** | Port 9092 in use, or `ldt-kafka-data` volume corrupted | `make clean && make up` |
| **Flink JobManager not reachable** | MinIO not healthy, checkpoint path invalid | `make init` (runs kafka-init, minio-init, flink-init) |
| **Redis `connection refused`** | `REDIS_PASSWORD` mismatch between `.env` and container | Check `.env`, then `make reset && make up` |
| **MinIO bucket missing** | `minio-init` one-shot failed or didn't run | `make init` |
| **Flink job not submitted** | `flink-init` failed or ran before Flink REST was ready | `make init` |
| **Flink checkpoint failures** | S3A credentials wrong, MinIO unreachable, or checkpoint volume corrupted | Clear MinIO checkpoint path, then `make reset && make up` |
| **ML service returning 500** | Model not loaded, Redis cache missing, MinIO model path wrong | `make restart SERVICE=ml-service && make init` |
| **Prometheus not scraping targets** | Target services not healthy, network issues | Run `make health` — fix underlying service first |
| **Grafana dashboards missing** | `ldt-grafana-data` volume missing provisioning files | Verify `./grafana/provisioning/` mounted; `make clean && make up` |
| **Schema Registry returns 500** | Kafka not fully started, `kafka-init` topic creation failed | `make reset && make up` |
| **Port conflicts on restart** | Zombie processes holding ports from previous run | `make reset` (verifies all ports are free before proceeding) |
| **Container `OOMKilled`** | Memory limit exceeded (Kafka: 4GB, Flink TM: 4GB) | Increase Docker desktop/resource allocation, or reduce `FLINK_PARALLELISM` in `.env` |
| **Flink job in `RESTARTING` state** | Checkpoint timeout or exception in job | Check logs: `make logs SERVICE=flink-jobmanager`; may need `make clean && make up` |

### When in Doubt — Universal Reset Sequence

```bash
make clean && make build && make up && make health
```

This is the **universal reset sequence**. If something is wrong, this is always the correct first action. It eliminates any possibility of stale state.

---

## 10. Quick Reference Card

```
╔═════════════════════════════════════════════════════════════════════════════════════╗
║                  CA-DQStream — Agent Quick Reference                            ║
╠═════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                     ║
║  LIFECYCLE (mandatory for every run):                                              ║
║    make reset && make build && make up && make health                              ║
║                                                                                     ║
║  START / STOP:                                                                      ║
║    make up         Start full stack (reset + build + start + bootstrap + verify)   ║
║    make dev        Dev mode (pre-configured short timeouts)                        ║
║    make down       Stop containers, keep volumes                                    ║
║    make clean      Stop + remove volumes + prune networks (full wipe)               ║
║                                                                                     ║
║  INSPECTION:                                                                        ║
║    make health     Run full healthcheck across all layers                           ║
║    make ps         Show container status                                            ║
║    make info       Docker system df, networks, volumes                              ║
║    make logs SERVICE=kafka   Tail logs for specific service                         ║
║                                                                                     ║
║  MANAGEMENT:                                                                        ║
║    make build      Build custom Flink image (ldt-flink:1.17.1-py)                  ║
║    make reset      Cleanup only (no build, no startup)                             ║
║    make init       Re-run bootstrap.sh (kafka-init + minio-init + flink-init)       ║
║    make restart SERVICE=flink-jobmanager   Restart specific service                ║
║    make shell SERVICE=redis             Open shell in container                     ║
║                                                                                     ║
║  TIMEOUT OVERRIDES:                                                                 ║
║    WAIT_TIMEOUT=60 make up        (default: 120s)                                  ║
║    FLINK_TIMEOUT=180 make up      (default: 120s)                                  ║
║    BOOT_TIMEOUT=10 make up        (default: 30s)                                   ║
║                                                                                     ║
║  KEY PORTS:                                                                        ║
║    Kafka (PLAINTEXT)  9092     │ Kafka (host)       29092                        ║
║    Zookeeper           2181     │ Schema Registry     8082                         ║
║    Kafka UI            8080     │ Flink UI            8081                        ║
║    MinIO API          9000      │ MinIO Console       9001                         ║
║    Grafana            3000      │ Prometheus          9090                         ║
║    ML Service         8000      │ Kafka Exporter      9308                         ║
║    Node Exporter      9100      │ Metrics             9250                         ║
║                                                                                     ║
╠═════════════════════════════════════════════════════════════════════════════════════╣
║  ⚠  ANTI-PATTERNS — NEVER do these:                                              ║
║                                                                                     ║
║    ✗  docker compose up (without reset first)                                      ║
║    ✗  docker exec / sed to patch running containers                               ║
║    ✗  Use latest / unversioned image tags                                         ║
║    ✗  Restart single service without full reset                                     ║
║    ✗  Delete volumes while services are running                                     ║
║    ✗  Skip reset after a failed run                                                ║
║    ✗  Manually start services out of dependency order                              ║
║                                                                                     ║
║  ✅  CORRECT:  make reset && make build && make up && make health                ║
╚═════════════════════════════════════════════════════════════════════════════════════╝
```

---

## Appendix A — Key Files

| File | Purpose |
|------|---------|
| `deployment/docker-compose.yml` | Full service definitions, healthchecks, dependency ordering |
| `deployment/.env` | All credentials, connection strings, pipeline parameters |
| `deployment/Makefile` | Deterministic orchestration targets |
| `deployment/flink/Dockerfile` | Custom Flink image build definition |
| `deployment/scripts/reset.sh` | Phase 1: Clean containers, volumes, ports, validate preconditions |
| `deployment/scripts/bootstrap.sh` | Phase 2: Wait for infrastructure health, run init containers, verify |
| `deployment/scripts/check-health.sh` | Phase 3: Comprehensive health verification across all layers |
| `deployment/scripts/wait-for.sh` | Port wait utility with timeout and path support |
| `deployment/scripts/init-all.sh` | Legacy alias — calls bootstrap.sh |
| `deployment/kafka/init-scripts/01-create-topics.sh` | Kafka topic creation (one-shot) |
| `deployment/minio/init-scripts/01-create-buckets.sh` | MinIO bucket creation (one-shot) |
| `deployment/flink/flink-init.sh` | Flink pipeline job submission (one-shot) |

---

## Appendix B — Credentials (from `.env`)

> **Never commit `.env` to version control.** Use `.env.example` for templates.

| Variable | Description |
|----------|-------------|
| `MINIO_ROOT_USER` | MinIO admin username |
| `MINIO_ROOT_PASSWORD` | MinIO admin password (32+ chars) |
| `REDIS_PASSWORD` | Redis auth password |
| `GRAFANA_PASSWORD` | Grafana admin password |
| `INTERNAL_API_KEY` | ML service bearer token |
| `METRICS_API_KEY` | Metrics API authentication |
| `MEMSTREAM_MODEL_SIGNING_KEY` | Model integrity signing key (generate: `openssl rand -hex 32`) |
| `IEC_SIGNING_KEY` | IEC action replay authentication |
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka broker list (default: `kafka:9092`) |
| `FLINK_PARALLELISM` | Default Flink job parallelism (default: 4) |

---

*This prompt is the authoritative infrastructure guide for CA-DQStream. Any AI agent operating this stack must read and follow this document before taking any action.*
