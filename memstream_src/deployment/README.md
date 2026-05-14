# CA-DQStream + MemStream Deployment

Enterprise-grade Docker Compose deployment for the CA-DQStream (Context-Aware Data Quality Stream) system with integrated MemStream ML anomaly detection.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CA-DQStream Architecture                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│  │   Kafka     │    │   Flink     │    │   MinIO     │                  │
│  │  Cluster    │───▶│  Cluster    │───▶│  Storage    │                  │
│  │  (6 comps)  │    │  (2 nodes)  │    │             │                  │
│  └─────────────┘    └─────────────┘    └─────────────┘                  │
│         │                 │                   │                        │
│         ▼                 ▼                   ▼                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│  │  Producer   │    │   Redis     │    │  Grafana    │                  │
│  │  (Data In)  │    │  (Cache)    │    │  Dashboard  │                  │
│  └─────────────┘    └─────────────┘    └─────────────┘                  │
│                           │                                               │
│                           ▼                                               │
│                    ┌─────────────┐    ┌─────────────┐                  │
│                    │  Prometheus │    │   ML        │                  │
│                    │  Monitoring │    │  Service    │                  │
│                    └─────────────┘    └─────────────┘                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Services (18 Total)

### Kafka Cluster (6 services)
| Service | Port | Description |
|---------|------|-------------|
| zookeeper | 2181 | Apache ZooKeeper for Kafka coordination |
| kafka | 9092, 29092 | Apache Kafka broker |
| schema-registry | 8082 | Confluent Schema Registry |
| kafka-ui | 8080 | Kafka UI management interface |
| kafka-exporter | 9308 | Prometheus metrics exporter for Kafka |
| kafka-init | - | Topic initialization job |

### Flink Cluster (2 services)
| Service | Port | Description |
|---------|------|-------------|
| flink-jobmanager | 8081, 9249 | Flink JobManager with REST API and metrics |
| flink-taskmanager | 9249 | Flink TaskManager for parallel processing |

### Storage (2 services)
| Service | Port | Description |
|---------|------|-------------|
| minio | 9000, 9001 | S3-compatible object storage with console |
| minio-init | - | Bucket initialization job |

### Cache (1 service)
| Service | Port | Description |
|---------|------|-------------|
| redis | 6379 | Redis cache for MemStream feature store |

### Observability (4 services)
| Service | Port | Description |
|---------|------|-------------|
| prometheus | 9090 | Metrics collection and storage |
| alertmanager | 9093 | Alert routing and notification |
| grafana | 3000 | Visualization dashboards |
| cadqstream-metrics | 9250 | Custom metrics exporter |

### Data Sources (1 service)
| Service | Port | Description |
|---------|------|-------------|
| kafka-producer | - | NYC Taxi data producer |

### ML Service (1 service)
| Service | Port | Description |
|---------|------|-------------|
| ml-service | 8000 | MemStream model serving |
| stats-writer | - | Statistics writer to MinIO |

## Quick Start

### Prerequisites
- Docker Engine 24.0+
- Docker Compose V2 (standalone or plugin)
- Minimum 8GB RAM
- 20GB disk space

### Setup

1. **Clone and navigate to deployment:**
```bash
cd memstream_src/deployment
```

2. **Copy environment file:**
```bash
cp .env.example .env
```

3. **Generate secure keys:**
```bash
# Generate MEMSTREAM_MODEL_SIGNING_KEY
openssl rand -hex 32

# Generate IEC_SIGNING_KEY
openssl rand -hex 32

# Generate REDIS_PASSWORD
openssl rand -hex 32

# Generate MINIO_SECRET_KEY
openssl rand -hex 32
```

4. **Prepare data directory:**
```bash
mkdir -p ../../data
# Place your parquet files in ../../data/
# Example: yellow_tripdata_2024-01.parquet
```

5. **Start the cluster:**
```bash
# Start all services
docker compose up -d

# Or with full logs
docker compose up -d --build
```

6. **Verify services:**
```bash
docker compose ps
```

## Kafka Topics

The following topics are automatically created:

| Topic | Partitions | Retention | Description |
|-------|------------|-----------|-------------|
| taxi-nyc-raw | 4 | 7 days | Raw NYC taxi data stream |
| dq-stream-anomalies | 4 | 7 days | Detected anomalies from MemStream |
| dq-meta-stream | 4 | 7 days | Data quality metadata |
| dq-hard-rule-violations | 4 | 7 days | Hard rule violations |
| dq-stream-processed | 4 | 7 days | Processed/validated data |
| iec-action-replay | 1 | 30 days | IEC action replay buffer |

## MinIO Buckets

| Bucket | Description |
|--------|-------------|
| cadqstream-raw | Raw input data archives |
| cadqstream-violations | Violation records |
| cadqstream-anomalies | Anomaly detection results |
| cadqstream-metrics | System metrics archives |
| cadqstream-drift | Data drift detection results |

## Accessing Services

| Service | URL | Credentials |
|---------|-----|-------------|
| Kafka UI | http://localhost:8080 | - |
| Flink Dashboard | http://localhost:8081 | - |
| MinIO Console | http://localhost:9001 | minioadmin / (MINIO_SECRET_KEY) |
| Grafana | http://localhost:3000 | admin / (GRAFANA_PASSWORD) |
| Prometheus | http://localhost:9090 | - |
| ML Service | http://localhost:8000 | - |

## Monitoring

### Grafana Dashboards
Pre-configured dashboards for:
- Pipeline Overview
- Data Quality Metrics
- Kafka Consumer Lag
- Flink Job Health
- Anomaly Detection Performance

### Alerting
Prometheus alerting rules configured for:
- Kafka broker down
- Flink job failures
- High consumer lag
- Redis connection issues
- MinIO bucket capacity

## Stopping the Cluster

```bash
# Stop all services
docker compose down

# Stop and remove volumes (data loss!)
docker compose down -v

# Stop and remove containers + images
docker compose down --rmi all
```

## Troubleshooting

### Kafka not starting
```bash
# Check ZooKeeper status
docker compose logs zookeeper

# Verify Kafka can connect to ZooKeeper
docker compose exec kafka kafka-broker-api-versions --bootstrap-server localhost:9092
```

### Flink job not submitting
```bash
# Check JobManager logs
docker compose logs flink-jobmanager

# Access Flink UI at http://localhost:8081
```

### MinIO bucket issues
```bash
# Verify buckets exist
docker compose exec minio mc ls local/

# Check MinIO logs
docker compose logs minio
```

### Redis connection issues
```bash
# Test Redis connection
docker compose exec redis redis-cli -a $REDIS_PASSWORD ping
```

## Development

### Rebuilding custom images
```bash
docker compose build cadqstream-metrics
docker compose build ml-service
docker compose build stats-writer
```

### Running specific services only
```bash
# Start only Kafka cluster
docker compose up -d zookeeper kafka kafka-init

# Start only observability stack
docker compose up -d prometheus grafana alertmanager
```

## File Structure

```
deployment/
├── docker-compose.yml          # Main compose file (18 services)
├── .env.example                # Environment template
├── README.md                   # This file
├── config/
│   ├── prometheus.yml          # Prometheus scrape config
│   ├── alert-rules/            # Prometheus alert rules
│   └── alertmanager.yml        # Alertmanager config
├── grafana/
│   └── provisioning/           # Grafana provisioning
├── operators/
│   └── deployment/
│       └── Dockerfile          # Flink operator image
├── kafka/
│   ├── producer.py             # Data producer script
│   └── init-scripts/          # Topic initialization
├── ml-service/
│   ├── Dockerfile
│   └── ml_service.py
├── cadqstream-metrics/
│   └── Dockerfile
└── stats-writer/
    ├── Dockerfile
    └── stats_writer.py
```

## License

Internal use only - CA-DQStream + MemStream Research Project
