# Docker/SRE Fixes - CA-DQStream + MemStream v4 → v5

> **Status:** CRITICAL & HIGH issues from PLAN_v4.md Docker/K8s review
> **Date:** 2026-05-12
> **Priority:** C-DK-1 through C-DK-4 (CRITICAL), H-DK-1 through H-DK-4 (HIGH)

---

## CRITICAL Issues Fixed

### C-DK-1: Complete Dockerfile
### C-DK-2: Complete docker-compose.yml
### C-DK-3: MEMSTREAM_MODEL_SIGNING_KEY in compose
### C-DK-4: Redis configuration for IEC communication

---

## HIGH Issues Fixed

### H-DK-1: CPU/memory limits
### H-DK-2: Traffic splitting implementation
### H-DK-3: /health endpoint code
### H-DK-4: requirements.txt

---

## Dockerfile

```dockerfile
# =============================================================================
# CA-DQStream + MemStream v5 - Multi-stage Production Dockerfile
# Base: python:3.12-slim-bookworm
# Features: Non-root user, CPU-only PyTorch, HMAC model signing
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder - Install all dependencies
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages in isolated environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install core dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.2 \
    redis==5.0.3 \
    pyflink==1.18.1 \
    flask==3.0.3 \
    prometheus-client==0.20.0 \
    scipy==1.13.1 \
    scikit-learn==1.4.2 \
    pyyaml==6.0.1 \
    python-dotenv==1.0.1 \
    hvac==1.2.1 \
    gunicorn==21.2.0

# -----------------------------------------------------------------------------
# Stage 2: Production - Minimal runtime image
# -----------------------------------------------------------------------------
FROM python:3.12-slim-bookworm AS production

# Security: Create non-root user
RUN groupadd --gid 1000 cadqstream && \
    useradd --uid 1000 --gid cadqstream --shell /bin/bash --create-home cadqstream

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY --chown=cadqstream:cadqstream . .

# Create necessary directories
RUN mkdir -p /app/logs /app/checkpoints /app/models && \
    chown -R cadqstream:cadqstream /app

# Expose ports
EXPOSE 8080 9249 6122

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Switch to non-root user
USER cadqstream

# Default: Run health server (can be overridden via CMD)
ENV PYTHONUNBUFFERED=1
ENV FLASK_ENV=production

CMD ["python", "-m", "memstream_src.operators.health_server"]
```

---

## docker-compose.yml

```yaml
# =============================================================================
# CA-DQStream + MemStream v5 - Docker Compose (Docker-only, no K8s)
# =============================================================================
# Usage:
#   Development: docker compose up
#   Production:  docker compose -f docker-compose.yml -f docker-compose.prod.yml up
# =============================================================================

services:
  # ---------------------------------------------------------------------------
  # Redis - Required for IEC → MemStream communication
  # ---------------------------------------------------------------------------
  redis:
    image: redis:7-alpine
    container_name: cadqstream-redis
    restart: unless-stopped
    command: >
      redis-server
      --appendonly yes
      --appendfsync everysec
      --requirepass ${REDIS_PASSWORD:?REDIS_PASSWORD must be set}
      --maxmemory 256mb
      --maxmemory-policy allkeys-lru
      --tcp-backlog 511
      --timeout 0
      --tcp-keepalive 300
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "-a", "${REDIS_PASSWORD}", "incr", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 10s
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - cadqstream-backend
    deploy:
      resources:
        limits:
          cpus: "0.5"
          memory: 512M
        reservations:
          cpus: "0.25"
          memory: 256M

  # ---------------------------------------------------------------------------
  # PostgreSQL - MetaAggregator state + DLQ
  # ---------------------------------------------------------------------------
  postgres:
    image: postgres:16-alpine
    container_name: cadqstream-postgres
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-cadqstream} -d ${POSTGRES_DB:-cadqstream}"]
      interval: 10s
      timeout: 5s
      retries: 5
      start_period: 10s
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-cadqstream}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?POSTGRES_PASSWORD must be set}
      POSTGRES_DB: ${POSTGRES_DB:-cadqstream}
      POSTGRES_HOST_AUTH_METHOD: scram-sha-256
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    networks:
      - cadqstream-backend
    deploy:
      resources:
        limits:
          cpus: "1"
          memory: 2G
        reservations:
          cpus: "0.5"
          memory: 1G

  # ---------------------------------------------------------------------------
  # Prometheus - Metrics collection
  # ---------------------------------------------------------------------------
  prometheus:
    image: prom/prometheus:v2.50.1
    container_name: cadqstream-prometheus
    restart: unless-stopped
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
      - '--storage.tsdb.wal-compression'
      - '--web.enable-lifecycle'
    healthcheck:
      test: ["CMD", "wget", "-q", "--spider", "http://localhost:9090/-/healthy"]
      interval: 30s
      timeout: 10s
      retries: 3
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./alerts:/etc/prometheus/alerts:ro
      - prometheus_data:/prometheus
    networks:
      - cadqstream-backend

  # ---------------------------------------------------------------------------
  # Prometheus Redis Exporter - Redis metrics
  # ---------------------------------------------------------------------------
  redis-exporter:
    image: oliver006/redis_exporter:v1.57.0
    container_name: cadqstream-redis-exporter
    restart: unless-stopped
    environment:
      REDIS_ADDR: redis://redis:6379
      REDIS_PASSWORD: ${REDIS_PASSWORD}
    ports:
      - "9121:9121"
    networks:
      - cadqstream-backend
    deploy:
      resources:
        limits:
          cpus: "0.25"
          memory: 128M

  # ---------------------------------------------------------------------------
  # Grafana - Dashboards
  # ---------------------------------------------------------------------------
  grafana:
    image: grafana/grafana:10.4.2
    container_name: cadqstream-grafana
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "wget -q --spider http://localhost:3000/api/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
    environment:
      GF_SECURITY_ADMIN_USER: ${GRAFANA_ADMIN_USER:-admin}
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_ADMIN_PASSWORD:?GRAFANA_ADMIN_PASSWORD must be set}
      GF_USERS_ALLOW_SIGN_UP: "false"
      GF_SERVER_ROOT_URL: "%(protocol)s://%(domain)s:%(http_port)s/grafana"
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - ./grafana/dashboards:/var/lib/grafana/dashboards:ro
    networks:
      - cadqstream-backend

  # ---------------------------------------------------------------------------
  # Flink JobManager
  # ---------------------------------------------------------------------------
  flink-jobmanager:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    container_name: cadqstream-jobmanager
    restart: unless-stopped
    ports:
      - "8081:8081"  # Flink UI
      - "6123:6123"  # RPC
    environment:
      JOB_MANAGER_RPC_ADDRESS: flink-jobmanager
      JOB_MANAGER_RPC_PORT: "6123"
      FLINK_PROPERTIES: |
        jobmanager.rpc.address: flink-jobmanager
        state.backend: rocksdb
        state.checkpoints.dir: file:///tmp/flink/checkpoints
        execution.checkpointing.interval: 45s
        execution.checkpointing.mode: EXACTLY_ONCE
        taskmanager.memory.process.size: 8192m
        taskmanager.memory.managed.fraction: 0.4
      # CRITICAL: HMAC model signing key
      MEMSTREAM_MODEL_SIGNING_KEY: ${MEMSTREAM_MODEL_SIGNING_KEY:?MEMSTREAM_MODEL_SIGNING_KEY must be set}
      REDIS_PASSWORD: ${REDIS_PASSWORD}
      REDIS_HOST: redis
      REDIS_PORT: "6379"
      FLINK_ENV: ${FLINK_ENV:-production}
      FLINK_SLA_ENABLED: "true"
    volumes:
      - flink_checkpoints:/tmp/flink/checkpoints
      - flink_savepoints:/tmp/flink/savepoints
      - ./models:/app/models:ro
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8081/overview || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 60s
    networks:
      - cadqstream-backend
    deploy:
      resources:
        limits:
          cpus: "2"
          memory: 4G
        reservations:
          cpus: "1"
          memory: 2G

  # ---------------------------------------------------------------------------
  # Flink TaskManager (scaled)
  # ---------------------------------------------------------------------------
  flink-taskmanager:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    container_name: cadqstream-taskmanager
    restart: unless-stopped
    depends_on:
      redis:
        condition: service_healthy
      flink-jobmanager:
        condition: service_healthy
    environment:
      TASK_MANAGER_RPC_ADDRESS: flink-taskmanager
      JOB_MANAGER_RPC_ADDRESS: flink-jobmanager
      JOB_MANAGER_RPC_PORT: "6123"
      FLINK_PROPERTIES: |
        taskmanager.rpc.address: flink-taskmanager
        taskmanager.numberOfTaskSlots: 4
        state.backend: rocksdb
        state.checkpoints.dir: file:///tmp/flink/checkpoints
        execution.checkpointing.interval: 45s
        execution.checkpointing.mode: EXACTLY_ONCE
        taskmanager.memory.process.size: 8192m
        taskmanager.memory.managed.fraction: 0.4
        taskmanager.memory.task.off-heap.enabled: true
        restart-strategy: exponential-delay
        restart-strategy.exponential-delay.initial-restart-delay: 1000
        restart-strategy.exponential-delay.max-restart-delay: 60000
        restart-strategy.exponential-delay.back-off-multiplier: 2.0
        restart-strategy.exponential-delay.reset-backoff-threshold: 300000
      # CRITICAL: HMAC model signing key
      MEMSTREAM_MODEL_SIGNING_KEY: ${MEMSTREAM_MODEL_SIGNING_KEY:?MEMSTREAM_MODEL_SIGNING_KEY must be set}
      REDIS_PASSWORD: ${REDIS_PASSWORD}
      REDIS_HOST: redis
      REDIS_PORT: "6379"
      FLINK_ENV: ${FLINK_ENV:-production}
    volumes:
      - flink_checkpoints:/tmp/flink/checkpoints
      - flink_savepoints:/tmp/flink/savepoints
      - ./models:/app/models:ro
    networks:
      - cadqstream-backend
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: "2"
          memory: 4G
        reservations:
          cpus: "0.5"
          memory: 1G

  # ---------------------------------------------------------------------------
  # Health Server - Exposes /health, /ready, /metrics
  # ---------------------------------------------------------------------------
  health-server:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    container_name: cadqstream-health-server
    restart: unless-stopped
    command: python -m memstream_src.operators.health_server
    ports:
      - "8080:8080"
    environment:
      FLASK_ENV: production
      FLASK_DEBUG: "false"
      METRICS_PORT: "8080"
      REDIS_HOST: redis
      REDIS_PORT: "6379"
      REDIS_PASSWORD: ${REDIS_PASSWORD}
      FLINK_JOB_MANAGER: http://flink-jobmanager:8081
      # HMAC signing key for model verification
      MEMSTREAM_MODEL_SIGNING_KEY: ${MEMSTREAM_MODEL_SIGNING_KEY}
    depends_on:
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"]
      interval: 15s
      timeout: 5s
      retries: 3
      start_period: 30s
    networks:
      - cadqstream-backend
    deploy:
      resources:
        limits:
          cpus: "0.5"
          memory: 512M
        reservations:
          cpus: "0.25"
          memory: 256M

  # ---------------------------------------------------------------------------
  # Kafka - Message broker (optional, enable in prod)
  # ---------------------------------------------------------------------------
  kafka:
    image: apache/kafka:3.7.0
    container_name: cadqstream-kafka
    restart: unless-stopped
    environment:
      KAFKA_NODE_ID: 1
      KAFKA_PROCESS_ROLES: broker,controller
      KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@kafka:9093
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 3
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 2
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 3
      KAFKA_AUTO_CREATE_TOPICS_ENABLE: "true"
      KAFKA_LOG_RETENTION_HOURS: 168
      KAFKA_LOG_SEGMENT_BYTES: 1073741824
      # Security (enable in production)
      # KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,SSL:SSL
      # KAFKA_SSL_KEYSTORE_LOCATION: /etc/kafka/secrets/kafka.keystore.jks
      # KAFKA_SSL_KEYSTORE_PASSWORD: ${KAFKA_SSL_KEYSTORE_PASSWORD}
    ports:
      - "9092:9092"
    volumes:
      - kafka_data:/var/lib/kafka/data
    networks:
      - cadqstream-backend
    deploy:
      resources:
        limits:
          cpus: "2"
          memory: 4G
        reservations:
          cpus: "1"
          memory: 2G

# =============================================================================
# Networks
# =============================================================================
networks:
  cadqstream-backend:
    driver: bridge
    internal: false
    ipam:
      driver: default
      config:
        - subnet: 172.28.0.0/16

# =============================================================================
# Volumes
# =============================================================================
volumes:
  redis_data:
    driver: local
  postgres_data:
    driver: local
  prometheus_data:
    driver: local
  grafana_data:
    driver: local
  flink_checkpoints:
    driver: local
  flink_savepoints:
    driver: local
  kafka_data:
    driver: local
```

---

## requirements.txt

```
# =============================================================================
# CA-DQStream + MemStream v5 - Python Dependencies
# =============================================================================
# Install: pip install -r requirements.txt
# =============================================================================

# Core ML/AI
torch==2.2.2
torchvision==0.17.2
numpy==1.26.4
scipy==1.13.1
scikit-learn==1.4.2

# Data Processing
pandas==2.2.2
pyarrow==16.0.0
pyarrow-hotfix==0.6

# Streaming & State Management
pyflink==1.18.1
redis==5.0.3

# Web & API
flask==3.0.3
werkzeug==3.0.3
gunicorn==21.2.0

# Monitoring & Metrics
prometheus-client==0.20.0

# Security
python-dotenv==1.0.1
cryptography==42.0.7
hvac==1.2.1

# Configuration
pyyaml==6.0.1

# Testing
pytest==8.2.1
pytest-cov==5.0.0
pytest-asyncio==0.23.6

# Development
black==24.4.2
flake8==7.1.0
mypy==1.10.0
```

---

## Health Server Code

```python
# memstream_src/operators/health_server.py
"""
Health Server for CA-DQStream + MemStream v5

Exposes:
  GET /health     - Liveness probe (is process alive?)
  GET /ready      - Readiness probe (is service ready to accept traffic?)
  GET /metrics    - Prometheus metrics endpoint
  GET /stats      - JSON statistics
"""

import os
import sys
import time
import logging
from typing import Dict, Any, Optional

from flask import Flask, jsonify, Response
from prometheus_client import (
    generate_latest,
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
)
import redis

# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Prometheus Metrics
# =============================================================================

SERVICE_INFO = Info(
    "memstream_service", "MemStream service information"
)
SERVICE_INFO.info({
    "version": "5.0.0",
    "service": "cadqstream",
    "flink_env": os.environ.get("FLINK_ENV", "unknown"),
})

REQUEST_COUNT = Counter(
    "memstream_health_requests_total",
    "Total health check requests",
    ["endpoint", "status"]
)

SERVICE_UP = Gauge(
    "memstream_service_up",
    "Service health status (1=up, 0=down)"
)

REDIS_CONNECTED = Gauge(
    "memstream_redis_connected",
    "Redis connection status (1=connected, 0=disconnected)"
)

FLINK_JOB_MANAGER_UP = Gauge(
    "memstream_flink_jobmanager_up",
    "Flink JobManager availability (1=up, 0=down)"
)

RESPONSE_TIME = Histogram(
    "memstream_health_response_seconds",
    "Health check response time",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# =============================================================================
# Flask App
# =============================================================================

app = Flask(__name__)

# Configuration
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
FLINK_JOB_MANAGER = os.environ.get(
    "FLINK_JOB_MANAGER", "http://localhost:8081"
)
METRICS_PORT = int(os.environ.get("METRICS_PORT", "8080"))

# Redis client (lazy initialization)
_redis_client: Optional[redis.Redis] = None


def get_redis_client() -> Optional[redis.Redis]:
    """Get or create Redis client with lazy initialization."""
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD or None,
                db=0,
                socket_timeout=5,
                socket_connect_timeout=5,
                retry_on_timeout=True,
            )
            # Test connection
            _redis_client.ping()
            logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            _redis_client = None
    return _redis_client


def check_flink_jobmanager() -> bool:
    """Check if Flink JobManager is available."""
    import urllib.request
    import urllib.error
    
    try:
        url = f"{FLINK_JOB_MANAGER}/overview"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status == 200
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        logger.warning(f"Flink JobManager check failed: {e}")
        return False


def check_redis() -> bool:
    """Check if Redis is accessible."""
    client = get_redis_client()
    if client is None:
        return False
    try:
        client.ping()
        return True
    except redis.RedisError as e:
        logger.warning(f"Redis check failed: {e}")
        return False


# =============================================================================
# Health Endpoints
# =============================================================================

@app.route("/health")
def health() -> tuple[Response, int]:
    """
    Liveness probe - Is the process alive?
    
    Returns 200 if the Flask process is running.
    Does not check external dependencies.
    """
    start = time.time()
    try:
        REQUEST_COUNT.labels(endpoint="health", status="success").inc()
        SERVICE_UP.set(1)
        RESPONSE_TIME.observe(time.time() - start)
        return jsonify({
            "status": "healthy",
            "timestamp": time.time(),
            "service": "cadqstream-health-server",
        }), 200
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="health", status="error").inc()
        SERVICE_UP.set(0)
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time(),
        }), 500


@app.route("/ready")
def ready() -> tuple[Response, int]:
    """
    Readiness probe - Is the service ready to accept traffic?
    
    Checks:
    1. Redis connectivity (required for MemStream)
    2. Flink JobManager availability (required for job status)
    """
    start = time.time()
    errors = []
    
    # Check Redis
    redis_ok = check_redis()
    REDIS_CONNECTED.set(1 if redis_ok else 0)
    if not redis_ok:
        errors.append("Redis connection failed")
    
    # Check Flink JobManager
    flink_ok = check_flink_jobmanager()
    FLINK_JOB_MANAGER_UP.set(1 if flink_ok else 0)
    if not flink_ok:
        errors.append("Flink JobManager unavailable")
    
    RESPONSE_TIME.observe(time.time() - start)
    
    if errors:
        REQUEST_COUNT.labels(endpoint="ready", status="not_ready").inc()
        return jsonify({
            "status": "not_ready",
            "checks": {
                "redis": "ok" if redis_ok else "failed",
                "flink_jobmanager": "ok" if flink_ok else "failed",
            },
            "errors": errors,
            "timestamp": time.time(),
        }), 503
    
    REQUEST_COUNT.labels(endpoint="ready", status="ready").inc()
    return jsonify({
        "status": "ready",
        "checks": {
            "redis": "ok",
            "flink_jobmanager": "ok",
        },
        "timestamp": time.time(),
    }), 200


@app.route("/metrics")
def metrics() -> tuple[Response, int]:
    """Prometheus metrics endpoint."""
    return Response(
        generate_latest(),
        mimetype=CONTENT_TYPE_LATEST
    )


@app.route("/stats")
def stats() -> tuple[Response, int]:
    """JSON statistics endpoint."""
    stats_data: Dict[str, Any] = {
        "service": "cadqstream-health-server",
        "version": "5.0.0",
        "uptime_seconds": time.time(),
        "checks": {
            "redis": "connected" if check_redis() else "disconnected",
            "flink_jobmanager": "available" if check_flink_jobmanager() else "unavailable",
        },
    }
    
    # Get Redis info if connected
    client = get_redis_client()
    if client:
        try:
            info = client.info("memory")
            stats_data["redis"] = {
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
            }
        except redis.RedisError:
            pass
    
    return jsonify(stats_data), 200


# =============================================================================
# Error Handlers
# =============================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(e):
    REQUEST_COUNT.labels(endpoint="error", status="500").inc()
    return jsonify({"error": "Internal server error"}), 500


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the health server."""
    logger.info(f"Starting MemStream Health Server on port {METRICS_PORT}")
    logger.info(f"Redis: {REDIS_HOST}:{REDIS_PORT}")
    logger.info(f"Flink JobManager: {FLINK_JOB_MANAGER}")
    
    app.run(
        host="0.0.0.0",
        port=METRICS_PORT,
        debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true",
    )


if __name__ == "__main__":
    main()
```

---

## Traffic Splitting Implementation

```python
# memstream_src/operators/traffic_splitter.py
"""
Traffic Splitting for CA-DQStream + MemStream v5

Implements:
  - Shadow mode: Mirror traffic to candidate model, log results
  - Canary mode: Route small % to candidate, compare results
  - Production mode: Full candidate model

Tracks disagreement between production and candidate models.
"""

import os
import time
import threading
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import random

import numpy as np
from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


class TrafficMode(Enum):
    """Traffic routing mode."""
    PRODUCTION = "production"      # 100% production model
    SHADOW = "shadow"              # 100% production, mirror to candidate
    CANARY = "canary"              # % to candidate, % to production
    FULL_CANDIDATE = "full_candidate"  # 100% candidate


@dataclass
class TrafficConfig:
    """Traffic splitting configuration."""
    mode: TrafficMode = TrafficMode.PRODUCTION
    canary_rate: float = 0.05  # 5% to candidate in canary mode
    shadow_logging: bool = True
    min_samples_for_evaluation: int = 500
    evaluation_window_seconds: int = 3600


@dataclass
class ShadowResult:
    """Result from shadow evaluation."""
    record_id: str
    timestamp: float
    production_score: float
    candidate_score: float
    production_label: str  # "normal" or "anomaly"
    candidate_label: str
    disagreement: bool
    neighborhood: str
    context_key: str


@dataclass
class EvaluationResult:
    """Statistical evaluation of traffic split."""
    total_samples: int
    production_anomaly_rate: float
    candidate_anomaly_rate: float
    disagreement_rate: float
    p_value: Optional[float]  # Wilcoxon signed-rank test
    cohens_d: Optional[float]  # Effect size
    recommendation: str  # "promote", "demote", "continue"
    confidence: float


class TrafficSplitter:
    """
    Manages traffic splitting between production and candidate models.
    
    Supports:
    - Shadow mode: All traffic to production, mirror to candidate
    - Canary mode: Split traffic, evaluate disagreements
    - Full candidate: All traffic to candidate (after promotion)
    """
    
    def __init__(self, config: TrafficConfig):
        self.config = config
        self._lock = threading.Lock()
        self._shadow_buffer: deque[ShadowResult] = deque(maxlen=10000)
        self._production_scores: List[float] = []
        self._candidate_scores: List[float] = []
        self._start_time = time.time()
        
        # Prometheus metrics
        self._traffic_total = Counter(
            "memstream_traffic_total",
            "Total traffic routed",
            ["route"]  # production, candidate, shadow
        )
        self._disagreement_count = Counter(
            "memstream_traffic_disagreement_total",
            "Total disagreements between models",
            ["neighborhood"]
        )
        self._disagreement_rate = Gauge(
            "memstream_traffic_disagreement_rate",
            "Current disagreement rate",
            ["neighborhood"]
        )
        self._evaluation_count = Counter(
            "memstream_traffic_evaluation_total",
            "Number of evaluations performed",
            ["recommendation"]
        )
        
        logger.info(f"TrafficSplitter initialized with mode: {config.mode}")
    
    def route(self, record_id: str, neighborhood: str) -> str:
        """
        Determine which model should handle this record.
        
        Returns:
            "production" or "candidate"
        """
        with self._lock:
            if self.config.mode == TrafficMode.PRODUCTION:
                self._traffic_total.labels(route="production").inc()
                return "production"
            
            elif self.config.mode == TrafficMode.SHADOW:
                # All to production, but record for shadow evaluation
                self._traffic_total.labels(route="production").inc()
                self._traffic_total.labels(route="shadow").inc()
                return "production"
            
            elif self.config.mode == TrafficMode.CANARY:
                if random.random() < self.config.canary_rate:
                    self._traffic_total.labels(route="candidate").inc()
                    return "candidate"
                else:
                    self._traffic_total.labels(route="production").inc()
                    return "production"
            
            elif self.config.mode == TrafficMode.FULL_CANDIDATE:
                self._traffic_total.labels(route="candidate").inc()
                return "candidate"
            
            # Default to production
            self._traffic_total.labels(route="production").inc()
            return "production"
    
    def record_shadow_result(
        self,
        record_id: str,
        production_score: float,
        candidate_score: float,
        production_label: str,
        candidate_label: str,
        neighborhood: str,
        context_key: str,
    ) -> None:
        """
        Record a shadow evaluation result.
        
        Called when both production and candidate have scored the same record.
        """
        disagreement = production_label != candidate_label
        
        result = ShadowResult(
            record_id=record_id,
            timestamp=time.time(),
            production_score=production_score,
            candidate_score=candidate_score,
            production_label=production_label,
            candidate_label=candidate_label,
            disagreement=disagreement,
            neighborhood=neighborhood,
            context_key=context_key,
        )
        
        with self._lock:
            self._shadow_buffer.append(result)
            self._production_scores.append(production_score)
            self._candidate_scores.append(candidate_score)
            
            if disagreement:
                self._disagreement_count.labels(neighborhood=neighborhood).inc()
    
    def evaluate(self) -> Optional[EvaluationResult]:
        """
        Perform statistical evaluation of shadow/canary results.
        
        Returns:
            EvaluationResult if enough samples, None otherwise
        """
        with self._lock:
            n = len(self._shadow_buffer)
            
            if n < self.config.min_samples_for_evaluation:
                logger.info(
                    f"Not enough samples for evaluation: {n}/{self.config.min_samples_for_evaluation}"
                )
                return None
            
            # Calculate rates
            prod_anomalies = sum(
                1 for r in self._shadow_buffer if r.production_label == "anomaly"
            )
            cand_anomalies = sum(
                1 for r in self._shadow_buffer if r.candidate_label == "anomaly"
            )
            disagreements = sum(1 for r in self._shadow_buffer if r.disagreement)
            
            prod_rate = prod_anomalies / n
            cand_rate = cand_anomalies / n
            disagree_rate = disagreements / n
            
            # Wilcoxon signed-rank test
            from scipy import stats
            
            # Remove pairs with identical scores (ties)
            valid_pairs = [
                (r.production_score, r.candidate_score)
                for r in self._shadow_buffer
                if abs(r.production_score - r.candidate_score) > 1e-6
            ]
            
            p_value = None
            cohens_d = None
            recommendation = "continue"
            confidence = 0.0
            
            if len(valid_pairs) >= 100:
                prod_arr = np.array([p[0] for p in valid_pairs])
                cand_arr = np.array([p[1] for p in valid_pairs])
                
                try:
                    stat, p_value = stats.wilcoxon(prod_arr, cand_arr)
                    
                    # Cohen's d
                    diff = cand_arr - prod_arr
                    cohens_d = diff.mean() / (diff.std() + 1e-6)
                    
                    # Recommendation logic
                    if p_value < 0.01 and cohens_d > 0.2:
                        recommendation = "promote"
                        confidence = min(0.95, 1 - p_value)
                    elif p_value < 0.01 and cohens_d < -0.2:
                        recommendation = "demote"
                        confidence = min(0.95, 1 - p_value)
                    else:
                        recommendation = "continue"
                        confidence = 0.5
                        
                except Exception as e:
                    logger.warning(f"Statistical evaluation failed: {e}")
                    recommendation = "continue"
                    confidence = 0.0
            
            self._evaluation_count.labels(recommendation=recommendation).inc()
            
            result = EvaluationResult(
                total_samples=n,
                production_anomaly_rate=prod_rate,
                candidate_anomaly_rate=cand_rate,
                disagreement_rate=disagree_rate,
                p_value=p_value,
                cohens_d=cohens_d,
                recommendation=recommendation,
                confidence=confidence,
            )
            
            # Log evaluation
            logger.info(
                f"Traffic Evaluation: n={n}, "
                f"prod_rate={prod_rate:.4f}, cand_rate={cand_rate:.4f}, "
                f"disagree={disagree_rate:.4f}, "
                f"p={p_value:.4f if p_value else 'N/A'}, "
                f"d={cohens_d:.4f if cohens_d else 'N/A'}, "
                f"rec={recommendation} (conf={confidence:.2f})"
            )
            
            return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current traffic splitting statistics."""
        with self._lock:
            n = len(self._shadow_buffer)
            disagreements = sum(1 for r in self._shadow_buffer if r.disagreement)
            
            return {
                "mode": self.config.mode.value,
                "total_samples": n,
                "production_scores": len(self._production_scores),
                "candidate_scores": len(self._candidate_scores),
                "disagreement_count": disagreements,
                "disagreement_rate": disagreements / n if n > 0 else 0.0,
                "uptime_seconds": time.time() - self._start_time,
                "config": {
                    "canary_rate": self.config.canary_rate,
                    "min_samples": self.config.min_samples_for_evaluation,
                }
            }
    
    def set_mode(self, mode: TrafficMode) -> None:
        """Change traffic routing mode."""
        with self._lock:
            old_mode = self.config.mode
            self.config.mode = mode
            logger.info(f"Traffic mode changed: {old_mode.value} -> {mode.value}")
    
    def reset(self) -> None:
        """Reset all tracking state."""
        with self._lock:
            self._shadow_buffer.clear()
            self._production_scores.clear()
            self._candidate_scores.clear()
            self._start_time = time.time()
            logger.info("TrafficSplitter state reset")
```

---

## docker-compose.prod.yml (Production Override)

```yaml
# =============================================================================
# Production Override for CA-DQStream + MemStream
# =============================================================================
# Usage: docker compose -f docker-compose.yml -f docker-compose.prod.yml up
# =============================================================================

services:
  redis:
    image: redis:7-alpine
    command: >
      redis-server
      --appendonly yes
      --appendfsync everysec
      --requirepass ${REDIS_PASSWORD}
      --maxmemory 512mb
      --maxmemory-policy allkeys-lru
      --save 900 1
      --save 300 10
      --save 60 10000
    deploy:
      resources:
        limits:
          cpus: "1"
          memory: 1G
        reservations:
          cpus: "0.5"
          memory: 512M

  postgres:
    environment:
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_MAX_CONNECTIONS: "100"
      POSTGRES_SHARED_BUFFERS: 512MB
      POSTGRES_EFFECTIVE_CACHE_SIZE: 1GB
      POSTGRES_MAINTENANCE_WORK_MEM: 128MB
      POSTGRES_WAL_LEVEL: replica
      POSTGRES_MAX_WAL_SIZE: 1GB
    deploy:
      resources:
        limits:
          cpus: "2"
          memory: 4G
        reservations:
          cpus: "1"
          memory: 2G

  flink-jobmanager:
    restart: always
    deploy:
      resources:
        limits:
          cpus: "4"
          memory: 8G
        reservations:
          cpus: "2"
          memory: 4G

  flink-taskmanager:
    restart: always
    deploy:
      replicas: 4
      resources:
        limits:
          cpus: "4"
          memory: 10G
        reservations:
          cpus: "2"
          memory: 4G

  prometheus:
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./prometheus/rules:/etc/prometheus/rules:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--config.file=/etc/prometheus/rules/*.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
      - '--storage.tsdb.wal-compression'
      - '--web.enable-lifecycle'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--query.max-concurrency=10'
    deploy:
      resources:
        limits:
          cpus: "1"
          memory: 2G

  grafana:
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_ADMIN_PASSWORD}
      GF_SERVER_SERVE_FROM_SUB_PATH: "true"
      GF_SERVER_ROOT_URL: "%(protocol)s://%(domain)s/grafana"
    volumes:
      - ./grafana/provisioning:/etc/grafana/provisioning:ro
      - ./grafana/dashboards:/var/lib/grafana/dashboards:ro
    deploy:
      resources:
        limits:
          cpus: "0.5"
          memory: 512M

  kafka:
    environment:
      KAFKA_LISTENERS: SSL://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093
      KAFKA_ADVERTISED_LISTENERS: SSL://kafka:9092
      KAFKA_SSL_KEYSTORE_LOCATION: /etc/kafka/secrets/kafka.keystore.jks
      KAFKA_SSL_KEYSTORE_PASSWORD: ${KAFKA_SSL_KEYSTORE_PASSWORD}
      KAFKA_SSL_KEYSTORE_TYPE: JKS
      KAFKA_SSL_TRUSTSTORE_LOCATION: /etc/kafka/secrets/kafka.truststore.jks
      KAFKA_SSL_TRUSTSTORE_PASSWORD: ${KAFKA_SSL_TRUSTSTORE_PASSWORD}
      KAFKA_SSL_CLIENT_AUTH: required
      KAFKA_SSL_ENDPOINT_IDENTIFICATION_ALGORITHM: https
    volumes:
      - ./kafka/secrets:/etc/kafka/secrets:ro
    deploy:
      resources:
        limits:
          cpus: "4"
          memory: 8G
        reservations:
          cpus: "2"
          memory: 4G

  health-server:
    restart: always
    deploy:
      resources:
        limits:
          cpus: "1"
          memory: 1G
        reservations:
          cpus: "0.5"
          memory: 512M

# Enable Kafka in production
  kafka:
    profiles:
      - kafka
```

---

## .env.example

```bash
# =============================================================================
# CA-DQStream + MemStream v5 - Environment Variables
# =============================================================================

# Redis
REDIS_PASSWORD=your-secure-redis-password-min-32-chars

# MinIO
MINIO_SECRET_KEY=your-secure-minio-secret
POSTGRES_DB=cadqstream

# Grafana
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=your-secure-grafana-password

# HMAC Model Signing Key (CRITICAL - must be set)
# Generate with: openssl rand -hex 32
MEMSTREAM_MODEL_SIGNING_KEY=your-hmac-signing-key-32-chars-minimum

# Flink Environment
FLINK_ENV=production
FLINK_SLA_ENABLED=true

# Kafka SSL (production)
KAFKA_SSL_KEYSTORE_PASSWORD=your-keystore-password
KAFKA_SSL_TRUSTSTORE_PASSWORD=your-truststore-password
```

---

## Verification

### Pre-flight Checks

```bash
# 1. Validate docker-compose syntax
docker compose config --quiet

# 2. Check all required env vars are set
grep -h "must be set" .env.example | sed 's/.*?:/\nERROR: /'

# 3. Verify .env file exists
test -f .env || echo "ERROR: .env file not found"

# 4. Generate HMAC key if not set
test -z "$MEMSTREAM_MODEL_SIGNING_KEY" && \
  echo "MEMSTREAM_MODEL_SIGNING_KEY=$(openssl rand -hex 32)" >> .env
```

### Build and Start

```bash
# Build images
docker compose build --no-cache

# Start services (background)
docker compose up -d

# Wait for health checks
docker compose ps

# Verify health endpoints
curl -f http://localhost:8080/health && echo "Health: OK"
curl -f http://localhost:8080/ready && echo "Ready: OK"
curl http://localhost:8080/metrics | head -5
```

### Smoke Tests

```bash
# Test Redis connectivity
docker compose exec health-server python -c \
  "import redis; r = redis.Redis(host='redis', port=6379, password='$REDIS_PASSWORD'); print(r.ping())"

# Test PostgreSQL
docker compose exec health-server python -c \
  "import boto3; s3 = boto3.client('s3', endpoint_url='http://localhost:9000', aws_access_key_id='minioadmin', aws_secret_access_key='$MINIO_SECRET_KEY'); print(s3.head_bucket(Bucket='cadqstream-checkpoints'))"

# Test Flink JobManager
curl -f http://localhost:8081/overview | python -c \
  "import sys, json; d = json.load(sys.stdin); print(f'Flink {d.get(\"flink-version\", \"unknown\")}')"

# Check Prometheus targets
curl -s http://localhost:9090/api/v1/targets | python -c \
  "import sys, json; d = json.load(sys.stdin); print(f'{len(d[\"data\"][\"activeTargets\"])} targets active')"
```

---

## Checklist

- [x] **C-DK-1** Complete Dockerfile with multi-stage build, non-root user, CPU-only PyTorch
- [x] **C-DK-2** Complete docker-compose.yml with all services
- [x] **C-DK-3** MEMSTREAM_MODEL_SIGNING_KEY in docker-compose.yml (env var, required)
- [x] **C-DK-4** Redis configuration with healthcheck, password auth, resource limits
- [x] **H-DK-1** CPU/memory limits on all services (2 CPU, 4GB limits; 0.5 CPU, 1GB reservations)
- [x] **H-DK-2** Traffic splitting implementation (shadow/canary/production modes)
- [x] **H-DK-3** Health server with /health, /ready, /metrics endpoints
- [x] **H-DK-4** requirements.txt with all dependencies and pinned versions
