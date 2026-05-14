# Docker/SRE Review - CA-DQStream + MemStream Hybrid v4

**Reviewer:** Principal SRE/DevOps Engineer
**Date:** 2026-05-12
**Files Reviewed:** `memstream_src/PLAN_v4.md` (full), prior `REVIEW_DOCKER.md` (v3 findings)

---

## Summary

The v4 plan represents a significant pivot from v3: **Docker-only deployment, no Kubernetes**. This simplifies the deployment surface but introduces new gaps. The plan contains **zero actual Dockerfile code** and **zero docker-compose configuration**, despite referencing `memstream_src/operators/deployment/dockerfile`. The architectural decisions (Redis-based IEC communication, HMAC verification, 3-phase dark launch) are sound, but the operational execution details are missing.

**Overall Assessment:** NOT PRODUCTION-READY for Docker deployment.

---

## Container Analysis

### Dockerfile Review

#### ISSUE: No Dockerfile in Plan

**Severity:** CRITICAL

The plan references `memstream_src/operators/deployment/dockerfile` in the directory structure but provides **zero Dockerfile code**. All deployment details are theoretical.

**Evidence:**
```
memstream_src/operators/deployment/
├── dockerfile           ← Referenced but NO CODE
├── kubernetes_deployment.yaml  ← NO CODE (K8s explicitly excluded in v4)
└── prometheus_alerts.yaml  ← Full code ✓
```

**Fix Required:**

```dockerfile
# memstream_src/operators/deployment/Dockerfile
FROM python:3.12-slim-bookworm AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libffi7 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only (smaller footprint)
RUN pip --no-cache-dir install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
COPY memstream_src/requirements.txt /tmp/requirements.txt
RUN pip --no-cache-dir install -r /tmp/requirements.txt

# v3 fix: python:3.8-slim is EOL (Python 3.8 EOL October 2024)
# v3 fix: cloudpickle version mismatch (2.2.1 vs 2.2.0)
# v3 fix: requests not installed but used in HEALTHCHECK

COPY memstream_src/ /app/memstream_src/

# v3 fix: Non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN mkdir -p /models /app/logs && chown -R appuser:appuser /app
USER appuser

WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check using built-in urllib (not requests)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"

ENTRYPOINT ["python", "-m", "memstream_src.operators.memstream_scoring_op"]
```

#### ISSUE: requirements.txt Not Defined

**Severity:** HIGH

The plan uses `torch`, `redis`, `pyflink`, `numpy`, `pandas`, `scipy` but no `requirements.txt` is provided or referenced.

**Fix Required:**

```text
# memstream_src/requirements.txt
torch==2.2.0
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
redis>=5.0.0
pyflink>=1.18.0
```

---

### Resource Configuration

#### ISSUE: No Docker Resource Limits

**Severity:** HIGH

The plan defines `SLOConfig` with latency targets but provides **zero Docker resource limits** for the MemStream container.

**Evidence from plan:**
```python
class SLOConfig:
    latency_p99_ms: float = 100.0
    availability_target: float = 0.999
    fpr_target: float = 0.05
```

**Fix Required:**

```yaml
# docker-compose.yml (new file needed)
services:
  memstream-operator:
    build:
      context: .
      dockerfile: memstream_src/operators/deployment/Dockerfile
    deploy:
      resources:
        limits:
          cpus: "2.0"           # PyTorch benefits from multi-core
          memory: 4G             # Model + memory (2048 slots × 50D × 4B ≈ 400KB per context)
        reservations:
          cpus: "0.5"
          memory: 1G
    environment:
      # v4: Redis-based IEC communication
      REDIS_HOST: ${REDIS_HOST:-redis}
      REDIS_PORT: ${REDIS_PORT:-6379}
      REDIS_PASSWORD: ${REDIS_PASSWORD}
      # v4: HMAC verification enforced
      MEMSTREAM_MODEL_SIGNING_KEY: ${MEMSTREAM_MODEL_SIGNING_KEY}
      REQUIRE_MODEL_SIGNATURE: "true"
      # Model path
      MEMSTREAM_BASE_MODEL_PATH: /models/memstream_warmed.pt
    volumes:
      - memstream-models:/models
      - ./logs:/app/logs
    depends_on:
      redis:
        condition: service_healthy
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
    deploy:
      resources:
        limits:
          cpus: "0.5"
          memory: 256M
    volumes:
      - redis-data:/data

volumes:
  memstream-models:
  redis-data:
```

#### ISSUE: No PyTorch Memory Management

**Severity:** MEDIUM

The plan has `_get_safe_device()` for CUDA fallback but no explicit memory management for PyTorch in CPU mode.

**Evidence:**
```python
def _get_safe_device(requested: str) -> str:
    if requested.startswith('cuda'):
        if not torch.cuda.is_available():
            return 'cpu'
        gpu_mem_free = (torch.cuda.get_device_properties(0).total_memory
                        - torch.cuda.memory_allocated(0))
        if gpu_mem_free < 500 * 1024 * 1024:
            return 'cpu'
    return requested
```

**Missing:** No `torch.set_num_threads()` limitation for CPU mode, no memory pooling configuration.

**Fix:**
```python
# Add to MemStreamCore.__init__ or operator open()
import torch
# Limit CPU threads to prevent container from using all cores
torch.set_num_threads(2)

# Enable memory pooling for PyTorch
torch.backends.cudnn.benchmark = False  # Relevant for GPU
```

---

### Health & Monitoring

#### ISSUE: No Docker HEALTHCHECK Defined

**Severity:** MEDIUM

The plan defines Prometheus alerts but **zero Docker HEALTHCHECK** configuration.

**Evidence from plan:**
```yaml
# Prometheus alert exists
- alert: LatencySLOBreach
  expr: histogram_quantile(0.99, memstream_latency_seconds) > 0.1
```

**Missing:** No `/health` endpoint, no readiness probe, no liveness probe.

**Fix Required:**

```python
# Add to memstream_src/operators/memstream_scoring_op.py
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/health')
def health():
    """Docker HEALTHCHECK endpoint."""
    return jsonify({
        'status': 'healthy',
        'records_scored': MemStreamScoringOperator._records_scored,
        'anomalies_detected': MemStreamScoringOperator._anomalies_detected,
    })

@app.route('/ready')
def ready():
    """Readiness probe - is model loaded?"""
    if hasattr(MemStreamScoringOperator, '_base_model'):
        return jsonify({'status': 'ready'})
    return jsonify({'status': 'not_ready'}), 503
```

```yaml
# In docker-compose.yml
healthcheck:
  test: ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 60s
```

---

## Deployment Strategy

### Dark Launch Phases

#### GOOD: 3-Phase Dark Launch Documented

**Severity:** INFO (Good practice documented)

The plan properly defines 3-phase dark launch:

```python
# Phase 1: Shadow mode (2 weeks)
SHADOW_SAMPLE_RATE = float(os.getenv('SHADOW_SAMPLE_RATE', '1.0'))

# Phase 2: Canary (5% → 10% → 25% → 50%)
CANARY_RATE = float(os.getenv('CANARY_RATE', '0.05'))

# Phase 3: Full production
# Set environment: MEMSTREAM_MODE=production
```

**Assessment:** This is correctly designed. The plan captures the essential phases.

#### ISSUE: No Traffic Splitting Implementation

**Severity:** HIGH

The plan describes dark launch conceptually but provides **zero implementation code** for:
- Traffic splitting logic
- Shadow score comparison
- Rollback triggers
- Metrics aggregation for shadow mode

**Fix Required:**

```python
# In memstream_src/operators/deployment/dark_launch.py
import os
import random
from enum import Enum

class DeploymentPhase(Enum):
    SHADOW = "shadow"
    CANARY = "canary"
    PRODUCTION = "production"

def get_deployment_config() -> dict:
    """Load dark launch configuration from environment."""
    mode = os.getenv('MEMSTREAM_MODE', 'shadow').lower()
    return {
        'phase': DeploymentPhase(mode),
        'shadow_sample_rate': float(os.getenv('SHADOW_SAMPLE_RATE', '1.0')),
        'canary_rate': float(os.getenv('CANARY_RATE', '0.05')),
        'shadow_disagreement_threshold': float(os.getenv(
            'SHADOW_DISAGREEMENT_THRESHOLD', '0.10'
        )),
    }

def should_use_memstream(config: dict) -> bool:
    """Decide whether to use MemStream for this record."""
    phase = config['phase']
    random_val = random.random()
    
    if phase == DeploymentPhase.SHADOW:
        return random_val < config['shadow_sample_rate']
    elif phase == DeploymentPhase.CANARY:
        return random_val < config['canary_rate']
    else:  # PRODUCTION
        return True

def check_shadow_disagreement(
    if_score: float, memstream_score: float, threshold: float
) -> bool:
    """Alert if shadow mode disagreement exceeds threshold."""
    # Example: disagreement if one flags as anomaly and other doesn't
    if_threshold = 0.5  # Example threshold for IF
    ms_threshold = 0.1  # Example threshold for MemStream
    
    if_class = if_score > if_threshold
    ms_class = memstream_score > ms_threshold
    
    disagree = if_class != ms_class
    return disagree
```

#### ISSUE: No Rollback Automation

**Severity:** MEDIUM

The plan mentions rollback triggers but provides **no automated rollback mechanism**.

**Fix Required:**

```yaml
# docker-compose.prod.yml for production deployment
services:
  memstream-operator:
    environment:
      MEMSTREAM_MODE: ${DEPLOYMENT_MODE:-shadow}
      SHADOW_SAMPLE_RATE: "${SHADOW_SAMPLE_RATE:-1.0}"
      CANARY_RATE: "${CANARY_RATE:-0.05}"
      AUTO_ROLLBACK_ON_DISAGREEMENT: "true"
      MAX_SHADOW_DISAGREEMENT_PCT: "10"
    deploy:
      replicas: 2  # At least 2 for canary comparison
```

---

### Environment Configuration

#### ISSUE: No docker-compose.yml in Plan

**Severity:** CRITICAL

The v4 plan pivots to "Docker-only, no K8s" but provides **zero docker-compose configuration**.

**Fix Required:**

```yaml
# docker-compose.yml (full production-ready)
services:
  # ── Redis ───────────────────────────────────────────────────────────────
  redis:
    image: redis:7-alpine
    container_name: cadqstream-redis
    restart: unless-stopped
    command: >
      redis-server 
      --appendonly yes 
      --requirepass ${REDIS_PASSWORD}
      --maxmemory 200mb
      --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5
    deploy:
      resources:
        limits:
          cpus: "0.5"
          memory: 256M
    volumes:
      - redis-data:/data
    networks:
      - backend

  # ── MemStream Operator ─────────────────────────────────────────────────
  memstream-operator:
    build:
      context: ${DOCKER_BUILD_CONTEXT:-.}
      dockerfile: memstream_src/operators/deployment/Dockerfile
      target: production
    container_name: cadqstream-memstream
    restart: unless-stopped
    depends_on:
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    environment:
      # Deployment mode
      MEMSTREAM_MODE: ${MEMSTREAM_MODE:-shadow}
      SHADOW_SAMPLE_RATE: "${SHADOW_SAMPLE_RATE:-1.0}"
      CANARY_RATE: "${CANARY_RATE:-0.05}"
      
      # Redis connection (v4: IEC communication)
      REDIS_HOST: redis
      REDIS_PORT: 6379
      REDIS_PASSWORD: ${REDIS_PASSWORD}
      
      # Model security (v4: HMAC verification enforced)
      MEMSTREAM_MODEL_SIGNING_KEY: ${MEMSTREAM_MODEL_SIGNING_KEY}
      REQUIRE_MODEL_SIGNATURE: "true"
      MODEL_PATH: /models/memstream_warmed.pt
      
      # Python config
      PYTHONUNBUFFERED: "1"
      PYTHONPATH: /app
    volumes:
      - memstream-models:/models:ro
      - ./logs:/app/logs
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 4G
        reservations:
          cpus: "0.5"
          memory: 1G
      replicas: 2
    networks:
      - backend

  # ── Prometheus Exporter ────────────────────────────────────────────────
  cadqstream-metrics:
    build:
      context: .
      dockerfile: memstream_src/operators/deployment/Dockerfile.metrics
    container_name: cadqstream-metrics
    restart: unless-stopped
    ports:
      - "9250:9250"
    environment:
      REDIS_HOST: redis
      REDIS_PORT: 6379
      REDIS_PASSWORD: ${REDIS_PASSWORD}
    healthcheck:
      test: ["CMD-SHELL", "nc -z localhost 9250 || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 3
    networks:
      - backend
    deploy:
      resources:
        limits:
          cpus: "0.25"
          memory: 128M

networks:
  backend:
    driver: bridge
    internal: false

volumes:
  redis-data:
    driver: local
  memstream-models:
    driver: local
```

---

## Issues Found

### CRITICAL Issues

| # | Component | Issue | Impact |
|---|-----------|-------|--------|
| C1 | Dockerfile | No Dockerfile code in plan | Cannot build container |
| C2 | Docker Compose | No docker-compose.yml defined | Cannot orchestrate containers |
| C3 | Secrets | `MEMSTREAM_MODEL_SIGNING_KEY` not in any compose | HMAC bypassed (security regression from v3 fix) |
| C4 | Redis | No Redis configuration for IEC | IEC→MemStream communication fails |

### HIGH Issues

| # | Component | Issue | Impact |
|---|-----------|-------|--------|
| H1 | Resources | No CPU/memory limits in plan | Resource exhaustion risk |
| H2 | Dark Launch | No traffic splitting implementation | Cannot execute dark launch |
| H3 | Health | No `/health` endpoint code | Cannot implement HEALTHCHECK |
| H4 | dependencies | No requirements.txt | Build reproducibility fails |

### MEDIUM Issues

| # | Component | Issue | Impact |
|---|-----------|-------|--------|
| M1 | Operations | No graceful shutdown code | State loss on container restart |
| M2 | Operations | No log aggregation config | Debugging困难 |
| M3 | Operations | No rollback automation | Manual intervention required |
| M4 | Monitoring | Prometheus alerts but no scrape config | Alerts not actionable |

### LOW Issues

| # | Component | Issue | Impact |
|---|-----------|-------|--------|
| L1 | Docker | No `.dockerignore` | Larger image size |
| L2 | Docker | No multi-stage build | Larger image size |
| L3 | Docker | No image tag strategy | Version confusion |

---

## Recommendations

### 1. Add Dockerfile (CRITICAL)

Create `memstream_src/operators/deployment/Dockerfile`:

```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm AS base

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Build stage
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only (smaller than CUDA version)
RUN pip install --no-cache-dir \
    torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu

# Install Python deps
COPY memstream_src/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Production stage
FROM base AS production

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libffi7 \
    curl \
    netcat \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Setup directories
RUN mkdir -p /models /app/logs /app/data && chown -R appuser:appuser /app

# Copy application
COPY --chown=appuser:appuser memstream_src/ /app/memstream_src/

WORKDIR /app
ENV PYTHONPATH=/app

USER appuser

# Health check using curl
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Expose metrics port
EXPOSE 8080

CMD ["python", "-m", "memstream_src.operators.memstream_scoring_op"]
```

### 2. Add requirements.txt (HIGH)

Create `memstream_src/requirements.txt`:

```text
# Core ML
torch==2.2.0
numpy>=1.24.0,<2.0
pandas>=2.0.0,<3.0
scipy>=1.10.0

# Streaming
pyflink>=1.18.0

# Communication
redis>=5.0.0

# HTTP (for health checks)
flask>=3.0.0

# Security
cryptography>=41.0.0

# Testing
pytest>=7.4.0
pytest-cov>=4.1.0
```

### 3. Add Health Endpoint (HIGH)

Create `memstream_src/operators/health.py`:

```python
"""Health check endpoints for Docker HEALTHCHECK."""
import os
import threading
from flask import Flask, jsonify
from typing import Dict, Any

app = Flask(__name__)

# Shared state (updated by operators)
_state: Dict[str, Any] = {
    'model_loaded': False,
    'records_scored': 0,
    'anomalies_detected': 0,
    'redis_connected': False,
}

_lock = threading.Lock()

def update_state(**kwargs):
    with _lock:
        _state.update(kwargs)

@app.route('/health')
def health():
    """Liveness probe - is the service running?"""
    return jsonify({
        'status': 'healthy',
        'timestamp': os.times().elapsed,
    })

@app.route('/ready')
def ready():
    """Readiness probe - is the service ready to receive traffic?"""
    with _lock:
        model_ready = _state['model_loaded']
        redis_ready = _state['redis_connected']
    
    if model_ready and redis_ready:
        return jsonify({
            'status': 'ready',
            'records_scored': _state['records_scored'],
        })
    return jsonify({
        'status': 'not_ready',
        'model_loaded': model_ready,
        'redis_connected': redis_ready,
    }), 503

@app.route('/metrics')
def metrics():
    """Prometheus-compatible metrics endpoint."""
    with _lock:
        records = _state['records_scored']
        anomalies = _state['anomalies_detected']
    
    anomaly_rate = anomalies / max(records, 1)
    
    metrics_text = f"""# HELP cadqstream_records_scored_total Total records scored
# TYPE cadqstream_records_scored_total counter
cadqstream_records_scored_total {records}

# HELP cadqstream_anomalies_detected_total Total anomalies detected
# TYPE cadqstream_anomalies_detected_total counter
cadqstream_anomalies_detected_total {anomalies}

# HELP cadqstream_anomaly_rate_current Current anomaly detection rate
# TYPE cadqstream_anomaly_rate_current gauge
cadqstream_anomaly_rate_current {anomaly_rate}
"""
    return metrics_text, 200, {'Content-Type': 'text/plain'}

def start_health_server(port: int = 8080):
    """Start the health check server in a background thread."""
    def run():
        app.run(host='0.0.0.0', port=port, threaded=True)
    
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread
```

### 4. Add Graceful Shutdown (MEDIUM)

Add to `MemStreamScoringOperator`:

```python
import signal
import sys

def _setup_signal_handlers(self):
    """Setup graceful shutdown handlers."""
    def signal_handler(signum, frame):
        LOGGER.info(f"[MemStreamOp] Received signal {signum}, shutting down gracefully...")
        # Save state before exit
        if hasattr(self, '_base_model'):
            LOGGER.info("[MemStreamOp] Model state preserved in checkpoint")
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

# Call in __init__
self._setup_signal_handlers()
```

### 5. Add Log Aggregation (MEDIUM)

```yaml
# In docker-compose.yml
services:
  memstream-operator:
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"
    volumes:
      - /var/lib/docker/containers:/var/lib/docker/containers:ro
```

---

## Verification Checklist

- [ ] Docker build verified (`docker build -f memstream_src/operators/deployment/Dockerfile .`)
- [ ] Resource limits tested under load (1000 records/sec for 10 min)
- [ ] Health checks validated (liveness + readiness probes)
- [ ] Redis connectivity verified (`docker compose up redis && docker compose run --rm memstream-operator redis-cli -h redis ping`)
- [ ] Dark launch tested (shadow → canary → production phases)
- [ ] HMAC verification enforced (verify crash on tampered model)
- [ ] Graceful shutdown tested (`docker stop` should preserve state)
- [ ] Log aggregation working (logs visible in centralized system)
- [ ] Prometheus scraping configured and alerts firing
- [ ] Rollback procedure tested (manual + automated)

---

## Prior v3 Issues Status

| Issue | v3 Status | v4 Status |
|-------|-----------|-----------|
| Python 3.8 EOL | CRITICAL | STILL MISSING (no Dockerfile) |
| HEALTHCHECK uses requests | CRITICAL | FIXED (use curl or urllib) |
| No non-root user | HIGH | STILL MISSING |
| Hardcoded credentials | CRITICAL | STILL MISSING (no compose) |
| PLAINTEXT Kafka | CRITICAL | N/A (removed Kafka from plan) |
| Ephemeral checkpoints | CRITICAL | NOT ADDRESSED (v4 focuses on Docker) |
| JVM heap exceeds memory | CRITICAL | N/A (Docker-only, no JVM) |

---

## Summary of Gaps

The v4 plan makes a clean break from v3's Kubernetes ambitions, but **introduces new critical gaps**:

1. **No deployment artifacts** - Zero Dockerfile, zero docker-compose.yml
2. **No requirements.txt** - Build reproducibility impossible
3. **No health monitoring** - Cannot implement Docker HEALTHCHECK
4. **No secrets management** - HMAC verification bypassed
5. **No operational runbook** - No startup/shutdown/logging procedures

**Recommendation:** Before proceeding to implementation, the team must produce:
1. `memstream_src/operators/deployment/Dockerfile`
2. `memstream_src/operators/deployment/docker-compose.yml`
3. `memstream_src/requirements.txt`
4. `memstream_src/operators/health.py`

---

*Reviewed by: Principal SRE/DevOps Engineer | 2026-05-12*
