"""
Health Server for CA-DQStream + MemStream v5.

FIXES in v5:
- H-DK-3: Health endpoint with /health, /ready, /metrics

Exposes:
  GET /health     - Liveness probe
  GET /ready     - Readiness probe
  GET /metrics    - Prometheus metrics endpoint
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

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Prometheus metrics
SERVICE_INFO = Info("memstream_service", "MemStream service information")
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

SERVICE_UP = Gauge("memstream_service_up", "Service health status")
REDIS_CONNECTED = Gauge("memstream_redis_connected", "Redis connection status")
RESPONSE_TIME = Histogram(
    "memstream_health_response_seconds",
    "Health check response time",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

app = Flask(__name__)

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.environ.get("REDIS_PASSWORD", "")
FLINK_JOB_MANAGER = os.environ.get("FLINK_JOB_MANAGER", "http://localhost:8081")
METRICS_PORT = int(os.environ.get("METRICS_PORT", "8080"))

_redis_client: Optional[redis.Redis] = None


def get_redis_client() -> Optional[redis.Redis]:
    """Get or create Redis client."""
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD or None,
                socket_timeout=5,
                socket_connect_timeout=5,
            )
            _redis_client.ping()
        except redis.ConnectionError:
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
    except (urllib.error.URLError, urllib.error.HTTPError):
        return False


@app.route("/health")
def health() -> tuple[Response, int]:
    """Liveness probe."""
    start = time.time()
    REQUEST_COUNT.labels(endpoint="health", status="success").inc()
    SERVICE_UP.set(1)
    RESPONSE_TIME.observe(time.time() - start)
    return jsonify({"status": "healthy", "timestamp": time.time()}), 200


@app.route("/ready")
def ready() -> tuple[Response, int]:
    """Readiness probe."""
    start = time.time()
    errors = []
    
    redis_ok = get_redis_client() is not None
    REDIS_CONNECTED.set(1 if redis_ok else 0)
    if not redis_ok:
        errors.append("Redis connection failed")
    
    flink_ok = check_flink_jobmanager()
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
        }), 503
    
    REQUEST_COUNT.labels(endpoint="ready", status="ready").inc()
    return jsonify({
        "status": "ready",
        "checks": {
            "redis": "ok",
            "flink_jobmanager": "ok",
        },
    }), 200


@app.route("/metrics")
def metrics() -> tuple[Response, int]:
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route("/stats")
def stats() -> tuple[Response, int]:
    """JSON statistics endpoint."""
    return jsonify({
        "service": "cadqstream-health-server",
        "version": "5.0.0",
        "timestamp": time.time(),
        "checks": {
            "redis": "connected" if get_redis_client() else "disconnected",
            "flink_jobmanager": "available" if check_flink_jobmanager() else "unavailable",
        },
    }), 200


def main():
    logger.info(f"Starting Health Server on port {METRICS_PORT}")
    app.run(host="0.0.0.0", port=METRICS_PORT)


if __name__ == "__main__":
    main()
