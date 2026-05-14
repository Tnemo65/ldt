# Monitoring Fixes - CA-DQStream + MemStream v4 → v5

> **Purpose:** Fix all CRITICAL and HIGH monitoring/observability issues identified in PLAN_v4.md
> **Date:** 2026-05-12
> **Status:** Implementation-ready

---

## Overview

This document provides complete implementations for all monitoring-related fixes:

| Issue | Priority | Description | Section |
|-------|----------|-------------|---------|
| C-MON-1 | CRITICAL | Prometheus metrics not exposed | Section 1 |
| C-MON-2 | CRITICAL | OpenTelemetry tracing missing | Section 2 |
| C-MON-3 | CRITICAL | No error budget tracking | Section 3 |
| C-MON-4 | CRITICAL | Anomaly rate metric not exposed | Section 1 |
| C-MON-5 | CRITICAL | HMAC failure not metriced | Section 1 |
| H-MON-6 | HIGH | No availability metric | Section 1 |
| H-MON-7 | HIGH | No FPR tracking metric | Section 1 |
| H-MON-8 | HIGH | Redis beta polling health | Section 1 |
| H-MON-9 | HIGH | No checkpoint metrics | Section 4 |
| H-MON-10 | HIGH | Logging not JSON-structured | Section 5 |

---

## Section 1: C-MON-1, C-MON-4, C-MON-5, H-MON-6, H-MON-7, H-MON-8 — Prometheus Metrics Instrumentation

### File: `memstream_src/monitoring/metrics.py`

```python
"""
Prometheus Metrics Instrumentation for CA-DQStream + MemStream.

Provides comprehensive metrics collection for:
- MemStream scoring latency and throughput
- Anomaly detection rates
- Error tracking (HMAC failures, model errors)
- IEC operator metrics
- SLO burn-rate tracking
- Redis beta polling health
- Checkpoint success/failure rates

Usage:
    from monitoring.metrics import MemStreamMetrics, IECMetrics, get_metrics_registry
    
    # In operator __init__:
    self.metrics = MemStreamMetrics()
    
    # In process_element:
    with self.metrics.scoring_latency.time(neighborhood, scoring_method):
        score = self.memstream.score_one(features)
    self.metrics.records_scored.labels(neighborhood, result).inc()
"""

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    CollectorRegistry,
    REGISTRY,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from contextlib import contextmanager
from typing import Optional, Dict, Any
import time
import threading


class SLOConfig:
    """SLO configuration for burn-rate calculations."""
    
    def __init__(
        self,
        latency_p99_target_ms: float = 100.0,
        error_rate_target: float = 0.001,  # 0.1%
        availability_target: float = 0.999,  # 99.9%
        window_hours: int = 1,
    ):
        self.latency_p99_target_ms = latency_p99_target_ms
        self.error_rate_target = error_rate_target
        self.availability_target = availability_target
        self.window_hours = window_hours
        
        # Error budget calculations (based on 30-day window)
        self.total_budget_seconds = 30 * 24 * 3600
        self.error_budget_seconds = self.total_budget_seconds * error_rate_target
        self.availability_budget_seconds = self.total_budget_seconds * (1 - availability_target)


class MemStreamMetrics:
    """
    Prometheus metrics for MemStream scoring operator.
    
    Metrics exposed:
    - Scoring latency histogram (seconds)
    - Records scored counter (by neighborhood, result)
    - Scoring errors counter (by error_type, neighborhood)
    - Memory utilization gauge (by neighborhood, context_key)
    - Beta threshold gauge (by neighborhood, context_key)
    - Anomaly rate gauge (by neighborhood)
    - HMAC failure counter (model_beta, security)
    - Redis connection failures counter
    - Redis polling latency histogram
    - Error budget remaining gauge (SLO tracking)
    """
    
    def __init__(self, registry: CollectorRegistry = REGISTRY):
        self._registry = registry
        self._lock = threading.Lock()
        
        # ─────────────────────────────────────────────────────────────
        # C-MON-1: Core scoring metrics
        # ─────────────────────────────────────────────────────────────
        
        # Latency histogram with buckets optimized for streaming workloads
        self.scoring_latency = Histogram(
            name="memstream_scoring_latency_seconds",
            documentation="MemStream score_one() latency in seconds",
            labelnames=["neighborhood", "scoring_method"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
            registry=registry,
        )
        
        # Records scored counter
        self.records_scored = Counter(
            name="memstream_records_scored_total",
            documentation="Total records scored by neighborhood and result",
            labelnames=["neighborhood", "scoring_result"],  # result: normal|anomaly|error
            registry=registry,
        )
        
        # ─────────────────────────────────────────────────────────────
        # C-MON-5: HMAC security failure tracking
        # ─────────────────────────────────────────────────────────────
        
        self.hmac_failures = Counter(
            name="memstream_hmac_failures_total",
            documentation="HMAC validation failures by source",
            labelnames=["source"],  # model_beta | serialization
            registry=registry,
        )
        
        # ─────────────────────────────────────────────────────────────
        # C-MON-4: Anomaly rate tracking
        # ─────────────────────────────────────────────────────────────
        
        self.anomaly_rate = Gauge(
            name="memstream_anomaly_rate",
            documentation="Current anomaly rate (rolling window) per neighborhood",
            labelnames=["neighborhood"],
            registry=registry,
        )
        
        # Anomaly rate histogram for percentile calculations
        self.anomaly_rate_histogram = Histogram(
            name="memstream_anomaly_rate_histogram",
            documentation="Anomaly rate distribution",
            buckets=[0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0],
            registry=registry,
        )
        
        # ─────────────────────────────────────────────────────────────
        # H-MON-6: Availability metrics
        # ─────────────────────────────────────────────────────────────
        
        self.total_requests = Counter(
            name="memstream_total_requests_total",
            documentation="Total requests received",
            labelnames=["neighborhood"],
            registry=registry,
        )
        
        self.error_count = Counter(
            name="memstream_error_count_total",
            documentation="Total errors by error category",
            labelnames=["neighborhood", "error_type"],
            registry=registry,
        )
        
        self.availability = Gauge(
            name="memstream_availability_ratio",
            documentation="Current availability ratio (1 - error_rate)",
            labelnames=["neighborhood"],
            registry=registry,
        )
        
        # ─────────────────────────────────────────────────────────────
        # H-MON-8: Redis beta polling health
        # ─────────────────────────────────────────────────────────────
        
        self.redis_connection_failures = Counter(
            name="memstream_redis_connection_failures_total",
            documentation="Redis connection failures",
            labelnames=["operation"],  # get | set | ping
            registry=registry,
        )
        
        self.redis_latency = Histogram(
            name="memstream_redis_latency_seconds",
            documentation="Redis operation latency",
            labelnames=["operation"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
            registry=registry,
        )
        
        # ─────────────────────────────────────────────────────────────
        # SLO burn-rate tracking (C-MON-3)
        # ─────────────────────────────────────────────────────────────
        
        self.slo_latency_budget_remaining = Gauge(
            name="memstream_slo_latency_budget_remaining_ms",
            documentation="Remaining latency budget (ms) in current SLO window",
            labelnames=["neighborhood"],
            registry=registry,
        )
        
        self.slo_error_budget_remaining = Gauge(
            name="memstream_slo_error_budget_remaining_seconds",
            documentation="Remaining error budget (seconds) in current SLO window",
            labelnames=["neighborhood"],
            registry=registry,
        )
        
        self.slo_burn_rate = Gauge(
            name="memstream_slo_burn_rate",
            documentation="Current SLO burn rate (1.0 = healthy, >1.0 = burning budget)",
            labelnames=["neighborhood", "slo_type"],  # latency | error | availability
            registry=registry,
        )
        
        # ─────────────────────────────────────────────────────────────
        # Error categorization
        # ─────────────────────────────────────────────────────────────
        
        self.scoring_errors = Counter(
            name="memstream_scoring_errors_total",
            documentation="Scoring errors by type and neighborhood",
            labelnames=["error_type", "neighborhood"],
            registry=registry,
        )
        
        # ─────────────────────────────────────────────────────────────
        # Memory & model state
        # ─────────────────────────────────────────────────────────────
        
        self.memory_utilization = Gauge(
            name="memstream_memory_utilization",
            documentation="Memory slot utilization (0.0 to 1.0)",
            labelnames=["neighborhood", "context_key"],
            registry=registry,
        )
        
        self.beta_threshold = Gauge(
            name="memstream_beta_threshold",
            documentation="Current beta threshold per context",
            labelnames=["neighborhood", "context_key"],
            registry=registry,
        )
        
        self.model_load_timestamp = Gauge(
            name="memstream_model_load_timestamp_seconds",
            documentation="Unix timestamp when model was loaded",
            registry=registry,
        )
        
        self.model_version = Info(
            name="memstream_model",
            documentation="Model version and metadata",
            registry=registry,
        )
        
        # Internal state for rate calculations
        self._last_error_budget_update = time.time()
        self._error_budget_consumed = {}  # neighborhood -> seconds consumed
        
    @contextmanager
    def scoring_latency_time(
        self, 
        neighborhood: str, 
        scoring_method: str = "online"
    ):
        """Context manager for timing scoring operations."""
        start = time.perf_counter()
        try:
            yield
        except Exception:
            self.scoring_errors.labels(error_type="scoring_exception", neighborhood=neighborhood).inc()
            raise
        finally:
            duration = time.perf_counter() - start
            self.scoring_latency.labels(
                neighborhood=neighborhood,
                scoring_method=scoring_method
            ).observe(duration)
    
    def record_scoring(
        self, 
        neighborhood: str, 
        result: str,  # "normal", "anomaly", "error"
        is_anomaly: bool = False,
    ):
        """Record a scoring event."""
        self.total_requests.labels(neighborhood=neighborhood).inc()
        self.records_scored.labels(neighborhood=neighborhood, scoring_result=result).inc()
        
        if is_anomaly:
            self.anomaly_rate.labels(neighborhood=neighborhood).inc()
            self.anomaly_rate_histogram.observe(1.0)
        else:
            self.anomaly_rate_histogram.observe(0.0)
    
    def record_hmac_failure(self, source: str = "model_beta"):
        """Record HMAC validation failure (C-MON-5)."""
        self.hmac_failures.labels(source=source).inc()
        self.scoring_errors.labels(error_type="hmac_failure", neighborhood="global").inc()
    
    def record_redis_failure(self, operation: str = "get"):
        """Record Redis connection failure (H-MON-8)."""
        self.redis_connection_failures.labels(operation=operation).inc()
    
    def record_redis_latency(self, operation: str, duration_seconds: float):
        """Record Redis operation latency (H-MON-8)."""
        self.redis_latency.labels(operation=operation).observe(duration_seconds)
    
    def update_error_budget(
        self, 
        neighborhood: str, 
        error_count: int, 
        window_seconds: float,
        slo_config: SLOConfig,
    ):
        """
        Update SLO error budget tracking (C-MON-3).
        
        Burn rate = (observed_error_rate / target_error_rate)
        - burn_rate < 1.0: Healthy, budget accumulating
        - burn_rate = 1.0: Consuming budget at expected rate
        - burn_rate > 1.0: Burning budget faster than expected
        """
        with self._lock:
            # Calculate observed error rate
            observed_error_rate = error_count / max(window_seconds, 1)
            
            # Calculate burn rate
            burn_rate = observed_error_rate / max(slo_config.error_rate_target, 1e-10)
            
            # Update burn rate gauge
            self.slo_burn_rate.labels(
                neighborhood=neighborhood, 
                slo_type="error"
            ).set(burn_rate)
            
            # Track budget consumption
            elapsed = time.time() - self._last_error_budget_update
            budget_consumed = elapsed * observed_error_rate
            self._error_budget_consumed[neighborhood] = (
                self._error_budget_consumed.get(neighborhood, 0) + budget_consumed
            )
            
            # Calculate remaining budget
            total_budget = slo_config.error_budget_seconds
            remaining = max(0, total_budget - self._error_budget_consumed[neighborhood])
            
            self.slo_error_budget_remaining.labels(neighborhood=neighborhood).set(remaining)
            
            self._last_error_budget_update = time.time()
    
    def update_availability(
        self, 
        neighborhood: str, 
        total: int, 
        errors: int,
    ):
        """Update availability metric (H-MON-6)."""
        if total > 0:
            availability_ratio = 1.0 - (errors / total)
            self.availability.labels(neighborhood=neighborhood).set(availability_ratio)
    
    def update_anomaly_rate(
        self, 
        neighborhood: str, 
        anomalies: int, 
        total: int,
    ):
        """Update anomaly rate gauge (C-MON-4)."""
        if total > 0:
            rate = anomalies / total
            self.anomaly_rate.labels(neighborhood=neighborhood).set(rate)
    
    def update_memory_utilization(
        self, 
        neighborhood: str, 
        context_key: str,
        used_slots: int,
        total_slots: int,
    ):
        """Update memory slot utilization gauge."""
        if total_slots > 0:
            utilization = used_slots / total_slots
            self.memory_utilization.labels(
                neighborhood=neighborhood, 
                context_key=context_key
            ).set(utilization)
    
    def update_beta_threshold(
        self, 
        neighborhood: str, 
        context_key: str,
        beta_value: float,
    ):
        """Update beta threshold gauge."""
        self.beta_threshold.labels(
            neighborhood=neighborhood, 
            context_key=context_key
        ).set(beta_value)
    
    def set_model_info(self, version: str, commit: str = ""):
        """Set model version info."""
        self.model_version.info({
            "version": version,
            "commit": commit,
        })
        self.model_load_timestamp.set_to_current_time()


class IECMetrics:
    """
    Prometheus metrics for IEC (Intelligent Evolution Controller) operator.
    """
    
    def __init__(self, registry: CollectorRegistry = REGISTRY):
        self._registry = registry
        
        self.drifts_detected = Counter(
            name="iec_drifts_detected_total",
            documentation="Total drift detections by neighborhood and metric",
            labelnames=["neighborhood", "metric_type"],
            registry=registry,
        )
        
        self.strategies_executed = Counter(
            name="iec_strategies_executed_total",
            documentation="IEC strategies executed",
            labelnames=["strategy", "severity"],
            registry=registry,
        )
        
        self.strategy_confidence = Histogram(
            name="iec_strategy_confidence",
            documentation="IEC strategy confidence scores",
            labelnames=["strategy"],
            buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
            registry=registry,
        )
        
        self.consecutive_actions = Gauge(
            name="iec_consecutive_actions_total",
            documentation="Consecutive IEC actions (for circuit breaker)",
            labelnames=["neighborhood"],
            registry=registry,
        )
        
        self.adwin_variance = Gauge(
            name="iec_adwin_variance",
            documentation="ADWIN variance estimate",
            labelnames=["neighborhood", "metric"],
            registry=registry,
        )
    
    def record_drift(self, neighborhood: str, metric_type: str):
        """Record a drift detection."""
        self.drifts_detected.labels(neighborhood=neighborhood, metric_type=metric_type).inc()
    
    def record_strategy(
        self, 
        strategy: str, 
        severity: str, 
        confidence: float,
    ):
        """Record strategy execution."""
        self.strategies_executed.labels(strategy=strategy, severity=severity).inc()
        self.strategy_confidence.labels(strategy=strategy).observe(confidence)
    
    def update_consecutive_actions(self, neighborhood: str, count: int):
        """Update consecutive action counter."""
        self.consecutive_actions.labels(neighborhood=neighborhood).set(count)


class CheckpointMetrics:
    """
    Prometheus metrics for Flink checkpoint tracking (H-MON-9).
    """
    
    def __init__(self, registry: CollectorRegistry = REGISTRY):
        self._registry = registry
        
        self.checkpoint_started = Counter(
            name="flink_checkpoint_started_total",
            documentation="Total checkpoints initiated",
            registry=registry,
        )
        
        self.checkpoint_completed = Counter(
            name="flink_checkpoint_completed_total",
            documentation="Total checkpoints completed successfully",
            registry=registry,
        )
        
        self.checkpoint_failed = Counter(
            name="flink_checkpoint_failed_total",
            documentation="Total checkpoints failed",
            labelnames=["reason"],
            registry=registry,
        )
        
        self.checkpoint_duration = Histogram(
            name="flink_checkpoint_duration_seconds",
            documentation="Checkpoint completion duration",
            buckets=[1, 5, 10, 30, 60, 120, 300, 600],
            registry=registry,
        )
        
        self.checkpoint_size = Histogram(
            name="flink_checkpoint_size_bytes",
            documentation="Checkpoint size in bytes",
            buckets=[1e6, 10e6, 100e6, 500e6, 1e9, 5e9, 10e9],  # 1MB to 10GB
            registry=registry,
        )
        
        self.checkpoint_align_duration = Histogram(
            name="flink_checkpoint_alignment_seconds",
            documentation="Time spent waiting for alignment",
            buckets=[0.01, 0.1, 0.5, 1, 5, 10, 30],
            registry=registry,
        )
        
        self.state_backend_size = Gauge(
            name="flink_state_backend_size_bytes",
            documentation="Current state backend size",
            registry=registry,
        )
    
    def record_checkpoint_start(self):
        """Record checkpoint initiation."""
        self.checkpoint_started.inc()
    
    def record_checkpoint_complete(
        self, 
        duration_seconds: float,
        size_bytes: int,
        alignment_seconds: float = 0,
    ):
        """Record successful checkpoint completion."""
        self.checkpoint_completed.inc()
        self.checkpoint_duration.observe(duration_seconds)
        self.checkpoint_size.observe(size_bytes)
        if alignment_seconds > 0:
            self.checkpoint_align_duration.observe(alignment_seconds)
    
    def record_checkpoint_failure(self, reason: str = "unknown"):
        """Record checkpoint failure."""
        self.checkpoint_failed.labels(reason=reason).inc()


class FPRMetrics:
    """
    Metrics for False Positive Rate tracking (H-MON-7).
    
    Requires IEC to provide labeled data with ground truth.
    """
    
    def __init__(self, registry: CollectorRegistry = REGISTRY):
        self._registry = registry
        
        self.fpr_estimate = Gauge(
            name="cadqstream_fpr_estimate",
            documentation="Estimated false positive rate",
            labelnames=["neighborhood", "context_key", "window"],
            registry=registry,
        )
        
        self.fpr_samples = Counter(
            name="cadqstream_fpr_samples_total",
            documentation="Total samples used for FPR estimation",
            labelnames=["neighborhood"],
            registry=registry,
        )
        
        self.fpr_confidence = Gauge(
            name="cadqstream_fpr_confidence",
            documentation="Confidence interval width for FPR estimate",
            labelnames=["neighborhood"],
            registry=registry,
        )
    
    def update_fpr(
        self, 
        neighborhood: str, 
        context_key: str,
        fpr: float,
        window: str = "1h",
        confidence_width: float = 0.0,
    ):
        """Update FPR estimate."""
        self.fpr_estimate.labels(
            neighborhood=neighborhood,
            context_key=context_key,
            window=window,
        ).set(fpr)
        if confidence_width > 0:
            self.fpr_confidence.labels(neighborhood=neighborhood).set(confidence_width)


# Global metrics instance (can be overridden in tests)
_global_metrics: Optional[MemStreamMetrics] = None
_global_iec_metrics: Optional[IECMetrics] = None
_global_checkpoint_metrics: Optional[CheckpointMetrics] = None
_global_fpr_metrics: Optional[FPRMetrics] = None
_lock = threading.Lock()


def get_metrics_registry() -> CollectorRegistry:
    """Get the global metrics registry."""
    return REGISTRY


def get_memstream_metrics() -> MemStreamMetrics:
    """Get or create the global MemStream metrics instance."""
    global _global_metrics
    with _lock:
        if _global_metrics is None:
            _global_metrics = MemStreamMetrics()
        return _global_metrics


def get_iec_metrics() -> IECMetrics:
    """Get or create the global IEC metrics instance."""
    global _global_iec_metrics
    with _lock:
        if _global_iec_metrics is None:
            _global_iec_metrics = IECMetrics()
        return _global_iec_metrics


def get_checkpoint_metrics() -> CheckpointMetrics:
    """Get or create the global checkpoint metrics instance."""
    global _global_checkpoint_metrics
    with _lock:
        if _global_checkpoint_metrics is None:
            _global_checkpoint_metrics = CheckpointMetrics()
        return _global_checkpoint_metrics


def get_fpr_metrics() -> FPRMetrics:
    """Get or create the global FPR metrics instance."""
    global _global_fpr_metrics
    with _lock:
        if _global_fpr_metrics is None:
            _global_fpr_metrics = FPRMetrics()
        return _global_fpr_metrics
```

### File: `memstream_src/operators/memstream_scoring_op_instrumented.py`

Integration example showing how to use the metrics in the MemStreamScoringOperator:

```python
"""
Example: Integrating MemStreamMetrics into MemStreamScoringOperator.

This shows where each metric should be incremented/called.
"""

# In __init__:
def __init__(self, config: dict):
    self.config = config
    # ... existing init code ...
    
    # Initialize metrics
    self.metrics = MemStreamMetrics()
    self.slo_config = SLOConfig(
        latency_p99_target_ms=100.0,
        error_rate_target=0.001,
        availability_target=0.999,
    )
    
    # Counters for windowed calculations
    self._window_errors = 0
    self._window_total = 0
    self._window_anomalies = 0
    self._window_start = time.time()


# In process_element:
def process_element(self, record, context):
    neighborhood = record.get('PULocationID', 'unknown')
    
    # Start tracing span (C-MON-2)
    with self.tracer.start_as_current_span(
        "memstream_score",
        attributes={
            "neighborhood": neighborhood,
            "context_key": context_key,
        }
    ):
        try:
            # Extract features
            features = self.vectorizer.transform(record)
            if features is None:
                return None
            
            # Time the scoring operation
            with self.metrics.scoring_latency_time(neighborhood, "online"):
                score, is_anomaly = self._score_with_tracing(features, record)
            
            # Record metrics
            self.metrics.record_scoring(
                neighborhood=neighborhood,
                result="anomaly" if is_anomaly else "normal",
                is_anomaly=is_anomaly,
            )
            
            # Update windowed stats
            self._window_total += 1
            if is_anomaly:
                self._window_anomalies += 1
            
            # Update anomaly rate periodically
            if self._window_total % 1000 == 0:
                self.metrics.update_anomaly_rate(
                    neighborhood, 
                    self._window_anomalies, 
                    self._window_total
                )
            
            # Update memory utilization
            self.metrics.update_memory_utilization(
                neighborhood,
                context_key,
                self.memstream.memory.used_slots,
                self.memstream.memory.total_slots,
            )
            
            # Update beta threshold
            self.metrics.update_beta_threshold(
                neighborhood,
                context_key,
                self.memstream.get_beta(context_key),
            )
            
            return self._emit_result(record, score, is_anomaly)
            
        except Exception as e:
            span = trace.get_current_span()
            span.record_exception(e)
            span.set_status(StatusCode.ERROR, str(e))
            
            self.metrics.scoring_errors.labels(
                error_type=type(e).__name__,
                neighborhood=neighborhood,
            ).inc()
            self.metrics.record_scoring(neighborhood, "error")
            self._window_errors += 1
            
            # Update availability
            self.metrics.update_availability(
                neighborhood,
                self._window_total,
                self._window_errors,
            )
            
            raise


# Redis beta polling with health metrics:
def _fetch_beta_from_redis(self, key: str) -> Optional[float]:
    """Fetch beta threshold from Redis with health metrics (H-MON-8)."""
    start = time.perf_counter()
    try:
        import redis
        value = self.redis_client.get(key)
        
        # Record latency
        duration = time.perf_counter() - start
        self.metrics.record_redis_latency("get", duration)
        
        return float(value) if value else None
        
    except redis.ConnectionError as e:
        self.metrics.record_redis_failure("get")
        logger.error("Redis connection failed", extra={"error": str(e)})
        return None
    except Exception as e:
        self.metrics.record_redis_failure("get")
        self.metrics.scoring_errors.labels(
            error_type="redis_error",
            neighborhood="global",
        ).inc()
        return None
```

---

## Section 2: C-MON-2 — OpenTelemetry Tracing

### File: `memstream_src/monitoring/tracing.py`

```python
"""
OpenTelemetry Tracing Setup for CA-DQStream + MemStream.

Provides distributed tracing with spans for:
- MemStream scoring pipeline
- Feature extraction
- Memory operations
- IEC decision making
- Kafka message production

Requires OTEL collector running (Jaeger/Zipkin compatible).

Usage:
    from monitoring.tracing import setup_tracing, get_tracer
    
    # In job initialization:
    setup_tracing(
        service_name="cadqstream-memstream",
        otlp_endpoint="http://otel-collector:4317",
    )
    
    tracer = get_tracer()
    
    # In operator:
    with tracer.start_as_current_span("score_record") as span:
        span.set_attribute("neighborhood", neighborhood)
        # ... scoring logic ...
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.trace import Status, StatusCode, SpanKind
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.context import Context
from opentelemetry.propagate import set_global_textmap
from typing import Optional, Dict, Any, Callable
import time
import functools
import logging

logger = logging.getLogger(__name__)

# Global tracer instance
_tracer: Optional[trace.Tracer] = None
_propagator = TraceContextTextMapPropagator()


def setup_tracing(
    service_name: str,
    service_version: str = "1.0.0",
    otlp_endpoint: Optional[str] = None,
    console_export: bool = False,
    sampling_rate: float = 1.0,
    environment: str = "production",
) -> trace.Tracer:
    """
    Initialize OpenTelemetry tracing.
    
    Args:
        service_name: Name of the service (e.g., "cadqstream-memstream")
        service_version: Version string
        otlp_endpoint: OTLP collector endpoint (e.g., "http://otel-collector:4317")
        console_export: Enable console export for debugging
        sampling_rate: Trace sampling rate (0.0 to 1.0)
        environment: Deployment environment
    
    Returns:
        Configured tracer instance
    """
    global _tracer
    
    # Create resource with service metadata
    resource = Resource.create({
        SERVICE_NAME: service_name,
        "service.version": service_version,
        "deployment.environment": environment,
        "host.name": _get_hostname(),
    })
    
    # Create tracer provider
    provider = TracerProvider(resource=resource)
    
    # Add OTLP exporter if endpoint specified
    if otlp_endpoint:
        try:
            otlp_exporter = OTLPSpanExporter(
                endpoint=otlp_endpoint,
                insecure=True,  # Use TLS in production
            )
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.info(f"OTLP exporter configured: {otlp_endpoint}")
        except Exception as e:
            logger.warning(f"Failed to configure OTLP exporter: {e}")
    
    # Add console exporter for debugging
    if console_export:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        logger.info("Console span exporter enabled")
    
    # Set global tracer provider
    trace.set_tracer_provider(provider)
    
    # Configure propagator for W3C TraceContext
    set_global_textmap(_propagator)
    
    # Get tracer
    _tracer = trace.get_tracer(
        instrumenting_module_name=__name__,
        tracer_provider=provider,
    )
    
    logger.info(f"Tracing initialized: service={service_name}, version={service_version}")
    
    return _tracer


def get_tracer() -> trace.Tracer:
    """Get the global tracer instance."""
    global _tracer
    if _tracer is None:
        # Return a no-op tracer if not initialized
        return trace.get_tracer(__name__)
    return _tracer


def _get_hostname() -> str:
    """Get hostname for resource attributes."""
    import socket
    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


class TracingContext:
    """
    Context manager for creating traced spans with common attributes.
    
    Usage:
        with TracingContext(tracer, "memstream_score") as ctx:
            ctx.set_attributes(
                neighborhood=neighborhood,
                context_key=context_key,
            )
            score = memstream.score_one(features)
            ctx.set_attribute("score", float(score))
    """
    
    def __init__(
        self,
        tracer: trace.Tracer,
        span_name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        self.tracer = tracer
        self.span_name = span_name
        self.kind = kind
        self.attributes = attributes or {}
        self.span: Optional[trace.Span] = None
    
    def __enter__(self):
        self.span = self.tracer.start_span(
            self.span_name,
            kind=self.kind,
        )
        for key, value in self.attributes.items():
            self.span.set_attribute(key, value)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.span.record_exception(exc_val)
            self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
        self.span.end()
        return False
    
    def set_attribute(self, key: str, value: Any):
        """Set span attribute."""
        if self.span:
            self.span.set_attribute(key, value)
    
    def set_attributes(self, **kwargs):
        """Set multiple span attributes."""
        for key, value in kwargs.items():
            self.set_attribute(key, value)
    
    def add_event(self, name: str, attributes: Optional[Dict] = None):
        """Add span event."""
        if self.span:
            self.span.add_event(name, attributes=attributes)


def traced(
    span_name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    kind: SpanKind = SpanKind.INTERNAL,
) -> Callable:
    """
    Decorator to add tracing to a function.
    
    Usage:
        @traced(span_name="memstream_score", attributes={"component": "scoring"})
        def score_one(self, features):
            # ... implementation ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            name = span_name or f"{func.__module__}.{func.__qualname__}"
            
            with tracer.start_as_current_span(name, kind=kind) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                
                # Add function arguments as span attributes
                span.set_attribute("function.name", func.__name__)
                
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        
        return wrapper
    return decorator


class MemStreamSpanAttributes:
    """Constants for MemStream span attribute names."""
    
    # Entity identifiers
    NEIGHBORHOOD = "memstream.neighborhood"
    CONTEXT_KEY = "memstream.context_key"
    TRIP_ID = "memstream.trip_id"
    
    # Scoring attributes
    SCORE = "memstream.score"
    IS_ANOMALY = "memstream.is_anomaly"
    SCORING_METHOD = "memstream.scoring_method"  # online | memory | warmup
    SCORE_LATENCY_MS = "memstream.score_latency_ms"
    
    # Memory attributes
    MEMORY_USED_SLOTS = "memstream.memory.used_slots"
    MEMORY_TOTAL_SLOTS = "memstream.memory.total_slots"
    MEMORY_UTILIZATION = "memstream.memory.utilization"
    
    # Beta threshold
    BETA_THRESHOLD = "memstream.beta_threshold"
    BETA_SOURCE = "memstream.beta_source"  # default | redis | iec
    
    # Feature attributes
    FEATURE_DIM = "memstream.feature_dim"
    FEATURE_EXTRACTION_MS = "memstream.feature_extraction_ms"
    
    # Error attributes
    ERROR_TYPE = "memstream.error_type"
    ERROR_MESSAGE = "memstream.error_message"
    
    # IEC attributes
    IEC_STRATEGY = "iec.strategy"
    IEC_CONFIDENCE = "iec.confidence"
    DRIFT_DETECTED = "iec.drift_detected"
    DRIFT_SEVERITY = "iec.drift_severity"


def create_scoring_span(
    neighborhood: str,
    context_key: str,
    scoring_method: str = "online",
) -> trace.Span:
    """
    Create a span for MemStream scoring with standard attributes.
    
    Args:
        neighborhood: Pickup location ID
        context_key: Context key (neighborhood × time_bucket)
        scoring_method: Method used for scoring
    
    Returns:
        Configured span
    """
    tracer = get_tracer()
    
    span = tracer.start_span(
        "memstream.score",
        kind=SpanKind.INTERNAL,
    )
    
    # Set standard attributes
    span.set_attribute(MemStreamSpanAttributes.NEIGHBORHOOD, neighborhood)
    span.set_attribute(MemStreamSpanAttributes.CONTEXT_KEY, context_key)
    span.set_attribute(MemStreamSpanAttributes.SCORING_METHOD, scoring_method)
    
    return span


def inject_trace_context(carrier: Dict) -> Dict:
    """
    Inject trace context into a carrier dict (e.g., Kafka headers).
    
    Args:
        carrier: Dict to inject context into
    
    Returns:
        Carrier with trace context
    """
    _propagator.inject(carrier)
    return carrier


def extract_trace_context(carrier: Dict) -> Context:
    """
    Extract trace context from a carrier dict.
    
    Args:
        carrier: Dict containing trace context
    
    Returns:
        Extracted context
    """
    return _propagator.extract(carrier)


def create_linked_span(
    span_name: str,
    trace_id: int,
    span_id: int,
) -> trace.Span:
    """
    Create a span linked to an external trace (e.g., Kafka message).
    
    Args:
        span_name: Name for the new span
        trace_id: External trace ID
        span_id: External span ID
    
    Returns:
        Linked span
    """
    tracer = get_tracer()
    
    from opentelemetry.trace import Link
    
    span_context = trace.SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=True,
        trace_flags=trace.TraceFlags(trace.TraceFlags.SAMPLED),
        trace_state=trace.TraceState(),
    )
    
    return tracer.start_span(
        span_name,
        links=[Link(span_context)],
    )
```

---

## Section 3: C-MON-3 — SLO Burn-Rate Calculation

### File: `memstream_src/monitoring/slo.py`

```python
"""
SLO Burn-Rate Calculation and Tracking for CA-DQStream + MemStream.

Implements multi-window burn-rate alerting:
- 1h window: Fast burn-rate (4x multiplier)
- 6h window: Medium burn-rate (2x multiplier)
- 3d window: Slow burn-rate (1x multiplier)

Multipliers based on error budget alerting best practices:
https://sre.google/workbook/alerting-on-slos/

Usage:
    from monitoring.slo import SLOBurnRateTracker, SLOAlertingPolicy
    
    tracker = SLOBurnRateTracker(
        slo_config=SLOConfig(
            latency_p99_target_ms=100.0,
            error_rate_target=0.001,
            availability_target=0.999,
        ),
        neighborhoods=["manhattan", "brooklyn", "queens"],
    )
    
    # In main loop:
    tracker.update(
        neighborhood="manhattan",
        latency_p99_ms=150.0,
        error_count=5,
        total_requests=10000,
    )
    
    alerts = tracker.check_alerts()
    for alert in alerts:
        publish_alert(alert)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import time
import threading
import math


class SLOType(Enum):
    """Types of SLOs being tracked."""
    LATENCY = "latency"
    ERROR_RATE = "error_rate"
    AVAILABILITY = "availability"


class BurnRateWindow(Enum):
    """Burn-rate calculation windows."""
    FAST_1H = "1h"
    MEDIUM_6H = "6h"
    SLOW_3D = "3d"


@dataclass
class SLOConfig:
    """Configuration for an SLO target."""
    
    latency_p99_target_ms: float = 100.0
    error_rate_target: float = 0.001  # 0.1%
    availability_target: float = 0.999  # 99.9%
    
    # Window sizes in seconds
    window_1h_seconds: float = 3600
    window_6h_seconds: float = 21600
    window_3d_seconds: float = 259200  # 3 days
    
    # Burn-rate thresholds (multiplier × target = alerting threshold)
    fast_burn_multiplier: float = 14.4   # 1h window: 5% budget burned/hour
    medium_burn_multiplier: float = 6.0  # 6h window: 10% budget burned/hour
    slow_burn_multiplier: float = 3.0    # 3d window: 100% budget burned in window


@dataclass
class WindowedMetrics:
    """Metrics aggregated over a time window."""
    
    window_seconds: float
    samples: int = 0
    error_count: int = 0
    latency_sum_ms: float = 0.0
    latency_max_ms: float = 0.0
    latency_p99_ms: float = 0.0  # Approximated via percentiles
    
    # For rolling percentiles
    latency_samples: List[float] = field(default_factory=list)
    
    def add_sample(self, latency_ms: float, is_error: bool = False):
        """Add a sample to the window."""
        self.samples += 1
        self.latency_sum_ms += latency_ms
        self.latency_max_ms = max(self.latency_max_ms, latency_ms)
        self.latency_samples.append(latency_ms)
        
        if is_error:
            self.error_count += 1
        
        # Keep samples within window (simplified - real impl needs timestamps)
        if len(self.latency_samples) > 10000:
            self.latency_samples = self.latency_samples[-5000:]
    
    def compute_p99(self) -> float:
        """Compute approximate p99 latency."""
        if not self.latency_samples:
            return 0.0
        sorted_samples = sorted(self.latency_samples)
        idx = int(len(sorted_samples) * 0.99)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]
    
    def reset(self):
        """Reset all metrics."""
        self.samples = 0
        self.error_count = 0
        self.latency_sum_ms = 0.0
        self.latency_max_ms = 0.0
        self.latency_samples.clear()


@dataclass
class SLOAlert:
    """Alert generated by SLO burn-rate breach."""
    
    alert_id: str
    neighborhood: str
    slo_type: SLOType
    burn_rate: float
    threshold: float
    budget_remaining_seconds: float
    budget_total_seconds: float
    window: BurnRateWindow
    severity: str  # warning | critical
    message: str
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            "alert_id": self.alert_id,
            "neighborhood": self.neighborhood,
            "slo_type": self.slo_type.value,
            "burn_rate": round(self.burn_rate, 2),
            "threshold": self.threshold,
            "budget_remaining_seconds": round(self.budget_remaining_seconds, 1),
            "budget_total_seconds": round(self.budget_total_seconds, 1),
            "window": self.window.value,
            "severity": self.severity,
            "message": self.message,
            "timestamp": self.timestamp,
        }


class SLOBurnRateTracker:
    """
    Multi-window SLO burn-rate tracker.
    
    Tracks burn rates across multiple time windows and generates
    alerts when burn rates exceed thresholds.
    
    Burn-rate = (observed_bad / total) / (target_bad / total_window)
    
    Example:
        - Target: 99.9% availability (0.1% errors)
        - 1h window: 3600 seconds
        - Budget: 3600 * 0.001 = 3.6 seconds of errors per hour
        
        If we observe 10 errors in 5 minutes (300s):
        - Burn rate = (10 errors / 300s) / (3.6 errors / 3600s)
        - Burn rate = 0.033 / 0.001 = 33x
        
        At 33x burn rate, we'd exhaust the 1h budget in ~2 minutes!
    """
    
    def __init__(
        self,
        slo_config: SLOConfig,
        neighborhoods: List[str],
    ):
        self.slo_config = slo_config
        self.neighborhoods = neighborhoods
        
        # Per-neighborhood, per-window metrics
        self._metrics: Dict[str, Dict[BurnRateWindow, WindowedMetrics]] = {}
        self._error_budget_consumed: Dict[str, float] = {}  # seconds consumed
        self._last_update: Dict[str, float] = {}
        
        # Alert tracking
        self._active_alerts: Dict[str, SLOAlert] = {}
        self._alert_history: List[SLOAlert] = []
        self._lock = threading.Lock()
        
        # Initialize metrics for each neighborhood
        for neighborhood in neighborhoods:
            self._init_neighborhood(neighborhood)
    
    def _init_neighborhood(self, neighborhood: str):
        """Initialize metrics tracking for a neighborhood."""
        self._metrics[neighborhood] = {
            BurnRateWindow.FAST_1H: WindowedMetrics(
                window_seconds=self.slo_config.window_1h_seconds
            ),
            BurnRateWindow.MEDIUM_6H: WindowedMetrics(
                window_seconds=self.slo_config.window_6h_seconds
            ),
            BurnRateWindow.SLOW_3D: WindowedMetrics(
                window_seconds=self.slo_config.window_3d_seconds
            ),
        }
        self._error_budget_consumed[neighborhood] = 0.0
        self._last_update[neighborhood] = time.time()
    
    def update(
        self,
        neighborhood: str,
        latency_p99_ms: Optional[float] = None,
        error_count: int = 0,
        total_requests: int = 0,
    ):
        """
        Update metrics for a neighborhood.
        
        Args:
            neighborhood: Neighborhood identifier
            latency_p99_ms: Measured p99 latency
            error_count: Number of errors in this sample period
            total_requests: Total requests in this sample period
        """
        with self._lock:
            if neighborhood not in self._metrics:
                self._init_neighborhood(neighborhood)
            
            now = time.time()
            elapsed = now - self._last_update[neighborhood]
            
            # Update each window
            for window, metrics in self._metrics[neighborhood].items():
                # Add samples to all windows
                if latency_p99_ms is not None:
                    metrics.add_sample(latency_p99_ms, is_error=False)
                
                # Update error rate (spread errors across windows)
                if error_count > 0 and total_requests > 0:
                    error_rate = error_count / total_requests
                    for _ in range(min(error_count, 100)):  # Limit iterations
                        metrics.add_sample(0, is_error=True)
            
            # Update error budget consumption
            if total_requests > 0:
                observed_error_rate = error_count / total_requests
                budget_consumed = elapsed * observed_error_rate
                self._error_budget_consumed[neighborhood] += budget_consumed
            
            self._last_update[neighborhood] = now
    
    def compute_burn_rates(self, neighborhood: str) -> Dict[BurnRateWindow, Dict[SLOType, float]]:
        """
        Compute burn rates across all windows and SLO types.
        
        Returns:
            Dict mapping (window, slo_type) to burn_rate
        """
        rates = {}
        
        for window, metrics in self._metrics.get(neighborhood, {}).items():
            if metrics.samples == 0:
                continue
            
            window_seconds = metrics.window_seconds
            
            # Error rate burn rate
            if metrics.samples > 0:
                observed_error_rate = metrics.error_count / metrics.samples
                target_error_rate = self.slo_config.error_rate_target
                
                # Normalize to same time scale
                error_burn_rate = observed_error_rate / target_error_rate
                
                rates[(window, SLOType.ERROR_RATE)] = error_burn_rate
            
            # Latency burn rate (if p99 available)
            if metrics.latency_samples:
                p99 = metrics.compute_p99()
                target_latency = self.slo_config.latency_p99_target_ms
                
                if target_latency > 0:
                    latency_burn_rate = p99 / target_latency
                    rates[(window, SLOType.LATENCY)] = latency_burn_rate
        
        return rates
    
    def check_alerts(self) -> List[SLOAlert]:
        """
        Check for SLO burn-rate alerts.
        
        Returns:
            List of new/ongoing alerts
        """
        alerts = []
        now = time.time()
        
        for neighborhood in self.neighborhoods:
            burn_rates = self.compute_burn_rates(neighborhood)
            
            for (window, slo_type), burn_rate in burn_rates.items():
                # Determine threshold based on window
                if window == BurnRateWindow.FAST_1H:
                    threshold = self.slo_config.fast_burn_multiplier
                elif window == BurnRateWindow.MEDIUM_6H:
                    threshold = self.slo_config.medium_burn_multiplier
                else:
                    threshold = self.slo_config.slow_burn_multiplier
                
                # Check if burn rate exceeds threshold
                if burn_rate > threshold:
                    # Calculate remaining budget
                    if slo_type == SLOType.ERROR_RATE:
                        total_budget = self.slo_config.error_rate_target * self.slo_config.window_3d_seconds
                        remaining = max(0, total_budget - self._error_budget_consumed[neighborhood])
                        message = (
                            f"Error budget burning at {burn_rate:.1f}x rate. "
                            f"Expected: {threshold:.1f}x. "
                            f"Budget remaining: {self._format_duration(remaining)}"
                        )
                    else:
                        remaining = 0
                        message = f"Latency SLO burning at {burn_rate:.1f}x rate"
                    
                    # Determine severity
                    severity = "critical" if burn_rate > threshold * 2 else "warning"
                    
                    alert = SLOAlert(
                        alert_id=f"{neighborhood}-{slo_type.value}-{window.value}",
                        neighborhood=neighborhood,
                        slo_type=slo_type,
                        burn_rate=burn_rate,
                        threshold=threshold,
                        budget_remaining_seconds=remaining,
                        budget_total_seconds=total_budget if slo_type == SLOType.ERROR_RATE else 0,
                        window=window,
                        severity=severity,
                        message=message,
                        timestamp=now,
                    )
                    
                    alerts.append(alert)
                    self._active_alerts[alert.alert_id] = alert
        
        # Track alert history
        self._alert_history.extend(alerts)
        if len(self._alert_history) > 10000:
            self._alert_history = self._alert_history[-5000:]
        
        return alerts
    
    def get_error_budget_status(self, neighborhood: str) -> Dict:
        """
        Get error budget status for a neighborhood.
        
        Returns:
            Dict with budget consumed, remaining, and projected exhaustion time
        """
        total_budget = self.slo_config.error_rate_target * self.slo_config.window_3d_seconds
        consumed = self._error_budget_consumed.get(neighborhood, 0)
        remaining = max(0, total_budget - consumed)
        
        # Project exhaustion
        now = time.time()
        elapsed = now - self._last_update.get(neighborhood, now)
        
        if elapsed > 0 and consumed > 0:
            burn_rate_per_second = consumed / elapsed
            if burn_rate_per_second > 0:
                exhaustion_seconds = remaining / burn_rate_per_second
            else:
                exhaustion_seconds = float('inf')
        else:
            exhaustion_seconds = float('inf')
        
        return {
            "neighborhood": neighborhood,
            "budget_total_seconds": round(total_budget, 1),
            "budget_consumed_seconds": round(consumed, 1),
            "budget_remaining_seconds": round(remaining, 1),
            "budget_remaining_percentage": round(100 * remaining / total_budget, 2),
            "exhaustion_projected_seconds": (
                round(exhaustion_seconds, 0) if exhaustion_seconds != float('inf') else None
            ),
            "healthy": remaining > (total_budget * 0.5),  # Healthy if >50% remaining
        }
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable form."""
        if seconds == float('inf') or seconds is None:
            return "never"
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        elif seconds < 86400:
            return f"{seconds/3600:.1f}h"
        else:
            return f"{seconds/86400:.1f}d"
    
    def reset_budget(self, neighborhood: str):
        """Reset error budget (e.g., after maintenance)."""
        with self._lock:
            self._error_budget_consumed[neighborhood] = 0.0
            for metrics in self._metrics.get(neighborhood, {}).values():
                metrics.reset()
            
            # Clear related alerts
            alert_ids_to_clear = [
                aid for aid, alert in self._active_alerts.items()
                if alert.neighborhood == neighborhood
            ]
            for aid in alert_ids_to_clear:
                del self._active_alerts[aid]
```

---

## Section 4: H-MON-9 — Checkpoint Metrics via Flink Listener

### File: `memstream_src/operators/checkpoint_listener.py`

```python
"""
Flink Checkpoint Metrics via CheckpointListener interface.

Implements CheckpointListener to capture:
- Checkpoint start/complete/failure events
- Checkpoint duration and size
- Alignment time for exactly-once semantics
- State backend size tracking

Usage:
    from operators.checkpoint_listener import CheckpointMetricsListener
    
    # In Flink job:
    env.get_checkpoint_config().set_externalized_checkpoint_cleanup(
        ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION
    )
    
    # Add listener to operators:
    operator = MemStreamScoringOperator()
    operator = CheckpointMetricsListener(operator, metrics)
"""

from pyflink.runtime.checkpoint import CheckpointListener, CheckpointMetadata
from pyflink.datastream.state import ValueStateDescriptor, StateBackend
from pyflink.common import typeinfo
from typing import Optional, Dict, Any, List
import time
import threading


class CheckpointMetricsListener(CheckpointListener):
    """
    Flink CheckpointListener that records checkpoint metrics.
    
    Implements the CheckpointListener interface to receive
    notifications about checkpoint completion.
    """
    
    def __init__(self, wrapped_operator, checkpoint_metrics):
        """
        Initialize checkpoint listener.
        
        Args:
            wrapped_operator: The operator to wrap
            checkpoint_metrics: CheckpointMetrics instance for recording
        """
        self._wrapped = wrapped_operator
        self._metrics = checkpoint_metrics
        self._pending_checkpoints: Dict[str, float] = {}
        self._completed_checkpoints: List[Dict] = []
        self._lock = threading.Lock()
    
    # ─────────────────────────────────────────────────────────────
    # CheckpointListener interface
    # ─────────────────────────────────────────────────────────────
    
    def notify_checkpoint_complete(
        self, 
        checkpoint_id: int,
        checkpoint_metadata: Optional[CheckpointMetadata] = None,
    ):
        """
        Called when a checkpoint completes.
        
        Args:
            checkpoint_id: ID of the completed checkpoint
            checkpoint_metadata: Metadata about the checkpoint
        """
        with self._lock:
            # Find matching pending checkpoint
            start_time = self._pending_checkpoints.pop(checkpoint_id, None)
            
            if start_time is not None:
                duration = time.time() - start_time
                
                # Extract checkpoint details if available
                size_bytes = 0
                alignment_time = 0
                
                if checkpoint_metadata:
                    # These fields depend on Flink version
                    size_bytes = getattr(checkpoint_metadata, 'state_size', 0)
                    alignment_time = getattr(checkpoint_metadata, 'alignment_duration', 0)
                
                # Record metrics
                self._metrics.record_checkpoint_complete(
                    duration_seconds=duration,
                    size_bytes=size_bytes,
                    alignment_seconds=alignment_time,
                )
                
                # Track completed checkpoints
                self._completed_checkpoints.append({
                    "checkpoint_id": checkpoint_id,
                    "duration": duration,
                    "size": size_bytes,
                    "timestamp": time.time(),
                })
                
                # Keep last 100 checkpoints for analysis
                if len(self._completed_checkpoints) > 100:
                    self._completed_checkpoints = self._completed_checkpoints[-50:]
    
    def notify_checkpoint_aborted(
        self,
        checkpoint_id: int,
        checkpoint_metadata: Optional[CheckpointMetadata] = None,
    ):
        """Called when a checkpoint is aborted."""
        with self._lock:
            self._pending_checkpoints.pop(checkpoint_id, None)
            self._metrics.record_checkpoint_failure(reason="aborted")
    
    def get_checkpoint_id(self) -> int:
        """
        Return the ID of the checkpoint to wait for.
        
        Returns 0 to disable waiting, or a specific checkpoint ID.
        """
        return 0  # Override to enable wait for specific checkpoint
    
    # ─────────────────────────────────────────────────────────────
    # Operator passthrough (delegate to wrapped operator)
    # ─────────────────────────────────────────────────────────────
    
    def open(self, runtime_context):
        """Open the operator."""
        if hasattr(self._wrapped, 'open'):
            return self._wrapped.open(runtime_context)
    
    def close(self):
        """Close the operator."""
        if hasattr(self._wrapped, 'close'):
            return self._wrapped.close()
    
    def process_element(self, *args, **kwargs):
        """Process element (delegate to wrapped)."""
        return self._wrapped.process_element(*args, **kwargs)


class CheckpointStateMetrics:
    """
    Track state backend size and checkpoint health.
    
    Use with Flink's REST API to get actual checkpoint sizes,
    or approximate based on state descriptors.
    """
    
    def __init__(self, checkpoint_metrics):
        self._metrics = checkpoint_metrics
        self._state_sizes: Dict[str, int] = {}  # operator -> bytes
        self._lock = threading.Lock()
    
    def update_state_size(self, operator_name: str, size_bytes: int):
        """Update known state size for an operator."""
        with self._lock:
            self._state_sizes[operator_name] = size_bytes
            total = sum(self._state_sizes.values())
            self._metrics.state_backend_size.set(total)
    
    def get_total_state_size(self) -> int:
        """Get total state backend size."""
        with self._lock:
            return sum(self._state_sizes.values())


# Flink job-level checkpoint configuration helper
def configure_checkpointing(env, config: Dict[str, Any]):
    """
    Configure checkpointing with proper settings.
    
    Args:
        env: Flink StreamExecutionEnvironment
        config: Checkpoint configuration dict
    """
    from pyflink.datastream.checkpoint_config import (
        CheckpointConfig,
        ExternalizedCheckpointCleanup,
        CheckpointingMode,
    )
    
    checkpoint_config = env.get_checkpoint_config()
    
    # Checkpointing mode
    mode = config.get("mode", "EXACTLY_ONCE")
    checkpoint_config.set_checkpointing_mode(
        CheckpointingMode.EXACTLY_ONCE if mode == "EXACTLY_ONCE" 
        else CheckpointingMode.AT_LEAST_ONCE
    )
    
    # Interval
    interval_ms = config.get("interval_seconds", 45) * 1000
    checkpoint_config.set_checkpoint_interval(interval_ms)
    
    # Timeout
    timeout_ms = config.get("timeout_seconds", 300) * 1000
    checkpoint_config.set_checkpoint_timeout(timeout_ms)
    
    # Min pause between checkpoints
    min_pause_ms = config.get("min_pause_seconds", 15) * 1000
    checkpoint_config.set_min_pause_between_checkpoints(min_pause_ms)
    
    # Max concurrent checkpoints
    max_concurrent = config.get("max_concurrent_checkpoints", 1)
    checkpoint_config.set_max_concurrent_checkpoints(max_concurrent)
    
    # Externalized checkpoint cleanup
    cleanup = config.get("externalized_cleanup", "RETAIN_ON_CANCELLATION")
    if cleanup == "DELETE_ON_CANCELLATION":
        checkpoint_config.set_externalized_checkpoint_cleanup(
            ExternalizedCheckpointCleanup.DELETE_ON_CANCELLATION
        )
    else:
        checkpoint_config.set_externalized_checkpoint_cleanup(
            ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION
        )
    
    # Enable unaligned checkpoints for exactly-once with slow sinks
    if config.get("unaligned_checkpoints", False):
        checkpoint_config.enable_unaligned_checkpoints()
    
    # Checkpoint retention
    if config.get("retain_after_cancellation", True):
        checkpoint_config.set_externalized_checkpoint_cleanup(
            ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION
        )
```

---

## Section 5: H-MON-10 — JSON Structured Logging

### File: `memstream_src/monitoring/logging_config.py`

```python
"""
JSON Structured Logging Configuration for CA-DQStream + MemStream.

Provides structured logging with:
- JSON output for log aggregation systems
- Correlation IDs for request tracing
- Standard fields (timestamp, level, service, etc.)
- Context managers for request-scoped logging

Supports:
- ELK Stack (Elasticsearch, Logstash, Kibana)
- Grafana Loki
- Datadog
- CloudWatch Logs
- Splunk

Usage:
    from monitoring.logging_config import setup_logging, get_logger
    
    # In application entry point:
    setup_logging(
        service_name="cadqstream-memstream",
        environment="production",
        log_level="INFO",
        json_output=True,
    )
    
    logger = get_logger(__name__)
    
    # In code:
    logger.info("Scoring record", extra={
        "neighborhood": "manhattan",
        "context_key": "manhattan_2024-01-15_10",
        "score": 0.75,
        "is_anomaly": False,
    })
"""

import logging
import sys
import json
import socket
import threading
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from contextvars import ContextVar
from functools import wraps
import uuid

# Context variables for request-scoped logging
_request_id_ctx: ContextVar[str] = ContextVar('request_id', default='')
_neighborhood_ctx: ContextVar[str] = ContextVar('neighborhood', default='')
_context_key_ctx: ContextVar[str] = ContextVar('context_key', default='')


class JsonFormatter(logging.Formatter):
    """
    JSON log formatter with standard fields.
    
    Output format:
    {
        "timestamp": "2024-01-15T10:30:00.000Z",
        "level": "INFO",
        "logger": "memstream.scoring",
        "message": "Scoring complete",
        "service": "cadqstream-memstream",
        "environment": "production",
        "host": "taskmanager-1",
        "request_id": "abc-123",
        "neighborhood": "manhattan",
        "context_key": "manhattan_2024-01-15_10",
        "duration_ms": 45.2,
        "extra_field": "value"
    }
    """
    
    # Standard fields always included
    RESERVED_FIELDS = {
        'timestamp', 'level', 'logger', 'message', 'service',
        'environment', 'host', 'request_id', 'neighborhood', 
        'context_key', 'trace_id', 'span_id', 'exception',
    }
    
    def __init__(
        self,
        service_name: str = "cadqstream",
        environment: str = "production",
        host: Optional[str] = None,
        include_stack_trace: bool = True,
        max_message_length: int = 10000,
    ):
        super().__init__()
        self.service_name = service_name
        self.environment = environment
        self.host = host or self._get_hostname()
        self.include_stack_trace = include_stack_trace
        self.max_message_length = max_message_length
    
    def _get_hostname(self) -> str:
        try:
            return socket.gethostname()
        except Exception:
            return "unknown"
    
    def format(self, record: logging.LogRecord) -> str:
        # Build standard fields
        log_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': self._truncate_message(record.getMessage()),
            'service': self.service_name,
            'environment': self.environment,
            'host': self.host,
        }
        
        # Add context variables
        request_id = _request_id_ctx.get()
        if request_id:
            log_data['request_id'] = request_id
        
        neighborhood = _neighborhood_ctx.get()
        if neighborhood:
            log_data['neighborhood'] = neighborhood
        
        context_key = _context_key_ctx.get()
        if context_key:
            log_data['context_key'] = context_key
        
        # Add trace context if available
        try:
            from opentelemetry import trace
            span = trace.get_current_span()
            if span and span.get_span_context().is_valid:
                ctx = span.get_span_context()
                log_data['trace_id'] = format(ctx.trace_id, '032x')
                log_data['span_id'] = format(ctx.span_id, '016x')
        except ImportError:
            pass  # OpenTelemetry not installed
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'created', 'filename',
                           'funcName', 'levelname', 'levelno', 'lineno',
                           'module', 'msecs', 'pathname', 'process',
                           'processName', 'relativeCreated', 'thread',
                           'threadName', 'exc_info', 'exc_text', 'stack_info',
                           'message'):
                
                # Skip reserved fields that we already handle
                if key in self.RESERVED_FIELDS:
                    continue
                
                # Serialize non-standard types
                log_data[key] = self._serialize_value(value)
        
        # Handle exception info
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
            }
            if self.include_stack_trace:
                log_data['exception']['stack_trace'] = traceback.format_exception(
                    *record.exc_info
                )
        
        return json.dumps(log_data, default=str)
    
    def _truncate_message(self, message: str) -> str:
        """Truncate message to max length."""
        if len(message) > self.max_message_length:
            return message[:self.max_message_length] + "...[truncated]"
        return message
    
    def _serialize_value(self, value: Any) -> Any:
        """Serialize a value to JSON-compatible type."""
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        elif isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        elif isinstance(value, set):
            return list(value)
        elif isinstance(value, datetime):
            return value.isoformat()
        else:
            try:
                return str(value)
            except Exception:
                return repr(value)


class StructuredLogger:
    """
    Wrapper around logging.Logger with structured logging helpers.
    
    Usage:
        logger = StructuredLogger(__name__)
        
        # Standard logging
        logger.info("Processing record")
        
        # Structured logging with context
        logger.info("Scoring complete",
            neighborhood="manhattan",
            context_key="manhattan_2024-01-15_10",
            score=0.75,
            latency_ms=45.2,
        )
        
        # With exception
        try:
            risky_operation()
        except Exception as e:
            logger.error("Operation failed",
                error_type=type(e).__name__,
                error_message=str(e),
                exc_info=True,
            )
    """
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        self._logger = logger or logging.getLogger(name)
        self.name = name
    
    def _log(
        self,
        level: int,
        msg: str,
        exc_info: bool = False,
        stack_info: bool = False,
        extra: Optional[Dict] = None,
        **kwargs
    ):
        """Internal log method with structured fields."""
        # Build extra dict from kwargs
        log_extra = extra.copy() if extra else {}
        for key, value in kwargs.items():
            log_extra[key] = value
        
        self._logger.log(level, msg, exc_info=exc_info, stack_info=stack_info, extra=log_extra)
    
    def debug(self, msg: str, **kwargs):
        self._log(logging.DEBUG, msg, **kwargs)
    
    def info(self, msg: str, **kwargs):
        self._log(logging.INFO, msg, **kwargs)
    
    def warning(self, msg: str, **kwargs):
        self._log(logging.WARNING, msg, **kwargs)
    
    def error(self, msg: str, **kwargs):
        self._log(logging.ERROR, msg, **kwargs)
    
    def critical(self, msg: str, **kwargs):
        self._log(logging.CRITICAL, msg, **kwargs)
    
    def exception(self, msg: str, **kwargs):
        kwargs.setdefault('exc_info', True)
        self._log(logging.ERROR, msg, **kwargs)
    
    def log(self, level: int, msg: str, **kwargs):
        self._log(level, msg, **kwargs)


def setup_logging(
    service_name: str = "cadqstream",
    environment: str = "production",
    log_level: str = "INFO",
    json_output: bool = True,
    log_format: Optional[str] = None,
    handlers: Optional[List[logging.Handler]] = None,
) -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        service_name: Name of the service
        environment: Deployment environment
        log_level: Minimum log level
        json_output: Output JSON format (for log aggregation)
        log_format: Custom format string (ignored if json_output=True)
        handlers: Custom handlers to add
    
    Returns:
        Root logger
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    if json_output:
        formatter = JsonFormatter(
            service_name=service_name,
            environment=environment,
        )
    else:
        formatter = logging.Formatter(
            log_format or '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Add custom handlers if provided
    if handlers:
        for handler in handlers:
            if json_output:
                handler.setFormatter(formatter)
            root_logger.addHandler(handler)
    
    # Suppress noisy loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('kafka').setLevel(logging.WARNING)
    logging.getLogger('pyflink').setLevel(logging.INFO)
    
    return root_logger


def get_logger(name: str) -> StructuredLogger:
    """Get a structured logger for a module."""
    return StructuredLogger(name)


def set_request_context(
    request_id: Optional[str] = None,
    neighborhood: Optional[str] = None,
    context_key: Optional[str] = None,
) -> str:
    """
    Set request-scoped logging context.
    
    Returns the request_id (generated if not provided).
    """
    if request_id is None:
        request_id = str(uuid.uuid4())
    
    _request_id_ctx.set(request_id)
    
    if neighborhood:
        _neighborhood_ctx.set(neighborhood)
    
    if context_key:
        _context_key_ctx.set(context_key)
    
    return request_id


def clear_request_context():
    """Clear request-scoped logging context."""
    _request_id_ctx.set('')
    _neighborhood_ctx.set('')
    _context_key_ctx.set('')


class RequestContextLogger:
    """
    Context manager for request-scoped logging.
    
    Usage:
        with RequestContextLogger(neighborhood="manhattan", context_key="mht_10"):
            logger = get_logger(__name__)
            logger.info("Processing batch")  # Auto-includes context
    """
    
    def __init__(
        self,
        request_id: Optional[str] = None,
        neighborhood: Optional[str] = None,
        context_key: Optional[str] = None,
    ):
        self.request_id = request_id
        self.neighborhood = neighborhood
        self.context_key = context_key
        self._token = None
    
    def __enter__(self):
        self._token = set_request_context(
            request_id=self.request_id,
            neighborhood=self.neighborhood,
            context_key=self.context_key,
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        clear_request_context()
        return False


def log_execution_time(logger: StructuredLogger, operation: str):
    """
    Decorator to log function execution time.
    
    Usage:
        @log_execution_time(logger, "model_score")
        def score_one(self, features):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.perf_counter() - start) * 1000
                logger.debug(
                    f"{operation} completed",
                    operation=operation,
                    function=func.__name__,
                    duration_ms=round(duration_ms, 2),
                )
                return result
            except Exception as e:
                duration_ms = (time.perf_counter() - start) * 1000
                logger.error(
                    f"{operation} failed",
                    operation=operation,
                    function=func.__name__,
                    duration_ms=round(duration_ms, 2),
                    error_type=type(e).__name__,
                    exc_info=True,
                )
                raise
        return wrapper
    return decorator
```

---

## Section 6: Integration — Operator Integration Guide

### File: `memstream_src/operators/memstream_with_monitoring.py`

Integration example showing all monitoring components working together:

```python
"""
Integration Example: MemStreamScoringOperator with Full Monitoring.

This shows how to integrate:
1. Prometheus metrics (MemStreamMetrics)
2. OpenTelemetry tracing
3. JSON structured logging
4. SLO burn-rate tracking
5. Checkpoint metrics

Place this code in memstream_scoring_op.py after existing implementation.
"""

from monitoring.metrics import (
    MemStreamMetrics, 
    get_memstream_metrics,
    SLOConfig,
)
from monitoring.tracing import (
    setup_tracing,
    get_tracer,
    MemStreamSpanAttributes,
    traced,
    RequestContextLogger,
)
from monitoring.logging_config import (
    setup_logging,
    get_logger,
    set_request_context,
)
from monitoring.slo import SLOBurnRateTracker


# ─────────────────────────────────────────────────────────────────────────────
# Metrics + Tracing Setup (in job initialization)
# ─────────────────────────────────────────────────────────────────────────────

def setup_monitoring(
    service_name: str = "cadqstream-memstream",
    otlp_endpoint: str = "http://otel-collector:4317",
    environment: str = "production",
):
    """Initialize all monitoring components."""
    
    # Setup JSON structured logging
    setup_logging(
        service_name=service_name,
        environment=environment,
        log_level="INFO",
        json_output=True,
    )
    
    # Setup OpenTelemetry tracing
    setup_tracing(
        service_name=service_name,
        service_version="1.0.0",
        otlp_endpoint=otlp_endpoint,
        environment=environment,
    )
    
    # Create metrics instances
    metrics = get_memstream_metrics()
    metrics.set_model_info(version="1.0.0")
    
    # Create SLO tracker
    slo_config = SLOConfig(
        latency_p99_target_ms=100.0,
        error_rate_target=0.001,
        availability_target=0.999,
    )
    
    neighborhoods = [
        "manhattan", "brooklyn", "queens", 
        "bronx", "staten_island"
    ]
    slo_tracker = SLOBurnRateTracker(slo_config, neighborhoods)
    
    return metrics, slo_tracker


# ─────────────────────────────────────────────────────────────────────────────
# Instrumented Operator Methods
# ─────────────────────────────────────────────────────────────────────────────

class MemStreamScoringOperatorMonitored:
    """MemStreamScoringOperator with full monitoring instrumentation."""
    
    def __init__(self, config: dict, metrics: MemStreamMetrics):
        self.config = config
        self.metrics = metrics
        self.tracer = get_tracer()
        self.logger = get_logger(__name__)
        
        # SLO tracking state
        self._window_errors = 0
        self._window_total = 0
        self._window_anomalies = 0
        self._window_latencies = []
        self._window_start = time.time()
        
        # SLO config
        self._slo_config = SLOConfig(
            latency_p99_target_ms=100.0,
            error_rate_target=0.001,
        )
    
    @traced(span_name="memstream.process_record")
    def process_element_instrumented(self, record: dict, context):
        """Instrumented process_element with full observability."""
        
        # Extract context
        neighborhood = record.get('PULocationID', 'unknown')
        context_key = self._get_context_key(record)
        
        # Set logging context
        with RequestContextLogger(neighborhood=neighborhood, context_key=context_key):
            self.logger.info("Processing record", extra={
                "trip_id": record.get('trip_id', 'unknown'),
            })
            
            # Create scoring span
            with self.tracer.start_as_current_span("memstream_score") as span:
                span.set_attribute(MemStreamSpanAttributes.NEIGHBORHOOD, neighborhood)
                span.set_attribute(MemStreamSpanAttributes.CONTEXT_KEY, context_key)
                
                try:
                    # Feature extraction with tracing
                    with self.tracer.start_as_current_span("extract_features") as feat_span:
                        feat_start = time.perf_counter()
                        features = self.vectorizer.transform(record)
                        feat_duration = (time.perf_counter() - feat_start) * 1000
                        
                        feat_span.set_attribute(
                            MemStreamSpanAttributes.FEATURE_EXTRACTION_MS, 
                            feat_duration
                        )
                        feat_span.set_attribute(
                            MemStreamSpanAttributes.FEATURE_DIM,
                            len(features) if features is not None else 0
                        )
                    
                    if features is None:
                        self.logger.warning("Feature extraction failed", extra={
                            "record_trip_id": record.get('trip_id'),
                        })
                        return None
                    
                    # Scoring with latency tracking
                    with self.metrics.scoring_latency_time(neighborhood, "online"):
                        score, is_anomaly = self._score_with_span(
                            features, record, span
                        )
                    
                    # Update span with results
                    span.set_attribute(MemStreamSpanAttributes.SCORE, float(score))
                    span.set_attribute(MemStreamSpanAttributes.IS_ANOMALY, is_anomaly)
                    span.set_attribute(
                        MemStreamSpanAttributes.SCORE_LATENCY_MS,
                        time.perf_counter() - span.start_time
                    )
                    
                    # Record metrics
                    self._record_scoring_metrics(neighborhood, context_key, score, is_anomaly)
                    
                    self.logger.info("Scoring complete", extra={
                        "score": float(score),
                        "is_anomaly": is_anomaly,
                    })
                    
                    return self._emit_result(record, score, is_anomaly)
                    
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(
                        trace.Status(trace.StatusCode.ERROR), 
                        str(e)
                    )
                    
                    self.logger.exception("Scoring failed", extra={
                        "error_type": type(e).__name__,
                        "neighborhood": neighborhood,
                    })
                    
                    # Record error metrics
                    self.metrics.scoring_errors.labels(
                        error_type=type(e).__name__,
                        neighborhood=neighborhood,
                    ).inc()
                    self.metrics.record_scoring(neighborhood, "error")
                    self._window_errors += 1
                    
                    # Update SLO
                    self._update_slo(neighborhood)
                    
                    raise
    
    def _score_with_span(self, features, record, span):
        """Score with tracing attributes."""
        score, is_anomaly = self.memstream.score_one(features)
        
        # Add memory stats to span
        span.set_attribute(
            MemStreamSpanAttributes.MEMORY_USED_SLOTS,
            self.memstream.memory.used_slots
        )
        span.set_attribute(
            MemStreamSpanAttributes.MEMORY_TOTAL_SLOTS,
            self.memstream.memory.total_slots
        )
        
        return score, is_anomaly
    
    def _record_scoring_metrics(self, neighborhood, context_key, score, is_anomaly):
        """Record all scoring metrics."""
        
        # Basic scoring metrics
        self.metrics.record_scoring(
            neighborhood=neighborhood,
            result="anomaly" if is_anomaly else "normal",
            is_anomaly=is_anomaly,
        )
        
        # Update windowed stats
        self._window_total += 1
        if is_anomaly:
            self._window_anomalies += 1
        
        # Update memory utilization
        if hasattr(self.memstream, 'memory'):
            self.metrics.update_memory_utilization(
                neighborhood,
                context_key,
                self.memstream.memory.used_slots,
                self.memstream.memory.total_slots,
            )
        
        # Update beta threshold
        beta = self.memstream.get_beta(context_key)
        self.metrics.update_beta_threshold(neighborhood, context_key, beta)
        span = trace.get_current_span()
        if span:
            span.set_attribute(MemStreamSpanAttributes.BETA_THRESHOLD, beta)
            span.set_attribute(MemStreamSpanAttributes.BETA_SOURCE, "memory")
        
        # Update rates every 1000 records
        if self._window_total % 1000 == 0:
            self.metrics.update_anomaly_rate(
                neighborhood,
                self._window_anomalies,
                self._window_total
            )
            self.metrics.update_availability(
                neighborhood,
                self._window_total,
                self._window_errors
            )
            
            # Check SLO alerts
            self._check_slo_alerts(neighborhood)
    
    def _update_slo(self, neighborhood: str):
        """Update SLO burn-rate."""
        elapsed = time.time() - self._window_start
        
        if elapsed >= 60:  # Update every minute
            self.slo_tracker.update(
                neighborhood=neighborhood,
                error_count=self._window_errors,
                total_requests=self._window_total,
            )
            
            # Reset window
            self._window_errors = 0
            self._window_total = 0
            self._window_anomalies = 0
            self._window_start = time.time()
    
    def _check_slo_alerts(self, neighborhood: str):
        """Check and publish SLO alerts."""
        alerts = self.slo_tracker.check_alerts()
        
        for alert in alerts:
            if alert.neighborhood == neighborhood:
                self.logger.warning(
                    "SLO alert",
                    extra={
                        "alert_type": "slo_burn_rate",
                        "slo_type": alert.slo_type.value,
                        "burn_rate": alert.burn_rate,
                        "threshold": alert.threshold,
                        "severity": alert.severity,
                        "budget_remaining_seconds": alert.budget_remaining_seconds,
                    }
                )
                
                # Publish alert (e.g., to PagerDuty, Slack, etc.)
                self._publish_alert(alert)
```

---

## Section 7: Prometheus Alert Rules

### File: `deployment/prometheus/rules/memstream-alerts.yaml`

```yaml
# Prometheus alert rules for MemStream monitoring
# Enable these rules AFTER implementing all metrics

groups:
  - name: memstream_scoring
    interval: 30s
    rules:
      # C-MON-1 / H-MON-6: Latency SLO
      - alert: MemStreamLatencyP99High
        expr: |
          histogram_quantile(0.99, rate(memstream_scoring_latency_seconds_bucket[5m])) * 1000 > 100
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "MemStream p99 latency exceeds 100ms"
          description: "P99 latency is {{ $value | printf \"%.2f\" }}ms (threshold: 100ms)"
      
      - alert: MemStreamLatencyP99Critical
        expr: |
          histogram_quantile(0.99, rate(memstream_scoring_latency_seconds_bucket[5m])) * 1000 > 500
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "MemStream p99 latency severely degraded"
          description: "P99 latency is {{ $value | printf \"%.2f\" }}ms (threshold: 500ms)"
      
      # C-MON-3: SLO Burn Rate
      - alert: MemStreamSLOBudgetBurningFast
        expr: |
          memstream_slo_burn_rate > 14.4
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "MemStream SLO budget burning at high rate"
          description: "Burn rate is {{ $value | printf \"%.1f\" }}x (threshold: 14.4x for 1h window)"
      
      # H-MON-6: Availability
      - alert: MemStreamAvailabilityLow
        expr: |
          memstream_availability_ratio < 0.99
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "MemStream availability below 99%"
          description: "Current availability: {{ $value | printf \"%.4f\" }}"
      
      # C-MON-4: Anomaly Rate
      - alert: MemStreamAnomalyRateAnomalouslyLow
        expr: |
          rate(memstream_records_scored_total{scoring_result="anomaly"}[1h])
          / rate(memstream_records_scored_total[1h]) < 0.001
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "MemStream anomaly rate unexpectedly low"
          description: "Anomaly rate is {{ $value | printf \"%.4f\" }}, may indicate model degradation"
      
      - alert: MemStreamAnomalyRateAnomalouslyHigh
        expr: |
          rate(memstream_records_scored_total{scoring_result="anomaly"}[1h])
          / rate(memstream_records_scored_total[1h]) > 0.15
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "MemStream anomaly rate abnormally high"
          description: "Anomaly rate is {{ $value | printf \"%.4f\" }}, may indicate data quality issues"
      
      # C-MON-5: HMAC Failures
      - alert: MemStreamHMACFailuresDetected
        expr: |
          increase(memstream_hmac_failures_total[5m]) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "MemStream HMAC validation failures"
          description: "{{ $value }} HMAC failures in last 5 minutes - possible security issue"
      
      # H-MON-8: Redis Health
      - alert: MemStreamRedisConnectionFailures
        expr: |
          increase(memstream_redis_connection_failures_total[5m]) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "MemStream Redis connection failures"
          description: "{{ $value }} Redis connection failures - beta polling may be affected"
      
      - alert: MemStreamRedisLatencyHigh
        expr: |
          histogram_quantile(0.99, rate(memstream_redis_latency_seconds_bucket[5m])) * 1000 > 50
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "MemStream Redis latency high"
          description: "Redis p99 latency is {{ $value | printf \"%.2f\" }}ms"
      
      # H-MON-9: Checkpoint Health
      - alert: MemStreamCheckpointFailures
        expr: |
          increase(flink_checkpoint_failed_total[1h]) > 3
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "MemStream checkpoint failures detected"
          description: "{{ $value }} checkpoint failures in last hour"
      
      - alert: MemStreamCheckpointDurationHigh
        expr: |
          histogram_quantile(0.99, rate(flink_checkpoint_duration_seconds_bucket[1h])) > 300
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "MemStream checkpoint duration high"
          description: "Checkpoint p99 duration is {{ $value | printf \"%.0f\" }}s"
      
      # Memory Health
      - alert: MemStreamMemoryUtilizationHigh
        expr: |
          memstream_memory_utilization > 0.9
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "MemStream memory utilization high"
          description: "Memory utilization is {{ $value | printf \"%.2f\" }}%"
```

---

## Verification Checklist

After implementing all monitoring fixes, verify each item:

### Prometheus Metrics (C-MON-1, C-MON-4, C-MON-5, H-MON-6, H-MON-7, H-MON-8)

- [ ] `memstream_scoring_latency_seconds` histogram visible in Prometheus
- [ ] `memstream_records_scored_total` counter incrementing with `neighborhood` and `scoring_result` labels
- [ ] `memstream_hmac_failures_total` counter increments on HMAC failures
- [ ] `memstream_anomaly_rate` gauge updates per neighborhood
- [ ] `memstream_total_requests_total` and `memstream_error_count_total` counters exist
- [ ] `memstream_availability_ratio` gauge computes correctly
- [ ] `cadqstream_fpr_estimate` gauge updates if IEC provides labels
- [ ] `memstream_redis_connection_failures_total` counter increments on Redis failures
- [ ] `memstream_redis_latency_seconds` histogram records Redis operation latency
- [ ] `memstream_scoring_errors_total` categorizes errors by type

### OpenTelemetry Tracing (C-MON-2)

- [ ] Traces visible in Jaeger UI with correct span hierarchy
- [ ] Spans include `neighborhood`, `context_key`, `score`, `is_anomaly` attributes
- [ ] Trace context propagates through Kafka message headers
- [ ] Exception stack traces attached to error spans

### SLO Burn-Rate Tracking (C-MON-3)

- [ ] `memstream_slo_error_budget_remaining_seconds` gauge decrements
- [ ] `memstream_slo_burn_rate` gauge updates correctly
- [ ] Burn-rate alerts trigger at 14.4x (1h), 6x (6h), 3x (3d) thresholds

### Checkpoint Metrics (H-MON-9)

- [ ] `flink_checkpoint_started_total` counter increments
- [ ] `flink_checkpoint_completed_total` counter increments on success
- [ ] `flink_checkpoint_failed_total` counter increments on failure
- [ ] `flink_checkpoint_duration_seconds` histogram records durations

### JSON Structured Logging (H-MON-10)

- [ ] Logs are valid JSON in stdout/stderr
- [ ] Each log line includes `timestamp`, `level`, `service`, `environment`
- [ ] `request_id`, `neighborhood`, `context_key` appear in relevant logs
- [ ] Trace IDs (`trace_id`, `span_id`) appear when OTel is active
- [ ] Exception logs include `exception.type`, `exception.message`, `exception.stack_trace`

### Alert Rules

- [ ] All alerts in `memstream-alerts.yaml` evaluate correctly
- [ ] Alerts fire at appropriate thresholds
- [ ] Alert annotations provide actionable information

---

## Dependencies

Required Python packages:

```
prometheus_client>=0.17.0
opentelemetry-api>=1.20.0
opentelemetry-sdk>=1.20.0
opentelemetry-exporter-otlp>=1.20.0
python-json-logger>=2.0.0  # Optional, for compatibility
```

Install:

```bash
pip install prometheus_client opentelemetry-api opentelemetry-sdk \
    opentelemetry-exporter-otlp
```

---

## Files Created

| File | Purpose |
|------|---------|
| `memstream_src/monitoring/__init__.py` | Package init |
| `memstream_src/monitoring/metrics.py` | Prometheus metrics (Section 1) |
| `memstream_src/monitoring/tracing.py` | OpenTelemetry tracing (Section 2) |
| `memstream_src/monitoring/slo.py` | SLO burn-rate tracking (Section 3) |
| `memstream_src/monitoring/checkpoint_listener.py` | Checkpoint metrics (Section 4) |
| `memstream_src/monitoring/logging_config.py` | JSON logging (Section 5) |
| `memstream_src/operators/memstream_with_monitoring.py` | Integration example (Section 6) |
| `deployment/prometheus/rules/memstream-alerts.yaml` | Alert rules (Section 7) |
