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
    with self.metrics.scoring_latency_time(neighborhood, scoring_method):
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
