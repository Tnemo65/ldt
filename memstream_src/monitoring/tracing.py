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


def traced(
    span_name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
    kind: SpanKind = SpanKind.INTERNAL,
) -> Callable:
    """Decorator to add tracing to a function."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            name = span_name or f"{func.__module__}.{func.__qualname__}"
            with tracer.start_as_current_span(name, kind=kind) as span:
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, value)
                span.set_attribute("function.name", func.__name__)
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise
        return wrapper
    return decorator


class RequestContextLogger:
    """
    Context-aware logger that injects trace context into log records.
    """

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def _inject_context(self, kwargs: dict):
        """Inject trace context into log record."""
        tracer = get_tracer()
        if tracer:
            current_span = trace.get_current_span()
            if current_span and current_span.is_recording():
                ctx = current_span.get_span_context()
                if ctx.is_valid:
                    kwargs.setdefault('extra', {})['trace_id'] = format(ctx.trace_id, '032x')
                    kwargs['extra']['span_id'] = format(ctx.span_id, '016x')
        return kwargs

    def debug(self, msg: str, *args, **kwargs):
        self._logger.debug(msg, *args, **self._inject_context(kwargs))

    def info(self, msg: str, *args, **kwargs):
        self._logger.info(msg, *args, **self._inject_context(kwargs))

    def warning(self, msg: str, *args, **kwargs):
        self._logger.warning(msg, *args, **self._inject_context(kwargs))

    def error(self, msg: str, *args, **kwargs):
        self._logger.error(msg, *args, **self._inject_context(kwargs))

    def critical(self, msg: str, *args, **kwargs):
        self._logger.critical(msg, *args, **self._inject_context(kwargs))


class TracingContext:
    """
    Context manager for creating traced spans with common attributes.
    """
    pass


class MemStreamSpanAttributes:
    """Constants for MemStream span attribute names."""
    NEIGHBORHOOD = "memstream.neighborhood"
    CONTEXT_KEY = "memstream.context_key"
    TRIP_ID = "memstream.trip_id"
    SCORE = "memstream.score"
    IS_ANOMALY = "memstream.is_anomaly"
    SCORING_METHOD = "memstream.scoring_method"
    SCORE_LATENCY_MS = "memstream.score_latency_ms"
    MEMORY_USED_SLOTS = "memstream.memory.used_slots"
    MEMORY_TOTAL_SLOTS = "memstream.memory.total_slots"
    MEMORY_UTILIZATION = "memstream.memory.utilization"
    BETA_THRESHOLD = "memstream.beta_threshold"
    BETA_SOURCE = "memstream.beta_source"
    FEATURE_DIM = "memstream.feature_dim"
    FEATURE_EXTRACTION_MS = "memstream.feature_extraction_ms"
    ERROR_TYPE = "memstream.error_type"
    ERROR_MESSAGE = "memstream.error_message"
    IEC_STRATEGY = "iec.strategy"
    IEC_CONFIDENCE = "iec.confidence"
    DRIFT_DETECTED = "iec.drift_detected"
    DRIFT_SEVERITY = "iec.drift_severity"


def create_scoring_span(
    neighborhood: str,
    context_key: str,
    scoring_method: str = "online",
) -> trace.Span:
    tracer = get_tracer()
    span = tracer.start_span(
        "memstream.score",
        kind=SpanKind.INTERNAL,
    )
    span.set_attribute(MemStreamSpanAttributes.NEIGHBORHOOD, neighborhood)
    span.set_attribute(MemStreamSpanAttributes.CONTEXT_KEY, context_key)
    span.set_attribute(MemStreamSpanAttributes.SCORING_METHOD, scoring_method)
    return span


def inject_trace_context(carrier: Dict) -> Dict:
    _propagator.inject(carrier)
    return carrier


def extract_trace_context(carrier: Dict) -> Context:
    return _propagator.extract(carrier)


def create_linked_span(
    span_name: str,
    trace_id: int,
    span_id: int,
) -> trace.Span:
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
