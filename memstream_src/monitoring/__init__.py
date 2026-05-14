"""
Monitoring module for CA-DQStream + MemStream.

Provides:
- Prometheus metrics (metrics.py)
- OpenTelemetry tracing (tracing.py)
- SLO burn-rate tracking (slo.py)
- Checkpoint metrics (checkpoint_listener.py)
- JSON structured logging (logging_config.py)

Usage:
    from monitoring import setup_monitoring, get_logger, get_tracer, get_memstream_metrics
    
    # In job initialization:
    metrics, slo_tracker = setup_monitoring(
        service_name="cadqstream-memstream",
        environment="production",
    )
    
    # In operators:
    logger = get_logger(__name__)
    logger.info("Processing record", extra={"neighborhood": neighborhood})
"""

from monitoring.metrics import (
    MemStreamMetrics,
    IECMetrics,
    CheckpointMetrics,
    FPRMetrics,
    SLOConfig,
    get_memstream_metrics,
    get_iec_metrics,
    get_checkpoint_metrics,
    get_fpr_metrics,
)

from monitoring.tracing import (
    setup_tracing,
    get_tracer,
    traced,
    RequestContextLogger,
    MemStreamSpanAttributes,
)

from monitoring.logging_config import (
    setup_logging,
    get_logger,
    set_request_context,
    clear_request_context,
)

from monitoring.slo import (
    SLOBurnRateTracker,
    SLOConfig as SLOSettings,
)

__all__ = [
    # Metrics
    'MemStreamMetrics',
    'IECMetrics', 
    'CheckpointMetrics',
    'FPRMetrics',
    'SLOConfig',
    'get_memstream_metrics',
    'get_iec_metrics',
    'get_checkpoint_metrics',
    'get_fpr_metrics',
    # Tracing
    'setup_tracing',
    'get_tracer',
    'traced',
    'RequestContextLogger',
    'MemStreamSpanAttributes',
    # Logging
    'setup_logging',
    'get_logger',
    'set_request_context',
    'clear_request_context',
    # SLO
    'SLOBurnRateTracker',
]
