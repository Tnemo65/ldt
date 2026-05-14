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
import time

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
    handlers: Optional[list] = None,
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
