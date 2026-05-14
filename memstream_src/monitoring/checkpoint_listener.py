"""
Flink Checkpoint Metrics via CheckpointListener interface.

Implements CheckpointListener to capture:
- Checkpoint start/complete/failure events
- Checkpoint duration and size
- Alignment time for exactly-once semantics
- State backend size tracking

Usage:
    from monitoring.checkpoint_listener import CheckpointMetricsListener, configure_checkpointing
    
    # In Flink job:
    env.get_checkpoint_config().set_externalized_checkpoint_cleanup(
        ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION
    )
    
    # Add listener to operators:
    operator = MemStreamScoringOperator()
    operator = CheckpointMetricsListener(operator, metrics)
"""

from typing import Optional, Dict, Any, List
import time
import threading
import logging

logger = logging.getLogger(__name__)


class CheckpointMetricsListener:
    """
    Flink CheckpointListener that records checkpoint metrics.
    
    This is a base implementation that can be extended for specific Flink versions.
    The actual CheckpointListener interface implementation depends on PyFlink's API.
    
    In production, use this pattern:
    
    class CheckpointMetricsListener(CheckpointListener):
        def __init__(self, wrapped_operator, checkpoint_metrics):
            self._wrapped = wrapped_operator
            self._metrics = checkpoint_metrics
            self._pending_checkpoints = {}
            self._lock = threading.Lock()
        
        def notify_checkpoint_complete(self, checkpoint_id, checkpoint_metadata=None):
            with self._lock:
                start_time = self._pending_checkpoints.pop(checkpoint_id, None)
                if start_time is not None:
                    duration = time.time() - start_time
                    size_bytes = getattr(checkpoint_metadata, 'state_size', 0) if checkpoint_metadata else 0
                    self._metrics.record_checkpoint_complete(duration, size_bytes)
        
        def notify_checkpoint_aborted(self, checkpoint_id, checkpoint_metadata=None):
            with self._lock:
                self._pending_checkpoints.pop(checkpoint_id, None)
                self._metrics.record_checkpoint_failure("aborted")
        
        # Delegate all other methods to wrapped operator
        def open(self, runtime_context):
            if hasattr(self._wrapped, 'open'):
                return self._wrapped.open(runtime_context)
        
        def process_element(self, *args, **kwargs):
            return self._wrapped.process_element(*args, **kwargs)
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
        self._pending_checkpoints: Dict[int, float] = {}
        self._completed_checkpoints: List[Dict] = []
        self._lock = threading.Lock()
    
    def notify_checkpoint_complete(
        self, 
        checkpoint_id: int,
        checkpoint_metadata: Optional[Any] = None,
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
        checkpoint_metadata: Optional[Any] = None,
    ):
        """Called when a checkpoint is aborted."""
        with self._lock:
            self._pending_checkpoints.pop(checkpoint_id, None)
            self._metrics.record_checkpoint_failure(reason="aborted")
    
    def record_checkpoint_start(self, checkpoint_id: int):
        """Record that a checkpoint was initiated."""
        with self._lock:
            self._pending_checkpoints[checkpoint_id] = time.time()
            self._metrics.record_checkpoint_start()
    
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


def configure_checkpointing(env, config: Dict[str, Any]):
    """
    Configure checkpointing with proper settings.
    
    Args:
        env: Flink StreamExecutionEnvironment
        config: Checkpoint configuration dict
    """
    try:
        from pyflink.datastream.checkpoint_config import (
            CheckpointConfig,
            ExternalizedCheckpointCleanup,
            CheckpointingMode,
        )
    except ImportError:
        logger.warning("PyFlink checkpoint config not available")
        return
    
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
