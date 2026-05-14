"""
IEC Feedback Operator - Layer 4 (Broadcast).

FIXES in v5:
- C-FL-1: Circuit breaker state in BroadcastState
- C-FL-3: Added hashlib, hmac imports
- C-SEC-1: HMAC key enforcement at startup

Uses KeyedBroadcastProcessFunction for IEC beta adjustments.
Receives adjust_beta/action_replay/stream_from_memory/fine_tune_ae from Kafka.
Broadcasts circuit breaker state across all parallel subtasks.

IEC Logic Integration:
- Uses IECController from src.core.iec for:
  - MultiInstanceADWIN (36 instances: 6 neighborhoods x 6 metrics)
  - DriftAggregator for severity assessment
  - StrategyPredictor (METER hypernetwork or fallback rules)
  - StrategyExecutor for adaptation strategies
"""

from pyflink.datastream import KeyedBroadcastProcessFunction
from pyflink.datastream.state import MapStateDescriptor
from pyflink.common.typeinfo import BasicTypeInfo
import pickle
import time
import hashlib
import hmac
import logging
import os
from typing import Dict, Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import IEC logic
from memstream_src.core.iec import IECController, IEC_CONFIG

LOGGER = logging.getLogger('cadqstream-iec-feedback')
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


# =============================================================================
# Configuration
# =============================================================================

IEC_SIGNING_KEY = os.getenv('IEC_SIGNING_KEY')

# C-SEC-1: Fail fast if signing key missing
if not IEC_SIGNING_KEY:
    raise RuntimeError(
        "[IECFeedback] FATAL: IEC_SIGNING_KEY environment variable is required. "
        "Unsigned beta updates are not permitted."
    )
if len(IEC_SIGNING_KEY) < 32:
    raise RuntimeError(
        f"[IECFeedback] FATAL: IEC_SIGNING_KEY must be at least 32 characters. "
        f"Got {len(IEC_SIGNING_KEY)}."
    )

REDIS_CONFIG = {
    'host': os.getenv('REDIS_HOST', 'redis'),
    'port': int(os.getenv('REDIS_PORT', '6379')),
    'db': 0,
    'password': os.getenv('REDIS_PASSWORD'),
    'ssl': os.getenv('REDIS_SSL', 'false').lower() == 'true',
}


class SLOConfig:
    """SLO configuration for IEC operations."""
    def __init__(self):
        self.latency_p99_ms = 100.0
        self.iec_cooldown_seconds = 300.0
        self.iec_max_consecutive = 10


# C-FL-2: Time-bounded Redis polling
BETA_POLL_INTERVAL_SECONDS = 10.0


# =============================================================================
# Broadcast State Descriptors
# =============================================================================

CIRCUIT_BREAKER_STATE_DESC = MapStateDescriptor(
    "iec_circuit_breaker",
    BasicTypeInfo.STRING_TYPE_INFO,
    BasicTypeInfo.STRING_TYPE_INFO
)

BETA_CACHE_STATE_DESC = MapStateDescriptor(
    "iec_beta_cache",
    BasicTypeInfo.STRING_TYPE_INFO,
    BasicTypeInfo.STRING_TYPE_INFO
)


# =============================================================================
# IEC Feedback Operator
# =============================================================================

class IECFeedbackOperator(KeyedBroadcastProcessFunction):
    """IEC Feedback Handler with checkpointable circuit breaker.
    
    C-FL-1 FIX: Circuit breaker state is now stored in BroadcastState.
    
    Now uses IECController from src.core.iec for full IEC logic:
    - MultiInstanceADWIN (36 instances)
    - DriftAggregator
    - StrategyPredictor (METER or fallback)
    - StrategyExecutor
    """
    
    def __init__(self, slo_config: SLOConfig = None, meter_model_path: Optional[str] = None):
        self.slo = slo_config or SLOConfig()
        self.meter_model_path = meter_model_path
        self._redis_client = None
        self._last_redis_poll = 0.0
        self._beta_cache = {}
        
        # IEC Controller for full IEC logic
        self._iec: Optional[IECController] = None
        
        # Action handlers (external actions from Kafka)
        self._action_handlers = {
            'adjust_beta': self._handle_adjust_beta,
            'stream_from_memory': self._handle_stream_from_memory,
            'fine_tune_ae': self._handle_fine_tune_ae,
        }
    
    def open(self, runtime_context):
        """Initialize Redis connection and IEC controller."""
        self._init_redis_client()
        
        # Initialize IEC Controller
        self._iec = IECController(
            meter_model_path=self.meter_model_path,
            redis_client=self._redis_client,
            config=IEC_CONFIG,
        )
        
        LOGGER.info(
            "[IECFeedback] Operator initialized with IEC Controller "
            "(%d ADWIN instances, %d neighborhoods)",
            len(IEC_CONFIG['neighborhoods']) * len(IEC_CONFIG['metrics']),
            len(IEC_CONFIG['neighborhoods'])
        )
    
    def _init_redis_client(self):
        """Initialize Redis connection."""
        try:
            import redis
            self._redis_client = redis.Redis(
                host=REDIS_CONFIG['host'],
                port=REDIS_CONFIG['port'],
                db=REDIS_CONFIG['db'],
                password=REDIS_CONFIG['password'],
                ssl=REDIS_CONFIG['ssl'],
                decode_responses=False,
                socket_timeout=5.0,
                socket_connect_timeout=5.0,
            )
            self._redis_client.ping()
            LOGGER.info(
                "[IECFeedback] Redis connected: %s:%d",
                REDIS_CONFIG['host'], REDIS_CONFIG['port']
            )
        except Exception as e:
            LOGGER.warning(
                "[IECFeedback] Redis connection failed: %s - will retry",
                e
            )
            self._redis_client = None
    
    # =========================================================================
    # Circuit Breaker (C-FL-1 FIX)
    # =========================================================================
    
    def _get_circuit_breaker_state(self, ctx) -> Tuple[float, int]:
        """Read circuit breaker state from BroadcastState."""
        cb_state = ctx.get_broadcast_state(CIRCUIT_BREAKER_STATE_DESC)
        
        last_time_str = cb_state.get('last_action_time', '0')
        consecutive_str = cb_state.get('consecutive_actions', '0')
        
        return float(last_time_str), int(consecutive_str)
    
    def _update_circuit_breaker_state(self, ctx, last_action_time: float, consecutive: int):
        """Update circuit breaker state in BroadcastState."""
        cb_state = ctx.get_broadcast_state(CIRCUIT_BREAKER_STATE_DESC)
        cb_state.put('last_action_time', str(last_action_time))
        cb_state.put('consecutive_actions', str(consecutive))
    
    def _check_circuit_breaker(self, ctx) -> Optional[str]:
        """Check if circuit breaker allows action."""
        last_time, consecutive = self._get_circuit_breaker_state(ctx)
        now = time.time()
        
        if now - last_time < self.slo.iec_cooldown_seconds:
            remaining = self.slo.iec_cooldown_seconds - (now - last_time)
            LOGGER.warning(
                "[IECFeedback] Circuit breaker: cooldown (%.1fs remaining)",
                remaining
            )
            return "cooldown_active"
        
        if consecutive >= self.slo.iec_max_consecutive:
            LOGGER.error(
                "[IECFeedback] CIRCUIT BREAKER TRIPPED - human review required!"
            )
            return "circuit_breaker_tripped"
        
        return None
    
    # =========================================================================
    # Action Handlers
    # =========================================================================
    
    def _handle_adjust_beta(self, action: Dict) -> Dict:
        """Handle beta adjustment action."""
        neighborhood = action.get('neighborhood', 'global')
        new_beta = action.get('beta_value')
        
        if new_beta is None:
            return {'status': 'error', 'message': 'missing beta_value'}
        
        try:
            client = self._get_redis_client()
            if client:
                beta_str = f"{new_beta:.6f}"
                sig = hmac.new(
                    IEC_SIGNING_KEY.encode(),
                    beta_str.encode(),
                    hashlib.sha256
                ).hexdigest()
                
                client.set(f'beta:{neighborhood}', f"{beta_str}:{sig}")
                self._beta_cache[neighborhood] = new_beta
                
                LOGGER.info(
                    "[IECFeedback] Beta updated: %s = %.4f",
                    neighborhood, new_beta
                )
                return {'status': 'ok', 'neighborhood': neighborhood, 'beta': new_beta}
            else:
                return {'status': 'error', 'message': 'Redis unavailable'}
        except Exception as e:
            LOGGER.error("[IECFeedback] Failed to update beta: %s", e)
            return {'status': 'error', 'message': str(e)}
    
    def _handle_stream_from_memory(self, action: Dict) -> Dict:
        """Handle stream_from_memory action."""
        neighborhood = action.get('neighborhood', 'global')
        LOGGER.info("[IECFeedback] Stream from memory: %s", neighborhood)
        return {'status': 'ok', 'action': 'stream_from_memory'}
    
    def _handle_fine_tune_ae(self, action: Dict) -> Dict:
        """Handle fine_tune_ae action."""
        neighborhood = action.get('neighborhood', 'global')
        learning_rate = action.get('learning_rate', 0.001)
        LOGGER.info(
            "[IECFeedback] Fine-tune AE: %s (lr=%.6f)",
            neighborhood, learning_rate
        )
        return {'status': 'ok', 'action': 'fine_tune_ae'}
    
    def _get_redis_client(self):
        if self._redis_client is None:
            self._init_redis_client()
        return self._redis_client
    
    # =========================================================================
    # IEC Meta-Metrics Processing (from Layer 3)
    # =========================================================================
    
    def process_meta_metrics(self, neighborhood: str, metrics: Dict) -> Dict:
        """
        Process meta-metrics from Layer 3 and return IEC decision.
        
        This is called when meta-metrics arrive from the scoring layer.
        It uses IECController for drift detection and strategy prediction.
        
        Args:
            neighborhood: Neighborhood name (e.g., 'manhattan')
            metrics: Dict with keys:
                - volume: Trip volume
                - null_rate: Null rate
                - violation_rate: Canary violation rate
                - anomaly_rate: ML anomaly rate
                - avg_anomaly_score: Average anomaly score
                - delta_score: |violation - anomaly| / sum
        
        Returns:
            IEC decision dict:
            {
                'neighborhood': str,
                'strategy': str,
                'confidence': float,
                'severity': str,
                'action_result': dict,
                ...
            }
        """
        if self._iec is None:
            LOGGER.warning("[IECFeedback] IECController not initialized")
            return {'status': 'error', 'message': 'IEC not initialized'}
        
        # Process through IEC controller
        decision = self._iec.process_meta_metrics(neighborhood, metrics)
        
        # Log the decision
        LOGGER.info(
            "[IECFeedback] Meta-metrics processed: neighborhood=%s, strategy=%s, "
            "severity=%s, confidence=%.2f, drifts=%d",
            neighborhood,
            decision.get('strategy', 'unknown'),
            decision.get('severity', 'unknown'),
            decision.get('confidence', 0.0),
            decision.get('n_drift_events', 0)
        )
        
        return decision
    
    def get_iec_stats(self) -> Dict:
        """Get IEC statistics."""
        if self._iec is None:
            return {'status': 'error', 'message': 'IEC not initialized'}
        return self._iec.get_stats()
    
    # =========================================================================
    # KeyedBroadcastProcessFunction Implementation
    # =========================================================================
    
    def process_broadcast_element(self, action: Dict, ctx, broadcaster):
        """Process broadcast element (IEC action).
        
        C-FL-1 FIX: Circuit breaker state is now in BroadcastState.
        """
        action_type = action.get('type', 'unknown')
        
        if action_type not in self._action_handlers:
            LOGGER.warning("[IECFeedback] Unknown action type: %s", action_type)
            return
        
        # Check circuit breaker from BroadcastState
        block_reason = self._check_circuit_breaker(ctx)
        if block_reason:
            LOGGER.warning(
                "[IECFeedback] Action %s blocked: %s",
                action_type, block_reason
            )
            return
        
        # Execute action
        handler = self._action_handlers[action_type]
        result = handler(action)
        
        # Update circuit breaker in BroadcastState
        now = time.time()
        last_time, consecutive = self._get_circuit_breaker_state(ctx)
        self._update_circuit_breaker_state(ctx, now, consecutive + 1)
        
        LOGGER.info(
            "[IECFeedback] Action %s: %s (consecutive: %d)",
            action_type, result.get('status', 'unknown'), consecutive + 1
        )
    
    def process_element(self, record: Dict, ctx, broadcaster):
        """Pass through main data stream."""
        yield record
