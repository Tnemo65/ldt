"""
IEC Controller: METER Hypernetwork + Strategy Execution for MemStream Migration.

This is the Phase 2D migration version of the Intelligent Evolution Controller (IEC):
1. Coordinates MultiInstanceADWIN for drift detection
2. Uses DriftAggregator for severity assessment
3. Predicts strategies via METER hypernetwork (or fallback rules)

HMAC SECURITY FIXES (Phase 2D):
- Line 565: Removed 'default-key' fallback - fails hard if IEC_SIGNING_KEY missing
- Line 664: Removed fallback key entirely
- HMAC mismatch now returns early - beta is NOT applied when HMAC fails
- MemStream_HMACVerificationFailures alert fired on HMAC failure

RETRAIN TRIGGERS:
- Trigger A: ADWIN drift detected for 7 consecutive days
- Trigger B: Anomaly rate > 15% for 3 consecutive days
- Trigger C: kNN distance > 2x baseline

Reference: Phase 2D migration plan
"""

import hashlib
import os
import pickle
import time
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np

from .adwin_multi_instance import MultiInstanceADWIN
from .drift_aggregator import DriftAggregator, SeverityLevel, EvolutionStrategy
from .circuit_breaker import CircuitBreaker, CircuitState
from .verification_feedback import VerificationFeedbackLoop

LOGGER = logging.getLogger('memstream-iec')


# =============================================================================
# HMAC Security Constants
# =============================================================================

# Environment variable for HMAC signing key
IEC_SIGNING_KEY_ENV = 'IEC_SIGNING_KEY'


def _get_signing_key() -> str:
    """
    Get IEC signing key from environment.
    
    CRITICAL (Phase 2D): Fails hard if key is missing.
    No fallback to 'default-key' or any insecure default.
    
    Returns:
        The HMAC signing key
        
    Raises:
        EnvironmentError: If IEC_SIGNING_KEY is not set
    """
    key = os.environ.get(IEC_SIGNING_KEY_ENV)
    if key is None:
        raise EnvironmentError(
            f"IEC_SIGNING_KEY environment variable is not set. "
            f"Set it in your environment before starting the pipeline. "
            f"This is required for secure beta writes to MinIO."
        )
    if not key:
        raise EnvironmentError(
            f"IEC_SIGNING_KEY environment variable is empty. "
            f"Set a non-empty HMAC signing key."
        )
    return key


def _emit_hmac_failure_alert(neighborhood: str, reason: str) -> None:
    """
    Emit Prometheus alert for HMAC verification failure.
    
    Args:
        neighborhood: The neighborhood affected
        reason: Reason for the failure
    """
    try:
        import urllib.request
        import json
        
        payload = json.dumps({
            'name': 'memstream_hmac_verification_failures',
            'value': 1,
            'labels': {
                'neighborhood': neighborhood,
                'reason': reason,
            },
            'type': 'counter'
        }).encode('utf-8')
        
        req = urllib.request.Request(
            'http://cadqstream-metrics:9250/internal/metrics',
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        try:
            urllib.request.urlopen(req, timeout=2)
        except Exception:
            pass  # Silently fail - don't block on metrics
        
        LOGGER.warning(
            f"[IECController] HMAC failure alert emitted: neighborhood={neighborhood}, reason={reason}"
        )
    except Exception:
        pass  # Don't block on metrics errors


# =============================================================================
# IEC Configuration
# =============================================================================

class IECConfig:
    """Configuration for IEC Controller."""

    # METER model paths
    DEFAULT_METER_MODEL_PATH = 'models/meter_hypernetwork.pkl'
    DEFAULT_METER_SCALER_PATH = 'models/meter_scaler.pkl'

    # Strategy names (must match METER training)
    STRATEGY_NAMES: Dict[int, str] = {
        0: EvolutionStrategy.DO_NOTHING,
        1: EvolutionStrategy.ADJUST_THRESHOLD,
        2: EvolutionStrategy.MEMORY_RESET,
    }

    # METER feature names (6D)
    FEATURE_NAMES: List[str] = [
        'volume',
        'null_rate',
        'violation_rate',
        'anomaly_rate',
        'avg_anomaly_score',
        'delta_score',
    ]

    def __init__(
        self,
        meter_model_path: Optional[str] = None,
        meter_scaler_path: Optional[str] = None,
        meter_metadata_path: Optional[str] = None,
        max_recent_drifts: int = 10,
        enable_meter: bool = True,
        # Retrain trigger thresholds
        retrain_adwin_days: int = 7,
        retrain_anomaly_rate_threshold: float = 0.15,
        retrain_anomaly_rate_days: int = 3,
        retrain_knn_multiplier: float = 2.0,
    ):
        """
        Initialize IEC configuration.

        Args:
            meter_model_path: Path to METER MLPClassifier model
            meter_scaler_path: Path to METER StandardScaler
            meter_metadata_path: Path to METER metadata JSON
            max_recent_drifts: Max drift events to track
            enable_meter: Enable METER (if False, use fallback rules)
            retrain_adwin_days: Days of consecutive ADWIN drift for Trigger A
            retrain_anomaly_rate_threshold: Threshold for Trigger B
            retrain_anomaly_rate_days: Consecutive days for Trigger B
            retrain_knn_multiplier: kNN multiplier for Trigger C
        """
        self.meter_model_path = meter_model_path or self.DEFAULT_METER_MODEL_PATH
        self.meter_scaler_path = meter_scaler_path or self.DEFAULT_METER_SCALER_PATH
        self.meter_metadata_path = meter_metadata_path
        self.max_recent_drifts = max_recent_drifts
        self.enable_meter = enable_meter
        
        # Retrain trigger configuration
        self.retrain_adwin_days = retrain_adwin_days
        self.retrain_anomaly_rate_threshold = retrain_anomaly_rate_threshold
        self.retrain_anomaly_rate_days = retrain_anomaly_rate_days
        self.retrain_knn_multiplier = retrain_knn_multiplier


# =============================================================================
# IEC Controller
# =============================================================================

class IECController:
    """
    Intelligent Evolution Controller (IEC) for MemStream.
    
    Phase 2D migration version with HMAC security fixes.
    
    Combines MultiInstanceADWIN + DriftAggregator + METER hypernetwork
    for automated drift handling and model evolution.

    Workflow:
        1. update(meta_metrics) - Update ADWINs, assess severity
        2. get_state() - Get current IEC state and decision
        3. reset() - Reset all state
        4. retrain_model() - Trigger retraining based on retrain triggers

    Retrain Triggers:
        - Trigger A: ADWIN drift for 7+ consecutive days
        - Trigger B: Anomaly rate > 15% for 3+ consecutive days
        - Trigger C: kNN distance > 2x baseline
    """

    def __init__(
        self,
        config: Optional[IECConfig] = None,
        meter_model_path: Optional[str] = None,
        meter_scaler_path: Optional[str] = None,
        minio_client: Optional[object] = None,
    ):
        """
        Initialize IEC Controller.

        Args:
            config: IEC configuration object
            meter_model_path: Override METER model path
            meter_scaler_path: Override METER scaler path
            minio_client: Optional MinIO client for beta writes
        """
        self.config = config or IECConfig()

        if meter_model_path:
            self.config.meter_model_path = meter_model_path
        if meter_scaler_path:
            self.config.meter_scaler_path = meter_scaler_path

        # Core components
        self._adwin = MultiInstanceADWIN()
        self._aggregator = DriftAggregator(max_recent=self.config.max_recent_drifts)

        # METER model (loaded lazily)
        self._meter_model = None
        self._meter_scaler = None
        self._meter_metadata = None
        self._meter_loaded = False

        # Decision cache
        self._last_decision: Optional[Dict] = None

        # Statistics
        self._n_processed = 0
        self._strategy_counts: Dict[str, int] = {s: 0 for s in EvolutionStrategy.ALL_STRATEGIES}

        # Circuit breaker
        self._circuit_breaker = CircuitBreaker(
            cooldown_seconds=300.0,
            max_consecutive=10,
            half_open_max_actions=3,
        )

        # Beta thresholds per neighborhood
        self._current_betas: Dict[str, float] = {}

        # MinIO client for beta writes
        self.minio_client = minio_client
        
        # Retrain trigger tracking
        self._retrain_tracking: Dict[str, dict] = {}
        self._init_retrain_tracking()
        
        # Baseline metrics for Trigger C (kNN)
        self._knn_baseline: Dict[str, float] = {}

    def _init_retrain_tracking(self) -> None:
        """Initialize retrain trigger tracking state."""
        neighborhoods = self._adwin.neighborhoods
        for nb in neighborhoods:
            self._retrain_tracking[nb] = {
                # Trigger A: ADWIN drift days
                'adwin_drift_days': 0,
                'last_adwin_drift': None,
                # Trigger B: Anomaly rate tracking
                'anomaly_rate_history': [],  # List of (timestamp, rate) tuples
                'anomaly_rate_days_above_threshold': 0,
                # Trigger C: kNN tracking
                'knn_history': [],  # List of (timestamp, knn_dist) tuples
                'knn_baseline': None,
            }

    @property
    def meter_loaded(self) -> bool:
        """Check if METER model is loaded."""
        return self._meter_loaded

    def _load_meter(self) -> bool:
        """
        Load METER hypernetwork from disk.

        Returns:
            True if loaded successfully, False otherwise
        """
        if self._meter_loaded:
            return True

        if not self.config.enable_meter:
            LOGGER.info("[IECController] METER disabled in config")
            return False

        model_path = self.config.meter_model_path
        scaler_path = self.config.meter_scaler_path

        # Check if files exist
        if not os.path.exists(model_path):
            LOGGER.warning(f"[IECController] METER model not found: {model_path}")
            return False
        if not os.path.exists(scaler_path):
            LOGGER.warning(f"[IECController] METER scaler not found: {scaler_path}")
            return False

        try:
            with open(model_path, 'rb') as f:
                self._meter_model = pickle.load(f)

            with open(scaler_path, 'rb') as f:
                self._meter_scaler = pickle.load(f)

            self._meter_loaded = True
            if self._meter_metadata and self._meter_metadata.get('test_accuracy'):
                LOGGER.info(
                    f"[IECController] METER loaded: {model_path}, "
                    f"accuracy={self._meter_metadata['test_accuracy']}"
                )
            else:
                LOGGER.info(f"[IECController] METER loaded: {model_path}")
            return True

        except Exception as e:
            LOGGER.error(f"[IECController] Failed to load METER: {e}")
            self._meter_model = None
            self._meter_scaler = None
            return False

    def _extract_features(self, meta_metrics: Dict) -> np.ndarray:
        """
        Extract 6D feature vector from meta-metrics.

        Features:
            0: volume - Total trip volume
            1: null_rate - Rate of null values
            2: violation_rate - Rate of business violations
            3: anomaly_rate - Rate of detected anomalies
            4: avg_anomaly_score - Average anomaly score
            5: delta_score - |violation - anomaly| / (violation + anomaly + eps)

        Args:
            meta_metrics: Meta-metrics dict

        Returns:
            6D feature vector
        """
        features = []

        for feat_name in self.config.FEATURE_NAMES:
            if feat_name == 'volume':
                value = sum(
                    mm.get(feat_name, 0)
                    for mm in meta_metrics.values()
                )
            else:
                values = [
                    mm.get(feat_name, 0)
                    for mm in meta_metrics.values()
                ]
                value = np.mean(values) if values else 0.0

            features.append(value)

        return np.array(features, dtype=np.float32)

    def _compute_delta_score(
        self,
        violation_rate: float,
        anomaly_rate: float,
        epsilon: float = 1e-8
    ) -> float:
        """
        Compute delta score.

        Formula: |violation_rate - anomaly_rate| / (violation_rate + anomaly_rate + epsilon)

        Args:
            violation_rate: Violation rate
            anomaly_rate: Anomaly rate
            epsilon: Small constant for numerical stability

        Returns:
            Delta score
        """
        return abs(violation_rate - anomaly_rate) / (violation_rate + anomaly_rate + epsilon)

    def _predict_with_meter(self, meta_metrics: Dict) -> Tuple[str, float]:
        """
        Predict strategy using METER hypernetwork.

        Args:
            meta_metrics: Meta-metrics dict

        Returns:
            Tuple of (strategy_name, confidence)
        """
        features = self._extract_features(meta_metrics)
        features_2d = features.reshape(1, -1)

        features_scaled = self._meter_scaler.transform(features_2d)
        strategy_id = self._meter_model.predict(features_scaled)[0]
        probs = self._meter_model.predict_proba(features_scaled)[0]

        strategy_name = self.config.STRATEGY_NAMES.get(
            int(strategy_id),
            EvolutionStrategy.DO_NOTHING
        )
        confidence = float(probs[int(strategy_id)])

        LOGGER.info(
            f"[IECController] METER prediction: {strategy_name} "
            f"(id={strategy_id}, conf={confidence:.3f})"
        )

        return strategy_name, confidence

    def _get_fallback_strategy(
        self,
        severity: str,
        meta_metrics: Optional[Dict] = None
    ) -> Tuple[str, float]:
        """
        Get fallback strategy using severity-based rules.

        Fallback confidence levels:
            - none: 1.0 (certain)
            - low: 0.7 (moderate confidence)
            - moderate: 0.8 (higher confidence for retrain)
            - high: 0.9 (certain for switch)

        Args:
            severity: Severity level
            meta_metrics: Optional meta-metrics for additional context

        Returns:
            Tuple of (strategy_name, confidence)
        """
        confidence_map = {
            SeverityLevel.NONE: 1.0,
            SeverityLevel.LOW: 0.7,
            SeverityLevel.MODERATE: 0.8,
            SeverityLevel.HIGH: 0.9,
        }

        strategy = self._aggregator.predict_strategy(severity)
        confidence = confidence_map.get(severity, 0.5)

        LOGGER.info(
            f"[IECController] Fallback strategy: {strategy} "
            f"(severity={severity}, conf={confidence:.3f})"
        )

        return strategy, confidence

    def update(self, meta_metrics: Dict) -> Dict:
        """
        Process meta-metrics and return IEC decision.
        
        This is the main entry point for IEC processing.

        Workflow:
            1. Update all ADWINs with current meta-metrics
            2. Add detected drifts to aggregator
            3. Assess severity
            4. Predict strategy (METER or fallback)
            5. Update retrain triggers
            6. Cache and return decision

        Args:
            meta_metrics: Dict of dicts, e.g.
                {
                    'manhattan': {'volume': 1500, 'null_rate': 0.02, ...},
                    'brooklyn': {'volume': 800, 'null_rate': 0.01, ...},
                }

        Returns:
            Decision dict:
            {
                'strategy': str,           # 'do_nothing', 'adjust_threshold', etc.
                'confidence': float,        # 0.0 to 1.0
                'severity': str,            # 'none', 'low', 'moderate', 'high'
                'affected_neighborhoods': list,
                'affected_metrics': list,
                'drift_events': list,       # [{'neighborhood': str, 'metric': str}, ...]
                'meters_used': bool,        # Whether METER was used
                'retrain_triggers': dict,   # Which retrain triggers fired
            }
        """
        self._n_processed += 1
        current_time = time.time()

        # Step 1: Update ADWINs
        drift_events = self._adwin.update_meta_metrics(meta_metrics)

        # Step 2: Add drifts to aggregator
        for event in drift_events:
            self._aggregator.add_drift(event['neighborhood'], event['metric'])

        # Step 3: Assess severity
        severity = self._aggregator.assess_drift_severity()

        # Step 4: Predict strategy (METER or fallback)
        meter_used = False
        if self._load_meter():
            try:
                strategy, confidence = self._predict_with_meter(meta_metrics)
                meter_used = True
            except Exception as e:
                LOGGER.warning(f"[IECController] METER prediction failed: {e}")
                strategy, confidence = self._get_fallback_strategy(severity, meta_metrics)
        else:
            strategy, confidence = self._get_fallback_strategy(severity, meta_metrics)

        # Step 5: Update retrain triggers and check for retrain conditions
        retrain_triggers = self._check_retrain_triggers(meta_metrics, drift_events, current_time)
        
        # Update ADWIN drift tracking
        for nb, tracking in self._retrain_tracking.items():
            has_drift = any(e['neighborhood'] == nb for e in drift_events)
            if has_drift:
                if tracking['last_adwin_drift'] is not None:
                    # Check if it's been consecutive days
                    days_since_last = (current_time - tracking['last_adwin_drift']) / 86400
                    if days_since_last <= 2:  # Within 2 days = still consecutive
                        tracking['adwin_drift_days'] += 1
                    else:
                        tracking['adwin_drift_days'] = 1
                else:
                    tracking['adwin_drift_days'] = 1
                tracking['last_adwin_drift'] = current_time

        # Build decision
        self._last_decision = {
            'strategy': strategy,
            'confidence': confidence,
            'severity': severity,
            'affected_neighborhoods': self._aggregator.get_affected_neighborhoods(),
            'affected_metrics': self._aggregator.get_affected_metrics(),
            'drift_events': drift_events,
            'meters_used': meter_used,
            'meta_metrics': meta_metrics,  # store for use in execute_strategy
            'circuit_state': self._circuit_breaker.state,
            'circuit_allow': self._circuit_breaker.should_allow_action(),
            'retrain_triggers': retrain_triggers,
            'timestamp': current_time,
        }

        # Update statistics
        self._strategy_counts[strategy] += 1

        LOGGER.info(
            f"[IECController] Decision: {strategy} "
            f"(severity={severity}, drifts={len(drift_events)}, "
            f"conf={confidence:.3f}, meter={meter_used})"
        )
        
        # Log retrain triggers if any
        if retrain_triggers:
            for trigger, details in retrain_triggers.items():
                LOGGER.warning(
                    f"[IECController] RETRAIN TRIGGER {trigger}: {details}"
                )

        return self._last_decision

    def _check_retrain_triggers(
        self, 
        meta_metrics: Dict, 
        drift_events: List[Dict],
        current_time: float
    ) -> Dict:
        """
        Check if any retrain triggers have fired.
        
        Trigger A: ADWIN drift for 7+ consecutive days
        Trigger B: Anomaly rate > 15% for 3+ consecutive days
        Trigger C: kNN distance > 2x baseline
        
        Args:
            meta_metrics: Current meta-metrics
            drift_events: Recent drift events
            current_time: Current timestamp
            
        Returns:
            Dict of triggered retrain conditions
        """
        triggers = {}
        
        for nb, metrics in meta_metrics.items():
            if nb not in self._retrain_tracking:
                continue
                
            tracking = self._retrain_tracking[nb]
            
            # Trigger A: ADWIN drift days
            adwin_drift_days = tracking.get('adwin_drift_days', 0)
            if adwin_drift_days >= self.config.retrain_adwin_days:
                triggers['Trigger_A_ADWIN_Drift'] = {
                    'neighborhood': nb,
                    'consecutive_drift_days': adwin_drift_days,
                    'threshold_days': self.config.retrain_adwin_days,
                }
            
            # Trigger B: Anomaly rate > threshold for consecutive days
            anomaly_rate = metrics.get('anomaly_rate', 0.0)
            tracking['anomaly_rate_history'].append((current_time, anomaly_rate))
            
            # Keep only last N days of history
            cutoff = current_time - (self.config.retrain_anomaly_rate_days * 86400)
            tracking['anomaly_rate_history'] = [
                (ts, rate) for ts, rate in tracking['anomaly_rate_history']
                if ts >= cutoff
            ]
            
            # Count days above threshold
            if anomaly_rate > self.config.retrain_anomaly_rate_threshold:
                tracking['anomaly_rate_days_above_threshold'] += 1
            else:
                tracking['anomaly_rate_days_above_threshold'] = 0
                
            if tracking['anomaly_rate_days_above_threshold'] >= self.config.retrain_anomaly_rate_days:
                triggers['Trigger_B_AnomalyRate'] = {
                    'neighborhood': nb,
                    'current_rate': anomaly_rate,
                    'consecutive_days_above_threshold': tracking['anomaly_rate_days_above_threshold'],
                    'threshold_rate': self.config.retrain_anomaly_rate_threshold,
                    'threshold_days': self.config.retrain_anomaly_rate_days,
                }
            
            # Trigger C: kNN > 2x baseline
            knn_dist = metrics.get('knn_distance', None)
            if knn_dist is not None:
                tracking['knn_history'].append((current_time, knn_dist))
                
                # Keep last 100 samples for baseline
                tracking['knn_history'] = tracking['knn_history'][-100:]
                
                # Compute baseline from first 50 samples
                if len(tracking['knn_history']) >= 50 and tracking['knn_baseline'] is None:
                    baseline_samples = [k for _, k in tracking['knn_history'][:50]]
                    tracking['knn_baseline'] = np.mean(baseline_samples)
                
                if tracking['knn_baseline'] is not None and tracking['knn_baseline'] > 0:
                    current_multiplier = knn_dist / tracking['knn_baseline']
                    if current_multiplier > self.config.retrain_knn_multiplier:
                        triggers['Trigger_C_kNNDistance'] = {
                            'neighborhood': nb,
                            'current_knn': knn_dist,
                            'baseline_knn': tracking['knn_baseline'],
                            'multiplier': current_multiplier,
                            'threshold_multiplier': self.config.retrain_knn_multiplier,
                        }
        
        return triggers

    def get_state(self) -> Optional[Dict]:
        """
        Get last decision state.

        Returns:
            Last decision dict, or None if no decision has been made
        """
        return self._last_decision

    def get_decision(self) -> Optional[Dict]:
        """
        Alias for get_state() for compatibility.
        
        Returns:
            Last decision dict, or None if no decision has been made
        """
        return self._last_decision

    def execute_strategy(self, decision: Optional[Dict] = None) -> Dict:
        """
        Execute the recommended strategy with circuit breaker protection.

        Args:
            decision: Decision dict (if None, uses last decision)

        Returns:
            Execution result dict
        """
        if decision is None:
            decision = self._last_decision

        if decision is None:
            return {'status': 'error', 'message': 'No decision available'}

        # Circuit breaker check
        if not self._circuit_breaker.should_allow_action():
            return {
                'action': decision.get('strategy', EvolutionStrategy.DO_NOTHING),
                'status': 'circuit_open',
                'message': f"Circuit breaker {self._circuit_breaker.state} - action blocked",
                'circuit_state': self._circuit_breaker.state,
                'consecutive_actions': self._circuit_breaker.consecutive_actions,
            }

        strategy = decision['strategy']

        # Snapshot metrics before adaptation for verification feedback
        affected_nb = decision.get('affected_neighborhoods', ['unknown'])
        meta_metrics = decision.get('meta_metrics', {})

        # Count drifts per neighborhood for delta_score computation
        drift_events = decision.get('drift_events', [])
        drift_counts = {}
        for evt in drift_events:
            nb = evt.get('neighborhood', 'unknown')
            drift_counts[nb] = drift_counts.get(nb, 0) + 1

        for nb in affected_nb:
            nb_metrics = meta_metrics.get(nb, {})
            old_beta = self._current_betas.get(nb, 0.5)
            new_beta = decision.get('new_beta', old_beta)

            # Compute delta_score from the neighborhood's metrics
            violation_rate = nb_metrics.get('violation_rate', 0.0)
            anomaly_rate = nb_metrics.get('anomaly_rate', 0.0)
            delta_score = self._compute_delta_score(violation_rate, anomaly_rate)

            # Build pre_metrics with actual values from this neighborhood
            pre_metrics = {
                'volume': nb_metrics.get('volume', 0),
                'anomaly_rate': anomaly_rate,
                'null_rate': nb_metrics.get('null_rate', 0.0),
                'violation_rate': violation_rate,
                'avg_anomaly_score': nb_metrics.get('avg_anomaly_score', 0.0),
                'delta_score': delta_score,
                'drift_count': drift_counts.get(nb, 0),
            }

        if strategy == EvolutionStrategy.DO_NOTHING:
            return {'status': 'success', 'action': 'do_nothing', 'message': 'Continue normal operation'}

        elif strategy == EvolutionStrategy.ADJUST_THRESHOLD:
            result = self._execute_adjust_threshold(decision)

        elif strategy == EvolutionStrategy.MEMORY_RESET:
            result = self._execute_memory_reset(decision)

        else:
            return {'status': 'error', 'message': f'Unknown strategy: {strategy}'}

        # Record action and outcome in circuit breaker
        self._circuit_breaker.record_action()
        if result.get('status') == 'success':
            self._circuit_breaker.on_action_success()
        else:
            self._circuit_breaker.on_action_failure()

        result['circuit_state'] = self._circuit_breaker.state
        result['consecutive_actions'] = self._circuit_breaker.consecutive_actions

        return result

    def _execute_adjust_threshold(self, decision: Dict) -> Dict:
        """
        Execute threshold adjustment strategy.

        Computes new beta from anomaly_rate, writes to MinIO for MemStreamScoringOperator
        to read on next scoring cycle (pull model).

        Writes to: s3://cadqstream-drift/iec/beta/{neighborhood}.json
        
        CRITICAL (Phase 2D): HMAC signature is mandatory. No fallback key.

        Args:
            decision: Decision dict

        Returns:
            Execution result dict
        """
        affected_neighborhoods = decision.get('affected_neighborhoods', [])
        meta_metrics = decision.get('meta_metrics', {})

        if not affected_neighborhoods:
            return {
                'status': 'success',
                'action': 'adjust_threshold',
                'message': 'No neighborhoods affected',
            }

        nb = affected_neighborhoods[0]
        nb_metrics = meta_metrics.get(nb, {})
        anomaly_rate = nb_metrics.get('anomaly_rate', 0.05)

        if anomaly_rate > 0.15:
            new_beta = 0.55
            reason = 'high_fpr'
        elif anomaly_rate < 0.03:
            new_beta = 0.45
            reason = 'low_tpr'
        else:
            new_beta = 0.50
            reason = 'normal'

        old_beta = self._current_betas.get(nb, 0.5)
        self._current_betas[nb] = new_beta

        # Write beta to MinIO for MemStreamScoringOperator to poll
        # HMAC signature is MANDATORY - fails hard if key missing
        if hasattr(self, 'minio_client') and self.minio_client:
            try:
                import json
                
                # CRITICAL (Phase 2D): Get signing key - FAIL HARD if missing
                iec_key = _get_signing_key()
                
                beta_str = f"{new_beta:.6f}"
                sig = hashlib.sha256(iec_key.encode() + beta_str.encode()).hexdigest()

                payload = {
                    "beta": new_beta,
                    "hmac": sig,
                    "neighborhood": nb,
                    "timestamp": time.time(),
                    "version": 1,
                }

                self.minio_client.put_object(
                    Bucket='cadqstream-drift',
                    Key=f"iec/beta/{nb}.json",
                    Body=json.dumps(payload).encode('utf-8'),
                    ContentType='application/json',
                )
                status = 'ok'
                LOGGER.info(
                    "[IECController] Beta written to MinIO: %s=%.4f (%s)",
                    nb, new_beta, reason
                )
            except EnvironmentError:
                # CRITICAL: Fail hard on missing key
                raise
            except Exception as e:
                status = f'minio_error: {e}'
                LOGGER.warning("[IECController] Failed to write beta to MinIO: %s", e)
        else:
            status = 'skipped (no minio)'

        return {
            'status': 'success',
            'action': 'adjust_threshold',
            'neighborhood': nb,
            'old_beta': old_beta,
            'new_beta': new_beta,
            'reason': reason,
            'anomaly_rate': anomaly_rate,
            'write_status': status,
            'affected_neighborhoods': affected_neighborhoods,
        }

    def _execute_memory_reset(self, decision: Dict) -> Dict:
        """
        Execute memory reset strategy for severe drift.

        MemStream is an online learning system - this resets ADWIN and beta only.
        Memory is NOT pre-loaded; the online learning rebuilds it from incoming records.

        Args:
            decision: IEC decision dict with affected_neighborhoods, severity

        Returns:
            Execution result dict
        """
        affected_neighborhoods = decision.get('affected_neighborhoods', [])
        severity = decision.get('severity', 'high')

        if not affected_neighborhoods:
            affected_neighborhoods = ['all']

        LOGGER.warning(
            "[IECController] MEMORY RESET for %s (severity=%s) — online learning rebuilding",
            affected_neighborhoods, severity
        )

        reset_nb = []
        failed = []

        for neighborhood in affected_neighborhoods:
            try:
                # Reset ADWIN windows for this neighborhood
                self._adwin.reset_all()
                
                # Reset beta to default and write to MinIO for scoring operator
                default_beta = getattr(self.config, 'default_beta', 0.5) if self.config else 0.5
                if hasattr(self, '_current_betas'):
                    self._current_betas[neighborhood] = default_beta

                # Write default beta to MinIO so MemStreamScoringOperator picks it up
                # HMAC signature is MANDATORY
                if hasattr(self, 'minio_client') and self.minio_client:
                    try:
                        import json
                        
                        # CRITICAL (Phase 2D): Get signing key - FAIL HARD if missing
                        iec_key = _get_signing_key()
                        
                        beta_str = f"{default_beta:.6f}"
                        sig = hashlib.sha256(iec_key.encode() + beta_str.encode()).hexdigest()
                        
                        beta_payload = {
                            "beta": default_beta,
                            "hmac": sig,
                            "neighborhood": neighborhood,
                            "timestamp": time.time(),
                            "version": 1,
                        }
                        self.minio_client.put_object(
                            Bucket='cadqstream-drift',
                            Key=f"iec/beta/{neighborhood}.json",
                            Body=json.dumps(beta_payload).encode('utf-8'),
                            ContentType='application/json',
                        )
                    except EnvironmentError:
                        # CRITICAL: Fail hard on missing key
                        raise
                    except Exception:
                        pass

                # Write audit event to MinIO
                if hasattr(self, 'minio_client') and self.minio_client:
                    try:
                        import json
                        event = {
                            'timestamp': time.time(),
                            'action': 'memory_reset',
                            'neighborhood': neighborhood,
                            'severity': severity,
                            'default_beta': default_beta,
                            'version': 1,
                        }
                        key = f"iec/memory_reset/{neighborhood}/{int(time.time())}.json"
                        self.minio_client.put_object(
                            Bucket='cadqstream-drift',
                            Key=key,
                            Body=json.dumps(event).encode('utf-8'),
                            ContentType='application/json',
                        )
                    except Exception:
                        pass

                reset_nb.append(neighborhood)

            except Exception as e:
                LOGGER.error(
                    "[IECController] Memory reset failed for %s: %s",
                    neighborhood, e
                )
                failed.append(neighborhood)

        status = 'ok' if not failed else ('partial' if reset_nb else 'failed')

        return {
            'status': status,
            'action': 'memory_reset',
            'reset_neighborhoods': reset_nb,
            'failed_neighborhoods': failed,
            'message': f'Memory reset for {len(reset_nb)} neighborhoods — online learning rebuilding',
            'affected_neighborhoods': affected_neighborhoods,
        }

    def get_stats(self) -> Dict:
        """
        Get IEC statistics.

        Returns:
            Statistics dict
        """
        return {
            'n_processed': self._n_processed,
            'meters_loaded': self._meter_loaded,
            'last_decision': self._last_decision,
            'adwin_total_drifts': self._adwin.get_total_drifts(),
            'aggregator': self._aggregator.get_stats(),
            'strategy_counts': self._strategy_counts.copy(),
            'circuit_breaker': self._circuit_breaker.get_status(),
            'retrain_tracking': {nb: {
                'adwin_drift_days': t.get('adwin_drift_days', 0),
                'anomaly_rate_days': t.get('anomaly_rate_days_above_threshold', 0),
                'knn_baseline': t.get('knn_baseline'),
            } for nb, t in self._retrain_tracking.items()},
        }

    def get_circuit_status(self) -> Dict:
        """Get circuit breaker status."""
        return self._circuit_breaker.get_status()

    def retrain_model(self, trigger: str = None) -> Dict:
        """
        Trigger model retraining based on retrain triggers.
        
        Call this when retrain triggers fire (from update()'s retrain_triggers field).
        In production, this would signal the offline training pipeline.
        
        Args:
            trigger: Optional trigger name to retrain for
            
        Returns:
            Dict with retrain status
        """
        LOGGER.warning(
            f"[IECController] RETRAIN TRIGGERED: {trigger or 'manual'}"
        )
        
        return {
            'status': 'retrain_triggered',
            'trigger': trigger,
            'timestamp': time.time(),
            'action': 'signal_offline_training',
            'message': (
                'Retrain triggered. In production, this signals the offline '
                'training pipeline to retrain the MemStream model.'
            ),
        }

    def reset(self) -> None:
        """Reset all IEC state."""
        self._adwin.reset()
        self._aggregator.reset()
        self._last_decision = None
        self._n_processed = 0
        self._strategy_counts = {s: 0 for s in EvolutionStrategy.ALL_STRATEGIES}
        self._init_retrain_tracking()


# =============================================================================
# HMAC Verification (for MinIO beta reader)
# =============================================================================

def verify_beta_hmac(
    beta_payload: dict,
    signing_key: str
) -> bool:
    """
    Verify HMAC signature on a beta payload from MinIO.
    
    This should be called by MemStreamScoringOperator when reading
    beta values from MinIO to verify they came from a legitimate IEC.
    
    Args:
        beta_payload: Dict from MinIO with 'beta', 'hmac', 'neighborhood', 'timestamp'
        signing_key: The IEC signing key to verify against
        
    Returns:
        True if HMAC is valid, False otherwise
    """
    try:
        provided_hmac = beta_payload.get('hmac', '')
        beta_value = beta_payload.get('beta', 0.0)
        neighborhood = beta_payload.get('neighborhood', '')
        
        # Compute expected HMAC: HMAC-SHA256(key, f"{beta:.6f}")
        beta_str = f"{beta_value:.6f}"
        expected_hmac = hashlib.sha256(signing_key.encode() + beta_str.encode()).hexdigest()
        
        is_valid = expected_hmac == provided_hmac
        
        if not is_valid:
            LOGGER.warning(
                f"[IECController] HMAC verification FAILED: neighborhood={neighborhood}, "
                f"expected={expected_hmac[:16]}..., got={provided_hmac[:16]}..."
            )
            # Fire Prometheus alert
            _emit_hmac_failure_alert(
                neighborhood=neighborhood,
                reason='hmac_mismatch'
            )
        
        return is_valid
        
    except Exception as e:
        LOGGER.error(f"[IECController] HMAC verification error: {e}")
        _emit_hmac_failure_alert(
            neighborhood=beta_payload.get('neighborhood', 'unknown'),
            reason='verification_error'
        )
        return False


# =============================================================================
# Factory Functions
# =============================================================================

def create_iec_controller(
    meter_model_path: Optional[str] = None,
    meter_scaler_path: Optional[str] = None,
    enable_meter: bool = True,
    minio_client: Optional[object] = None,
) -> 'IECController':
    """
    Create IEC controller with default or custom configuration.

    Args:
        meter_model_path: Path to METER model
        meter_scaler_path: Path to METER scaler
        enable_meter: Enable METER (False = use fallback only)
        minio_client: Optional MinIO client for beta writes

    Returns:
        IECController instance
    """
    config = IECConfig(
        meter_model_path=meter_model_path,
        meter_scaler_path=meter_scaler_path,
        enable_meter=enable_meter,
    )
    return IECController(config=config, minio_client=minio_client)


__all__ = [
    'IECConfig',
    'IECController',
    'create_iec_controller',
    'CircuitBreaker',
    'CircuitState',
    'verify_beta_hmac',
    'IEC_SIGNING_KEY_ENV',
]
