"""
IEC Controller: METER Hypernetwork + Strategy Execution for CA-DQStream.

This module implements the Intelligent Evolution Controller (IEC) that:
1. Coordinates MultiInstanceADWIN for drift detection
2. Uses DriftAggregator for severity assessment
3. Predicts strategies via METER hypernetwork (or fallback rules)

Reference: original_flow.md Section 5.6 (METER Hypernetwork)
"""

import hashlib
import os
import pickle
import time
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np

from .adwin_multi_instance import MultiInstanceADWIN, ADWIN
from .drift_aggregator import DriftAggregator, SeverityLevel, EvolutionStrategy
from .circuit_breaker import CircuitBreaker, CircuitState
from .verification_feedback import VerificationFeedbackLoop

LOGGER = logging.getLogger('cadqstream-eia')


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
        # 3: SWITCH_MODEL removed — not applicable to online learning
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
    ):
        """
        Initialize IEC configuration.

        Args:
            meter_model_path: Path to METER MLPClassifier model
            meter_scaler_path: Path to METER StandardScaler
            meter_metadata_path: Path to METER metadata JSON
            max_recent_drifts: Max drift events to track
            enable_meter: Enable METER (if False, use fallback rules)
        """
        self.meter_model_path = meter_model_path or self.DEFAULT_METER_MODEL_PATH
        self.meter_scaler_path = meter_scaler_path or self.DEFAULT_METER_SCALER_PATH
        self.meter_metadata_path = meter_metadata_path
        self.max_recent_drifts = max_recent_drifts
        self.enable_meter = enable_meter


# =============================================================================
# IEC Controller
# =============================================================================

class IECController:
    """
    Intelligent Evolution Controller (IEC).

    Combines MultiInstanceADWIN + DriftAggregator + METER hypernetwork
    for automated drift handling and model evolution.

    Workflow:
        1. process_meta_metrics(meta_metrics) - Update ADWINs, assess severity
        2. get_decision() - Get current IEC decision
        3. execute_strategy(decision) - Apply the chosen strategy

    Example:
        >>> iec = IECController()
        >>> meta_metrics = {
        ...     'manhattan': {'volume': 1500, 'null_rate': 0.02, ...},
        ...     'brooklyn': {'volume': 800, 'null_rate': 0.01, ...},
        ... }
        >>> decision = iec.process_meta_metrics(meta_metrics)
        >>> print(f"Strategy: {decision['strategy']}, Confidence: {decision['confidence']:.2f}")
        >>> if decision['strategy'] != 'do_nothing':
        ...     iec.execute_strategy(decision)
    """

    def __init__(
        self,
        config: Optional[IECConfig] = None,
        meter_model_path: Optional[str] = None,
        meter_scaler_path: Optional[str] = None,
    ):
        """
        Initialize IEC Controller.

        Args:
            config: IEC configuration object
            meter_model_path: Override METER model path
            meter_scaler_path: Override METER scaler path
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
        self._current_betas: Dict[str, float] = {}  # neighborhood -> current beta

        # Verification feedback loop
        self._verification_loop = VerificationFeedbackLoop(
            verification_windows=3,
            improvement_threshold=0.01,
            failure_threshold=3,
            rollback_threshold=5,
        )

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

            if self._meter_metadata and self._meter_metadata.get('test_accuracy'):
                LOGGER.info(
                    f"[IECController] METER loaded: {model_path}, "
                    f"accuracy={self._meter_metadata['test_accuracy']}"
                )
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

        Fallback confidence levels (from original_flow.md):
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

    def process_meta_metrics(self, meta_metrics: Dict) -> Dict:
        """
        Process meta-metrics and return IEC decision.

        This is the main entry point for IEC processing.

        Workflow:
            1. Update all ADWINs with current meta-metrics
            2. Add detected drifts to aggregator
            3. Assess severity
            4. Predict strategy (METER or fallback)
            5. Cache and return decision

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
            }
        """
        self._n_processed += 1

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
        }

        # Update statistics
        self._strategy_counts[strategy] += 1

        LOGGER.info(
            f"[IECController] Decision: {strategy} "
            f"(severity={severity}, drifts={len(drift_events)}, "
            f"conf={confidence:.3f}, meter={meter_used})"
        )

        return self._last_decision

    def get_decision(self) -> Optional[Dict]:
        """
        Get last decision.

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

            self._verification_loop.snapshot_before_adaptation(
                neighborhood=nb,
                strategy=strategy if hasattr(strategy, 'value') else strategy,
                old_beta=old_beta,
                new_beta=new_beta,
                metrics=pre_metrics,
            )

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
        if hasattr(self, 'minio_client') and self.minio_client:
            try:
                import json
                iec_key = os.getenv('IEC_SIGNING_KEY', 'default-key')
                beta_str = f"{new_beta:.6f}"
                sig = hmac.new(
                    iec_key.encode(),
                    beta_str.encode(),
                    hashlib.sha256
                ).hexdigest()

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

        MemStream is an online learning system — this resets ADWIN and beta only.
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
                nb_idx = self._neighborhood_to_index.get(neighborhood, -1) if hasattr(self, '_neighborhood_to_index') else -1
                if nb_idx >= 0 and hasattr(self, '_adwin_instances') and self._adwin_instances:
                    adwin = self._adwin_instances.get(nb_idx)
                    if adwin is not None:
                        if hasattr(adwin, 'reset_all'):
                            adwin.reset_all()
                        elif hasattr(adwin, 'reset'):
                            adwin.reset()
                        LOGGER.info(
                            "[IECController] ADWIN reset for %s (idx=%d)",
                            neighborhood, nb_idx
                        )

                # Reset beta to default and write to MinIO for scoring operator
                default_beta = getattr(self.config, 'default_beta', 0.5) if self.config else 0.5
                if hasattr(self, '_ms_core') and self._ms_core is not None:
                    self._ms_core.set_beta(default_beta)
                if hasattr(self, '_current_betas'):
                    self._current_betas[neighborhood] = default_beta

                # Write default beta to MinIO so MemStreamScoringOperator picks it up
                if hasattr(self, 'minio_client') and self.minio_client:
                    try:
                        import json as _json
                        iec_key = os.getenv('IEC_SIGNING_KEY', 'default-key')
                        beta_str = f"{default_beta:.6f}"
                        sig = hmac.new(
                            iec_key.encode(),
                            beta_str.encode(),
                            hashlib.sha256
                        ).hexdigest()
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
                            Body=_json.dumps(beta_payload).encode('utf-8'),
                            ContentType='application/json',
                        )
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
        }

    def get_circuit_status(self) -> Dict:
        """Get circuit breaker status."""
        return self._circuit_breaker.get_status()

    def record_verification(self, meta_metrics: Dict) -> Optional[Dict]:
        """Record a verification window and check adaptation outcomes.
        
        Call this from MetaAggregator after emitting metrics.
        Returns verification result if a pending adaptation is ready to be verified.
        """
        return self._verification_loop.record_verification_window(meta_metrics)

    def get_verification_status(self) -> Dict:
        """Get verification loop status."""
        return self._verification_loop.get_status()

    def reset(self) -> None:
        """Reset all IEC state."""
        self._adwin.reset()
        self._aggregator.reset()
        self._last_decision = None
        self._n_processed = 0
        self._strategy_counts = {s: 0 for s in EvolutionStrategy.ALL_STRATEGIES}


# =============================================================================
# Factory Functions
# =============================================================================

def create_iec_controller(
    meter_model_path: Optional[str] = None,
    meter_scaler_path: Optional[str] = None,
    enable_meter: bool = True,
) -> IECController:
    """
    Create IEC controller with default or custom configuration.

    Args:
        meter_model_path: Path to METER model
        meter_scaler_path: Path to METER scaler
        enable_meter: Enable METER (False = use fallback only)

    Returns:
        IECController instance
    """
    config = IECConfig(
        meter_model_path=meter_model_path,
        meter_scaler_path=meter_scaler_path,
        enable_meter=enable_meter,
    )
    return IECController(config=config)


__all__ = [
    'IECConfig',
    'IECController',
    'create_iec_controller',
    'CircuitBreaker',
    'CircuitState',
    'VerificationFeedbackLoop',
]
