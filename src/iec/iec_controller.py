"""
IEC Controller: Sequential Pipeline Phase 3.

Simplified from Phase 2D:
- NO METER hypernetwork
- NO adjust_threshold strategy
- NO beta writes
- 2 strategies: do_nothing, quick_retrain

Workflow:
    1. update(meta_metrics) - Update ADWINs, assess severity
    2. execute_strategy() - Either do_nothing OR write retrain signal to MinIO
    3. reset() - Reset all state

Retrain Triggers (any fires -> quick_retrain):
    - Trigger A: ADWIN drift detected for 7+ consecutive days
    - Trigger B: Anomaly rate > 15% for 3+ consecutive days
    - Trigger C: kNN distance > 2x baseline

Retrain Signal written to: cadqstream-drift/iec/retrain/{neighborhood}/{timestamp}.json
Consumed by: action-replay-worker -> ml-service /api/retrain
"""

import hashlib
import os
import time
from typing import Dict, List, Optional
import logging

import numpy as np

from .adwin_multi_instance import MultiInstanceADWIN
from .drift_aggregator import DriftAggregator, SeverityLevel, EvolutionStrategy
from .circuit_breaker import CircuitBreaker, CircuitState

LOGGER = logging.getLogger('memstream-iec')


# =============================================================================
# HMAC Security (for retrain signal verification)
# =============================================================================

IEC_SIGNING_KEY_ENV = 'IEC_SIGNING_KEY'


def _get_signing_key() -> str:
    """
    Get signing key for retrain signals.

    Reads MEMSTREAM_MODEL_SIGNING_KEY (primary, matches MemStreamScoringOperator)
    as the primary key, with IEC_SIGNING_KEY as fallback for backward compat.
    """
    key = os.environ.get('MEMSTREAM_MODEL_SIGNING_KEY') or os.environ.get(IEC_SIGNING_KEY_ENV)
    if not key:
        raise ValueError(
            "MEMSTREAM_MODEL_SIGNING_KEY environment variable must be set. "
            "This key is used to sign retrain signals written to MinIO and must "
            "match the key used by MemStreamScoringOperator for HMAC verification."
        )
    return key


def _emit_failure_alert(neighborhood: str, reason: str, metric_name: str = 'iecc') -> None:
    """Emit a non-blocking Prometheus alert."""
    try:
        import urllib.request
        import json

        payload = json.dumps({
            'name': f'{metric_name}_failures',
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
            pass
    except Exception:
        pass


# =============================================================================
# IEC Configuration
# =============================================================================

class IECConfig:
    """Configuration for IEC Controller (Phase 3: Sequential Pipeline)."""

    # Feature names (6D) used for severity assessment
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
        max_recent_drifts: int = 10,
        # Retrain trigger thresholds
        retrain_adwin_days: int = 7,
        retrain_anomaly_rate_threshold: float = 0.15,
        retrain_anomaly_rate_days: int = 3,
        retrain_knn_multiplier: float = 2.0,
        # ADWIN per-neighborhood configuration
        adwin_delta: float = 0.002,
    ):
        """
        Initialize IEC configuration.

        Args:
            max_recent_drifts: Max drift events to track
            retrain_adwin_days: Days of consecutive ADWIN drift for Trigger A
            retrain_anomaly_rate_threshold: Threshold for Trigger B
            retrain_anomaly_rate_days: Consecutive days for Trigger B
            retrain_knn_multiplier: kNN multiplier for Trigger C
            adwin_delta: ADWIN delta parameter (sensitivity)
        """
        self.max_recent_drifts = max_recent_drifts
        self.retrain_adwin_days = retrain_adwin_days
        self.retrain_anomaly_rate_threshold = retrain_anomaly_rate_threshold
        self.retrain_anomaly_rate_days = retrain_anomaly_rate_days
        self.retrain_knn_multiplier = retrain_knn_multiplier
        self.adwin_delta = adwin_delta


# =============================================================================
# IEC Controller
# =============================================================================

class IECController:
    """
    Intelligent Evolution Controller (IEC) for MemStream Sequential Pipeline.

    Phase 3 version — simplified to 2 strategies.

    Components:
    - MultiInstanceADWIN: drift detection per neighborhood x metric
    - DriftAggregator: severity assessment
    - CircuitBreaker: prevents action storms

    Strategies:
    - do_nothing: no drift or minor drift
    - quick_retrain: severe drift — retrain MemStream AE + reset ADWIN thresholds per grid

    Flow:
        1. update(meta_metrics) - Update ADWINs, assess severity, check retrain triggers
        2. execute_strategy() - do_nothing or write retrain signal to MinIO
        3. reset() - Reset all state
    """

    def __init__(
        self,
        config: Optional[IECConfig] = None,
        minio_client: Optional[object] = None,
    ):
        """
        Initialize IEC Controller.

        Args:
            config: IEC configuration object
            minio_client: MinIO client for writing retrain signals
        """
        self.config = config or IECConfig()

        # Core components
        self._adwin = MultiInstanceADWIN()
        self._aggregator = DriftAggregator(max_recent=self.config.max_recent_drifts)

        # Decision cache
        self._last_decision: Optional[Dict] = None

        # Statistics
        self._n_processed = 0
        self._strategy_counts: Dict[str, int] = {
            s: 0 for s in EvolutionStrategy.ALL_STRATEGIES
        }

        # Circuit breaker
        self._circuit_breaker = CircuitBreaker(
            cooldown_seconds=300.0,
            max_consecutive=10,
            half_open_max_actions=3,
        )

        # MinIO client for retrain signal writes
        self.minio_client = minio_client

        # Retrain trigger tracking per neighborhood
        # NOTE: _retrain_tracking is NOT persisted to Flink state.
        # On restart, these counters reset to zero:
        #   - adwin_drift_days: Trigger A won't fire for 7 days after restart
        #   - anomaly_rate_days_above_threshold: Trigger B won't fire for 3 days after restart
        #   - knn_baseline: Will be recalculated from new memory state (acceptable)
        # If persistence is needed, integrate with Flink keyed state (ValueState).
        self._retrain_tracking: Dict[str, dict] = {}
        self._init_retrain_tracking()

    def _init_retrain_tracking(self) -> None:
        """Initialize retrain trigger tracking state."""
        neighborhoods = self._adwin.neighborhoods
        for nb in neighborhoods:
            self._retrain_tracking[nb] = {
                'adwin_drift_days': 0,
                'last_adwin_drift': None,
                'anomaly_rate_history': [],
                'anomaly_rate_days_above_threshold': 0,
                'knn_history': [],
                'knn_baseline': None,
            }

    def _compute_delta_score(
        self,
        violation_rate: float,
        anomaly_rate: float,
        epsilon: float = 1e-8
    ) -> float:
        """
        Compute delta score: |violation - anomaly| / (violation + anomaly + eps).
        Measures divergence between Canary and MemStream detection.
        """
        return abs(violation_rate - anomaly_rate) / (violation_rate + anomaly_rate + epsilon)

    def _get_fallback_strategy(
        self,
        severity: str,
    ) -> tuple:
        """
        Get strategy using severity-based rules (no METER).

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
            f"[IECController] Strategy: {strategy} "
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
            4. Predict strategy (severity-based)
            5. Check retrain triggers
            6. Cache and return decision

        Args:
            meta_metrics: Dict keyed by neighborhood, e.g.
                {
                    'manhattan': {'volume': 1500, 'null_rate': 0.02, ...},
                    'brooklyn': {'volume': 800, 'null_rate': 0.01, ...},
                }

        Returns:
            Decision dict:
            {
                'strategy': str,           # 'do_nothing' or 'quick_retrain'
                'confidence': float,
                'severity': str,
                'affected_neighborhoods': list,
                'affected_metrics': list,
                'drift_events': list,
                'retrain_triggers': dict,
                'timestamp': float,
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

        # Step 4: Predict strategy (severity-based, no METER)
        strategy, confidence = self._get_fallback_strategy(severity)

        # Step 5: Check retrain triggers
        retrain_triggers = self._check_retrain_triggers(meta_metrics, drift_events, current_time)

        # Update ADWIN drift days tracking
        for nb, tracking in self._retrain_tracking.items():
            has_drift = any(e['neighborhood'] == nb for e in drift_events)
            if has_drift:
                if tracking['last_adwin_drift'] is not None:
                    days_since_last = (current_time - tracking['last_adwin_drift']) / 86400
                    if days_since_last <= 2:
                        tracking['adwin_drift_days'] += 1
                    else:
                        tracking['adwin_drift_days'] = 1
                else:
                    tracking['adwin_drift_days'] = 1
                tracking['last_adwin_drift'] = current_time

        # Override strategy to quick_retrain if any trigger fired
        if retrain_triggers:
            strategy = EvolutionStrategy.QUICK_RETRAIN
            confidence = 0.95
            LOGGER.warning(
                f"[IECController] Retrain triggers fired — forcing strategy to quick_retrain"
            )

        # Build decision
        self._last_decision = {
            'strategy': strategy,
            'confidence': confidence,
            'severity': severity,
            'affected_neighborhoods': self._aggregator.get_affected_neighborhoods(),
            'affected_metrics': self._aggregator.get_affected_metrics(),
            'drift_events': drift_events,
            'meta_metrics': meta_metrics,
            'circuit_state': self._circuit_breaker.state,
            'circuit_allow': self._circuit_breaker.should_allow_action(),
            'retrain_triggers': retrain_triggers,
            'timestamp': current_time,
        }

        self._strategy_counts[strategy] += 1

        LOGGER.info(
            f"[IECController] Decision: {strategy} "
            f"(severity={severity}, drifts={len(drift_events)}, "
            f"conf={confidence:.3f}, triggers={list(retrain_triggers.keys())})"
        )

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
        Check if any retrain trigger has fired.

        Triggers:
            A: ADWIN drift for 7+ consecutive days
            B: Anomaly rate > 15% for 3+ consecutive days
            C: kNN distance > 2x baseline

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

            # Trigger B: Anomaly rate consecutive days
            anomaly_rate = metrics.get('anomaly_rate', 0.0)
            tracking['anomaly_rate_history'].append((current_time, anomaly_rate))

            cutoff = current_time - (self.config.retrain_anomaly_rate_days * 86400)
            tracking['anomaly_rate_history'] = [
                (ts, rate) for ts, rate in tracking['anomaly_rate_history']
                if ts >= cutoff
            ]

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
                tracking['knn_history'] = tracking['knn_history'][-100:]

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
        """Get last decision state."""
        return self._last_decision

    def get_decision(self) -> Optional[Dict]:
        """Alias for get_state()."""
        return self._last_decision

    def execute_strategy(self, decision: Optional[Dict] = None) -> Dict:
        """
        Execute the recommended strategy with circuit breaker protection.

        Phase 3 strategies:
            - do_nothing: return success, no action
            - quick_retrain: write retrain signal to MinIO

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

        if strategy == EvolutionStrategy.DO_NOTHING:
            return {
                'status': 'success',
                'action': 'do_nothing',
                'message': 'Continue normal operation'
            }

        elif strategy == EvolutionStrategy.QUICK_RETRAIN:
            result = self._execute_quick_retrain(decision)

            # Record action in circuit breaker — ONLY for actual adaptations (quick_retrain)
            # NOT for do_nothing (which runs on every MetaAggregator window and would
            # cause the circuit to trip after ~10 windows of normal operation)
            self._circuit_breaker.record_action()
            if result.get('status') == 'success':
                self._circuit_breaker.on_action_success()
                # Reset kNN tracking so baseline recalculates from new memory state after retrain
                for nb in decision.get('affected_neighborhoods', []):
                    if nb in self._retrain_tracking:
                        self._retrain_tracking[nb]['knn_history'] = []
                        self._retrain_tracking[nb]['knn_baseline'] = None
            else:
                self._circuit_breaker.on_action_failure()

            result['circuit_state'] = self._circuit_breaker.state
            result['consecutive_actions'] = self._circuit_breaker.consecutive_actions
            return result

        else:
            return {'status': 'error', 'message': f'Unknown strategy: {strategy}'}

    def _execute_quick_retrain(self, decision: Dict) -> Dict:
        """
        Execute quick_retrain strategy.

        Writes a retrain signal to MinIO for action-replay-worker to consume.
        Signal path: s3://cadqstream-drift/iec/retrain/{neighborhood}/{timestamp}.json

        The retrain signal triggers:
            1. MemStreamScoringOperator: reset memory + re-warmup for affected neighborhoods
            2. ml-service: retrain MemStream AE model
            3. ADWIN: reset thresholds per neighborhood

        Args:
            decision: Decision dict

        Returns:
            Execution result dict
        """
        affected_neighborhoods = decision.get('affected_neighborhoods', [])
        retrain_triggers = decision.get('retrain_triggers', {})
        severity = decision.get('severity', 'high')
        current_time = decision.get('timestamp', time.time())

        if not affected_neighborhoods:
            affected_neighborhoods = ['all']

        LOGGER.warning(
            f"[IECController] QUICK RETRAIN for {affected_neighborhoods} "
            f"(severity={severity}, triggers={list(retrain_triggers.keys())})"
        )

        retrain_nb = []
        failed = []

        for neighborhood in affected_neighborhoods:
            try:
                self._write_retrain_signal(
                    neighborhood=neighborhood,
                    severity=severity,
                    triggers=retrain_triggers,
                    current_time=current_time,
                )
                retrain_nb.append(neighborhood)
            except Exception as e:
                LOGGER.error(
                    f"[IECController] Failed to write retrain signal for {neighborhood}: {e}"
                )
                _emit_failure_alert(neighborhood, str(e), 'iecc_retrain')
                failed.append(neighborhood)

        status = 'ok' if not failed else ('partial' if retrain_nb else 'failed')

        return {
            'status': status,
            'action': 'quick_retrain',
            'retrain_neighborhoods': retrain_nb,
            'failed_neighborhoods': failed,
            'severity': severity,
            'triggers': list(retrain_triggers.keys()),
            'message': f'Retrain signal written for {len(retrain_nb)} neighborhoods',
            'affected_neighborhoods': affected_neighborhoods,
        }

    def _write_retrain_signal(
        self,
        neighborhood: str,
        severity: str,
        triggers: Dict,
        current_time: float,
    ) -> None:
        """
        Write a retrain signal to MinIO.

        Path: cadqstream-drift/iec/retrain/{neighborhood}/{timestamp}.json

        Format:
            {
                "neighborhood": str,
                "severity": str,
                "triggers": list[str],
                "timestamp": float,
                "action": "quick_retrain",
                "version": 1,
                "hmac": str
            }

        Args:
            neighborhood: Affected neighborhood
            severity: Drift severity level
            triggers: Active retrain triggers
            current_time: Current timestamp
        """
        import json

        if self.minio_client is None:
            raise RuntimeError(
                f"[IECController] Cannot write retrain signal: MinIO client unavailable. "
                f"Retrain decision will not be executed for neighborhood={neighborhood}. "
                f"Ensure MinIO is configured and accessible."
            )

        iec_key = _get_signing_key()

        signal = {
            'neighborhood': neighborhood,
            'severity': severity,
            'triggers': list(triggers.keys()),
            'trigger_details': triggers,
            'timestamp': current_time,
            'action': 'quick_retrain',
            'version': 1,
        }

        # Sign the signal
        signal_bytes = json.dumps(signal, sort_keys=True).encode('utf-8')
        sig = hashlib.sha256(iec_key.encode() + signal_bytes).hexdigest()
        signal['hmac'] = sig

        key = f"iec/retrain/{neighborhood}/{int(current_time * 1000)}.json"
        self.minio_client.put_object(
            Bucket='cadqstream-drift',
            Key=key,
            Body=json.dumps(signal).encode('utf-8'),
            ContentType='application/json',
        )

        LOGGER.info(
            f"[IECController] Retrain signal written to MinIO: {key} "
            f"(neighborhood={neighborhood}, triggers={list(triggers.keys())})"
        )

    def get_stats(self) -> Dict:
        """Get IEC statistics."""
        return {
            'n_processed': self._n_processed,
            'last_decision': self._last_decision,
            'adwin_total_drifts': self._adwin.get_total_drifts(),
            'aggregator': self._aggregator.get_stats(),
            'strategy_counts': self._strategy_counts.copy(),
            'circuit_breaker': self._circuit_breaker.get_status(),
            'retrain_tracking': {
                nb: {
                    'adwin_drift_days': t.get('adwin_drift_days', 0),
                    'anomaly_rate_days': t.get('anomaly_rate_days_above_threshold', 0),
                    'knn_baseline': t.get('knn_baseline'),
                }
                for nb, t in self._retrain_tracking.items()
            },
        }

    def get_circuit_status(self) -> Dict:
        """Get circuit breaker status."""
        return self._circuit_breaker.get_status()

    def reset(self) -> None:
        """Reset all IEC state."""
        self._adwin.reset()
        self._aggregator.reset()
        self._circuit_breaker.reset()
        self._last_decision = None
        self._n_processed = 0
        self._strategy_counts = {s: 0 for s in EvolutionStrategy.ALL_STRATEGIES}
        self._init_retrain_tracking()


# =============================================================================
# Retrain Signal Verification (for consumers)
# =============================================================================

def verify_retrain_signal(
    signal: dict,
    signing_key: str
) -> bool:
    """
    Verify HMAC signature on a retrain signal from MinIO.

    Called by action-replay-worker when consuming retrain signals.

    Args:
        signal: Dict from MinIO
        signing_key: The IEC signing key

    Returns:
        True if HMAC is valid, False otherwise
    """
    try:
        provided_hmac = signal.get('hmac', '')
        signal_bytes = json.dumps(
            {k: v for k, v in signal.items() if k != 'hmac'},
            sort_keys=True
        ).encode('utf-8')
        expected_hmac = hashlib.sha256(signing_key.encode() + signal_bytes).hexdigest()

        is_valid = expected_hmac == provided_hmac

        if not is_valid:
            _emit_failure_alert(
                neighborhood=signal.get('neighborhood', 'unknown'),
                reason='hmac_mismatch',
                metric_name='iecc_retrain'
            )

        return is_valid
    except Exception:
        return False


# =============================================================================
# Factory Functions
# =============================================================================

def create_iec_controller(
    minio_client: Optional[object] = None,
) -> 'IECController':
    """
    Create IEC controller with default configuration.

    Args:
        minio_client: MinIO client for retrain signal writes

    Returns:
        IECController instance
    """
    return IECController(minio_client=minio_client)


__all__ = [
    'IECConfig',
    'IECController',
    'create_iec_controller',
    'CircuitBreaker',
    'CircuitState',
    'verify_retrain_signal',
    'IEC_SIGNING_KEY_ENV',
]
