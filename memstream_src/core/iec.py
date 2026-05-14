"""
IEC - Intelligent Evolution Controller (Unified Entry Point)

This module provides the complete IEC implementation by combining:
1. MultiInstanceADWIN - 36 instances (6 neighborhoods x 6 metrics)
2. DriftAggregator - Severity assessment
3. StrategyPredictor - METER hypernetwork (or fallback rules)
4. StrategyExecutor - Execute adaptation strategies

Reference: original_flow.md lines 599-1020

NOTE: All state is in-memory + MinIO checkpoints.
"""

import os
import time
import json
import hashlib
import hmac
from collections import deque
from typing import Dict, List, Tuple, Optional

import numpy as np

# Re-export from existing modules
from .adwin_multi_instance import ADWIN as _ADWIN, MultiInstanceADWIN as _MultiInstanceADWIN
from .drift_aggregator import (
    SeverityLevel,
    EvolutionStrategy,
    DriftAggregator as _DriftAggregator,
)

# =============================================================================
# IEC CONFIG
# =============================================================================

IEC_CONFIG = {
    # Neighborhoods (6)
    'neighborhoods': [
        'manhattan',
        'brooklyn',
        'queens',
        'bronx',
        'staten_island',
        'airport',
    ],
    # Metrics for ADWIN instances (6)
    'metrics': [
        'volume',             # Data volume
        'null_rate',         # Null rate (highest sensitivity)
        'violation_rate',    # Canary violation rate
        'anomaly_rate',      # ML anomaly rate
        'avg_anomaly_score', # Average anomaly score
        'delta_score',       # |violation_rate - anomaly_rate| / sum
    ],
    # ADWIN delta per metric (sensitivity)
    'adwin_delta': {
        'volume': 0.005,
        'null_rate': 0.001,   # Highest sensitivity
        'violation_rate': 0.002,
        'anomaly_rate': 0.002,
        'avg_anomaly_score': 0.003,
        'delta_score': 0.002,
    },
    # Severity thresholds (drift count -> severity)
    'severity_thresholds': {
        'none': 0,
        'low': 3,
        'moderate': 6,
        'high': float('inf'),
    },
    # Strategy thresholds (anomaly_rate -> threshold adjustment)
    'anomaly_rate_high': 0.15,    # >15% -> increase threshold
    'anomaly_rate_low': 0.03,    # <3% -> decrease threshold
    'threshold_high': 0.55,
    'threshold_low': 0.45,
    'threshold_default': 0.50,
}

# =============================================================================
# ADWIN Wrapper (for IEC namespace)
# =============================================================================

class ADWIN(_ADWIN):
    """ADWIN drift detector (wrapper with IEC configuration)."""
    pass


class MultiInstanceADWIN(_MultiInstanceADWIN):
    """ADWIN instances for IEC: 6 neighborhoods x 6 metrics = 36 instances.
    
    Reference: original_flow.md lines 715-726
    """
    pass


# =============================================================================
# DriftAggregator Wrapper
# =============================================================================

class DriftAggregator(_DriftAggregator):
    """Aggregate drift events and assess severity.
    
    Reference: original_flow.md lines 728-738 (severity assessment)
    """
    pass


# =============================================================================
# Strategy Prediction (Fallback Rule-based)
# =============================================================================

class StrategyPredictor:
    """
    Predict adaptation strategy based on severity.
    
    NOTE: METER hypernetwork (sklearn MLP) is optional.
    If available, use it. Otherwise, fall back to rule-based.
    
    Reference: original_flow.md lines 997-1011
    """
    
    def __init__(self, meter_model_path: Optional[str] = None):
        self.meter_model = None
        self.meter_scaler = None
        self.meter_metadata = None
        
        # Try to load METER model
        if meter_model_path:
            self._load_meter(meter_model_path)
    
    def _load_meter(self, model_path: str) -> bool:
        """Try to load METER hypernetwork."""
        import pickle
        
        scaler_path = model_path.replace('_hypernetwork.pkl', '_scaler.pkl')
        
        if not os.path.exists(model_path):
            import logging
            logging.getLogger('cadqstream-eia').warning(
                f"[StrategyPredictor] METER model not found: {model_path}"
            )
            return False
        if not os.path.exists(scaler_path):
            import logging
            logging.getLogger('cadqstream-eia').warning(
                f"[StrategyPredictor] METER scaler not found: {scaler_path}"
            )
            return False
        
        try:
            with open(model_path, 'rb') as f:
                self.meter_model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                self.meter_scaler = pickle.load(f)
            
            import logging
            logging.getLogger('cadqstream-eia').info(
                f"[StrategyPredictor] METER model loaded: {model_path}"
            )
            return True
        except Exception as e:
            import logging
            logging.getLogger('cadqstream-eia').error(
                f"[StrategyPredictor] Failed to load METER: {e}"
            )
            self.meter_model = None
            self.meter_scaler = None
            return False
    
    def predict(self, meta_metrics: Dict) -> Tuple[str, float]:
        """
        Predict strategy and confidence.
        
        Returns:
            (strategy, confidence)
            strategy: 'do_nothing', 'adjust_threshold', or 'memory_reset'
            confidence: 0.0 to 1.0
        """
        if self.meter_model is not None and self.meter_scaler is not None:
            return self._predict_meter(meta_metrics)
        else:
            return self._predict_fallback(meta_metrics)
    
    def _predict_meter(self, meta_metrics: Dict) -> Tuple[str, float]:
        """Use METER hypernetwork for prediction."""
        features = np.array([[
            meta_metrics.get('volume', 1000),
            meta_metrics.get('null_rate', 0.0),
            meta_metrics.get('violation_rate', 0.0),
            meta_metrics.get('anomaly_rate', 0.0),
            meta_metrics.get('avg_anomaly_score', 0.0),
            meta_metrics.get('delta_score', 0.0),
        ]], dtype=np.float32)
        
        features_scaled = self.meter_scaler.transform(features)
        strategy_id = self.meter_model.predict(features_scaled)[0]
        probs = self.meter_model.predict_proba(features_scaled)[0]
        confidence = float(probs[int(strategy_id)])
        
        strategies = {
            0: EvolutionStrategy.DO_NOTHING,
            1: EvolutionStrategy.ADJUST_THRESHOLD,
            2: EvolutionStrategy.MEMORY_RESET,
            # 3: SWITCH_MODEL removed — not applicable to online learning
        }
        
        strategy_id = int(strategy_id)
        if strategy_id >= 2:
            # METER was trained with 4 classes; map 2→memory_reset, 3→memory_reset
            strategy_id = 2
        
        return strategies.get(strategy_id, EvolutionStrategy.DO_NOTHING), confidence
    
    def _predict_fallback(self, meta_metrics: Dict) -> Tuple[str, float]:
        """
        Fallback rule-based strategy prediction for online MemStream.

        Reference: original_flow.md lines 1004-1011

        Rules:
        - anomaly_rate > 0.15 → ADJUST_THRESHOLD (too many anomalies, raise beta)
        - anomaly_rate < 0.03 → ADJUST_THRESHOLD (too few anomalies, lower beta)
        - else → DO_NOTHING
        """
        anomaly_rate = meta_metrics.get('anomaly_rate', 0.05)

        if anomaly_rate > IEC_CONFIG['anomaly_rate_high']:
            return EvolutionStrategy.ADJUST_THRESHOLD, 0.8
        elif anomaly_rate < IEC_CONFIG['anomaly_rate_low']:
            return EvolutionStrategy.ADJUST_THRESHOLD, 0.7
        else:
            return EvolutionStrategy.DO_NOTHING, 1.0


# =============================================================================
# Strategy Executor
# =============================================================================

class StrategyExecutor:
    """
    Execute adaptation strategies.
    
    Reference: original_flow.md lines 900-906
    
    NOTE: No Redis dependency - returns action dict for Kafka output.
    """
    
    def __init__(self, redis_client=None, minio_client=None):
        self.redis_client = redis_client
        self.minio_client = minio_client
        self.strategy_handlers = {
            EvolutionStrategy.DO_NOTHING: self._do_nothing,
            EvolutionStrategy.ADJUST_THRESHOLD: self._adjust_threshold,
            EvolutionStrategy.MEMORY_RESET: self._memory_reset,
        }
    
    def execute(self, strategy: str, meta_metrics: Dict = None) -> Dict:
        """Execute the given strategy."""
        handler = self.strategy_handlers.get(strategy, self._do_nothing)
        return handler(meta_metrics or {})
    
    def _do_nothing(self, meta_metrics: Dict) -> Dict:
        """No action needed."""
        return {
            'action': EvolutionStrategy.DO_NOTHING,
            'message': 'No adaptation needed',
            'status': 'ok',
        }
    
    def _adjust_threshold(self, meta_metrics: Dict) -> Dict:
        """
        Adjust anomaly threshold based on anomaly_rate.
        
        Reference: original_flow.md lines 682-690
        """
        anomaly_rate = meta_metrics.get('anomaly_rate', 0.05)
        neighborhood = meta_metrics.get('neighborhood', 'global')
        
        if anomaly_rate > IEC_CONFIG['anomaly_rate_high']:
            new_beta = IEC_CONFIG['threshold_high']
            reason = 'high_fpr'
        elif anomaly_rate < IEC_CONFIG['anomaly_rate_low']:
            new_beta = IEC_CONFIG['threshold_low']
            reason = 'low_tpr'
        else:
            new_beta = IEC_CONFIG['threshold_default']
            reason = 'normal'
        
        # Update MinIO beta if available
        if self.minio_client:
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
                "neighborhood": neighborhood,
                "timestamp": time.time(),
                "version": 1,
            }

            try:
                self.minio_client.put_object(
                    Bucket='cadqstream-drift',
                    Key=f"iec/beta/{neighborhood}.json",
                    Body=json.dumps(payload).encode('utf-8'),
                    ContentType='application/json',
                )
                status = 'ok'
            except Exception as e:
                status = f'minio_error: {e}'
        else:
            status = 'skipped (no minio)'
        
        return {
            'action': EvolutionStrategy.ADJUST_THRESHOLD,
            'neighborhood': neighborhood,
            'old_beta': IEC_CONFIG['threshold_default'],
            'new_beta': new_beta,
            'reason': reason,
            'anomaly_rate': anomaly_rate,
            'status': status,
        }
    
    def _memory_reset(self, decision: Dict) -> Dict:
        """
        Reset MemStream memory for severely drifted neighborhoods.

        MemStream is an online learning system — memory resets are SEVERE_DRIFT responses.
        After reset, the online learning mechanism rebuilds memory from incoming records.
        This is NOT retraining — it's emergency memory wipe.

        Algorithm:
        1. Reset ADWIN windows for affected neighborhoods (clear stale drift history)
        2. Reset beta to default (fresh start for threshold adaptation)
        3. Write memory reset event to MinIO for audit trail

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
            "[IECController] MEMORY RESET for %s (severity=%s) — online learning will rebuild",
            affected_neighborhoods, severity
        )

        reset_nb = []
        failed = []

        for neighborhood in affected_neighborhoods:
            try:
                # Reset ADWIN windows for this neighborhood
                if hasattr(self, '_adwin_instances') and self._adwin_instances:
                    nb_idx = self._neighborhood_to_index.get(neighborhood, -1) if hasattr(self, '_neighborhood_to_index') else -1
                    if nb_idx >= 0 and nb_idx in self._adwin_instances:
                        adwin = self._adwin_instances[nb_idx]
                        if hasattr(adwin, 'reset'):
                            adwin.reset()
                            LOGGER.info(
                                "[IECController] ADWIN reset for neighborhood %s (idx=%d)",
                                neighborhood, nb_idx
                            )
                        elif hasattr(adwin, 'reset_all'):
                            adwin.reset_all()

                # Reset beta to default
                default_beta = self._config.get('default_beta', 0.5) if self._config else 0.5
                if hasattr(self, '_ms_core') and self._ms_core is not None:
                    self._ms_core.set_beta(default_beta)
                if hasattr(self, '_current_betas'):
                    self._current_betas[neighborhood] = default_beta

                # Write audit event to MinIO
                if self.minio_client:
                    try:
                        import json
                        event = {
                            'timestamp': time.time(),
                            'action': EvolutionStrategy.MEMORY_RESET,
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
                        pass  # Non-critical

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
            'action': EvolutionStrategy.MEMORY_RESET,
            'reset_neighborhoods': reset_nb,
            'failed_neighborhoods': failed,
            'message': f'Memory reset for {len(reset_nb)} neighborhoods — online learning rebuilding',
            'affected_neighborhoods': affected_neighborhoods,
        }


# =============================================================================
# IEC Controller (Unified API)
# =============================================================================

class IECController:
    """
    Main IEC controller combining all components.
    
    Reference: original_flow.md lines 598-650
    
    Example:
        >>> iec = IECController()
        >>> meta_metrics = {
        ...     'manhattan': {'volume': 1500, 'null_rate': 0.02, ...},
        ...     'brooklyn': {'volume': 800, 'null_rate': 0.01, ...},
        ... }
        >>> decision = iec.process_meta_metrics(meta_metrics)
        >>> print(f"Strategy: {decision['strategy']}, Confidence: {decision['confidence']:.2f}")
        >>> if decision['strategy'] != 'do_nothing':
        ...     result = iec.execute_strategy(decision)
    """
    
    def __init__(
        self,
        meter_model_path: Optional[str] = None,
        redis_client=None,
        minio_client=None,
        config: Optional[Dict] = None
    ):
        self.config = config or IEC_CONFIG
        self.redis_client = redis_client
        self.minio_client = minio_client

        # Core components
        self.adwin_u = _MultiInstanceADWIN()
        self.drift_aggregator = _DriftAggregator(
            max_recent=50,
            severity_thresholds=self.config.get('severity_thresholds', {})
        )
        self.strategy_predictor = StrategyPredictor(meter_model_path)
        self.strategy_executor = StrategyExecutor(redis_client, minio_client)
        
        # Statistics
        self._n_processed = 0
        self._last_decision: Optional[Dict] = None
    
    def process_meta_metrics(self, neighborhood: str, metrics: Dict) -> Dict:
        """
        Process meta-metrics from Layer 3 (per-neighborhood).
        
        Args:
            neighborhood: Neighborhood name
            metrics: Dict with volume, null_rate, violation_rate, etc.
        
        Returns:
            Dict with IEC decision and action result
        """
        # Step 1: Update ADWIN instances
        drift_events = []
        for metric in self.config['metrics']:
            value = metrics.get(metric, 0.0)
            if self.adwin_u.update(neighborhood, metric, value):
                drift_events.append({
                    'neighborhood': neighborhood,
                    'metric': metric,
                    'timestamp': time.time(),
                })
        
        # Step 2: Add drift events to aggregator
        for event in drift_events:
            self.drift_aggregator.add_drift(
                event['neighborhood'],
                event['metric']
            )
        
        # Step 3: Assess severity
        severity = self.drift_aggregator.assess_drift_severity()
        
        # Step 4: Predict strategy
        strategy, confidence = self.strategy_predictor.predict(metrics)
        
        # Step 5: Execute strategy
        action_result = self.strategy_executor.execute(strategy, {
            **metrics,
            'neighborhood': neighborhood,
            'severity': severity,
        })
        
        # Step 6: Prepare IEC decision
        self._last_decision = {
            'neighborhood': neighborhood,
            'n_drift_events': len(drift_events),
            'severity': severity,
            'strategy': strategy,
            'confidence': confidence,
            'action_result': action_result,
            'timestamp': time.time(),
            'affected_neighborhoods': self.drift_aggregator.get_affected_neighborhoods(),
        }
        
        self._n_processed += 1
        
        return self._last_decision
    
    def get_decision(self) -> Optional[Dict]:
        """Get the last IEC decision."""
        return self._last_decision
    
    def execute_strategy(self, decision: Optional[Dict] = None) -> Dict:
        """Execute the recommended strategy."""
        if decision is None:
            decision = self._last_decision
        
        if decision is None:
            return {'status': 'error', 'message': 'No decision available'}
        
        return self.strategy_executor.execute(
            decision['strategy'],
            decision.get('action_result', {})
        )
    
    def get_stats(self) -> Dict:
        """Get IEC statistics."""
        return {
            'n_processed': self._n_processed,
            'drift_aggregator': self.drift_aggregator.get_stats(),
            'n_adwin_instances': len(self.adwin_u._adwins),
            'total_drifts': self.adwin_u.get_total_drifts(),
        }
    
    def reset(self) -> None:
        """Reset all IEC state."""
        self.adwin_u.reset()
        self.drift_aggregator.reset()
        self._last_decision = None
        self._n_processed = 0


# =============================================================================
# Factory Function
# =============================================================================

def create_iec_controller(
    meter_model_path: Optional[str] = None,
    redis_client=None,
    minio_client=None,
) -> IECController:
    """
    Create IEC controller with default configuration.

    Args:
        meter_model_path: Path to METER hypernetwork model
        redis_client: Redis client for beta updates (deprecated)
        minio_client: MinIO client for beta updates

    Returns:
        IECController instance
    """
    return IECController(
        meter_model_path=meter_model_path,
        redis_client=redis_client,
        minio_client=minio_client,
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Config
    'IEC_CONFIG',
    # Components
    'ADWIN',
    'MultiInstanceADWIN',
    'DriftAggregator',
    'StrategyPredictor',
    'StrategyExecutor',
    'IECController',
    # Utilities
    'SeverityLevel',
    'EvolutionStrategy',
    'create_iec_controller',
]
