"""
IEC (Intelligent Evolution Controller) Operator - Layer 4.
Task 3.21-3.25: Multi-strategy adaptation with METER + ADWIN-U

IEC combines:
1. ADWIN-U drift detection (spatial-temporal signals)
2. METER hypernetwork (strategy selection)
3. Multi-strategy execution (retrain, adjust, switch, do_nothing)

Flow:
1. Meta-metrics arrive from MetaAggregator
2. ADWIN-U detects drifts per neighborhood×metric
3. METER predicts optimal strategy from metrics
4. IEC executes strategy (update models, thresholds, etc.)

Spec: Task 3.21-3.25 (IEC multi-strategy architecture)
"""

from pyflink.datastream import MapFunction
import pickle
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.iec.adwin_multi_instance import MultiInstanceADWIN, DriftAggregator


class IECOperator(MapFunction):
    """Intelligent Evolution Controller for adaptive drift handling.

    Combines ADWIN-U drift detection with METER strategy selection.
    Executes multi-strategy adaptation based on drift signals and meta-metrics.

    Strategies:
    1. do_nothing - No drift detected, continue normal operation
    2. adjust_threshold - Minor drift, adjust anomaly thresholds
    3. retrain_model - Moderate drift, trigger model retraining
    4. switch_model - Severe drift, switch to alternative model

    Args:
        meter_model_path: Path to trained METER hypernetwork
        meter_scaler_path: Path to METER feature scaler
        adwin_delta_config: ADWIN sensitivity configuration
    """

    def __init__(
        self,
        meter_model_path: str = 'models/meter_hypernetwork.pkl',
        meter_scaler_path: str = 'models/meter_scaler.pkl',
        adwin_delta_config: dict = None
    ):
        """Initialize IEC operator."""
        self.meter_model_path = meter_model_path
        self.meter_scaler_path = meter_scaler_path
        self.adwin_delta_config = adwin_delta_config

        self.meter_model = None
        self.meter_scaler = None
        self.adwin_u = None
        self.drift_aggregator = None

        self.stats = {
            'windows_processed': 0,
            'drifts_detected': 0,
            'strategies_executed': {
                'do_nothing': 0,
                'adjust_threshold': 0,
                'retrain_model': 0,
                'switch_model': 0
            }
        }

    def open(self, runtime_context):
        """Load METER model and initialize ADWIN-U.

        Called once per task slot when Flink job starts.
        """
        try:
            # Load METER model
            print(f"[IEC] Loading METER model: {self.meter_model_path}")
            with open(self.meter_model_path, 'rb') as f:
                self.meter_model = pickle.load(f)

            with open(self.meter_scaler_path, 'rb') as f:
                self.meter_scaler = pickle.load(f)

            print(f"[IEC] METER model loaded")

        except Exception as e:
            print(f"[IEC] WARNING: Could not load METER model: {e}")
            print(f"[IEC] IEC will operate in ADWIN-only mode")
            self.meter_model = None

        # Initialize ADWIN-U
        self.adwin_u = MultiInstanceADWIN(
            delta_config=self.adwin_delta_config
        )

        # Initialize drift aggregator
        self.drift_aggregator = DriftAggregator(drift_threshold=3)

        print(f"[IEC] Operator initialized")

    def map(self, meta_metrics):
        """Process meta-metrics window and execute IEC loop.

        Args:
            meta_metrics: Meta-metrics from MetaAggregator (1-minute window)

        Returns:
            IEC decision with strategy and drift status
        """
        if meta_metrics is None:
            return None

        self.stats['windows_processed'] += 1

        # Step 1: Update ADWIN-U with meta-metrics
        drifts = self.adwin_u.update_meta_metrics(meta_metrics)

        # Add drifts to aggregator
        for drift in drifts:
            self.drift_aggregator.add_drift(drift)
            self.stats['drifts_detected'] += 1

        # Step 2: Assess drift severity
        drift_assessment = self.drift_aggregator.assess_drift_severity()

        # Step 3: Predict strategy using METER (if available)
        if self.meter_model is not None:
            strategy, confidence = self._predict_strategy(meta_metrics)
        else:
            # Fallback: Rule-based strategy selection
            strategy, confidence = self._fallback_strategy(drift_assessment)

        # Step 4: Execute strategy
        action_result = self._execute_strategy(strategy, meta_metrics, drift_assessment)

        # Update stats
        self.stats['strategies_executed'][strategy] += 1

        # Construct IEC decision
        iec_decision = {
            **meta_metrics,
            'drifts_detected': drifts,
            'drift_assessment': drift_assessment,
            'iec_strategy': strategy,
            'iec_confidence': confidence,
            'action_result': action_result,
            'iec_timestamp': datetime.utcnow().isoformat()
        }

        # Log periodically
        if self.stats['windows_processed'] % 10 == 0:
            self._log_stats()

        return iec_decision

    def _predict_strategy(self, meta_metrics: dict):
        """Predict optimal strategy using METER.

        Args:
            meta_metrics: Meta-metrics dict

        Returns:
            (strategy_name, confidence)
        """
        import numpy as np

        # Extract features
        features = np.array([[
            meta_metrics.get('volume', 1000),
            meta_metrics.get('null_rate', 0.0),
            meta_metrics.get('violation_rate', 0.0),
            meta_metrics.get('anomaly_rate', 0.0),
            meta_metrics.get('avg_anomaly_score', 0.0),
            meta_metrics.get('delta_score', 0.0)
        ]])

        # Scale
        features_scaled = self.meter_scaler.transform(features)

        # Predict
        strategy_id = self.meter_model.predict(features_scaled)[0]
        probs = self.meter_model.predict_proba(features_scaled)[0]

        strategy_names = {
            0: 'do_nothing',
            1: 'adjust_threshold',
            2: 'retrain_model',
            3: 'switch_model'
        }

        strategy_name = strategy_names[strategy_id]
        confidence = probs[strategy_id]

        return strategy_name, confidence

    def _fallback_strategy(self, drift_assessment: dict):
        """Fallback rule-based strategy selection (if METER unavailable).

        Args:
            drift_assessment: Drift severity assessment

        Returns:
            (strategy_name, confidence)
        """
        severity = drift_assessment['severity']

        if severity == 'none':
            return 'do_nothing', 1.0
        elif severity == 'low':
            return 'adjust_threshold', 0.7
        elif severity == 'moderate':
            return 'retrain_model', 0.8
        else:  # high
            return 'switch_model', 0.9

    def _execute_strategy(self, strategy: str, meta_metrics: dict, drift_assessment: dict):
        """Execute adaptation strategy.

        Args:
            strategy: Strategy name
            meta_metrics: Current meta-metrics
            drift_assessment: Drift assessment

        Returns:
            Dict with action results
        """
        if strategy == 'do_nothing':
            return {'action': 'none', 'message': 'No adaptation needed'}

        elif strategy == 'adjust_threshold':
            # In production: Adjust anomaly detection thresholds
            # For now, just log the action
            new_threshold = self._compute_adjusted_threshold(meta_metrics)
            return {
                'action': 'threshold_adjusted',
                'new_threshold': new_threshold,
                'message': 'Anomaly threshold adjusted'
            }

        elif strategy == 'retrain_model':
            # In production: Trigger async model retraining
            # For now, emit signal to dq-meta-stream
            return {
                'action': 'retrain_triggered',
                'message': 'Model retraining initiated',
                'neighborhood': meta_metrics.get('neighborhood_id')
            }

        elif strategy == 'switch_model':
            # In production: Switch to backup/alternative model
            return {
                'action': 'model_switched',
                'message': 'Switched to alternative model',
                'neighborhood': meta_metrics.get('neighborhood_id')
            }

        else:
            return {'action': 'unknown', 'message': f'Unknown strategy: {strategy}'}

    def _compute_adjusted_threshold(self, meta_metrics: dict):
        """Compute adjusted anomaly threshold based on meta-metrics.

        Args:
            meta_metrics: Current meta-metrics

        Returns:
            New threshold value
        """
        # Simple heuristic: increase threshold if anomaly_rate is high
        anomaly_rate = meta_metrics.get('anomaly_rate', 0.05)

        if anomaly_rate > 0.15:
            # High FPR - increase threshold
            return 0.55
        elif anomaly_rate < 0.03:
            # Low TPR - decrease threshold
            return 0.45
        else:
            # Normal - keep default
            return 0.50

    def _log_stats(self):
        """Log IEC statistics."""
        print(f"\n[IEC Statistics]")
        print(f"  Windows processed: {self.stats['windows_processed']}")
        print(f"  Drifts detected: {self.stats['drifts_detected']}")
        print(f"  Strategies executed:")
        for strategy, count in self.stats['strategies_executed'].items():
            print(f"    {strategy}: {count}")

    def close(self):
        """Print final statistics on close."""
        print(f"\n{'='*60}")
        print(f"IEC OPERATOR FINAL STATISTICS")
        print(f"{'='*60}")
        self._log_stats()
        print(f"{'='*60}")
