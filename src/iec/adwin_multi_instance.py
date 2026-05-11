"""
ADWIN-U: Multi-Instance ADWIN for Drift Detection.
Task 3.21-3.25: Multiple ADWIN instances per neighborhood × metric

ADWIN-U Architecture:
- 6 neighborhoods × 6 metrics = 36 instances (base)
- Optional: 7 neighborhoods × 6 metrics = 42 instances (with 'unknown')
- Each instance monitors one metric stream per neighborhood
- Delta (sensitivity) configurable per metric type

Metrics monitored:
1. volume
2. null_rate
3. violation_rate
4. anomaly_rate
5. avg_anomaly_score
6. delta_score

Spec: Task 3.21-3.25 (ADWIN-U multi-instance pattern)
"""

from river.drift import ADWIN
from datetime import datetime
from typing import Dict, List, Optional
import logging

LOGGER = logging.getLogger('cadqstream-adwin-u')


class MultiInstanceADWIN:
    """ADWIN-U: Multiple ADWIN instances for spatial-temporal drift detection.

    Creates separate ADWIN instance for each (neighborhood, metric) pair.
    Enables fine-grained drift detection with per-metric sensitivity tuning.

    Args:
        neighborhoods: List of neighborhood IDs to monitor
        metrics: List of metric names to monitor
        delta_config: Dict mapping metric names to delta values (sensitivity)
        default_delta: Default delta for metrics not in config
    """

    def __init__(
        self,
        neighborhoods: List[str] = None,
        metrics: List[str] = None,
        delta_config: Dict[str, float] = None,
        default_delta: float = 0.002
    ):
        """Initialize multi-instance ADWIN."""

        # Default neighborhoods
        if neighborhoods is None:
            neighborhoods = [
                'manhattan', 'brooklyn', 'queens',
                'bronx', 'staten_island', 'airport'
            ]

        # Default metrics (6 from MetaAggregator)
        if metrics is None:
            metrics = [
                'volume', 'null_rate', 'violation_rate',
                'anomaly_rate', 'avg_anomaly_score', 'delta_score'
            ]

        # Default delta configuration (per metric sensitivity)
        if delta_config is None:
            delta_config = {
                'volume': 0.005,  # Less sensitive (high variance)
                'null_rate': 0.001,  # More sensitive (critical)
                'violation_rate': 0.002,  # Moderate sensitivity
                'anomaly_rate': 0.002,  # Moderate sensitivity
                'avg_anomaly_score': 0.003,  # Less sensitive
                'delta_score': 0.002  # Moderate sensitivity
            }

        self.neighborhoods = neighborhoods
        self.metrics = metrics
        self.delta_config = delta_config
        self.default_delta = default_delta

        # Create ADWIN instances
        self.adwin_instances = {}

        for neighborhood in neighborhoods:
            for metric in metrics:
                key = f"{neighborhood}_{metric}"
                delta = delta_config.get(metric, default_delta)

                self.adwin_instances[key] = ADWIN(delta=delta)

        LOGGER.info("[ADWIN-U] Initialized %d instances (neighborhoods=%d, metrics=%d)",
                   len(self.adwin_instances), len(neighborhoods), len(metrics))

    def update(self, neighborhood: str, metric_name: str, value: float):
        """Update ADWIN instance and check for drift.

        Args:
            neighborhood: Neighborhood ID
            metric_name: Metric name
            value: Metric value

        Returns:
            Dict with drift detection result
        """
        key = f"{neighborhood}_{metric_name}"

        if key not in self.adwin_instances:
            # Unknown neighborhood or metric - skip
            return {'drift_detected': False}

        # Update ADWIN
        adwin = self.adwin_instances[key]
        adwin.update(value)

        # Check for drift
        if adwin.drift_detected:
            return {
                'drift_detected': True,
                'neighborhood': neighborhood,
                'metric': metric_name,
                'value': value,
                'timestamp': datetime.utcnow().isoformat(),
                'drift_key': key
            }

        return {'drift_detected': False}

    def update_meta_metrics(self, meta_metrics: dict):
        """Update all ADWIN instances from MetaAggregator output.

        Args:
            meta_metrics: Dict with neighborhood_id and 6 metrics

        Returns:
            List of drift detections
        """
        neighborhood = meta_metrics.get('neighborhood_id', 'unknown')

        drifts = []

        for metric_name in self.metrics:
            value = meta_metrics.get(metric_name)

            if value is None:
                continue

            # Update ADWIN
            drift_result = self.update(neighborhood, metric_name, value)

            if drift_result['drift_detected']:
                drifts.append(drift_result)

        return drifts

    def get_statistics(self):
        """Get statistics from all ADWIN instances.

        Returns:
            Dict with instance counts and drift statistics
        """
        total_instances = len(self.adwin_instances)
        drifts_detected = sum(
            1 for adwin in self.adwin_instances.values()
            if adwin.drift_detected
        )

        return {
            'total_instances': total_instances,
            'neighborhoods': len(self.neighborhoods),
            'metrics': len(self.metrics),
            'drifts_detected': drifts_detected
        }

    def reset_instance(self, neighborhood: str, metric_name: str):
        """Reset specific ADWIN instance.

        Args:
            neighborhood: Neighborhood ID
            metric_name: Metric name
        """
        key = f"{neighborhood}_{metric_name}"

        if key in self.adwin_instances:
            delta = self.delta_config.get(metric_name, self.default_delta)
            self.adwin_instances[key] = ADWIN(delta=delta)

    def reset_all(self):
        """Reset all ADWIN instances."""
        for key in self.adwin_instances:
            neighborhood, metric_name = key.rsplit('_', 1)
            delta = self.delta_config.get(metric_name, self.default_delta)
            self.adwin_instances[key] = ADWIN(delta=delta)


class DriftAggregator:
    """Aggregate drift signals across multiple ADWIN instances.

    Provides higher-level drift assessment by combining signals from
    multiple neighborhoods and metrics.
    """

    def __init__(self, drift_threshold: int = 3):
        """Initialize drift aggregator.

        Args:
            drift_threshold: Number of concurrent drifts to trigger alert
        """
        self.drift_threshold = drift_threshold
        self.recent_drifts = []
        self.drift_history = []

    def add_drift(self, drift_event: dict):
        """Add drift event to aggregator.

        Args:
            drift_event: Drift detection result from ADWIN
        """
        self.recent_drifts.append(drift_event)
        self.drift_history.append(drift_event)

        # Keep only recent drifts (last 10 minutes)
        # In production, would use proper time-based windowing
        if len(self.recent_drifts) > 100:
            self.recent_drifts = self.recent_drifts[-100:]

    def assess_drift_severity(self):
        """Assess overall drift severity.

        Returns:
            Dict with severity assessment
        """
        n_recent = len(self.recent_drifts)

        if n_recent == 0:
            severity = 'none'
        elif n_recent < self.drift_threshold:
            severity = 'low'
        elif n_recent < self.drift_threshold * 2:
            severity = 'moderate'
        else:
            severity = 'high'

        # Count affected neighborhoods
        affected_neighborhoods = set(
            d['neighborhood'] for d in self.recent_drifts
        )

        # Count affected metrics
        affected_metrics = set(
            d['metric'] for d in self.recent_drifts
        )

        return {
            'severity': severity,
            'drift_count': n_recent,
            'affected_neighborhoods': list(affected_neighborhoods),
            'affected_metrics': list(affected_metrics),
            'threshold': self.drift_threshold
        }

    def clear_recent(self):
        """Clear recent drifts (e.g., after adaptation)."""
        self.recent_drifts = []
