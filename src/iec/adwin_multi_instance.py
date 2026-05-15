"""
SimpleADWIN-U: Multi-Instance SimpleADWIN for Drift Detection.
Task 3.21-3.25: Multiple SimpleADWIN instances per neighborhood × metric

SimpleADWIN-U Architecture:
- 10 neighborhoods × 6 metrics = 60 instances (Phase 1D expansion)
- Each instance monitors one metric stream per neighborhood
- Delta (sensitivity) configurable per metric type

Metrics monitored:
1. volume
2. null_rate
3. violation_rate
4. anomaly_rate
5. avg_anomaly_score
6. delta_score

Spec: Phase 1D — SimpleADWIN Expansion (6→10 neighborhoods)
"""

from datetime import datetime
from typing import Dict, List, Optional
import logging

from src.ml.memstream_core import SimpleADWIN

LOGGER = logging.getLogger('cadqstream-adwin-u')


class MultiInstanceADWIN:
    """SimpleADWIN-U: Multiple SimpleADWIN instances for spatial-temporal drift detection.

    Creates separate SimpleADWIN instance for each (neighborhood, metric) pair.
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
        default_delta: float = 0.002,
        max_window: int = 500
    ):
        """Initialize multi-instance SimpleADWIN.

        Args:
            neighborhoods: List of neighborhood IDs to monitor.
                Defaults to 10 NYC neighborhoods (Phase 1D).
            metrics: List of metric names to monitor.
                Defaults to 6 metrics from MetaAggregator.
            delta_config: Per-metric delta (sensitivity) overrides.
                Lower delta = more conservative (fewer false positives).
            default_delta: Delta for metrics not in delta_config.
            max_window: Maximum window size per SimpleADWIN instance.
        """

        # Default neighborhoods (10 — Phase 1D expansion)
        if neighborhoods is None:
            neighborhoods = [
                'manhattan', 'brooklyn', 'queens_lower', 'queens_upper',
                'bronx', 'staten_island', 'ewr', 'jfk', 'nalp', 'unknown'
            ]

        # Default metrics (6 from MetaAggregator)
        if metrics is None:
            metrics = [
                'volume', 'null_rate', 'violation_rate',
                'anomaly_rate', 'avg_anomaly_score', 'delta_score'
            ]

        # Phase 1D selective delta tuning
        # Critical metrics: null_rate stays at 0.001 (most sensitive)
        # Other metrics: tuned to reduce noise while keeping sensitivity
        if delta_config is None:
            delta_config = {
                'volume':             0.005,   # Less sensitive (high variance)
                'null_rate':          0.001,   # CRITICAL — DO NOT CHANGE
                'violation_rate':     0.005,   # Moderate sensitivity
                'anomaly_rate':       0.005,   # Moderate sensitivity
                'avg_anomaly_score':  0.005,   # Moderate sensitivity
                'delta_score':        0.005,   # Moderate sensitivity
            }

        self.neighborhoods = neighborhoods
        self.metrics = metrics
        self.delta_config = delta_config
        self.default_delta = default_delta
        self.max_window = max_window

        # Create SimpleADWIN instances (60 total: 10 neighborhoods × 6 metrics)
        self.adwin_instances = {}
        
        # Track drift history for get_total_drifts()
        self.drift_history: List[Dict] = []

        for neighborhood in neighborhoods:
            for metric in metrics:
                key = f"{neighborhood}_{metric}"
                delta = delta_config.get(metric, default_delta)

                self.adwin_instances[key] = SimpleADWIN(delta=delta, max_window=max_window)

        LOGGER.info(
            "[SimpleADWIN-U] Initialized %d instances (neighborhoods=%d, metrics=%d, max_window=%d)",
            len(self.adwin_instances), len(neighborhoods), len(metrics), max_window
        )

    def update(self, neighborhood: str, metric_name: str, value: float):
        """Update SimpleADWIN instance and check for drift.

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

        # Update SimpleADWIN
        adwin = self.adwin_instances[key]
        adwin.update(value)

        # Check for drift
        if adwin.drift_detected:
            drift_event = {
                'drift_detected': True,
                'neighborhood': neighborhood,
                'metric': metric_name,
                'value': value,
                'timestamp': datetime.utcnow().isoformat(),
                'drift_key': key
            }
            self.drift_history.append(drift_event)
            return drift_event

        return {'drift_detected': False}

    def update_meta_metrics(self, meta_metrics: dict):
        """Update all SimpleADWIN instances from MetaAggregator output.

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

            # Update SimpleADWIN
            drift_result = self.update(neighborhood, metric_name, value)

            if drift_result['drift_detected']:
                drifts.append(drift_result)

        return drifts

    def get_statistics(self):
        """Get statistics from all SimpleADWIN instances.

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
        """Reset specific SimpleADWIN instance.

        Args:
            neighborhood: Neighborhood ID
            metric_name: Metric name
        """
        key = f"{neighborhood}_{metric_name}"

        if key in self.adwin_instances:
            delta = self.delta_config.get(metric_name, self.default_delta)
            self.adwin_instances[key] = SimpleADWIN(delta=delta, max_window=self.max_window)

    def reset_all(self):
        """Reset all SimpleADWIN instances and clear drift history."""
        for key in self.adwin_instances:
            neighborhood, metric_name = key.rsplit('_', 1)
            delta = self.delta_config.get(metric_name, self.default_delta)
            self.adwin_instances[key] = SimpleADWIN(delta=delta, max_window=self.max_window)
        self.drift_history.clear()

    def get_total_drifts(self) -> int:
        """
        Get total number of drift events across all SimpleADWIN instances.
        
        Returns:
            Total drift count (cumulative across instances).
        """
        return len(self.drift_history) if hasattr(self, 'drift_history') else 0

    def get_adwin_instance(self, neighborhood: str, metric_name: str) -> Optional[object]:
        """
        Get specific SimpleADWIN instance for a neighborhood and metric.
        
        Args:
            neighborhood: Neighborhood name
            metric_name: Metric name
            
        Returns:
            SimpleADWIN instance or None if not found
        """
        key = f"{neighborhood}_{metric_name}"
        return self.adwin_instances.get(key)


class DriftAggregator:
    """Aggregate drift signals across multiple SimpleADWIN instances.

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
            drift_event: Drift detection result from SimpleADWIN
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
