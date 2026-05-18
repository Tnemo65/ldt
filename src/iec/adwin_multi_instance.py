"""
MultiInstanceADWIN: Multi-Instance ADWIN-U for Drift Detection.

Task 3.21-3.25: Multiple ADWIN-U instances per neighborhood x metric

ADWIN-U Architecture:
- 10 neighborhoods x 6 metrics = 60 instances
- Each instance monitors one metric stream per neighborhood
- Delta (sensitivity) configurable per metric type
- Supports higher-order statistics: mean, variance, skewness, kurtosis

Metrics monitored:
1. volume             -> statistic='variance'  (detect volume spikes)
2. null_rate          -> statistic='skewness'  (detect NULL burst asymmetry)
3. violation_rate     -> statistic='mean'       (detect rule violation level changes)
4. anomaly_rate       -> statistic='mean'      (detect ML behavior changes)
5. avg_anomaly_score  -> statistic='mean'      (detect concept drift via score shifts)
6. delta_score        -> statistic='mean'      (detect canary-vs-ML disagreement shifts)

Spec: Phase 1D - ADWIN-U Expansion (ADWIN -> ADWIN-U upgrade)
"""

from collections import deque
from datetime import datetime
from typing import Dict, List, Optional
import logging
import math

from src.ml.adwin_u import ADWIN_U

LOGGER = logging.getLogger('cadqstream-adwin-u')


class MultiInstanceADWIN:
    """MultiInstanceADWIN: Multiple ADWIN-U instances for spatial-temporal drift detection.

    Creates separate ADWIN-U instance for each (neighborhood, metric) pair.
    Enables fine-grained drift detection with per-metric sensitivity tuning
    and higher-order statistical tests (mean, variance, skewness, kurtosis).

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
        max_window: int = 500,
        use_secondary_check: bool = False,
    ):
        """Initialize multi-instance ADWIN-U.

        Args:
            neighborhoods: List of neighborhood IDs to monitor.
                Defaults to 10 NYC neighborhoods.
            metrics: List of metric names to monitor.
                Defaults to 6 metrics from MetaAggregator.
            delta_config: Per-metric delta (sensitivity) overrides.
                Lower delta = more conservative (fewer false positives).
            default_delta: Delta for metrics not in delta_config.
            max_window: Maximum window size per ADWIN-U instance.
            use_secondary_check: Require secondary statistic confirmation.
                Reduces false positives but increases detection latency.
        """

        # Default neighborhoods (10 - Phase 1D expansion)
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
                'null_rate':          0.001,   # CRITICAL - DO NOT CHANGE
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
        self.use_secondary_check = use_secondary_check

        # Create ADWIN-U instances (60 total: 10 neighborhoods x 6 metrics)
        self.adwin_instances = {}

        # Track drift history for get_total_drifts()
        self.drift_history: deque = deque(maxlen=1000)

        for neighborhood in neighborhoods:
            for metric in metrics:
                key = f"{neighborhood}_{metric}"
                delta = delta_config.get(metric, default_delta)

                self.adwin_instances[key] = ADWIN_U.for_metric(
                    metric_name=metric,
                    delta=delta,
                    use_secondary_check=use_secondary_check,
                )

        LOGGER.info(
            "[MultiInstanceADWIN] Initialized %d ADWIN-U instances "
            "(neighborhoods=%d, metrics=%d, max_window=%d, secondary_check=%s)",
            len(self.adwin_instances), len(neighborhoods), len(metrics),
            max_window, use_secondary_check
        )

    def update(self, neighborhood: str, metric_name: str, value: float):
        """Update ADWIN-U instance and check for drift.

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

        # Update ADWIN-U
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
                'drift_key': key,
                'drift_type': adwin.last_drift_type,
                'drift_type_name': adwin.drift_type_name(adwin.last_drift_type),
                'drift_magnitude': adwin.last_drift_magnitude,
                'statistic': adwin.statistic,
            }
            self.drift_history.append(drift_event)
            return drift_event

        return {'drift_detected': False}

    def update_meta_metrics(self, meta_metrics: dict):
        """Update all ADWIN-U instances from MetaAggregator output.

        Args:
            meta_metrics: Dict keyed by neighborhood name, e.g.
                {
                    'manhattan': {'volume': 1500, 'null_rate': 0.02, ...},
                    'brooklyn': {'volume': 800, 'null_rate': 0.01, ...},
                }

        Returns:
            List of drift detections
        """
        drifts = []

        for neighborhood, metrics in meta_metrics.items():
            if neighborhood == '_dlq':
                continue

            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)) and not math.isnan(float(value)):
                    drift_result = self.update(neighborhood, metric_name, float(value))
                    if drift_result['drift_detected']:
                        drifts.append(drift_result)

        return drifts

    def get_statistics(self):
        """Get statistics from all ADWIN-U instances.

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
        """Reset specific ADWIN-U instance.

        Args:
            neighborhood: Neighborhood ID
            metric_name: Metric name
        """
        key = f"{neighborhood}_{metric_name}"

        if key in self.adwin_instances:
            delta = self.delta_config.get(metric_name, self.default_delta)
            self.adwin_instances[key] = ADWIN_U.for_metric(
                metric_name=metric_name,
                delta=delta,
                use_secondary_check=self.use_secondary_check,
            )

    def reset_all(self):
        """Reset all ADWIN-U instances and clear drift history."""
        for key in self.adwin_instances:
            # Key format: 'neighborhood_metric' where neighborhood may contain underscores
            # (e.g., 'queens_lower_volume', 'staten_island_anomaly_rate').
            # Split off the last underscore to get the metric name.
            metric_name = key.rsplit('_', 1)[-1]
            neighborhood = key[:-(len(metric_name) + 1)]
            delta = self.delta_config.get(metric_name, self.default_delta)
            self.adwin_instances[key] = ADWIN_U.for_metric(
                metric_name=metric_name,
                delta=delta,
                use_secondary_check=self.use_secondary_check,
            )
        self.drift_history.clear()

    def get_total_drifts(self) -> int:
        """
        Get total number of drift events across all ADWIN-U instances.

        Returns:
            Total drift count (cumulative across instances).
        """
        return len(self.drift_history) if hasattr(self, 'drift_history') else 0

    def get_adwin_instance(self, neighborhood: str, metric_name: str):
        """
        Get specific ADWIN-U instance for a neighborhood and metric.

        Args:
            neighborhood: Neighborhood name
            metric_name: Metric name

        Returns:
            ADWIN_U instance or None if not found
        """
        key = f"{neighborhood}_{metric_name}"
        return self.adwin_instances.get(key)

    def get_drift_type_counts(self) -> Dict[str, int]:
        """
        Get count of each drift type detected.

        Returns:
            Dict mapping drift type names to counts.
        """
        counts = {
            'mean_shift': 0,
            'variance_shift': 0,
            'skewness_shift': 0,
            'kurtosis_shift': 0,
            'none': 0,
        }
        for event in self.drift_history:
            dt_name = event.get('drift_type_name', 'none')
            if dt_name in counts:
                counts[dt_name] += 1
        return counts

    def get_recent_drifts(self, limit: int = 50) -> List[dict]:
        """
        Get most recent drift events.

        Args:
            limit: Maximum number of events to return.

        Returns:
            List of recent drift event dicts.
        """
        return list(self.drift_history)[-limit:]


__all__ = ['MultiInstanceADWIN']
