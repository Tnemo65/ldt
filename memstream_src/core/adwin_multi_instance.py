"""
Multi-instance ADWIN for IEC (36 instances: 6 neighborhoods × 6 metrics).

This module implements ADWIN-U (Adaptive Windowing) for drift detection across
multiple NYC taxi neighborhoods and data quality metrics.

Reference: original_flow.md Section 5.3.1, Zhang et al. (Bifet & Gavalda, 2007)
"""

from typing import Dict, List, Optional, Tuple
import math
import logging

from collections import deque

LOGGER = logging.getLogger('cadqstream-eia')


# =============================================================================
# ADWIN Core Implementation
# =============================================================================

class ADWIN:
    """
    ADWIN-U: Adaptive Windowing for Drift Detection.

    Detects concept drift by monitoring the mean of a data stream using
    a split-window statistical test with confidence-based thresholding.

    Algorithm:
        1. Append value to sliding window
        2. If n > 100: check drift at all split points from n/4 to 3n/4
           - For each split: left = window[:split], right = window[split:]
           - If |mean1 - mean2| > epsilon_cut: drift detected
           - epsilon_cut = (2/m) * sqrt(delta), where m = harmonic mean of n1, n2
        3. If len(window) > max_window: evict oldest values
        4. Return drift_detected

    Parameters:
        delta: Confidence parameter (smaller = more sensitive to drift)
               Recommended: 0.001 (high sensitivity) to 0.005 (low sensitivity)
        max_window: Maximum window size (default: 1000)
        min_check_size: Minimum window size before checking (default: 100)
        min_split_size: Minimum samples per sub-window (default: 20)
    """

    def __init__(
        self,
        delta: float = 0.002,
        max_window: int = 1000,
        min_check_size: int = 100,
        min_split_size: int = 20
    ) -> None:
        self.delta = delta
        self.max_window = max_window
        self.min_check_size = min_check_size
        self.min_split_size = min_split_size

        self._window: deque = deque(maxlen=max_window)
        self._total: float = 0.0
        self._n: int = 0

    def update(self, value: float) -> bool:
        """
        Add value to window and check for drift.

        Args:
            value: New data point

        Returns:
            True if drift detected, False otherwise
        """
        drift_detected = False

        self._window.append(value)
        self._total += value
        self._n += 1

        if self._n > self.min_check_size:
            mean = self._total / self._n
            drift_detected = self._detect_drift(mean)

        return drift_detected

    def _detect_drift(self, overall_mean: float) -> bool:
        """
        Detect drift using ADWIN's variance-based split-window test.

        Tests all split points from n/4 to 3n/4. For each split:
        - Computes means of left and right sub-windows
        - Computes epsilon_cut based on delta and harmonic mean of sizes
        - If |mean1 - mean2| > epsilon_cut, drift is detected
        """
        n = len(self._window)
        if n < self.min_check_size:
            return False

        window_list = list(self._window)

        for split in range(n // 4, 3 * n // 4):
            left = window_list[:split]
            right = window_list[split:]

            n1, n2 = len(left), len(right)
            if n1 < self.min_split_size or n2 < self.min_split_size:
                continue

            mean1 = sum(left) / n1
            mean2 = sum(right) / n2

            m = 1.0 / (1.0 / n1 + 1.0 / n2)
            epsilon_cut = (2.0 / m) * math.sqrt(self.delta)

            if abs(mean1 - mean2) > epsilon_cut:
                self._cut_window(split)
                return True

        return False

    def _cut_window(self, split_point: int) -> None:
        """
        Cut window at split point, removing older elements.
        """
        n = len(self._window)
        self._window = deque(list(self._window)[split_point:], maxlen=self.max_window)
        self._total = sum(self._window)
        self._n = len(self._window)

    def reset(self) -> None:
        """Reset the detector state."""
        self._window.clear()
        self._total = 0.0
        self._n = 0

    @property
    def window_size(self) -> int:
        """Current window size."""
        return self._n

    @property
    def window_mean(self) -> float:
        """Current window mean."""
        return self._total / self._n if self._n > 0 else 0.0

    def get_stats(self) -> Dict:
        """Get detector statistics."""
        return {
            'n': self._n,
            'mean': self.window_mean,
            'max_window': self.max_window,
            'delta': self.delta,
        }


# =============================================================================
# MultiInstanceADWIN: 36 ADWIN instances
# =============================================================================

class MultiInstanceADWIN:
    """
    Multi-instance ADWIN for IEC (36 instances: 6 neighborhoods × 6 metrics).

    Each neighborhood × metric combination has an independent ADWIN detector
    with tuned sensitivity (delta) parameters.

    Neighborhoods:
        manhattan, brooklyn, queens, bronx, staten_island, airport

    Metrics:
        volume: Trip volume per time window
        null_rate: Rate of null/missing values
        violation_rate: Rate of business rule violations
        anomaly_rate: Rate of detected anomalies
        avg_anomaly_score: Average anomaly score
        delta_score: |violation_rate - anomaly_rate| / (violation_rate + anomaly_rate + epsilon)

    Delta Configuration (sensitivity):
        - Low delta (0.001): High sensitivity — for critical metrics
        - Medium delta (0.002-0.003): Balanced sensitivity
        - High delta (0.005): Low sensitivity — for high-variance metrics

    Example:
        >>> adwin = MultiInstanceADWIN()
        >>> drift_events = adwin.update_meta_metrics({
        ...     'manhattan': {'volume': 1500, 'null_rate': 0.02, ...},
        ...     'brooklyn': {'volume': 800, 'null_rate': 0.01, ...},
        ... })
        >>> if drift_events:
        ...     print(f"Drift detected: {drift_events}")
    """

    NEIGHBORHOODS: List[str] = [
        'manhattan',
        'brooklyn',
        'queens',
        'bronx',
        'staten_island',
        'airport',
    ]

    METRICS: List[str] = [
        'volume',
        'null_rate',
        'violation_rate',
        'anomaly_rate',
        'avg_anomaly_score',
        'delta_score',
    ]

    DELTA_CONFIG: Dict[str, float] = {
        'volume': 0.005,            # Low sensitivity — high variance
        'null_rate': 0.001,          # Highest sensitivity — critical indicator
        'violation_rate': 0.002,     # Medium sensitivity
        'anomaly_rate': 0.002,      # Medium sensitivity
        'avg_anomaly_score': 0.003, # Low sensitivity
        'delta_score': 0.002,       # Medium sensitivity
    }

    def __init__(self, max_window: int = 1000):
        """
        Initialize 36 ADWIN instances.

        Args:
            max_window: Maximum window size for each ADWIN
        """
        self.max_window = max_window
        self._adwins: Dict[str, ADWIN] = {}
        self._drift_counts: Dict[str, int] = {}

        for nb in self.NEIGHBORHOODS:
            for metric in self.METRICS:
                key = self._make_key(nb, metric)
                delta = self.DELTA_CONFIG.get(metric, 0.002)
                self._adwins[key] = ADWIN(delta=delta, max_window=max_window)
                self._drift_counts[key] = 0

    @staticmethod
    def _make_key(neighborhood: str, metric: str) -> str:
        """Create ADWIN instance key."""
        return f"{neighborhood}_{metric}"

    def update(self, neighborhood: str, metric: str, value: float) -> bool:
        """
        Update ADWIN for specific neighborhood/metric and check for drift.

        Args:
            neighborhood: Neighborhood name
            metric: Metric name
            value: New metric value

        Returns:
            True if drift detected, False otherwise
        """
        key = self._make_key(neighborhood, metric)
        if key not in self._adwins:
            LOGGER.warning(f"[MultiInstanceADWIN] Unknown key: {key}")
            return False

        drift_detected = self._adwins[key].update(value)
        if drift_detected:
            self._drift_counts[key] += 1
            LOGGER.info(
                f"[MultiInstanceADWIN] Drift detected: {neighborhood}/{metric}"
            )

        return drift_detected

    def update_meta_metrics(self, meta_metrics: Dict) -> List[Dict]:
        """
        Update all ADWINs from meta-metrics dict and return drift events.

        Args:
            meta_metrics: Dict of dicts, e.g.
                {
                    'manhattan': {'volume': 1500, 'null_rate': 0.02, ...},
                    'brooklyn': {'volume': 800, 'null_rate': 0.01, ...},
                }

        Returns:
            List of drift events, e.g.
                [{'neighborhood': 'manhattan', 'metric': 'null_rate'}, ...]
        """
        drifts = []

        for nb in self.NEIGHBORHOODS:
            nb_metrics = meta_metrics.get(nb, {})
            for metric in self.METRICS:
                value = nb_metrics.get(metric, 0)
                if self.update(nb, metric, value):
                    drifts.append({'neighborhood': nb, 'metric': metric})

        return drifts

    def reset(self, neighborhood: Optional[str] = None, metric: Optional[str] = None) -> None:
        """
        Reset ADWIN state.

        Args:
            neighborhood: Specific neighborhood to reset, or None for all
            metric: Specific metric to reset, or None for all
        """
        if neighborhood and metric:
            key = self._make_key(neighborhood, metric)
            if key in self._adwins:
                self._adwins[key].reset()
        elif neighborhood:
            for m in self.METRICS:
                key = self._make_key(neighborhood, m)
                if key in self._adwins:
                    self._adwins[key].reset()
        else:
            for adwin in self._adwins.values():
                adwin.reset()

    def get_drift_count(self, neighborhood: str, metric: str) -> int:
        """Get cumulative drift count for specific neighborhood/metric."""
        key = self._make_key(neighborhood, metric)
        return self._drift_counts.get(key, 0)

    def get_total_drifts(self) -> int:
        """Get total cumulative drift count across all instances."""
        return sum(self._drift_counts.values())

    def get_stats(self, neighborhood: Optional[str] = None) -> Dict:
        """
        Get ADWIN statistics.

        Args:
            neighborhood: Specific neighborhood, or None for all

        Returns:
            Dict of statistics
        """
        if neighborhood:
            stats = {}
            for metric in self.METRICS:
                key = self._make_key(neighborhood, metric)
                stats[metric] = self._adwins[key].get_stats()
            return stats

        return {
            nb: {
                metric: self._adwins[self._make_key(nb, metric)].get_stats()
                for metric in self.METRICS
            }
            for nb in self.NEIGHBORHOODS
        }


__all__ = ['ADWIN', 'MultiInstanceADWIN']
