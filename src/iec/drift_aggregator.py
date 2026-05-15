"""
DriftAggregator: Severity assessment + strategy prediction for IEC.

This module aggregates drift events from MultiInstanceADWIN and determines
the appropriate evolution strategy based on drift severity.

Phase 2D migration: copied from memstream_src/core/drift_aggregator.py

Reference: original_flow.md Section 5.6 (METER Hypernetwork)
"""

import time
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple
import logging

LOGGER = logging.getLogger('memstream-iec')


# =============================================================================
# Severity Levels
# =============================================================================

class SeverityLevel:
    """Severity level constants and utilities."""

    NONE = 'none'
    LOW = 'low'
    MODERATE = 'moderate'
    HIGH = 'high'

    LEVEL_ORDER = [NONE, LOW, MODERATE, HIGH]

    @classmethod
    def compare(cls, a: str, b: str) -> int:
        """
        Compare two severity levels.

        Returns:
            -1 if a < b, 0 if a == b, 1 if a > b
        """
        try:
            idx_a = cls.LEVEL_ORDER.index(a)
            idx_b = cls.LEVEL_ORDER.index(b)
            return idx_a - idx_b
        except ValueError:
            return 0


# =============================================================================
# Evolution Strategies
# =============================================================================

class EvolutionStrategy:
    """Evolution strategy constants for online MemStream adaptation.

    MemStream is an online learning system — no offline retraining or model switching.
    Only two strategies are applicable:
    - DO_NOTHING: no drift detected, continue normal operation
    - ADJUST_THRESHOLD: drift detected, adjust beta (ContextBeta threshold)
    - MEMORY_RESET: severe drift detected, reset memory and let online learning rebuild
    """

    DO_NOTHING = 'do_nothing'          # No drift — continue normal operation
    ADJUST_THRESHOLD = 'adjust_threshold'  # Drift detected — adjust beta threshold
    MEMORY_RESET = 'memory_reset'     # Severe drift — reset memory, rebuild online

    ALL_STRATEGIES = [DO_NOTHING, ADJUST_THRESHOLD, MEMORY_RESET]


# =============================================================================
# Severity Thresholds
# =============================================================================

DEFAULT_SEVERITY_THRESHOLDS = {
    SeverityLevel.NONE: 0,
    SeverityLevel.LOW: 3,
    SeverityLevel.MODERATE: 6,
    SeverityLevel.HIGH: 10,
}


# =============================================================================
# DriftAggregator
# =============================================================================

class DriftAggregator:
    """
    Aggregate drift events and assess severity for IEC decision-making.

    This class tracks recent drift events across all neighborhoods and metrics,
    computes severity based on drift frequency, and maps severity to evolution
    strategies.

    Severity Assessment:
        - Based on count of recent drift events (within rolling window)
        - Severity thresholds determine strategy mapping

    Strategy Mapping:
        none (0 drifts)       -> do_nothing
        low (1-2 drifts)      -> adjust_threshold
        moderate (3-5 drifts) -> adjust_threshold (online learning adapts)
        high (6+ drifts)     -> memory_reset (severe drift, reset + rebuild)

    Example:
        >>> aggregator = DriftAggregator()
        >>> aggregator.add_drift('manhattan', 'null_rate')
        >>> aggregator.add_drift('brooklyn', 'violation_rate')
        >>> severity = aggregator.assess_drift_severity()
        >>> strategy = aggregator.predict_strategy(severity)
        >>> print(f"Severity: {severity}, Strategy: {strategy}")
    """

    def __init__(
        self,
        max_recent: int = 10,
        severity_thresholds: Optional[Dict[str, int]] = None
    ):
        """
        Initialize DriftAggregator.

        Args:
            max_recent: Maximum number of recent drift events to track
            severity_thresholds: Custom severity thresholds (drift_count -> severity)
        """
        self.max_recent = max_recent
        self.severity_thresholds = severity_thresholds or DEFAULT_SEVERITY_THRESHOLDS.copy()

        # Rolling window of recent drift events
        self._recent_drifts: deque = deque(maxlen=max_recent)

        # Cumulative counts per neighborhood and metric
        self._drift_counts_nb: Dict[str, int] = defaultdict(int)
        self._drift_counts_metric: Dict[str, int] = defaultdict(int)
        self._drift_counts_total: int = 0

    def add_drift(self, neighborhood: str, metric: str) -> None:
        """
        Add a drift event.

        Args:
            neighborhood: Neighborhood where drift was detected
            metric: Metric that triggered drift detection
        """
        self._recent_drifts.append({
            'neighborhood': neighborhood,
            'metric': metric,
            'timestamp': time.time(),
        })
        self._drift_counts_nb[neighborhood] += 1
        self._drift_counts_metric[metric] += 1
        self._drift_counts_total += 1

        LOGGER.info(
            f"[DriftAggregator] Drift event: {neighborhood}/{metric} "
            f"(total: {self._drift_counts_total})"
        )

    def assess_drift_severity(self) -> str:
        """
        Assess drift severity based on recent drift count.

        Uses rolling window of recent drift events to determine severity:
        - 0 events: none
        - 1-2 events: low
        - 3-5 events: moderate
        - 6+ events: high

        Returns:
            Severity level: 'none', 'low', 'moderate', or 'high'
        """
        n_recent = len(self._recent_drifts)

        if n_recent == 0:
            return SeverityLevel.NONE
        elif n_recent < 3:
            return SeverityLevel.LOW
        elif n_recent < 6:
            return SeverityLevel.MODERATE
        else:
            return SeverityLevel.HIGH

    def assess_drift_severity_detailed(self) -> Dict:
        """
        Get detailed severity assessment with counts.

        Returns:
            Dict with severity details:
            {
                'severity': str,
                'n_recent': int,
                'n_total': int,
                'affected_neighborhoods': int,
                'affected_metrics': int,
                'drift_rate': float,
            }
        """
        severity = self.assess_drift_severity()
        n_recent = len(self._recent_drifts)

        affected_nb = len(set(d['neighborhood'] for d in self._recent_drifts))
        affected_metrics = len(set(d['metric'] for d in self._recent_drifts))

        return {
            'severity': severity,
            'n_recent': n_recent,
            'n_total': self._drift_counts_total,
            'affected_neighborhoods': affected_nb,
            'affected_metrics': affected_metrics,
            'drift_rate': n_recent / self.max_recent if self.max_recent > 0 else 0.0,
        }

    def get_affected_neighborhoods(self) -> List[str]:
        """
        Get list of neighborhoods with recent drifts.

        Returns:
            List of neighborhood names
        """
        return list(set(d['neighborhood'] for d in self._recent_drifts))

    def get_affected_metrics(self) -> List[str]:
        """
        Get list of metrics with recent drifts.

        Returns:
            List of metric names
        """
        return list(set(d['metric'] for d in self._recent_drifts))

    def predict_strategy(self, severity: Optional[str] = None) -> str:
        """
        Map severity to evolution strategy.

        Args:
            severity: Severity level (if None, computed from recent drifts)

        Returns:
            Strategy name: 'do_nothing', 'adjust_threshold', or 'memory_reset'
        """
        if severity is None:
            severity = self.assess_drift_severity()

        strategy_map = {
            SeverityLevel.NONE: EvolutionStrategy.DO_NOTHING,
            SeverityLevel.LOW: EvolutionStrategy.ADJUST_THRESHOLD,
            SeverityLevel.MODERATE: EvolutionStrategy.ADJUST_THRESHOLD,
            SeverityLevel.HIGH: EvolutionStrategy.MEMORY_RESET,
        }

        return strategy_map.get(severity, EvolutionStrategy.DO_NOTHING)

    def predict_strategy_with_confidence(self, severity: Optional[str] = None) -> Tuple[str, float]:
        """
        Predict strategy with confidence score.

        Args:
            severity: Severity level (if None, computed from recent drifts)

        Returns:
            Tuple of (strategy, confidence)
        """
        if severity is None:
            severity = self.assess_drift_severity()

        confidence_map = {
            SeverityLevel.NONE: 1.0,
            SeverityLevel.LOW: 0.7,
            SeverityLevel.MODERATE: 0.8,
            SeverityLevel.HIGH: 0.9,
        }

        strategy = self.predict_strategy(severity)
        confidence = confidence_map.get(severity, 0.5)

        return strategy, confidence

    def get_stats(self) -> Dict:
        """
        Get aggregator statistics.

        Returns:
            Dict with statistics:
            {
                'n_recent': int,
                'n_total': int,
                'severity': str,
                'strategy': str,
                'affected_neighborhoods': list,
                'affected_metrics': list,
                'per_neighborhood': dict,
                'per_metric': dict,
            }
        """
        severity = self.assess_drift_severity()
        strategy = self.predict_strategy(severity)

        return {
            'n_recent': len(self._recent_drifts),
            'n_total': self._drift_counts_total,
            'severity': severity,
            'strategy': strategy,
            'affected_neighborhoods': self.get_affected_neighborhoods(),
            'affected_metrics': self.get_affected_metrics(),
            'per_neighborhood': dict(self._drift_counts_nb),
            'per_metric': dict(self._drift_counts_metric),
        }

    def reset(self) -> None:
        """Reset all drift tracking state."""
        self._recent_drifts.clear()
        self._drift_counts_nb.clear()
        self._drift_counts_metric.clear()
        self._drift_counts_total = 0

    def get_recent_drifts(self) -> List[Dict]:
        """Get list of recent drift events."""
        return list(self._recent_drifts)


__all__ = [
    'SeverityLevel',
    'EvolutionStrategy',
    'DriftAggregator',
]
