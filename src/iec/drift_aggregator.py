"""
DriftAggregator: Severity assessment + strategy prediction for IEC.

This module aggregates drift events from MultiInstanceADWIN and determines
the appropriate evolution strategy based on drift severity.

Phase 3 with KAIS Enhancement:
    - DO_NOTHING: no drift or minor drift — continue normal operation
    - QUICK_RETRAIN: severe drift detected — retrain MemStream AE + reset ADWIN thresholds

Severity Assessment (KAIS-enhanced):
    - Uses weighted drift score: severity = f(drift_count, drift_type, drift_magnitude)
    - Per KAIS 2025 paper: skewness/kurtosis drifts (rank 1) are more indicative
      of distributional shifts than mean/variance drifts (rank 4-5)
    - Drift type weights: skewness_shift=2.0, kurtosis_shift=1.8, variance=1.2, mean=1.0

Strategy Mapping:
    none (0 drifts)       -> do_nothing
    low (< 3 weighted)     -> do_nothing
    moderate (3-8)         -> do_nothing
    high (8+ weighted)     -> quick_retrain
"""

import time
from collections import deque, defaultdict
from typing import Dict, List, Optional, Tuple
import logging

LOGGER = logging.getLogger('memstream-iec')


# Per KAIS 2025 paper: skewness and kurtosis are rank 1 for unsupervised drift detection.
# Higher-order statistics capture distributional shifts that mean/variance miss.
# Weights reflect diagnostic value: skewness/kurtosis drifts are more severe.
DRIFT_TYPE_WEIGHTS = {
    'skewness_shift':   2.0,   # Rank 1 — asymmetric distribution shift
    'kurtosis_shift':    1.8,   # Rank 1 — tail-mass shift
    'variance_shift':   1.2,   # Rank 4 — spread change
    'mean_shift':        1.0,   # Baseline — level shift
    'none':              0.0,
}


# Severity thresholds (weighted score)
# weighted_score = sum(weight * min(magnitude, 2.0)) for each recent drift
_WEIGHTED_SEVERITY_NONE: float = 0.0
_WEIGHTED_SEVERITY_LOW: float = 3.0    # 1-2 skewness drifts
_WEIGHTED_SEVERITY_MODERATE: float = 8.0  # 3+ skewness or mixed drifts


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
        """Compare two severity levels. Returns -1 if a < b, 0 if equal, 1 if a > b."""
        try:
            idx_a = cls.LEVEL_ORDER.index(a)
            idx_b = cls.LEVEL_ORDER.index(b)
            return idx_a - idx_b
        except ValueError:
            return 0


# =============================================================================
# Evolution Strategies (Phase 3: Sequential Pipeline)
# =============================================================================

class EvolutionStrategy:
    """
    Evolution strategy constants for MemStream sequential pipeline.

    Only two strategies:
    - DO_NOTHING: no drift or minor drift, continue normal operation
    - QUICK_RETRAIN: severe drift, retrain MemStream AE + reset ADWIN thresholds per grid

    Removed from previous version: ADJUST_THRESHOLD, MEMORY_RESET, METER.
    """

    DO_NOTHING = 'do_nothing'
    QUICK_RETRAIN = 'quick_retrain'

    ALL_STRATEGIES = [DO_NOTHING, QUICK_RETRAIN]


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
    computes severity based on drift frequency, drift type, and drift magnitude,
    and maps severity to evolution strategies.

    KAIS Enhancement (Phase 3+):
      - Severity uses weighted score: f(drift_count, drift_type, drift_magnitude)
      - Drift type weights per KAIS 2025: skewness=2.0, kurtosis=1.8, variance=1.2, mean=1.0
      - This captures distributional shifts better than raw count

    Severity Assessment:
        weighted_score = sum(DRIFT_TYPE_WEIGHTS[dt] * min(magnitude, 2.0))
        - 0 events: none
        - < 3.0: low
        - 3.0 - 8.0: moderate
        - 8.0+: high

    Strategy Mapping (Phase 3):
        none (0 drifts)       -> do_nothing
        low (< 3 weighted)    -> do_nothing
        moderate (3-8)         -> do_nothing
        high (8+ weighted)     -> quick_retrain

    Example:
        >>> aggregator = DriftAggregator()
        >>> aggregator.add_drift('manhattan', 'null_rate',
        ...     drift_type_name='skewness_shift', drift_magnitude=0.5)
        >>> aggregator.add_drift('brooklyn', 'violation_rate',
        ...     drift_type_name='mean_shift', drift_magnitude=0.3)
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

    def add_drift(self, neighborhood: str, metric: str, drift_type: int = 0,
                  drift_type_name: str = 'none', drift_magnitude: float = 1.0) -> None:
        """Add a drift event.

        Args:
            neighborhood: Neighborhood ID
            metric: Metric name
            drift_type: ADWIN_U drift type constant (0=none, 1=mean, 2=variance, 3=skewness, 4=kurtosis)
            drift_type_name: Human-readable drift type name
            drift_magnitude: Magnitude of the drift (from ADWIN-U last_drift_magnitude).
                Used in weighted severity scoring. Default 1.0.
        """
        self._recent_drifts.append({
            'neighborhood': neighborhood,
            'metric': metric,
            'drift_type': drift_type,
            'drift_type_name': drift_type_name,
            'drift_magnitude': float(drift_magnitude),
            'timestamp': time.time(),
        })
        self._drift_counts_nb[neighborhood] += 1
        self._drift_counts_metric[metric] += 1
        self._drift_counts_total += 1

        LOGGER.info(
            f"[DriftAggregator] Drift event: {neighborhood}/{metric} "
            f"(type={drift_type_name}, mag={drift_magnitude:.4f}, total: {self._drift_counts_total})"
        )

    def assess_drift_severity(self) -> str:
        """
        Assess drift severity based on weighted drift score.

        Uses KAIS-enhanced formula:
            weighted_score = sum(DRIFT_TYPE_WEIGHTS[dt] * min(magnitude, 2.0))

        Per KAIS 2025 paper: skewness/kurtosis drifts carry more diagnostic weight
        because they capture higher-order distributional shifts.

        Thresholds:
            - 0 events: none
            - < 3.0: low (1-2 skewness drifts with small magnitude, or 2-3 mean drifts)
            - 3.0 - 8.0: moderate (mixed drift types, moderate magnitude)
            - 8.0+: high (multiple skewness/kurtosis drifts or high-magnitude drifts)

        Returns:
            Severity level: 'none', 'low', 'moderate', or 'high'
        """
        if not self._recent_drifts:
            return SeverityLevel.NONE

        weighted_score = 0.0
        for d in self._recent_drifts:
            dt_name = d.get('drift_type_name', 'none')
            weight = DRIFT_TYPE_WEIGHTS.get(dt_name, 1.0)
            magnitude = min(d.get('drift_magnitude', 1.0), 2.0)
            weighted_score += weight * magnitude

        if weighted_score < _WEIGHTED_SEVERITY_LOW:
            return SeverityLevel.LOW
        elif weighted_score < _WEIGHTED_SEVERITY_MODERATE:
            return SeverityLevel.MODERATE
        else:
            return SeverityLevel.HIGH

    def assess_drift_severity_detailed(self) -> Dict:
        """Get detailed severity assessment with counts, drift type breakdown, and weighted score."""
        severity = self.assess_drift_severity()
        n_recent = len(self._recent_drifts)

        affected_nb = len(set(d['neighborhood'] for d in self._recent_drifts))
        affected_metrics = len(set(d['metric'] for d in self._recent_drifts))

        # Count drift types
        drift_types = {}
        for d in self._recent_drifts:
            dt_name = d.get('drift_type_name', 'none')
            drift_types[dt_name] = drift_types.get(dt_name, 0) + 1

        # Compute weighted score (for transparency)
        weighted_score = 0.0
        for d in self._recent_drifts:
            dt_name = d.get('drift_type_name', 'none')
            weight = DRIFT_TYPE_WEIGHTS.get(dt_name, 1.0)
            magnitude = min(d.get('drift_magnitude', 1.0), 2.0)
            weighted_score += weight * magnitude

        return {
            'severity': severity,
            'n_recent': n_recent,
            'n_total': self._drift_counts_total,
            'affected_neighborhoods': affected_nb,
            'affected_metrics': affected_metrics,
            'drift_rate': n_recent / self.max_recent if self.max_recent > 0 else 0.0,
            'drift_types': drift_types,
            'weighted_score': round(weighted_score, 3),
            'weighted_severity_thresholds': {
                'low': _WEIGHTED_SEVERITY_LOW,
                'moderate': _WEIGHTED_SEVERITY_MODERATE,
            },
        }

    def get_affected_neighborhoods(self) -> List[str]:
        """Get list of neighborhoods with recent drifts."""
        return list(set(d['neighborhood'] for d in self._recent_drifts))

    def get_affected_metrics(self) -> List[str]:
        """Get list of metrics with recent drifts."""
        return list(set(d['metric'] for d in self._recent_drifts))

    def predict_strategy(self, severity: Optional[str] = None) -> str:
        """
        Map severity to evolution strategy (Phase 3: 2 strategies only).

        Args:
            severity: Severity level (if None, computed from recent drifts)

        Returns:
            Strategy name: 'do_nothing' or 'quick_retrain'
        """
        if severity is None:
            severity = self.assess_drift_severity()

        # Phase 3 mapping: only high severity triggers quick_retrain
        strategy_map = {
            SeverityLevel.NONE: EvolutionStrategy.DO_NOTHING,
            SeverityLevel.LOW: EvolutionStrategy.DO_NOTHING,
            SeverityLevel.MODERATE: EvolutionStrategy.DO_NOTHING,
            SeverityLevel.HIGH: EvolutionStrategy.QUICK_RETRAIN,
        }

        return strategy_map.get(severity, EvolutionStrategy.DO_NOTHING)

    def predict_strategy_with_confidence(self, severity: Optional[str] = None) -> Tuple[str, float]:
        """Predict strategy with confidence score."""
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
        """Get aggregator statistics."""
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
    'DRIFT_TYPE_WEIGHTS',
]
