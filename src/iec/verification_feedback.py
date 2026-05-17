"""
IEC Verification Feedback Loop.

*** RESERVED FOR PHASE 4 INTEGRATION ***

Phase 3 Status: This module is not used in the current Sequential Pipeline.
- Not imported by iec_controller.py
- Not instantiated by any operator
- Exported in __init__.py for future use

Phase 4 TODO:
- Integrate with IECController after quick_retrain execution
- Replace beta-based AdaptationRecord with strategy-based tracking
  (Phase 3 uses do_nothing/quick_retrain, not adjust_threshold)
- Implement verification windows post-retrain

Tracks whether IEC adaptations (threshold adjustments) actually improve metrics.
If no improvement after N verification windows, triggers escalation or rollback.

Phase 2D migration: copied from memstream_src/core/verification_feedback.py

Reference: no prior implementation exists — this is a new component.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

LOGGER = logging.getLogger('memstream-iec.verification')


@dataclass
class AdaptationRecord:
    """
    Record of a single IEC adaptation.

    Phase 4: Strategy-based (replaces Phase 2D beta-based tracking).
    """
    timestamp: float
    strategy: str
    neighborhood: str
    # Phase 4: old/new values depend on strategy type
    # For quick_retrain: old/new model_version or memory_snapshot_id
    old_value: float = 0.0
    new_value: float = 0.0
    anomaly_rate_before: float = 0.0
    anomaly_rate_after: float = 0.0
    drift_count_before: int = 0
    drift_count_after: int = 0
    improvement: float = 0.0  # positive = better, negative = worse
    verified: bool = False
    verdict: str = 'pending'  # 'pending', 'success', 'failed', 'neutral'


@dataclass
class VerificationWindow:
    """Metrics snapshot for a verification window."""
    timestamp: float
    neighborhood: str
    volume: int
    anomaly_rate: float
    null_rate: float
    violation_rate: float
    avg_anomaly_score: float
    delta_score: float
    drift_count: int


class VerificationFeedbackLoop:
    """
    Tracks adaptation outcomes and verifies effectiveness.

    Algorithm:
    1. Before adaptation: snapshot metrics
    2. After adaptation: wait N verification windows
    3. Compare post-adaptation metrics to pre-adaptation baseline
    4. If metrics worsened (improvement < 0): record failure
    5. If failures >= failure_threshold: escalate (reduce confidence, trigger audit)
    6. If consecutive failures >= rollback_threshold: trigger rollback

    Parameters:
        verification_windows: Number of MetaAggregator windows to wait before verifying (default: 3)
        improvement_threshold: Minimum improvement needed to count as success (default: 0.01)
        failure_threshold: Consecutive failures before escalating (default: 3)
        rollback_threshold: Consecutive failures before rollback (default: 5)
        max_history: Maximum adaptation records to keep (default: 500)
    """

    def __init__(
        self,
        verification_windows: int = 3,
        improvement_threshold: float = 0.01,
        failure_threshold: int = 3,
        rollback_threshold: int = 5,
        max_history: int = 500,
    ):
        self.verification_windows = verification_windows
        self.improvement_threshold = improvement_threshold
        self.failure_threshold = failure_threshold
        self.rollback_threshold = rollback_threshold
        self.max_history = max_history

        self._adaptation_history: deque = deque(maxlen=max_history)
        self._current_snapshot: Dict[str, VerificationWindow] = {}  # neighborhood -> snapshot
        self._pending_verification: Dict[str, AdaptationRecord] = {}  # neighborhood -> record
        self._consecutive_failures: Dict[str, int] = {}  # neighborhood -> count
        self._consecutive_successes: Dict[str, int] = {}  # neighborhood -> count
        self._window_counts: Dict[str, int] = {}  # neighborhood -> number of verification windows received

        self._total_adaptations = 0
        self._total_verified = 0
        self._total_success = 0
        self._total_failed = 0
        self._total_rollbacks = 0

    def snapshot_before_adaptation(
        self,
        neighborhood: str,
        strategy: str,
        old_value: float,
        new_value: float,
        metrics: Dict,
    ) -> None:
        """
        Record a snapshot before applying an adaptation.

        Args:
            neighborhood: The neighborhood being adapted
            strategy: The adaptation strategy (e.g., 'quick_retrain', 'do_nothing')
            old_value: Phase 4: model version or memory snapshot ID before adaptation
            new_value: Phase 4: model version or memory snapshot ID after adaptation
            metrics: Current meta-metrics from MetaAggregator
        """
        snapshot = VerificationWindow(
            timestamp=time.time(),
            neighborhood=neighborhood,
            volume=int(metrics.get('volume', 0)),
            anomaly_rate=float(metrics.get('anomaly_rate', 0.0)),
            null_rate=float(metrics.get('null_rate', 0.0)),
            violation_rate=float(metrics.get('violation_rate', 0.0)),
            avg_anomaly_score=float(metrics.get('avg_anomaly_score', 0.0)),
            delta_score=float(metrics.get('delta_score', 0.0)),
            drift_count=int(metrics.get('drift_count', 0)),
        )
        
        self._current_snapshot[neighborhood] = snapshot
        
        record = AdaptationRecord(
            timestamp=time.time(),
            strategy=strategy,
            neighborhood=neighborhood,
            old_value=old_value,
            new_value=new_value,
            anomaly_rate_before=snapshot.anomaly_rate,
            anomaly_rate_after=0.0,
            drift_count_before=snapshot.drift_count,
            drift_count_after=0,
            improvement=0.0,
        )
        
        self._pending_verification[neighborhood] = record
        self._window_counts[neighborhood] = 0
        self._total_adaptations += 1
        
        LOGGER.info(
            "[Verification] Snapshot recorded for %s: anomaly_rate=%.4f, drift_count=%d",
            neighborhood, snapshot.anomaly_rate, snapshot.drift_count
        )

    def record_verification_window(self, metrics: Dict) -> Optional[Dict]:
        """
        Record a new verification window metric and check pending verifications.

        Args:
            metrics: Current meta-metrics. Can be:
                - Dict keyed by neighborhood: {'manhattan': {...}, 'brooklyn': {...}}
                - Flat dict with 'neighborhood_id' field

        Returns:
            Dict with verification results if a pending adaptation is ready to be verified,
            or None if no pending verifications.
        """
        # Normalize: extract neighborhood from either format
        if 'neighborhood_id' in metrics:
            neighborhood = metrics.get('neighborhood_id', 'unknown')
        elif len(metrics) == 1 and isinstance(list(metrics.values())[0], dict):
            neighborhood = list(metrics.keys())[0]
        else:
            neighborhood = metrics.get('neighborhood', 'unknown')
        
        if neighborhood not in self._pending_verification:
            return None
        
        # Increment window counter
        self._window_counts[neighborhood] = self._window_counts.get(neighborhood, 0) + 1
        window_count = self._window_counts[neighborhood]
        
        # Only compute verdict after waiting for verification_windows
        if window_count < self.verification_windows:
            LOGGER.info(
                "[Verification] Window %d/%d for %s, still waiting...",
                window_count, self.verification_windows, neighborhood
            )
            return None
        
        record = self._pending_verification[neighborhood]
        before = self._current_snapshot.get(neighborhood)
        
        if before is None:
            return None
        
        # Compute improvement metrics
        # Positive improvement = anomaly_rate decreased or drift_count decreased
        anomaly_improvement = before.anomaly_rate - current_anomaly_rate
        drift_improvement = float(before.drift_count - current_drift_count)
        
        # Composite improvement score
        improvement = (anomaly_improvement * 0.7) + (drift_improvement * 0.3)
        
        record.anomaly_rate_after = current_anomaly_rate
        record.drift_count_after = current_drift_count
        record.improvement = improvement
        record.verified = True
        
        # Determine verdict
        if improvement > self.improvement_threshold:
            record.verdict = 'success'
            self._total_success += 1
            self._consecutive_failures[neighborhood] = 0
            self._consecutive_successes[neighborhood] = self._consecutive_successes.get(neighborhood, 0) + 1
        elif improvement < -self.improvement_threshold:
            record.verdict = 'failed'
            self._total_failed += 1
            self._consecutive_failures[neighborhood] = self._consecutive_failures.get(neighborhood, 0) + 1
            self._consecutive_successes[neighborhood] = 0
        else:
            record.verdict = 'neutral'
            self._consecutive_failures[neighborhood] = 0
            self._consecutive_successes[neighborhood] = 0
        
        # Move from pending to history
        del self._pending_verification[neighborhood]
        del self._window_counts[neighborhood]
        self._adaptation_history.append(record)
        self._total_verified += 1
        
        # Determine action
        consecutive_fails = self._consecutive_failures.get(neighborhood, 0)
        consecutive_successes = self._consecutive_successes.get(neighborhood, 0)
        
        result = {
            'neighborhood': neighborhood,
            'verdict': record.verdict,
            'improvement': improvement,
            'anomaly_rate_delta': anomaly_improvement,
            'drift_count_delta': drift_improvement,
            'consecutive_failures': consecutive_fails,
            'consecutive_successes': consecutive_successes,
            'strategy': record.strategy,
            'old_value': record.old_value,
            'new_value': record.new_value,
            'action': 'none',
        }
        
        # Escalation: too many consecutive failures
        if consecutive_fails >= self.failure_threshold and consecutive_fails < self.rollback_threshold:
            result['action'] = 'escalate'
            LOGGER.warning(
                "[Verification] ESCALATE %s: %d consecutive failures, "
                "improvement=%.4f, anomaly_rate delta=%.4f",
                neighborhood, consecutive_fails, improvement, anomaly_improvement
            )
        
        # Rollback: too many consecutive failures
        if consecutive_fails >= self.rollback_threshold:
            result['action'] = 'rollback'
            result['rollback_value'] = record.old_value
            self._total_rollbacks += 1
            LOGGER.error(
                "[Verification] ROLLBACK %s: %d consecutive failures, "
                "reverting from new_value %.4f to old_value %.4f",
                neighborhood, consecutive_fails, record.new_value, record.old_value
            )
        
        if record.verdict == 'success' and consecutive_successes >= 3:
            LOGGER.info(
                "[Verification] %s stabilized: %d consecutive successes",
                neighborhood, consecutive_successes
            )
        
        return result

    def get_status(self) -> Dict:
        """Get overall verification loop status."""
        total = len(self._adaptation_history)
        if total > 0:
            success_rate = self._total_success / total
            fail_rate = self._total_failed / total
        else:
            success_rate = 0.0
            fail_rate = 0.0
        
        return {
            'total_adaptations': self._total_adaptations,
            'total_verified': self._total_verified,
            'total_success': self._total_success,
            'total_failed': self._total_failed,
            'total_rollbacks': self._total_rollbacks,
            'success_rate': success_rate,
            'fail_rate': fail_rate,
            'pending_verifications': len(self._pending_verification),
            'neighborhood_failures': dict(self._consecutive_failures),
            'neighborhood_successes': dict(self._consecutive_successes),
        }

    def get_recent_history(self, n: int = 10) -> List[Dict]:
        """Get the N most recent adaptation records."""
        history = list(self._adaptation_history)
        recent = history[-n:] if len(history) > n else history
        return [
            {
                'timestamp': r.timestamp,
                'neighborhood': r.neighborhood,
                'strategy': r.strategy,
                'old_value': r.old_value,
                'new_value': r.new_value,
                'verdict': r.verdict,
                'improvement': r.improvement,
            }
            for r in reversed(recent)
        ]


__all__ = [
    'VerificationFeedbackLoop',
    'AdaptationRecord',
    'VerificationWindow',
]
