"""
VotingEnsembleMapper - Layer 3 Priority-Based Fusion.

Implements priority-based voting between canary and ML decisions.
Priority: Canary > ML

Logic:
    IF canary_violations is not empty:
        final_decision = 'ANOMALY'
        decision_source = 'canary_rule'
        confidence = 1.0

    ELIF is_anomaly == True (ML):
        final_decision = 'ANOMALY'
        decision_source = 'complex_ml'
        confidence = min(anomaly_score / threshold, 1.0)

    ELSE (both clean):
        final_decision = 'CLEAN'
        decision_source = 'both_agree'
        confidence = max(0.0, 1.0 - min(anomaly_score / threshold, 1.0))

Note: Records go through EITHER canary OR complex branch (via union),
      not both. A record from canary has canary_violations but NO is_anomaly.
      A record from complex has is_anomaly but NO canary_violations.

Reference: original_flow.md lines 570-598
"""

import logging
from typing import Dict, Optional

from enum import Enum

LOGGER = logging.getLogger('cadqstream.layer3')
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)


class VotingDecision(Enum):
    """Voting decision types."""
    CLEAN = 'CLEAN'
    ANOMALY = 'ANOMALY'


class DecisionSource(Enum):
    """Source of voting decision."""
    CANARY_RULE = 'canary_rule'
    COMPLEX_ML = 'complex_ml'
    BOTH_AGREE = 'both_agree'


def decide(record: Dict) -> Dict:
    """
    Standalone voting decision function.
    
    Args:
        record: Record with canary_violations and/or is_anomaly fields
        
    Returns:
        Record with added final_decision, decision_source, confidence
    """
    canary_violations = record.get('canary_violations', [])
    is_anomaly = record.get('is_anomaly', False)
    anomaly_score = record.get('anomaly_score', 0.0)
    threshold = record.get('threshold', 0.5)
    
    if canary_violations and len(canary_violations) > 0:
        return {
            **record,
            'final_decision': VotingDecision.ANOMALY.value,
            'decision_source': DecisionSource.CANARY_RULE.value,
            'confidence': 1.0,
        }
    
    elif is_anomaly:
        confidence = min(anomaly_score / threshold, 1.0) if threshold > 0 else 1.0
        return {
            **record,
            'final_decision': VotingDecision.ANOMALY.value,
            'decision_source': DecisionSource.COMPLEX_ML.value,
            'confidence': confidence,
        }
    
    else:
        confidence = max(0.0, 1.0 - min(anomaly_score / threshold, 1.0)) if threshold > 0 else 1.0
        return {
            **record,
            'final_decision': VotingDecision.CLEAN.value,
            'decision_source': DecisionSource.BOTH_AGREE.value,
            'confidence': confidence,
        }


class VotingAggregator:
    """
    Aggregator for voting statistics (used in windows).
    
    Collects voting decisions and computes aggregate statistics.
    """
    
    def __init__(self):
        self.total = 0
        self.anomaly = 0
        self.clean = 0
        self.canary_decisions = 0
        self.ml_decisions = 0
        self.sum_confidence = 0.0
    
    def add(self, record: Dict):
        """Add a record to aggregation."""
        self.total += 1
        
        final_decision = record.get('final_decision', 'CLEAN')
        decision_source = record.get('decision_source', 'unknown')
        confidence = record.get('confidence', 1.0)
        
        self.sum_confidence += confidence
        
        if final_decision == VotingDecision.ANOMALY.value:
            self.anomaly += 1
            if decision_source == DecisionSource.CANARY_RULE.value:
                self.canary_decisions += 1
            elif decision_source == DecisionSource.COMPLEX_ML.value:
                self.ml_decisions += 1
        else:
            self.clean += 1
    
    def get_stats(self) -> Dict:
        """Get aggregate statistics."""
        if self.total == 0:
            return {
                'total': 0,
                'anomaly_rate': 0.0,
                'canary_rate': 0.0,
                'ml_rate': 0.0,
                'avg_confidence': 1.0,
            }
        
        return {
            'total': self.total,
            'anomaly_rate': self.anomaly / self.total,
            'canary_rate': self.canary_decisions / self.total,
            'ml_rate': self.ml_decisions / self.total,
            'avg_confidence': self.sum_confidence / self.total,
        }


# Backward-compatibility alias: VotingEnsembleFunction was a KeyedProcessFunction
# that was removed in favor of VotingEnsembleMapper (MapFunction, stateless).
# Other modules (layer3_functions.py, layer3_job.py, tests) import this name.
# Using a simple wrapper that delegates to VotingEnsembleMapper.
class VotingEnsembleFunction:
    """
    Backward-compatibility alias for VotingEnsembleMapper.

    Deprecated: Use VotingEnsembleMapper directly.
    """

    def __init__(self):
        self._inner = VotingEnsembleMapper()

    def __call__(self, *args, **kwargs):
        return self._inner(*args, **kwargs)

    def __str__(self):
        return f"VotingEnsembleFunction(backward-compat → VotingEnsembleMapper)"

    def __repr__(self):
        return self.__str__()
