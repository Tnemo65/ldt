"""
Layer 3 Combined Functions - Voting Ensemble.

This module re-exports all Layer 3 operators for convenience.
Also provides factory functions for creating the complete Layer 3 pipeline.

Usage:
    from .layer3.layer3_functions import create_layer3_pipeline
    
    # Or import individual functions
    from .layer3 import VotingEnsembleFunction, MetaAggregator
"""

from typing import Dict, List, Tuple, Optional
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.common.typeinfo import Types

# Re-export all Layer 3 functions
from .voting_ensemble import (
    VotingEnsembleFunction,
    VotingEnsembleMapper,
    vote,
    get_decision_label,
    get_source_label,
    DECISION_ANOMALY,
    DECISION_CLEAN,
    SOURCE_CANARY_RULE,
    SOURCE_COMPLEX_ML,
    SOURCE_BOTH_AGREE,
)
from .meta_aggregator import (
    MetaAggregateFunction as MetaAggregator,
    MetaWindowProcessFunction as MetaAggregatorProcessWindowFunction,
    create_tumbling_window,
    calculate_delta_score,
    DEFAULT_WINDOW_SIZE_MS,
)

# For type hints
try:
    from pyflink.datastream import KeyedProcessFunction, ProcessFunction
except ImportError:
    KeyedProcessFunction = None
    ProcessFunction = None


# =============================================================================
# Pipeline Factory Functions
# =============================================================================

def create_layer3_pipeline(
    validated_stream,
    neighborhood_key_func=None,
    window_minutes: int = 5,
) -> Tuple:
    """
    Create complete Layer 3 pipeline.
    
    Pipeline Flow:
        Validated Stream (from Layer 1)
            → VotingEnsembleFunction (keyed by neighborhood)
            → MetaAggregator (tumbling windows per neighborhood)
    
    Args:
        validated_stream: DataStream[Dict] from Layer 1
        neighborhood_key_func: Function to extract neighborhood from record
                              Default: extract PULocationID-based neighborhood
        window_minutes: MetaAggregator window size in minutes
        
    Returns:
        Tuple of (decision_stream, metrics_stream)
            - decision_stream: Records with final decisions
            - metrics_stream: Aggregated per-neighborhood metrics
    """
    from pyflink.datastream import TimeCharacteristic
    from pyflink.datastream.window import TumblingEventTimeWindows
    from pyflink.common import Time
    
    # Default neighborhood extraction
    if neighborhood_key_func is None:
        neighborhood_key_func = lambda r: str(r.get('PULocationID', 'unknown'))
    
    # Voting Ensemble (keyed by neighborhood)
    decision_stream = (
        validated_stream
        .key_by(neighborhood_key_func)
        .process(VotingEnsembleFunction())
    )
    
    # Meta Aggregator (tumbling windows)
    metrics_stream = (
        decision_stream
        .key_by(neighborhood_key_func)
        .window(TumblingEventTimeWindows.of(Time.minutes(window_minutes)))
        .process(MetaAggregatorProcessWindowFunction())
    )
    
    return decision_stream, metrics_stream


def create_voting_stream(
    input_stream,
    neighborhood_key_func=None,
) -> 'DataStream':
    """
    Create voting ensemble stream (without meta aggregation).
    
    Args:
        input_stream: DataStream with canary_violations, is_anomaly, etc.
        neighborhood_key_func: Function to extract neighborhood
        
    Returns:
        DataStream with final_decision, decision_source, confidence
    """
    if neighborhood_key_func is None:
        neighborhood_key_func = lambda r: str(r.get('PULocationID', 'unknown'))
    
    return (
        input_stream
        .key_by(neighborhood_key_func)
        .process(VotingEnsembleFunction())
    )


def create_metrics_stream(
    decision_stream,
    neighborhood_key_func=None,
    window_minutes: int = 5,
) -> 'DataStream':
    """
    Create meta-aggregated metrics stream.
    
    Args:
        decision_stream: DataStream with final_decision already added
        neighborhood_key_func: Function to extract neighborhood
        window_minutes: Window size in minutes
        
    Returns:
        DataStream with aggregated metrics per neighborhood per window
    """
    from pyflink.datastream.window import TumblingEventTimeWindows
    from pyflink.common import Time
    
    if neighborhood_key_func is None:
        neighborhood_key_func = lambda r: str(r.get('PULocationID', 'unknown'))
    
    return (
        decision_stream
        .key_by(neighborhood_key_func)
        .window(TumblingEventTimeWindows.of(Time.minutes(window_minutes)))
        .process(MetaAggregatorProcessWindowFunction())
    )


# =============================================================================
# Output Tag Definitions
# =============================================================================

def get_output_tags():
    """
    Get OutputTags for Layer 3 side outputs.
    
    Returns:
        Dict of output tag names to OutputTag objects
    """
    from pyflink.datastream import OutputTag
    
    return {
        'decision_stream': OutputTag('decision_stream'),
        'metrics_stream': OutputTag('metrics_stream'),
        'anomaly_stream': OutputTag('anomaly_stream'),
        'clean_stream': OutputTag('clean_stream'),
    }


# =============================================================================
# Stream Splitter
# =============================================================================

class Layer3StreamSplitter:
    """
    Splits Layer 3 output into separate streams based on decision.
    
    Usage:
        splitter = Layer3StreamSplitter()
        anomaly_stream, clean_stream = splitter.split(decision_stream)
    """
    
    def __init__(self, neighborhood_key_func=None):
        self.neighborhood_key_func = neighborhood_key_func or (
            lambda r: str(r.get('PULocationID', 'unknown'))
        )
    
    def split(
        self,
        decision_stream,
    ) -> Tuple['DataStream', 'DataStream']:
        """
        Split decision stream into anomaly and clean streams.
        
        Args:
            decision_stream: DataStream with final_decision field
            
        Returns:
            Tuple of (anomaly_stream, clean_stream)
        """
        anomaly_stream = (
            decision_stream
            .filter(lambda r: r.get('final_decision') == DECISION_ANOMALY)
        )
        
        clean_stream = (
            decision_stream
            .filter(lambda r: r.get('final_decision') == DECISION_CLEAN)
        )
        
        return anomaly_stream, clean_stream
    
    def split_with_sources(
        self,
        decision_stream,
    ) -> Dict[str, 'DataStream']:
        """
        Split into separate streams by decision source.
        
        Returns:
            Dict of source_name -> DataStream
        """
        return {
            'canary_anomaly': decision_stream.filter(
                lambda r: r.get('decision_source') == SOURCE_CANARY_RULE
            ),
            'ml_anomaly': decision_stream.filter(
                lambda r: r.get('decision_source') == SOURCE_COMPLEX_ML
            ),
            'both_clean': decision_stream.filter(
                lambda r: r.get('decision_source') == SOURCE_BOTH_AGREE
            ),
        }


# =============================================================================
# Layer 3 Metrics
# =============================================================================

class Layer3Metrics:
    """
    Aggregated metrics for all Layer 3 operators.
    """
    
    def __init__(self):
        self.voting_total = 0
        self.voting_anomalies = 0
        self.voting_clean = 0
        self.voting_canary = 0
        self.voting_ml = 0
        
        self.windows_aggregated = 0
        self.total_window_records = 0
    
    def record_voting(
        self,
        final_decision: str,
        decision_source: str,
    ):
        """Record voting decision."""
        self.voting_total += 1
        
        if final_decision == DECISION_ANOMALY:
            self.voting_anomalies += 1
            if decision_source == SOURCE_CANARY_RULE:
                self.voting_canary += 1
            elif decision_source == SOURCE_COMPLEX_ML:
                self.voting_ml += 1
        else:
            self.voting_clean += 1
    
    def record_window(self, record_count: int):
        """Record window aggregation."""
        self.windows_aggregated += 1
        self.total_window_records += record_count
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        return {
            'voting': {
                'total': self.voting_total,
                'anomalies': self.voting_anomalies,
                'clean': self.voting_clean,
                'anomaly_rate': self.voting_anomalies / max(self.voting_total, 1),
                'canary_triggered': self.voting_canary,
                'ml_triggered': self.voting_ml,
            },
            'windows': {
                'aggregated': self.windows_aggregated,
                'total_records': self.total_window_records,
                'avg_records_per_window': (
                    self.total_window_records / max(self.windows_aggregated, 1)
                ),
            },
        }


# =============================================================================
# Delta Score Analysis
# =============================================================================

def analyze_delta_score(metrics: Dict) -> Dict:
    """
    Analyze delta score and provide interpretation.
    
    Delta Score Interpretation:
        - < 0.2: Low disagreement (canary and ML agree)
        - 0.2-0.5: Moderate disagreement
        - > 0.5: High disagreement (one system catching what other misses)
    
    Args:
        metrics: Dict with violation_rate and anomaly_rate
        
    Returns:
        Analysis dict with delta_score and interpretation
    """
    violation_rate = metrics.get('violation_rate', 0.0)
    anomaly_rate = metrics.get('anomaly_rate', 0.0)
    
    delta = calculate_delta_score(violation_rate, anomaly_rate)
    
    if delta < 0.2:
        interpretation = 'low_disagreement'
        description = 'Canary and ML agree well'
    elif delta < 0.5:
        interpretation = 'moderate_disagreement'
        description = 'Some divergence between systems'
    else:
        interpretation = 'high_disagreement'
        description = 'Significant divergence - review calibration'
    
    return {
        'delta_score': delta,
        'violation_rate': violation_rate,
        'anomaly_rate': anomaly_rate,
        'interpretation': interpretation,
        'description': description,
        'recommendation': _get_delta_recommendation(delta, violation_rate, anomaly_rate),
    }


def _get_delta_recommendation(
    delta: float,
    violation_rate: float,
    anomaly_rate: float,
) -> str:
    """Get recommendation based on delta score."""
    if delta < 0.2:
        if violation_rate > anomaly_rate:
            return "Canary more sensitive - consider tuning rules"
        elif anomaly_rate > violation_rate:
            return "ML more sensitive - consider adjusting threshold"
        else:
            return "Systems aligned - monitoring healthy"
    
    elif delta < 0.5:
        if violation_rate > anomaly_rate:
            return "Canary catching issues ML misses - review ML model"
        else:
            return "ML catching issues Canary misses - review rule thresholds"
    
    else:
        return "High disagreement - investigate calibration of both systems"


__all__ = [
    # Individual functions
    'VotingEnsembleFunction',
    'vote',
    'get_decision_label',
    'get_source_label',
    'DECISION_ANOMALY',
    'DECISION_CLEAN',
    'SOURCE_CANARY_RULE',
    'SOURCE_COMPLEX_ML',
    'SOURCE_BOTH_AGREE',
    'MetaAggregator',
    'MetaAggregatorProcessWindowFunction',
    'create_tumbling_window',
    'calculate_delta_score',
    'DEFAULT_WINDOW_SIZE_MS',
    # Pipeline factories
    'create_layer3_pipeline',
    'create_voting_stream',
    'create_metrics_stream',
    'get_output_tags',
    'Layer3StreamSplitter',
    # Metrics
    'Layer3Metrics',
    'analyze_delta_score',
]
