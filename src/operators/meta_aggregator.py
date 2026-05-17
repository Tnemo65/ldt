"""
Layer 3 MetaAggregator - Sequential Pipeline Phase 3.

Sequential pipeline (no dual branch, no voting):
1. SequentialFinalDecisionFunction: set final_decision per record
2. MetaAggregateFunction: aggregate over 1-minute windows per neighborhood
3. MetaWindowProcessFunction: add window metadata + compute delta_score

Final Decision Logic (Phase 3):
    - has_violation=True (Canary) -> final_decision='ANOMALY', source='canary_rule'
    - is_anomaly=True (MemStream) -> final_decision='ANOMALY', source='memstream_ml'
    - else -> final_decision='CLEAN', source='pass'

Meta-Metrics (6 signals per neighborhood per minute):
1. volume — record count
2. null_rate — % records with null fields
3. violation_rate — % Canary violations
4. anomaly_rate — % MemStream anomalies
5. avg_anomaly_score — mean MemStream score
6. delta_score — |violation_rate - anomaly_rate| / (violation_rate + anomaly_rate + eps)

These metrics feed ADWIN drift detection and IEC strategy selection.
"""

from pyflink.datastream import MapFunction, AggregateFunction, ProcessWindowFunction
from pyflink.datastream.window import TumblingEventTimeWindows
from pyflink.common.time import Time
from pyflink.common.typeinfo import Types
from datetime import datetime
import json
import logging
import urllib.request

LOGGER = logging.getLogger('cadqstream-meta-agg')


class SequentialFinalDecisionFunction(MapFunction):
    """Set final_decision in the sequential pipeline.

    No voting — final decision is determined by:
    1. Canary violations (has_violation=True) -> ANOMALY (canary_rule)
    2. MemStream anomalies (is_anomaly=True) -> ANOMALY (memstream_ml)
    3. Otherwise -> CLEAN (pass)

    Decision is stored in-place on the record for downstream MetaAggregator.
    """

    def map(self, value):
        """Compute final decision for a record.

        Args:
            value: Record with has_violation and is_anomaly fields

        Returns:
            Record with final_decision, decision_source, confidence fields added
        """
        if value is None:
            return {
                'final_decision': 'CLEAN',
                'decision_source': 'null_input',
                'confidence': 0.0,
                'has_violation': False,
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'seq_timestamp': '',
                '_dlq': True,
                '_dlq_reason': 'null_input',
                '_dlq_category': 'VALIDATION_ERROR',
                '_dlq_operator': 'SequentialFinalDecisionFunction',
                'trip_id': 'dlq_null',
                'tpep_pickup_datetime': '',
            }

        if not isinstance(value, dict):
            return {
                'final_decision': 'CLEAN',
                'decision_source': f'not_dict:{type(value).__name__}',
                'confidence': 0.0,
                'has_violation': False,
                'is_anomaly': False,
                'anomaly_score': 0.0,
                'seq_timestamp': '',
                '_dlq': True,
                '_dlq_reason': f'not_dict:{type(value).__name__}',
                '_dlq_category': 'VALIDATION_ERROR',
                '_dlq_operator': 'SequentialFinalDecisionFunction',
                'trip_id': f'dlq_{type(value).__name__[:20]}',
                'tpep_pickup_datetime': '',
            }

        has_violation = value.get('has_violation', False)
        is_anomaly = value.get('is_anomaly', False)
        anomaly_score = value.get('anomaly_score', 0.0)

        if has_violation:
            final_decision = 'ANOMALY'
            decision_source = 'canary_rule'
            confidence = 1.0
        elif is_anomaly:
            final_decision = 'ANOMALY'
            decision_source = 'memstream_ml'
            threshold = value.get('threshold', 1.0)
            confidence = min(anomaly_score / threshold if threshold > 0 else 0.5, 1.0)
        else:
            final_decision = 'CLEAN'
            decision_source = 'pass'
            confidence = 1.0

        value['final_decision'] = final_decision
        value['decision_source'] = decision_source
        value['confidence'] = float(confidence)
        value['seq_timestamp'] = value.get('tpep_pickup_datetime', '')

        return value


class MetaAggregateFunction(AggregateFunction):
    """Compute 6 meta-metrics per neighborhood per 1-minute window.

    Meta-Metrics:
    1. volume — record count
    2. null_rate — % null fields
    3. violation_rate — % Canary violations
    4. anomaly_rate — % final anomalies
    5. avg_anomaly_score — mean MemStream score
    6. delta_score — computed in MetaWindowProcessFunction

    These metrics feed into ADWIN drift detection and IEC strategy selection.
    """

    def create_accumulator(self):
        return {
            'volume': 0,
            'null_count': 0,
            'violation_count': 0,
            'anomaly_count': 0,
            'score_sum': 0.0,
            'score_count': 0
        }

    def add(self, value, accumulator):
        # value may be a tuple (neighborhood, record) from ExtractNeighborhoodFunction
        if isinstance(value, tuple):
            record = value[1]
        else:
            record = value

        if record is None:
            return accumulator

        accumulator['volume'] += 1

        if isinstance(record, dict) and any(
            record.get(field) is None
            for field in ['fare_amount', 'trip_distance', 'passenger_count']
        ):
            accumulator['null_count'] += 1

        if isinstance(record, dict) and record.get('has_violation', False):
            accumulator['violation_count'] += 1

        if isinstance(record, dict) and record.get('final_decision') == 'ANOMALY':
            accumulator['anomaly_count'] += 1

        if isinstance(record, dict):
            anomaly_score = record.get('anomaly_score')
            if anomaly_score is not None:
                accumulator['score_sum'] += anomaly_score
                accumulator['score_count'] += 1

        return accumulator

    def get_result(self, accumulator):
        volume = accumulator['volume']

        if volume == 0:
            return {
                'volume': 0,
                'null_rate': 0.0,
                'violation_rate': 0.0,
                'anomaly_rate': 0.0,
                'avg_anomaly_score': 0.0,
                'delta_score': 0.0,
                '_dlq': True,
                '_dlq_reason': 'empty_window',
            }

        null_rate = accumulator['null_count'] / volume
        violation_rate = accumulator['violation_count'] / volume
        anomaly_rate = accumulator['anomaly_count'] / volume

        avg_score = 0.0
        if accumulator['score_count'] > 0:
            avg_score = accumulator['score_sum'] / accumulator['score_count']

        return {
            'volume': volume,
            'null_rate': null_rate,
            'violation_rate': violation_rate,
            'anomaly_rate': anomaly_rate,
            'avg_anomaly_score': avg_score,
            'delta_score': 0.0
        }

    def merge(self, acc1, acc2):
        return {
            'volume': acc1['volume'] + acc2['volume'],
            'null_count': acc1['null_count'] + acc2['null_count'],
            'violation_count': acc1['violation_count'] + acc2['violation_count'],
            'anomaly_count': acc1['anomaly_count'] + acc2['anomaly_count'],
            'score_sum': acc1['score_sum'] + acc2['score_sum'],
            'score_count': acc1['score_count'] + acc2['score_count']
        }


class MetaWindowProcessFunction(ProcessWindowFunction):
    """Add window metadata and compute delta_score.

    delta_score = |violation_rate - anomaly_rate| / (violation_rate + anomaly_rate + eps)
    Measures divergence between Canary (rule-based) and MemStream (ML-based) detection.
    """

    def __init__(self):
        pass

    def process(self, key, context, elements):
        """Process window results."""
        metrics = next(iter(elements))

        if metrics is None:
            yield {
                'volume': 0,
                'null_rate': 0.0,
                'violation_rate': 0.0,
                'anomaly_rate': 0.0,
                'avg_anomaly_score': 0.0,
                'delta_score': 0.0,
                'window_start': datetime.fromtimestamp(context.window().start / 1000).isoformat(),
                'window_end': datetime.fromtimestamp(context.window().end / 1000).isoformat(),
                'neighborhood_id': key,
                '_dlq': True,
                '_dlq_reason': 'null_metrics',
            }
            return

        violation_rate = metrics.get('violation_rate', 0.0)
        anomaly_rate = metrics.get('anomaly_rate', 0.0)
        epsilon = 1e-6

        delta_score = abs(violation_rate - anomaly_rate) / (violation_rate + anomaly_rate + epsilon)
        metrics['delta_score'] = delta_score

        window = context.window()
        metrics['window_start'] = datetime.fromtimestamp(window.start / 1000).isoformat()
        metrics['window_end'] = datetime.fromtimestamp(window.end / 1000).isoformat()
        metrics['neighborhood_id'] = key

        yield metrics


def extract_neighborhood_key(record: dict) -> str:
    """Extract neighborhood key for spatial grouping.

    Returns:
        Neighborhood key (e.g., 'manhattan', 'brooklyn', etc.)
    """
    zone_id = record.get('PULocationID', 0)

    if zone_id <= 50:
        return 'manhattan'
    elif zone_id <= 100:
        return 'brooklyn'
    elif zone_id <= 150:
        return 'queens'
    elif zone_id <= 200:
        return 'bronx'
    elif zone_id in [132, 138]:
        return 'airport'
    else:
        return 'staten_island'
