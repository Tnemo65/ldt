"""
Layer 3 MetaAggregator - Voting Ensemble with Meta-Metrics.
Task 3.11-3.15: Combine Canary + Complex decisions, compute drift signals

Two-Stage Operation:
1. Voting Ensemble: Combine Canary (rules) + Complex (ML) per record
2. Meta-Metrics: Aggregate over 1-minute windows per neighborhood

Voting Logic:
- If Canary has violations → ANOMALY (overrides ML)
- If ML score > threshold → ANOMALY
- Otherwise → CLEAN

Meta-Metrics (6 signals per neighborhood per minute):
1. volume - Record count
2. null_rate - % records with null fields
3. violation_rate - % Canary violations
4. anomaly_rate - % ML anomalies
5. avg_anomaly_score - Mean ML score
6. delta_score - Change from previous window

Spec: Task 3.11-3.15 (Voting + windowed aggregation)
"""

from pyflink.datastream import MapFunction, AggregateFunction, ProcessWindowFunction
from pyflink.datastream.window import TumblingEventTimeWindows
from pyflink.common.time import Time
from pyflink.common.typeinfo import Types
from datetime import datetime


class VotingEnsembleFunction(MapFunction):
    """Combine Canary (rule-based) + Complex (ML-based) decisions.

    Voting Logic:
    - Priority 1: Canary violations (hard business rules)
    - Priority 2: ML anomaly score > threshold
    - Otherwise: CLEAN

    Args:
        canary_weight: Weight for Canary branch (default 1.0 = override)
        complex_weight: Weight for Complex branch (default 1.0)
    """

    def __init__(self, canary_weight=1.0, complex_weight=1.0):
        """Initialize voting ensemble."""
        self.canary_weight = canary_weight
        self.complex_weight = complex_weight
        self.records_processed = 0
        self.anomalies_detected = 0

    def map(self, value):
        """Apply voting ensemble to merged record.

        Args:
            value: Merged record from Rendezvous (has both Canary + Complex fields)

        Returns:
            Record with final_decision, confidence, and decision_source
        """
        if value is None:
            return None

        self.records_processed += 1

        # Extract Canary decision
        has_violation = value.get('has_violation', False)
        violations = value.get('canary_violations', [])

        # Extract Complex decision
        is_ml_anomaly = value.get('is_anomaly', False)
        anomaly_score = value.get('anomaly_score', 0.0)
        threshold = value.get('threshold', 0.5)

        # Voting logic
        if has_violation:
            # Priority 1: Canary violations override ML
            final_decision = 'ANOMALY'
            decision_source = 'canary_rule'
            confidence = 1.0  # High confidence (hard rule)
            self.anomalies_detected += 1

        elif is_ml_anomaly:
            # Priority 2: ML anomaly detection
            final_decision = 'ANOMALY'
            decision_source = 'complex_ml'
            # Confidence based on how far score exceeds threshold
            confidence = min((anomaly_score / threshold) if threshold > 0 else 0.5, 1.0)
            self.anomalies_detected += 1

        else:
            # Clean record
            final_decision = 'CLEAN'
            decision_source = 'both_agree'
            # Confidence based on how far below threshold
            confidence = 1.0 - (anomaly_score / threshold) if threshold > 0 else 0.5

        # Enrich record with voting results
        value['final_decision'] = final_decision
        value['decision_source'] = decision_source
        value['confidence'] = float(confidence)
        value['voting_timestamp'] = datetime.utcnow().isoformat()

        # Log stats periodically
        if self.records_processed % 10000 == 0:
            anomaly_rate = self.anomalies_detected / self.records_processed * 100
            print(f"[VotingEnsemble] Processed: {self.records_processed:,}, "
                  f"Anomalies: {self.anomalies_detected:,} ({anomaly_rate:.2f}%)")

        return value


class MetaAggregateFunction(AggregateFunction):
    """Compute 6 meta-metrics per neighborhood per 1-minute window.

    Meta-Metrics:
    1. volume - Record count
    2. null_rate - % null fields
    3. violation_rate - % Canary violations
    4. anomaly_rate - % final anomalies
    5. avg_anomaly_score - Mean ML score
    6. delta_score - Change from previous window (computed in ProcessWindowFunction)

    These metrics feed into drift detection (ADWIN-U) and IEC strategy selection.
    """

    def create_accumulator(self):
        """Create empty accumulator."""
        return {
            'volume': 0,
            'null_count': 0,
            'violation_count': 0,
            'anomaly_count': 0,
            'score_sum': 0.0,
            'score_count': 0
        }

    def add(self, value, accumulator):
        """Add record to accumulator.

        Args:
            value: Record with voting decision
            accumulator: Current accumulator state

        Returns:
            Updated accumulator
        """
        accumulator['volume'] += 1

        # Count nulls (simplified check)
        if any(value.get(field) is None for field in ['fare_amount', 'trip_distance', 'passenger_count']):
            accumulator['null_count'] += 1

        # Count Canary violations
        if value.get('has_violation', False):
            accumulator['violation_count'] += 1

        # Count final anomalies
        if value.get('final_decision') == 'ANOMALY':
            accumulator['anomaly_count'] += 1

        # Accumulate ML scores
        anomaly_score = value.get('anomaly_score')
        if anomaly_score is not None:
            accumulator['score_sum'] += anomaly_score
            accumulator['score_count'] += 1

        return accumulator

    def get_result(self, accumulator):
        """Compute final meta-metrics.

        Args:
            accumulator: Final accumulator state

        Returns:
            Dict with 6 meta-metrics (delta_score computed later)
        """
        volume = accumulator['volume']

        if volume == 0:
            return None

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
            'delta_score': 0.0  # Will be computed by ProcessWindowFunction
        }

    def merge(self, acc1, acc2):
        """Merge two accumulators (for parallelism).

        Args:
            acc1: First accumulator
            acc2: Second accumulator

        Returns:
            Merged accumulator
        """
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

    Computes delta_score (change from previous window) using ValueState.
    """

    def __init__(self):
        """Initialize process function."""
        self.prev_anomaly_rate_state = None

    def open(self, runtime_context):
        """Initialize state for tracking previous window."""
        from pyflink.datastream.state import ValueStateDescriptor

        descriptor = ValueStateDescriptor(
            "prev_anomaly_rate",
            Types.DOUBLE()
        )

        self.prev_anomaly_rate_state = runtime_context.get_state(descriptor)

    def process(self, key, context, elements):
        """Process window results.

        Args:
            key: Window key (e.g., neighborhood_id)
            context: Window context
            elements: Aggregated results (single element from AggregateFunction)

        Yields:
            Meta-metrics with window metadata and delta_score
        """
        # Get aggregated metrics
        metrics = next(iter(elements))

        if metrics is None:
            return

        # Compute delta_score
        current_anomaly_rate = metrics['anomaly_rate']
        prev_anomaly_rate = self.prev_anomaly_rate_state.value()

        if prev_anomaly_rate is None:
            delta_score = 0.0
        else:
            delta_score = current_anomaly_rate - prev_anomaly_rate

        # Update state for next window
        self.prev_anomaly_rate_state.update(current_anomaly_rate)

        # Add delta_score
        metrics['delta_score'] = delta_score

        # Add window metadata
        window = context.window()
        metrics['window_start'] = datetime.fromtimestamp(window.start / 1000).isoformat()
        metrics['window_end'] = datetime.fromtimestamp(window.end / 1000).isoformat()
        metrics['neighborhood_id'] = key  # Spatial grouping key

        yield metrics


def extract_neighborhood_key(record: dict) -> str:
    """Extract neighborhood key for spatial grouping.

    Args:
        record: Trip record

    Returns:
        Neighborhood key (e.g., 'manhattan', 'brooklyn', etc.)
    """
    zone_id = record.get('PULocationID', 0)

    # Simplified neighborhood mapping
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
