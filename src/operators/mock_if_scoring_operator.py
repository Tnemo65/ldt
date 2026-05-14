"""
Mock sklearn IsolationForest Scoring Operator for Layer 2 Complex Branch.

This mock operator is used when no trained ML model is available.
It scores records using simple heuristic rules instead of ML.

For production with trained model, use:
1. IFScoringOperator with BroadcastState (requires model loaded via Kafka)
2. Or: Pre-load model and use BroadcastProcessFunction pattern
"""

from pyflink.datastream import MapFunction
import logging
import hashlib

LOGGER = logging.getLogger('cadqstream-if-scoring')


class MockIFScoringOperator(MapFunction):
    """Score records with heuristic rules when ML model is unavailable.

    This mock provides basic anomaly scoring using:
    - Distance-based heuristics
    - Fare amount statistics
    - Passenger count anomalies
    - Speed anomalies
    """

    def __init__(self):
        pass

    def open(self, runtime_context):
        pass

    def map(self, value):
        """Score record with heuristic rules (mock ML scoring).

        Args:
            value: Record dict (already passed Layer 1 validation)

        Returns:
            Enriched record with anomaly_score, threshold, is_anomaly, context_key
        """
        if value is None:
            return None

        try:
            anomaly_score = 0.0
            anomaly_reasons = []

            # Heuristic 1: Extreme distance
            distance = float(value.get('trip_distance', 0))
            if distance > 50:
                anomaly_score += 0.3
                anomaly_reasons.append('extreme_distance')
            elif distance > 100:
                anomaly_score += 0.5
                anomaly_reasons.append('very_extreme_distance')

            # Heuristic 2: Extreme fare
            fare = float(value.get('fare_amount', 0))
            if fare > 100:
                anomaly_score += 0.2
                anomaly_reasons.append('high_fare')
            if fare > 500:
                anomaly_score += 0.3
                anomaly_reasons.append('extreme_fare')

            # Heuristic 3: Speed anomaly (fare/distance ratio)
            if distance > 0:
                fare_per_mile = fare / distance
                if fare_per_mile > 10:
                    anomaly_score += 0.2
                    anomaly_reasons.append('high_fare_per_mile')
                if fare_per_mile > 50:
                    anomaly_score += 0.3
                    anomaly_reasons.append('extreme_fare_per_mile')

            # Heuristic 4: Zero/near-zero distance with fare
            if distance < 0.1 and fare > 10:
                anomaly_score += 0.4
                anomaly_reasons.append('zero_distance_fare')

            # Heuristic 5: Unusual passenger count
            passengers = int(value.get('passenger_count', 1))
            if passengers == 0 or passengers > 6:
                anomaly_score += 0.2
                anomaly_reasons.append('unusual_passengers')

            # Heuristic 6: Congestion surcharge anomalies
            congestion = float(value.get('congestion_surcharge', 0))
            if congestion > 10:
                anomaly_score += 0.1
                anomaly_reasons.append('high_congestion')

            # Normalize score to [0, 1] range
            anomaly_score = min(anomaly_score, 1.0)

            # Context key for tracking
            context_key = f"mock_score_{int(anomaly_score * 10)}"

            # Threshold: 0.5 (score above this is anomaly)
            threshold = 0.5
            is_anomaly = anomaly_score > threshold

            return {
                **value,
                'anomaly_score': float(anomaly_score),
                'threshold': float(threshold),
                'is_anomaly': bool(is_anomaly),
                'context_key': context_key,
                'anomaly_reasons': anomaly_reasons,
                'ml_model': 'mock_heuristic_v1'
            }

        except Exception as e:
            LOGGER.error("[MockIFScoring] ERROR scoring record: %s", e)
            return {
                **value,
                'anomaly_score': -1.0,
                'threshold': 0.5,
                'is_anomaly': False,
                'context_key': 'error',
                'anomaly_reasons': [],
                'ml_model': 'mock_heuristic_v1',
                'scoring_error': str(e)
            }
