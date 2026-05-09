"""
iForestASD Scoring Operator for Layer 2 Complex Branch.
Spec: Task 2.6-2.10 (Broadcast State pattern with V1.9 bug fixes)

Scores records using iForestASD model with context-aware thresholds.
Model loaded from Broadcast State for hot-swappable updates.
"""

from pyflink.datastream import MapFunction
from pyflink.datastream.state import MapStateDescriptor
from pyflink.common.typeinfo import Types
import pickle
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.features.vectorizer import FeatureVectorizer


def get_context_key(record: dict, neighborhood_mapping: dict = None) -> str:
    """Generate 4D context key for threshold lookup.

    Context dimensions:
    - trip_type: short (<2mi), medium (2-10mi), long (>10mi)
    - time_window: morning_rush, midday, evening_rush, night
    - day_type: weekday, weekend
    - neighborhood: manhattan, brooklyn, queens, bronx, staten_island, airport

    Args:
        record: Trip record with pickup_datetime and distance
        neighborhood_mapping: Zone to neighborhood mapping (optional)

    Returns:
        Context key string (e.g., "medium_evening_rush_weekday_manhattan")
    """
    try:
        from datetime import datetime

        # Parse datetime
        pickup_dt = record.get('tpep_pickup_datetime')
        if isinstance(pickup_dt, str):
            pickup_dt = datetime.fromisoformat(pickup_dt)

        # Trip type (based on distance)
        distance = record.get('trip_distance', 0)
        if distance < 2:
            trip_type = 'short'
        elif distance < 10:
            trip_type = 'medium'
        else:
            trip_type = 'long'

        # Time window
        hour = pickup_dt.hour
        if 6 <= hour < 10:
            time_window = 'morning_rush'
        elif 17 <= hour < 20:
            time_window = 'evening_rush'
        elif 22 <= hour or hour < 6:
            time_window = 'night'
        else:
            time_window = 'midday'

        # Day type
        day_type = 'weekend' if pickup_dt.weekday() >= 5 else 'weekday'

        # Neighborhood (simplified mapping if not provided)
        zone_id = record.get('PULocationID', 0)
        if neighborhood_mapping:
            neighborhood = neighborhood_mapping.get(str(zone_id), 'unknown')
        else:
            # Simple heuristic mapping
            if zone_id <= 50:
                neighborhood = 'manhattan'
            elif zone_id <= 100:
                neighborhood = 'brooklyn'
            elif zone_id <= 150:
                neighborhood = 'queens'
            elif zone_id <= 200:
                neighborhood = 'bronx'
            elif zone_id in [132, 138]:  # JFK, LGA
                neighborhood = 'airport'
            else:
                neighborhood = 'staten_island'

        return f"{trip_type}_{time_window}_{day_type}_{neighborhood}"

    except Exception as e:
        return "unknown_unknown_unknown_unknown"


class IFScoringOperator(MapFunction):
    """Score records with iForestASD using Broadcast State model.

    V1.9 Spec:
    - Loads model from Broadcast State (hot-swappable)
    - Uses context-aware thresholds (4D clustering)
    - Returns enriched record with anomaly_score and is_anomaly flag

    Features:
    - 21D enhanced vectorization (15D base + 6D ratio)
    - StandardScaler normalization
    - Per-context thresholds for reduced FPR
    """

    def __init__(self):
        """Initialize operator (model loaded in open())."""
        self.model = None
        self.scaler = None
        self.thresholds = None
        self.vectorizer = None
        self.neighborhood_mapping = None

    def open(self, runtime_context):
        """Load model from Broadcast State.

        CRITICAL: This is called once per task slot when Flink job starts.
        Model must be preloaded into Broadcast State before job execution.

        V1.9: Broadcast State should be cleared before put() to avoid stale data.
        """
        try:
            # Get Broadcast State
            model_state_desc = MapStateDescriptor(
                "model_state",
                Types.STRING(),
                Types.PICKLED_BYTE_ARRAY()
            )

            broadcast_state = runtime_context.get_broadcast_state(model_state_desc)

            # Load model
            model_bytes = broadcast_state.get("current_model")
            if model_bytes is None:
                raise RuntimeError("Model not found in Broadcast State - run model loader first")

            self.model = pickle.loads(model_bytes)
            print(f"[IFScoringOperator] Model loaded: {self.model.n_estimators} trees")

            # Load scaler
            scaler_bytes = broadcast_state.get("scaler")
            if scaler_bytes is None:
                raise RuntimeError("Scaler not found in Broadcast State")

            self.scaler = pickle.loads(scaler_bytes)
            print(f"[IFScoringOperator] Scaler loaded")

            # Load thresholds
            threshold_json = broadcast_state.get("thresholds")
            if threshold_json is None:
                raise RuntimeError("Thresholds not found in Broadcast State")

            self.thresholds = json.loads(threshold_json.decode('utf-8'))
            print(f"[IFScoringOperator] Thresholds loaded: {len(self.thresholds.get('thresholds', {}))} contexts")

            # Load neighborhood mapping if available
            mapping_json = broadcast_state.get("neighborhood_mapping")
            if mapping_json:
                self.neighborhood_mapping = json.loads(mapping_json.decode('utf-8'))

            # Initialize vectorizer
            self.vectorizer = FeatureVectorizer()
            print(f"[IFScoringOperator] Vectorizer initialized (21D features)")

        except Exception as e:
            print(f"[IFScoringOperator] ERROR in open(): {e}")
            raise

    def map(self, value):
        """Score single record with iForestASD.

        Args:
            value: Record dict (already passed Layer 1 validation)

        Returns:
            Enriched record with anomaly_score, threshold, is_anomaly, context_key
        """
        if value is None:
            return None

        try:
            # Vectorize (21D enhanced features)
            features = self.vectorizer.transform(value)

            # Scale
            features_scaled = self.scaler.transform([features])[0]

            # sklearn IsolationForest: score_samples returns negative anomaly scores
            # Lower (more negative) = more anomalous
            # We negate to make higher = more anomalous (same semantics as before)
            raw_score = self.model.score_samples(features_scaled.reshape(1, -1))[0]
            anomaly_score = -raw_score  # Now higher = more anomalous (River-like semantics)

            # Get context-aware threshold (using negated threshold for comparison)
            context_key = get_context_key(value, self.neighborhood_mapping)
            threshold_val = self.thresholds.get('thresholds', {}).get(
                context_key,
                self.thresholds.get('global_threshold', 0.5)
            )
            # Compare negated score with negated threshold: score > threshold
            is_anomaly = anomaly_score > threshold_val

            # Return enriched record
            return {
                **value,
                'anomaly_score': float(anomaly_score),
                'threshold': float(threshold),
                'is_anomaly': bool(is_anomaly),
                'context_key': context_key
            }

        except Exception as e:
            # Log error but don't crash pipeline
            print(f"[IFScoringOperator] ERROR scoring record: {e}")

            # Return record with error flag
            return {
                **value,
                'anomaly_score': -1.0,
                'threshold': 0.0,
                'is_anomaly': False,
                'context_key': 'error',
                'scoring_error': str(e)
            }
