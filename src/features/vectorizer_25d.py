"""
25D Feature Vectorizer for NYC Taxi MemStream.

This module provides the 25D feature extraction that matches the MemStream
benchmark results (AUC-PR = 0.9996).

Feature structure (25D):
- Temporal (8D): Hour sin/cos, day-of-week sin/cos
- Monetary (7D): Fare, tips, tolls, surcharges
- Spatial (2D): Pickup/dropoff borough encoding
- Trip characteristics (8D): Distance, duration, speed, etc.

Reference: memstream_src/core/feature_extractor.py
"""

from typing import Dict, List, Optional
from datetime import datetime

import numpy as np


# =============================================================================
# Constants
# =============================================================================

# NYC Taxi Zone definitions
MANHATTAN_ZONES = set(range(1, 51))
BROOKLYN_ZONES = set(range(51, 101))
QUEENS_ZONES = set(range(101, 151))
BRONX_ZONES = set(range(151, 201))
AIRPORT_ZONES = {132, 138}  # JFK and LaGuardia
STATEN_ISLAND_ZONES = set(range(201, 264))

# Feature names (for reference)
FEATURE_NAMES = [
    'pickup_hour_sin',
    'pickup_hour_cos',
    'dropoff_hour_sin',
    'dropoff_hour_cos',
    'pickup_day_of_week_sin',
    'pickup_day_of_week_cos',
    'dropoff_day_of_week_sin',
    'dropoff_day_of_week_cos',
    'trip_distance',
    'fare_amount',
    'extra',
    'mta_tax',
    'tip_amount',
    'tolls_amount',
    'improvement_surcharge',
    'total_amount',
    'passenger_count',
    'trip_duration_minutes',
    'speed_mph',
    'fare_per_mile',
    'is_airport_trip',
    'is_rush_hour',
    'is_weekend',
    'pickup_borough_encoded',
    'dropoff_borough_encoded',
]

NUM_FEATURES = len(FEATURE_NAMES)  # 25D

# Borough encoding
BOROUGH_ENCODING = {
    'manhattan': 0,
    'brooklyn': 1,
    'queens': 2,
    'bronx': 3,
    'staten_island': 4,
    'airport': 5,
    'unknown': 6,
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_borough_from_zone(zone_id: int) -> str:
    """Map zone ID to borough name."""
    if zone_id in MANHATTAN_ZONES:
        return 'manhattan'
    elif zone_id in BROOKLYN_ZONES:
        return 'brooklyn'
    elif zone_id in QUEENS_ZONES:
        return 'queens'
    elif zone_id in BRONX_ZONES:
        return 'bronx'
    elif zone_id in AIRPORT_ZONES:
        return 'airport'
    elif zone_id in STATEN_ISLAND_ZONES:
        return 'staten_island'
    else:
        return 'unknown'


def parse_datetime(dt_str: str) -> Optional[datetime]:
    """Parse datetime string from various formats."""
    if not dt_str:
        return None
    
    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%d %H:%M:%S UTC',
        '%Y/%m/%d %H:%M:%S',
        '%m/%d/%Y %H:%M:%S',
        '%m/%d/%Y %I:%M:%S %p',
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(dt_str.strip(), fmt)
        except ValueError:
            continue
    
    try:
        return datetime.fromisoformat(dt_str.replace('/', '-'))
    except ValueError:
        return None


# =============================================================================
# FeatureVectorizer25D
# =============================================================================

class FeatureVectorizer25D:
    """
    25D FeatureVectorizer for NYC Taxi MemStream anomaly detection.

    Feature categories:
    - Temporal (8D): Hour and day-of-week (sin/cos encoded)
    - Monetary (7D): Fare, tips, tolls, surcharges
    - Spatial (2D): Pickup/dropoff borough encoding
    - Trip characteristics (8D): Distance, duration, speed, etc.

    Total: 25 dimensions
    """

    FEATURE_NAMES = FEATURE_NAMES
    DIM = NUM_FEATURES

    def __init__(self):
        """Initialize the feature vectorizer."""
        pass

    @property
    def num_features(self) -> int:
        """Return the number of features (25D)."""
        return self.DIM

    def transform(self, record: Dict) -> Optional[np.ndarray]:
        """
        Transform a single NYC taxi record into a 25D feature vector.

        Args:
            record: Dict with NYC taxi fields

        Returns:
            np.ndarray of shape (25,) or None if essential fields missing
        """
        try:
            features = self._extract_features(record)
            return features.astype(np.float32)
        except (ValueError, TypeError, KeyError):
            return None

    def transform_batch(self, records: List[Dict]) -> np.ndarray:
        """
        Transform a batch of NYC taxi records into feature matrix.

        Args:
            records: List of dicts with NYC taxi fields

        Returns:
            np.ndarray of shape (N, 25)
        """
        features_list = []
        for record in records:
            feat = self.transform(record)
            if feat is not None:
                features_list.append(feat)
            else:
                features_list.append(np.zeros(self.DIM, dtype=np.float32))

        return np.array(features_list, dtype=np.float32)

    def _extract_features(self, record: Dict) -> np.ndarray:
        """Extract raw features from a record."""
        features = np.zeros(self.DIM, dtype=np.float32)

        # --- Temporal features (8D) ---

        # Pickup time
        pickup_dt = parse_datetime(record.get('tpep_pickup_datetime', ''))
        if pickup_dt is not None:
            # Hour sin/cos encoding
            features[0] = np.sin(2 * np.pi * pickup_dt.hour / 24)
            features[1] = np.cos(2 * np.pi * pickup_dt.hour / 24)
            # Day of week sin/cos encoding
            features[4] = np.sin(2 * np.pi * pickup_dt.weekday() / 7)
            features[5] = np.cos(2 * np.pi * pickup_dt.weekday() / 7)

        # Dropoff time
        dropoff_dt = parse_datetime(record.get('tpep_dropoff_datetime', ''))
        if dropoff_dt is not None:
            # Hour sin/cos encoding
            features[2] = np.sin(2 * np.pi * dropoff_dt.hour / 24)
            features[3] = np.cos(2 * np.pi * dropoff_dt.hour / 24)
            # Day of week sin/cos encoding
            features[6] = np.sin(2 * np.pi * dropoff_dt.weekday() / 7)
            features[7] = np.cos(2 * np.pi * dropoff_dt.weekday() / 7)

        # --- Monetary features (7D) ---

        features[9] = float(record.get('fare_amount', 0))
        features[10] = float(record.get('extra', 0))
        features[11] = float(record.get('mta_tax', 0))
        features[12] = float(record.get('tip_amount', 0))
        features[13] = float(record.get('tolls_amount', 0))
        features[14] = float(record.get('improvement_surcharge', 0))
        features[15] = float(record.get('total_amount', 0))

        # --- Spatial features (2D) ---

        pickup_zone = int(float(record.get('PULocationID', 0)))
        dropoff_zone = int(float(record.get('DOLocationID', 0)))

        pickup_borough = get_borough_from_zone(pickup_zone)
        dropoff_borough = get_borough_from_zone(dropoff_zone)

        features[23] = BOROUGH_ENCODING.get(pickup_borough, 6)
        features[24] = BOROUGH_ENCODING.get(dropoff_borough, 6)

        # --- Trip characteristics (8D) ---

        features[8] = float(record.get('trip_distance', 0))  # trip_distance
        features[16] = float(record.get('passenger_count', 1))  # passenger_count

        # Duration in minutes
        if pickup_dt is not None and dropoff_dt is not None:
            duration_minutes = (dropoff_dt - pickup_dt).total_seconds() / 60
            features[17] = max(0, duration_minutes)  # trip_duration_minutes

            # Speed (mph)
            distance = float(record.get('trip_distance', 0))
            if duration_minutes > 0:
                hours = duration_minutes / 60
                features[18] = distance / hours if hours > 0 else 0  # speed_mph
        else:
            features[17] = 0
            features[18] = 0

        # Fare per mile
        distance = float(record.get('trip_distance', 0))
        fare = float(record.get('fare_amount', 0))
        features[19] = fare / distance if distance > 0 else 0  # fare_per_mile

        # Binary features
        features[20] = 1.0 if pickup_zone in AIRPORT_ZONES or dropoff_zone in AIRPORT_ZONES else 0.0  # is_airport_trip

        # Rush hour (7-9 AM or 5-8 PM on weekdays)
        if pickup_dt is not None:
            hour = pickup_dt.hour
            weekday = pickup_dt.weekday()
            is_rush = ((7 <= hour <= 9) or (17 <= hour <= 20)) and weekday < 5
            features[21] = 1.0 if is_rush else 0.0  # is_rush_hour
            features[22] = 1.0 if weekday >= 5 else 0.0  # is_weekend
        else:
            features[21] = 0.0
            features[22] = 0.0

        return features


# =============================================================================
# Quick Context Extraction (for drift detection per neighborhood)
# =============================================================================

def get_neighborhood_from_record(record: Dict) -> str:
    """Extract neighborhood from record for per-neighborhood processing."""
    zone_id = int(float(record.get('PULocationID', 1)))
    return get_borough_from_zone(zone_id)


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    'FeatureVectorizer25D',
    'get_borough_from_zone',
    'get_neighborhood_from_record',
    'parse_datetime',
    'FEATURE_NAMES',
    'NUM_FEATURES',
]
