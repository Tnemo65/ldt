"""15D Feature Vectorizer.
Spec: Lines 1446-1464
"""

import numpy as np
from datetime import datetime

class FeatureVectorizer:
    """Extract 15D feature vector from taxi trip record."""

    def transform(self, record: dict) -> np.ndarray:
        """Transform record to 15D numpy array.

        Features:
        1. trip_distance
        2. fare_amount
        3. passenger_count
        4. payment_type
        5. trip_duration (seconds)
        6. hour_of_day
        7. day_of_week
        8. is_weekend
        9. speed_mph
        10. fare_per_mile
        11. PULocationID
        12. DOLocationID
        13. is_airport_pickup
        14. is_airport_dropoff
        15. zone_distance (PU-DO Manhattan distance)
        """

        # Parse timestamps (handle both string and datetime objects)
        pickup = record['tpep_pickup_datetime']
        if isinstance(pickup, str):
            pickup = datetime.fromisoformat(pickup)
        dropoff = record['tpep_dropoff_datetime']
        if isinstance(dropoff, str):
            dropoff = datetime.fromisoformat(dropoff)

        # Derived features
        duration_sec = (dropoff - pickup).total_seconds()
        speed_mph = (record['trip_distance'] / (duration_sec / 3600)) \
                    if duration_sec > 0 else 0
        fare_per_mile = (record['fare_amount'] / record['trip_distance']) \
                        if record['trip_distance'] > 0 else 0

        # Airport zones (JFK: 132, 138; LGA: 137, 138; Newark: 1)
        airport_zones = {1, 132, 137, 138}
        is_airport_pu = 1 if record['PULocationID'] in airport_zones else 0
        is_airport_do = 1 if record['DOLocationID'] in airport_zones else 0

        # Zone distance (Manhattan distance approximation)
        zone_dist = abs(record['PULocationID'] - record['DOLocationID'])

        features = np.array([
            record['trip_distance'],
            record['fare_amount'],
            record['passenger_count'],
            record['payment_type'],
            duration_sec,
            pickup.hour,
            pickup.weekday(),
            1 if pickup.weekday() >= 5 else 0,
            speed_mph,
            fare_per_mile,
            record['PULocationID'],
            record['DOLocationID'],
            is_airport_pu,
            is_airport_do,
            zone_dist
        ], dtype=np.float64)

        return features

    def fit(self, X):
        """No-op for compatibility with sklearn API."""
        return self
