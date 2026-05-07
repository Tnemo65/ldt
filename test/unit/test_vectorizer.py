"""Feature vectorizer tests.
Spec: Lines 1446-1464 (15D vector)
"""

import pytest
import numpy as np
from src.features.vectorizer import FeatureVectorizer

def test_vectorizer_15d_output():
    """Vectorizer must produce 15D output."""
    vectorizer = FeatureVectorizer()

    record = {
        'trip_distance': 3.5,
        'fare_amount': 15.5,
        'passenger_count': 2,
        'payment_type': 1,
        'PULocationID': 161,
        'DOLocationID': 230,
        'tpep_pickup_datetime': '2024-01-15T10:30:00',
        'tpep_dropoff_datetime': '2024-01-15T10:45:00',
    }

    features = vectorizer.transform(record)
    assert len(features) == 15, f"Expected 15D, got {len(features)}D"

def test_vectorizer_no_null():
    """Vectorizer must not produce NULL/NaN."""
    vectorizer = FeatureVectorizer()

    record = {
        'trip_distance': 3.5,
        'fare_amount': 15.5,
        'passenger_count': 2,
        'payment_type': 1,
        'PULocationID': 161,
        'DOLocationID': 230,
        'tpep_pickup_datetime': '2024-01-15T10:30:00',
        'tpep_dropoff_datetime': '2024-01-15T10:45:00',
    }

    features = vectorizer.transform(record)
    assert not np.isnan(features).any(), "Features contain NaN"
