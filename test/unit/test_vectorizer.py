"""Feature vectorizer tests.
Tests 21D enhanced vectorizer with ratio features.
"""

import pytest
import numpy as np
from src.features.vectorizer import FeatureVectorizer

def test_vectorizer_21d_output():
    """Vectorizer must produce 21D output."""
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
        'total_amount': 20.0,
    }

    features = vectorizer.transform(record)
    assert len(features) == 21, f"Expected 21D, got {len(features)}D"

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
