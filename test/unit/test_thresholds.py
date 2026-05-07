"""Context threshold tests.
Spec: Lines 2901-2904
"""

import pytest
import json
from pathlib import Path

def test_threshold_file_exists():
    """Threshold matrix must exist."""
    path = Path('src/config/threshold_matrix.json')
    assert path.exists(), "Threshold file not found"

def test_4d_structure():
    """Threshold matrix must have 4D structure."""
    with open('src/config/threshold_matrix.json') as f:
        thresholds = json.load(f)

    assert 'thresholds' in thresholds
    assert 'global_threshold' in thresholds

    # Check 4D keys exist
    for key in thresholds['thresholds'].keys():
        parts = key.split('_')
        assert len(parts) >= 4, f"Key {key} not 4D (has {len(parts)} parts)"

def test_percentile_value():
    """Thresholds should be 95th percentile (>0)."""
    with open('src/config/threshold_matrix.json') as f:
        thresholds = json.load(f)

    for key, value in thresholds['thresholds'].items():
        assert value > 0, f"Threshold {key} invalid: {value}"
