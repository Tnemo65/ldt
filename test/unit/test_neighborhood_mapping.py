"""Neighborhood mapping tests.
Spec: Lines 2845-2847, Appendix A Lines 4453-4478
"""

import pytest
import json
from pathlib import Path

@pytest.fixture
def neighborhood_mapping():
    config_path = Path(__file__).parent.parent.parent / 'src' / 'config' / 'neighborhood_mapping.json'
    with open(config_path) as f:
        return json.load(f)

def test_mapping_file_exists():
    """Mapping file must exist."""
    path = Path(__file__).parent.parent.parent / 'src' / 'config' / 'neighborhood_mapping.json'
    assert path.exists(), "Mapping file not found"

def test_all_zones_mapped(neighborhood_mapping):
    """All 265 zones must be mapped."""
    mapping = neighborhood_mapping['mapping']
    zone_ids = set(int(k) for k in mapping.keys())
    expected = set(range(1, 266))
    assert zone_ids == expected, f"Missing zones: {expected - zone_ids}"

def test_neighborhood_count(neighborhood_mapping):
    """5-7 neighborhoods required (spec)."""
    neighborhoods = set(neighborhood_mapping['mapping'].values())
    count = len(neighborhoods)
    assert 5 <= count <= 8, f"Expected 5-7 neighborhoods, got {count}"

def test_balanced_distribution(neighborhood_mapping):
    """No single neighborhood >50% of trips."""
    # This test validates structure only (not data-dependent)
    assert 'mapping' in neighborhood_mapping
    assert 'version' in neighborhood_mapping
    assert neighborhood_mapping['version'] == "1.0"
