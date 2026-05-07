"""Synthetic anomaly injection tests.
Spec: Lines 2884-2894
"""

import pytest
import pandas as pd
from pathlib import Path

def test_injected_file_exists():
    """File with synthetic anomalies must exist."""
    path = Path('data/clean/jan_2024_with_50k_anomalies.parquet')
    assert path.exists(), "Injected file not found"

def test_labels_file_exists():
    """Anomaly labels CSV must exist."""
    path = Path('data/clean/anomaly_labels.csv')
    assert path.exists(), "Labels file not found"

def test_anomaly_count():
    """Exactly 50K anomalies injected."""
    labels = pd.read_csv('data/clean/anomaly_labels.csv')
    anomaly_count = (labels['is_anomaly'] == 1).sum()
    assert anomaly_count == 50_000, f"Expected 50K anomalies, got {anomaly_count:,}"

def test_five_scenarios():
    """5 fraud scenarios, 10K each."""
    labels = pd.read_csv('data/clean/anomaly_labels.csv')
    scenarios = labels[labels['is_anomaly'] == 1]['scenario'].value_counts()

    assert len(scenarios) == 5, f"Expected 5 scenarios, got {len(scenarios)}"

    for scenario, count in scenarios.items():
        assert count == 10_000, f"Scenario {scenario}: expected 10K, got {count:,}"
