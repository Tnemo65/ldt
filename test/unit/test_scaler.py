"""StandardScaler tests.
Spec: Lines 3663-3673 (CRITICAL: scaler.pkl must exist)
"""

import pytest
import pickle
from pathlib import Path
import numpy as np

def test_scaler_file_exists():
    """Scaler PKL must exist."""
    path = Path('models/scaler.pkl')
    assert path.exists(), "scaler.pkl not found"

def test_scaler_fitted():
    """Scaler must be fitted (mean/scale exist)."""
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    assert hasattr(scaler, 'mean_'), "Scaler not fitted (no mean_)"
    assert hasattr(scaler, 'scale_'), "Scaler not fitted (no scale_)"
    assert len(scaler.mean_) == 15, "Scaler not 15D"

def test_scaler_transform():
    """Scaler must transform 15D vectors."""
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    X = np.random.random((10, 15))
    X_scaled = scaler.transform(X)

    assert X_scaled.shape == (10, 15)
    # Verify transform produces finite values (no NaN/Inf)
    assert np.isfinite(X_scaled).all(), "Scaled features contain NaN or Inf"
