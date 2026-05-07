"""iForest training tests.
Spec: Lines 3414-3465 (HalfSpaceTrees on clean baseline)
"""

import pytest
from pathlib import Path
import pickle
import numpy as np


def test_training_script_exists():
    """Training script should exist."""
    script_path = Path('src/ml/train_iforest.py')
    assert script_path.exists(), f"Training script not found: {script_path}"


def test_clean_baseline_data_exists():
    """Clean baseline data should exist for training."""
    data_path = Path('data/clean/jan_2024_clean_baseline.parquet')
    assert data_path.exists(), f"Training data not found: {data_path}"


def test_scaler_exists():
    """StandardScaler should be fitted and saved."""
    scaler_path = Path('models/scaler.pkl')
    assert scaler_path.exists(), f"Scaler not found: {scaler_path}"

    # Load and verify it's a scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Check it has expected attributes
    assert hasattr(scaler, 'transform'), "Scaler should have transform method"
    assert hasattr(scaler, 'mean_'), "Scaler should be fitted (has mean_)"


def test_iforest_model_can_be_trained():
    """iForest model should be trainable with River."""
    from river.anomaly import HalfSpaceTrees

    # Create model with spec params
    model = HalfSpaceTrees(
        n_trees=100,
        height=8,
        window_size=256,
        seed=42
    )

    # Train on small sample
    sample_data = [
        {0: 1.5, 1: 2.3, 2: 0.8, 3: 4.1, 4: 3.2, 5: 1.1, 6: 2.7,
         7: 0.5, 8: 3.8, 9: 1.9, 10: 2.1, 11: 0.9, 12: 1.6, 13: 3.3, 14: 2.4}
        for _ in range(100)
    ]

    for features in sample_data:
        model.learn_one(features)

    # Should be able to score
    score = model.score_one(sample_data[0])
    assert isinstance(score, (int, float)), "Score should be numeric"
    assert score >= 0, "Score should be non-negative"


@pytest.mark.slow
def test_trained_model_exists():
    """After training, model file should exist."""
    model_path = Path('models/iforest_model.pkl')

    # This test will initially fail until training is run
    if not model_path.exists():
        pytest.skip("Model not yet trained - run training script first")

    # Load and verify
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Check it's a HalfSpaceTrees model
    assert hasattr(model, 'score_one'), "Model should have score_one method"
    assert hasattr(model, 'learn_one'), "Model should have learn_one method"


@pytest.mark.slow
def test_model_can_score():
    """Trained model should be able to score new records."""
    model_path = Path('models/iforest_model.pkl')

    if not model_path.exists():
        pytest.skip("Model not yet trained")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Test scoring
    test_features = {i: float(i) for i in range(15)}  # 15D features
    score = model.score_one(test_features)

    assert isinstance(score, float), "Score should be float"
    assert score >= 0, "Anomaly score should be non-negative"
