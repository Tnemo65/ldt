"""iForest training tests - sklearn IsolationForest.
Tests both sklearn IsolationForest and legacy River HalfSpaceTrees.
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

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    assert hasattr(scaler, 'transform'), "Scaler should have transform method"
    assert hasattr(scaler, 'mean_'), "Scaler should be fitted (has mean_)"


def test_iforest_model_can_be_trained():
    """sklearn IsolationForest should be trainable."""
    from sklearn.ensemble import IsolationForest

    model = IsolationForest(
        n_estimators=100,
        max_samples=256,
        contamination=0.001,
        random_state=42
    )

    sample_data = np.random.randn(100, 21)
    model.fit(sample_data)

    scores = model.score_samples(sample_data[:5])
    assert len(scores) == 5, "Should return 5 scores"
    assert scores.dtype == np.float64, "Scores should be float64"


@pytest.mark.slow
def test_trained_model_exists():
    """After training, model file should exist."""
    model_path = Path('models/iforest_model.pkl')

    if not model_path.exists():
        pytest.skip("Model not yet trained - run training script first")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # sklearn IsolationForest has score_samples, River has score_one
    has_sklearn_api = hasattr(model, 'score_samples')
    has_river_api = hasattr(model, 'score_one')
    assert has_sklearn_api or has_river_api, "Model should have scoring method"


@pytest.mark.slow
def test_model_can_score():
    """Trained model should be able to score new records."""
    model_path = Path('models/iforest_model.pkl')

    if not model_path.exists():
        pytest.skip("Model not yet trained")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    test_features = np.random.randn(1, 21)

    if hasattr(model, 'score_samples'):
        score = model.score_samples(test_features)
        assert isinstance(score[0], (np.floating, float))
    elif hasattr(model, 'score_one'):
        feat_dict = {i: float(v) for i, v in enumerate(test_features[0])}
        score = model.score_one(feat_dict)
        assert isinstance(score, (int, float))

