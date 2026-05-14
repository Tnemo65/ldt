#!/usr/bin/env python3
"""
METER Hypernetwork Training - Meta-Learning for Strategy Selection.
Task 3.16-3.20: Train MLP to predict optimal adaptation strategy from meta-metrics

METER (Meta-learning Ensemble for Temporal Event Recognition)
- Input: 6 meta-metrics (volume, null_rate, violation_rate, anomaly_rate, avg_score, delta_score)
- Output: Optimal strategy (retrain, adjust_threshold, switch_model, do_nothing)

Architecture:
- MLP: 6 → 64 → 32 → 16 → 4
- Activation: ReLU
- Solver: Adam
- Training: Supervised on historical drift scenarios

Usage:
  python src/ml/train_meter.py
  python src/ml/train_meter.py --scenarios data/drift_scenarios.csv
"""

import argparse
import sys
from pathlib import Path
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Strategy labels
STRATEGIES = {
    0: 'do_nothing',
    1: 'adjust_threshold',
    2: 'retrain_model',
    3: 'switch_model'
}


def generate_synthetic_drift_scenarios(n_samples=1000):
    """Generate synthetic drift scenarios for training.

    In production, these would be collected from real historical drifts.
    For now, we generate synthetic data based on expected patterns.

    Args:
        n_samples: Number of scenarios to generate

    Returns:
        DataFrame with meta-metrics and optimal strategy labels
    """
    print(f"\nGenerating {n_samples} synthetic drift scenarios...")

    scenarios = []

    for i in range(n_samples):
        # Scenario 1: Normal operation (40%)
        if np.random.rand() < 0.4:
            volume = np.random.randint(800, 1200)
            null_rate = np.random.uniform(0.0, 0.02)
            violation_rate = np.random.uniform(0.0, 0.05)
            anomaly_rate = np.random.uniform(0.03, 0.06)
            avg_score = np.random.uniform(0.3, 0.5)
            delta_score = np.random.uniform(-0.01, 0.01)
            strategy = 0  # do_nothing

        # Scenario 2: Minor drift - adjust threshold (30%)
        elif np.random.rand() < 0.7:
            volume = np.random.randint(700, 1300)
            null_rate = np.random.uniform(0.02, 0.05)
            violation_rate = np.random.uniform(0.05, 0.10)
            anomaly_rate = np.random.uniform(0.06, 0.12)
            avg_score = np.random.uniform(0.5, 0.7)
            delta_score = np.random.uniform(0.02, 0.05)
            strategy = 1  # adjust_threshold

        # Scenario 3: Moderate drift - retrain (20%)
        elif np.random.rand() < 0.85:
            volume = np.random.randint(600, 1400)
            null_rate = np.random.uniform(0.05, 0.10)
            violation_rate = np.random.uniform(0.10, 0.20)
            anomaly_rate = np.random.uniform(0.12, 0.25)
            avg_score = np.random.uniform(0.7, 0.85)
            delta_score = np.random.uniform(0.05, 0.10)
            strategy = 2  # retrain_model

        # Scenario 4: Severe drift - switch model (10%)
        else:
            volume = np.random.randint(500, 1500)
            null_rate = np.random.uniform(0.10, 0.20)
            violation_rate = np.random.uniform(0.20, 0.40)
            anomaly_rate = np.random.uniform(0.25, 0.50)
            avg_score = np.random.uniform(0.85, 1.0)
            delta_score = np.random.uniform(0.10, 0.30)
            strategy = 3  # switch_model

        scenarios.append({
            'volume': volume,
            'null_rate': null_rate,
            'violation_rate': violation_rate,
            'anomaly_rate': anomaly_rate,
            'avg_anomaly_score': avg_score,
            'delta_score': delta_score,
            'strategy': strategy
        })

    df = pd.DataFrame(scenarios)
    print(f"  ✓ Generated {len(df)} scenarios")
    print(f"\nStrategy distribution:")
    print(df['strategy'].value_counts().sort_index())

    return df


def train_meter_hypernetwork(
    scenarios_df: pd.DataFrame,
    output_dir: str = 'models',
    test_size: float = 0.2,
    random_state: int = 42
):
    """Train METER hypernetwork on drift scenarios.

    Args:
        scenarios_df: DataFrame with meta-metrics and strategy labels
        output_dir: Directory to save trained model
        test_size: Test set fraction
        random_state: Random seed

    Returns:
        Trained model, scaler, and metrics
    """
    print("="*60)
    print("METER Hypernetwork Training")
    print("="*60)

    # Prepare features and labels
    feature_cols = ['volume', 'null_rate', 'violation_rate',
                    'anomaly_rate', 'avg_anomaly_score', 'delta_score']

    X = scenarios_df[feature_cols].values
    y = scenarios_df['strategy'].values

    print(f"\nFeatures: {feature_cols}")
    print(f"Samples: {len(X)}")
    print(f"Classes: {len(np.unique(y))}")

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # Scale features
    print(f"\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train METER hypernetwork (MLP)
    print(f"\nTraining METER hypernetwork...")
    print(f"  Architecture: 6 → 64 → 32 → 16 → 4")
    print(f"  Activation: ReLU")
    print(f"  Solver: Adam")

    meter_model = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=random_state,
        verbose=False,
        early_stopping=True,
        validation_fraction=0.1
    )

    meter_model.fit(X_train_scaled, y_train)

    print(f"  ✓ Training complete")
    print(f"  Iterations: {meter_model.n_iter_}")

    # Evaluate
    print(f"\nEvaluating...")

    train_score = meter_model.score(X_train_scaled, y_train)
    test_score = meter_model.score(X_test_scaled, y_test)

    print(f"  Train accuracy: {train_score:.3f}")
    print(f"  Test accuracy: {test_score:.3f}")

    # Predictions
    y_pred = meter_model.predict(X_test_scaled)

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=[STRATEGIES[i] for i in range(4)]))

    # Confusion matrix
    print(f"Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Save model
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model_path = Path(output_dir) / 'meter_hypernetwork.pkl'
    scaler_path = Path(output_dir) / 'meter_scaler.pkl'

    with open(model_path, 'wb') as f:
        pickle.dump(meter_model, f)

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\n✅ Model saved:")
    print(f"  {model_path}")
    print(f"  {scaler_path}")

    # Save metadata
    metadata = {
        'model': 'METER Hypernetwork (MLP)',
        'architecture': '6 → 64 → 32 → 16 → 4',
        'input_features': feature_cols,
        'output_strategies': STRATEGIES,
        'train_accuracy': float(train_score),
        'test_accuracy': float(test_score),
        'n_samples': len(X),
        'n_iter': int(meter_model.n_iter_)
    }

    metadata_path = Path(output_dir) / 'meter_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  {metadata_path}")

    return meter_model, scaler, metadata


def predict_strategy(model, scaler, meta_metrics: dict):
    """Predict optimal strategy from meta-metrics.

    Args:
        model: Trained METER model
        scaler: Fitted StandardScaler
        meta_metrics: Dict with 6 meta-metrics

    Returns:
        Predicted strategy name and confidence
    """
    features = np.array([[
        meta_metrics['volume'],
        meta_metrics['null_rate'],
        meta_metrics['violation_rate'],
        meta_metrics['anomaly_rate'],
        meta_metrics['avg_anomaly_score'],
        meta_metrics['delta_score']
    ]])

    features_scaled = scaler.transform(features)

    # Predict
    strategy_id = model.predict(features_scaled)[0]
    strategy_name = STRATEGIES[strategy_id]

    # Get confidence (probability)
    probs = model.predict_proba(features_scaled)[0]
    confidence = probs[strategy_id]

    return strategy_name, confidence


def main():
    parser = argparse.ArgumentParser(description='Train METER hypernetwork')
    parser.add_argument(
        '--scenarios',
        type=str,
        default=None,
        help='Path to drift scenarios CSV (optional, will generate synthetic if not provided)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Output directory for model (default: models)'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=1000,
        help='Number of synthetic scenarios to generate (default: 1000)'
    )

    args = parser.parse_args()

    # Load or generate scenarios
    if args.scenarios and Path(args.scenarios).exists():
        print(f"Loading scenarios from: {args.scenarios}")
        scenarios_df = pd.read_csv(args.scenarios)
    else:
        scenarios_df = generate_synthetic_drift_scenarios(args.n_samples)

    # Train METER
    model, scaler, metadata = train_meter_hypernetwork(
        scenarios_df,
        output_dir=args.output_dir
    )

    # Test prediction
    print(f"\n{'='*60}")
    print("Test Prediction")
    print(f"{'='*60}")

    test_metrics = {
        'volume': 1000,
        'null_rate': 0.08,
        'violation_rate': 0.15,
        'anomaly_rate': 0.18,
        'avg_anomaly_score': 0.75,
        'delta_score': 0.08
    }

    strategy, confidence = predict_strategy(model, scaler, test_metrics)

    print(f"Input metrics: {test_metrics}")
    print(f"Predicted strategy: {strategy} (confidence: {confidence:.3f})")

    return 0


if __name__ == '__main__':
    exit(main())
