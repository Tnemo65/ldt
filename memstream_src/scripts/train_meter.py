#!/usr/bin/env python3
"""
METER Hypernetwork Training - Meta-Learning for Strategy Selection.
Step 3 of METER deployment pipeline.

Loads:
- Real drift scenario data from v9 benchmark (scripts/train_meter_from_benchmark.py)
- Augmented synthetic data (scripts/generate_meter_training_data.py)

Then trains sklearn MLP with:
- SMOTE balancing (or class weights fallback)
- 5-fold cross-validation
- Architecture selection (best of 3 configurations)
- Saves to models/meter_hypernetwork.pkl, models/meter_scaler.pkl, models/meter_metadata.json

Usage:
  python memstream_src/scripts/train_meter.py
  python memstream_src/scripts/train_meter.py --real-data data/meter_training_real.csv
  python memstream_src/scripts/train_meter.py --synthetic-data data/meter_training_synthetic.csv
"""

import argparse
import sys
from pathlib import Path
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

# For imports from src/ml/ when running from memstream_src/scripts/
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


STRATEGIES = {
    0: 'do_nothing',
    1: 'adjust_threshold',
    2: 'memory_reset',
    # 3: 'switch_model' removed — not applicable to online learning
}


def load_data(real_path=None, synthetic_path=None):
    """Load real and/or synthetic training data."""
    dfs = []

    if real_path and Path(real_path).exists():
        df_real = pd.read_csv(real_path)
        # Standardize column name
        if 'avg_score' in df_real.columns:
            df_real = df_real.rename(columns={'avg_score': 'avg_anomaly_score'})
        dfs.append(df_real)
        print(f"  Loaded real data: {len(df_real)} records")

    if synthetic_path and Path(synthetic_path).exists():
        df_synth = pd.read_csv(synthetic_path)
        if 'avg_score' in df_synth.columns:
            df_synth = df_synth.rename(columns={'avg_score': 'avg_anomaly_score'})
        dfs.append(df_synth)
        print(f"  Loaded synthetic data: {len(df_synth)} records")

    if not dfs:
        print("WARNING: No data files found. Generating synthetic data inline...")
        df_synth = generate_inline_synthetic(1000)
        dfs.append(df_synth)

    df = pd.concat(dfs, ignore_index=True)

    # Drop rows with strategy outside 0-3
    df = df[df['strategy'].isin([0, 1, 2, 3])]

    return df


def generate_inline_synthetic(n_samples=1000):
    """Generate synthetic data when no files are available."""
    np.random.seed(42)
    scenarios = []

    scenario_defs = [
        (0, 0.40, {'volume': (800, 1200), 'null_rate': (0.00, 0.02), 'violation_rate': (0.01, 0.05),
                    'anomaly_rate': (0.03, 0.06), 'avg_score': (0.35, 0.45)}),
        (1, 0.30, {'volume': (850, 1150), 'null_rate': (0.01, 0.03), 'violation_rate': (0.03, 0.07),
                    'anomaly_rate': (0.06, 0.12), 'avg_score': (0.48, 0.58)}),
        (2, 0.20, {'volume': (700, 900), 'null_rate': (0.03, 0.08), 'violation_rate': (0.08, 0.15),
                    'anomaly_rate': (0.15, 0.30), 'avg_score': (0.55, 0.70)}),
        (3, 0.10, {'volume': (400, 700), 'null_rate': (0.10, 0.25), 'violation_rate': (0.20, 0.40),
                    'anomaly_rate': (0.40, 0.70), 'avg_score': (0.80, 0.99)}),
    ]

    for strategy, frac, ranges in scenario_defs:
        n = int(n_samples * frac)
        for _ in range(n):
            vr = np.random.uniform(*ranges['violation_rate'])
            ar = np.random.uniform(*ranges['anomaly_rate'])
            delta = abs(vr - ar) / (vr + ar + 1e-6)
            scenarios.append({
                'volume': np.random.uniform(*ranges['volume']),
                'null_rate': np.random.uniform(*ranges['null_rate']),
                'violation_rate': vr,
                'anomaly_rate': ar,
                'avg_anomaly_score': np.random.uniform(*ranges['avg_score']),
                'delta_score': delta,
                'strategy': strategy,
            })

    return pd.DataFrame(scenarios)


def apply_smote_or_weights(X, y):
    """Apply SMOTE balancing or fall back to class weights."""
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        print("  SMOTE: Balanced classes via oversampling")
        return X_resampled, y_resampled
    except ImportError:
        print("  imbalanced-learn not available, using class weights instead")
        return X, y


def train_and_select_model(X_train, y_train, X_test, y_test):
    """Try multiple MLP architectures and select the best via CV."""
    feature_cols = ['volume', 'null_rate', 'violation_rate',
                    'anomaly_rate', 'avg_anomaly_score', 'delta_score']

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Architectures to try
    architectures = [
        (64, 32, 16),    # original
        (128, 64, 32),   # wider
        (32, 16),        # narrower
    ]

    best_score = 0
    best_model = None
    best_arch = None
    cv_results = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for arch in architectures:
        print(f"\n  Testing architecture {arch}...")

        model = MLPClassifier(
            hidden_layer_sizes=arch,
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42,
            verbose=False,
            early_stopping=True,
            validation_fraction=0.15,
        )

        # Cross-validation with sample weights
        cv_f1_scores = []
        for train_idx, val_idx in skf.split(X_train_s, y_train):
            X_cv_train = X_train_s[train_idx]
            y_cv_train = y_train[train_idx]
            X_cv_val = X_train_s[val_idx]
            y_cv_val = y_train[val_idx]

            cv_model = MLPClassifier(
                hidden_layer_sizes=arch,
                activation='relu',
                solver='adam',
                max_iter=1000,
                random_state=42,
                verbose=False,
                early_stopping=True,
                validation_fraction=0.15,
            )
            cv_model.fit(X_cv_train, y_cv_train)
            cv_pred = cv_model.predict(X_cv_val)
            cv_f1_scores.append(f1_score(y_cv_val, cv_pred, average='weighted'))

        cv_mean = float(np.mean(cv_f1_scores))
        cv_std = float(np.std(cv_f1_scores))

        # Train final model
        model.fit(X_train_s, y_train)
        test_score = float(model.score(X_test_s, y_test))
        test_f1 = float(f1_score(y_test, model.predict(X_test_s), average='weighted'))

        # Cross-validation F1
        cv_scores = cross_val_score(model, X_train_s, y_train, cv=skf, scoring='f1_weighted')
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))

        # Train final model
        model.fit(X_train_s, y_train)
        test_score = float(model.score(X_test_s, y_test))
        test_f1 = float(f1_score(y_test, model.predict(X_test_s), average='weighted'))

        print(f"    CV F1: {cv_mean:.3f} (+/- {cv_std:.3f})")
        print(f"    Test accuracy: {test_score:.3f}, Test F1: {test_f1:.3f}")

        cv_results.append({
            'arch': arch,
            'cv_f1_mean': cv_mean,
            'cv_f1_std': cv_std,
            'test_acc': test_score,
            'test_f1': test_f1,
        })

        if cv_mean > best_score:
            best_score = cv_mean
            best_model = model
            best_arch = arch

    print(f"\n  Best architecture: {best_arch} (CV F1={best_score:.3f})")
    return best_model, scaler, cv_results, best_arch


def main():
    parser = argparse.ArgumentParser(description='Train METER hypernetwork')
    parser.add_argument('--real-data', type=str,
                        default='data/meter_training_real.csv',
                        help='Path to real drift scenario CSV')
    parser.add_argument('--synthetic-data', type=str,
                        default='data/meter_training_synthetic.csv',
                        help='Path to synthetic training CSV')
    parser.add_argument('--output-dir', type=str,
                        default='models',
                        help='Output directory for model')
    parser.add_argument('--n-synthetic', type=int, default=1200,
                        help='N synthetic samples if no file exists')

    args = parser.parse_args()

    print("=" * 60)
    print("METER Hypernetwork Training (Step 3)")
    print("=" * 60)

    # Load data
    print("\nLoading training data...")
    real_path = Path(args.real_data)
    synth_path = Path(args.synthetic_data)

    if not real_path.exists() and not synth_path.exists():
        # Generate synthetic if neither exists
        print("  No data files found, generating synthetic...")
        df_synth = generate_inline_synthetic(args.n_synthetic)
        df = df_synth
    else:
        df = load_data(
            real_path=str(real_path) if real_path.exists() else None,
            synthetic_path=str(synth_path) if synth_path.exists() else None
        )

    print(f"\nTotal records: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    # Prepare features and labels
    feature_cols = ['volume', 'null_rate', 'violation_rate',
                    'avg_anomaly_score', 'anomaly_rate', 'delta_score']
    # Handle both column name variants
    if 'avg_score' in df.columns and 'avg_anomaly_score' not in df.columns:
        df = df.rename(columns={'avg_score': 'avg_anomaly_score'})
        feature_cols = ['volume', 'null_rate', 'violation_rate',
                       'avg_anomaly_score', 'anomaly_rate', 'delta_score']

    available_cols = [c for c in feature_cols if c in df.columns]
    if len(available_cols) < 6:
        print(f"WARNING: Only {len(available_cols)} features available: {available_cols}")
        feature_cols = available_cols

    X = df[feature_cols].values.astype(np.float64)
    y = df['strategy'].values.astype(np.int32)

    print(f"\nFeature columns: {feature_cols}")
    print(f"\nStrategy distribution:")
    for s, name in STRATEGIES.items():
        count = int((y == s).sum())
        print(f"  {s} ({name}): {count} ({count/len(y)*100:.1f}%)")

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

    # Train and select best architecture
    print("\nTraining and selecting best architecture...")
    model, scaler, cv_results, best_arch = train_and_select_model(
        X_train, y_train, X_test, y_test
    )

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation (on held-out test set)")
    print("=" * 60)

    X_test_s = scaler.transform(X_test)
    y_pred = model.predict(X_test_s)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=[STRATEGIES[i] for i in range(4)]))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    test_acc = float(model.score(X_test_s, y_test))
    test_f1 = float(f1_score(y_test, y_pred, average='weighted'))

    # Save model
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)

    model_path = output_dir / 'meter_hypernetwork.pkl'
    scaler_path = output_dir / 'meter_scaler.pkl'
    metadata_path = output_dir / 'meter_metadata.json'

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    metadata = {
        'model': 'METER Hypernetwork (sklearn MLPClassifier)',
        'architecture': str(best_arch),
        'input_features': feature_cols,
        'output_strategies': STRATEGIES,
        'train_accuracy': float(model.score(scaler.transform(X_train), y_train)),
        'test_accuracy': test_acc,
        'test_f1_weighted': test_f1,
        'n_samples_total': int(len(df)),
        'n_samples_train': int(len(X_train)),
        'cv_results': cv_results,
        'class_distribution': {
            STRATEGIES[k]: int((y == k).sum()) for k in STRATEGIES
        },
        'source_data': {
            'real': str(real_path) if real_path.exists() else None,
            'synthetic': str(synth_path) if synth_path.exists() else None,
        }
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 60}")
    print("METER MODEL SAVED")
    print(f"{'=' * 60}")
    print(f"  {model_path}")
    print(f"  {scaler_path}")
    print(f"  {metadata_path}")
    print(f"  Test accuracy: {test_acc:.3f}")
    print(f"  Test F1 (weighted): {test_f1:.3f}")
    print(f"  Architecture: {best_arch}")

    # Quick test predictions
    print(f"\n{'=' * 60}")
    print("Test Predictions (sample cases)")
    print(f"{'=' * 60}")

    test_cases = [
        {'volume': 1000, 'null_rate': 0.01, 'violation_rate': 0.03,
         'anomaly_rate': 0.05, 'avg_anomaly_score': 0.40, 'delta_score': 0.10,
         'desc': 'Normal operation'},
        {'volume': 950, 'null_rate': 0.02, 'violation_rate': 0.05,
         'anomaly_rate': 0.09, 'avg_anomaly_score': 0.52, 'delta_score': 0.12,
         'desc': 'Threshold drift'},
        {'volume': 800, 'null_rate': 0.06, 'violation_rate': 0.12,
         'anomaly_rate': 0.22, 'avg_anomaly_score': 0.62, 'delta_score': 0.20,
         'desc': 'Distribution shift'},
        {'volume': 550, 'null_rate': 0.15, 'violation_rate': 0.30,
         'anomaly_rate': 0.55, 'avg_anomaly_score': 0.88, 'delta_score': 0.45,
         'desc': 'Concept drift (severe)'},
    ]

    for tc in test_cases:
        feat = np.array([[tc[c] for c in feature_cols]])
        feat_s = scaler.transform(feat)
        pred_id = int(model.predict(feat_s)[0])
        probs = model.predict_proba(feat_s)[0]
        conf = float(probs[pred_id])
        print(f"  {tc['desc']:30s} -> {STRATEGIES[pred_id]} (conf={conf:.2f})")

    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    sys.exit(main())
