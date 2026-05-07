"""
Fit StandardScaler on Jan 2024 clean baseline.
Spec: Lines 3663-3673 (CRITICAL: prevents data leakage)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from sklearn.preprocessing import StandardScaler
from src.features.vectorizer import FeatureVectorizer

def fit_scaler(data_path: Path, output_path: Path, sample_size=100_000):
    """Fit StandardScaler on clean baseline (NO test data)."""

    print("Loading clean baseline...")
    df = pd.read_parquet(data_path)
    print(f"Records: {len(df):,}")

    # Sample for faster fitting (Phase 0 setup)
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
        print(f"Sampled: {len(df):,} records for fitting")

    # Vectorize features
    print("\nVectorizing features (15D)...")
    vectorizer = FeatureVectorizer()

    features = []
    for i, row in df.iterrows():
        vec = vectorizer.transform(row.to_dict())
        features.append(vec)

        if (i + 1) % 10000 == 0 and i < len(df) - 1:
            print(f"  Processed: {len(features):,} / {len(df):,}")

    X = np.array(features)
    print(f"Feature matrix: {X.shape}")

    # Fit scaler
    print("\nFitting StandardScaler...")
    scaler = StandardScaler()
    scaler.fit(X)

    print(f"Mean: {scaler.mean_[:5]} ...")
    print(f"Scale: {scaler.scale_[:5]} ...")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\n✅ Saved scaler to {output_path}")
    print(f"   Fitted on {len(X):,} samples")
    print(f"   Feature dim: {len(scaler.mean_)}D")

def main():
    data_path = Path('data/clean/jan_2024_clean_baseline.parquet')
    output_path = Path('models/scaler.pkl')
    fit_scaler(data_path, output_path)

if __name__ == "__main__":
    main()
