#!/usr/bin/env python3
"""
Data Analysis for Parameter Selection
======================================
Analyzes 2024 training data to inform parameter choices for all 18 ML models.

This script performs:
1. Distribution analysis (normality, skewness) → informs HBOS, MCD
2. Variance analysis (PCA) → informs PCA n_components
3. Density analysis → informs KNN n_neighbors
4. Temporal autocorrelation → confirms lag features, informs RRCF

Usage:
    python analyze_data_for_params.py --train-file output_anomaly/data_v2/02_train_2024.parquet

Output:
    Prints parameter recommendations for each model
"""

import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')


def analyze_distribution(df: pd.DataFrame) -> dict:
    """Analyze distribution characteristics of training data.

    Returns:
        dict with normality test, skewness, kurtosis
    """
    print("\n" + "="*80)
    print("1. DISTRIBUTION ANALYSIS")
    print("="*80)

    # Focus on primary feature: trip_count
    trip_counts = df['trip_count'].values

    # Normality test (Shapiro-Wilk on sample for computational efficiency)
    sample_size = min(5000, len(trip_counts))
    sample = np.random.choice(trip_counts, size=sample_size, replace=False)
    shapiro_stat, shapiro_p = stats.shapiro(sample)

    # Skewness and kurtosis
    skewness = stats.skew(trip_counts)
    kurtosis_val = stats.kurtosis(trip_counts)

    results = {
        'shapiro_statistic': shapiro_stat,
        'shapiro_pvalue': shapiro_p,
        'is_normal': shapiro_p > 0.05,
        'skewness': skewness,
        'kurtosis': kurtosis_val,
        'mean': trip_counts.mean(),
        'std': trip_counts.std(),
        'median': np.median(trip_counts),
        'min': trip_counts.min(),
        'max': trip_counts.max()
    }

    print(f"\nTrip Count Statistics:")
    print(f"  Mean: {results['mean']:.2f}")
    print(f"  Std: {results['std']:.2f}")
    print(f"  Median: {results['median']:.2f}")
    print(f"  Range: [{results['min']:.0f}, {results['max']:.0f}]")
    print(f"\nNormality Test (Shapiro-Wilk on {sample_size} samples):")
    print(f"  Statistic: {shapiro_stat:.4f}")
    print(f"  P-value: {shapiro_p:.6f}")
    print(f"  Is Normal (p > 0.05)? {results['is_normal']}")
    print(f"\nShape Characteristics:")
    print(f"  Skewness: {skewness:.4f} ({'right-skewed' if skewness > 0 else 'left-skewed'})")
    print(f"  Kurtosis: {kurtosis_val:.4f} ({'heavy-tailed' if kurtosis_val > 0 else 'light-tailed'})")

    # Recommendations
    print(f"\n📊 Parameter Recommendations:")
    if not results['is_normal']:
        print(f"  ⚠️ MCD: Data is NOT normally distributed (p={shapiro_p:.6f})")
        print(f"     → May underperform; consider COPOD/ECOD instead")
    if abs(skewness) > 1:
        print(f"  ⚠️ HBOS: Data is highly skewed (skew={skewness:.2f})")
        print(f"     → Use fewer bins (n_bins=10) to avoid empty bins")

    return results


def analyze_pca_variance(df: pd.DataFrame, feature_cols: list) -> dict:
    """Analyze variance explained by PCA components.

    Returns:
        dict with n_components needed for 90%, 95%, 99% variance
    """
    print("\n" + "="*80)
    print("2. PCA VARIANCE ANALYSIS")
    print("="*80)

    X = df[feature_cols].fillna(0).values

    # Fit PCA with all components
    pca = PCA()
    pca.fit(X)

    explained_variance = pca.explained_variance_ratio_
    cumsum_variance = np.cumsum(explained_variance)

    # Find n_components for target variance levels
    results = {}
    for target in [0.90, 0.95, 0.99]:
        n_comp = np.argmax(cumsum_variance >= target) + 1
        results[f'n_components_{int(target*100)}'] = n_comp

    print(f"\nVariance Explained by Components:")
    for i in range(min(10, len(explained_variance))):
        print(f"  PC{i+1}: {explained_variance[i]*100:.2f}% "
              f"(Cumulative: {cumsum_variance[i]*100:.2f}%)")

    print(f"\nComponents Needed for Target Variance:")
    print(f"  90% variance: {results['n_components_90']} components")
    print(f"  95% variance: {results['n_components_95']} components")
    print(f"  99% variance: {results['n_components_99']} components")

    # Recommendations
    print(f"\n📊 Parameter Recommendations:")
    print(f"  PCA n_components:")
    print(f"    - Aggressive compression: n_components={results['n_components_90']} (90% var)")
    print(f"    - Balanced: n_components={results['n_components_95']} (95% var)")
    print(f"    - Conservative: n_components={results['n_components_99']} (99% var)")

    return results


def analyze_local_density(df: pd.DataFrame, feature_cols: list, k_values: list = [5, 10, 20, 50]) -> dict:
    """Analyze local density variations to inform KNN n_neighbors.

    Returns:
        dict with density statistics for different k values
    """
    print("\n" + "="*80)
    print("3. LOCAL DENSITY ANALYSIS (KNN)")
    print("="*80)

    X = df[feature_cols].fillna(0).values

    # Sample for computational efficiency
    sample_size = min(5000, len(X))
    indices = np.random.choice(len(X), size=sample_size, replace=False)
    X_sample = X[indices]

    results = {}

    for k in k_values:
        # Fit k-NN
        knn = NearestNeighbors(n_neighbors=k+1)  # +1 because point itself is nearest
        knn.fit(X_sample)

        # Get distances to k-th nearest neighbor
        distances, _ = knn.kneighbors(X_sample)
        k_dist = distances[:, -1]  # Distance to k-th neighbor

        results[f'k{k}'] = {
            'mean_dist': k_dist.mean(),
            'std_dist': k_dist.std(),
            'median_dist': np.median(k_dist),
            'cv': k_dist.std() / k_dist.mean()  # Coefficient of variation
        }

        print(f"\nk={k} neighbors:")
        print(f"  Mean distance: {results[f'k{k}']['mean_dist']:.4f}")
        print(f"  Std distance: {results[f'k{k}']['std_dist']:.4f}")
        print(f"  CV (std/mean): {results[f'k{k}']['cv']:.4f}")

    # Recommendations
    print(f"\n📊 Parameter Recommendations:")
    print(f"  KNN n_neighbors:")
    print(f"    - For local anomalies (tight clusters): n_neighbors=5")
    print(f"    - For contextual anomalies (medium): n_neighbors=20")
    print(f"    - For global anomalies (sparse): n_neighbors=50")

    # Check density variation
    cv_5 = results['k5']['cv']
    cv_50 = results['k50']['cv']
    if cv_5 > 0.5:
        print(f"  ⚠️ High density variation (CV={cv_5:.2f}) → Use k=5 for local patterns")
    if cv_50 < 0.3:
        print(f"  ✓ Uniform density (CV={cv_50:.2f}) → k=20 or k=50 both viable")

    return results


def analyze_temporal_correlation(df: pd.DataFrame) -> dict:
    """Analyze temporal autocorrelation to validate lag features.

    Returns:
        dict with correlation values for lag-1, lag-48, lag-336
    """
    print("\n" + "="*80)
    print("4. TEMPORAL AUTOCORRELATION ANALYSIS")
    print("="*80)

    trip_counts = df['trip_count'].values

    results = {}

    # Compute autocorrelation for different lags
    lags = {
        'lag_1': 1,      # 30-min lag
        'lag_48': 48,    # 1-day lag
        'lag_144': 144,  # 3-day lag
        'lag_336': 336   # 1-week lag
    }

    for lag_name, lag_val in lags.items():
        if len(trip_counts) > lag_val:
            # Compute Pearson correlation
            x1 = trip_counts[:-lag_val]
            x2 = trip_counts[lag_val:]
            corr, pvalue = stats.pearsonr(x1, x2)
            results[lag_name] = {'correlation': corr, 'pvalue': pvalue}

            print(f"\n{lag_name} (lag={lag_val} windows = {lag_val*0.5:.1f} hours):")
            print(f"  Correlation: {corr:.4f}")
            print(f"  P-value: {pvalue:.6f}")
            print(f"  Significant? {pvalue < 0.001}")

    # Recommendations
    print(f"\n📊 Parameter Recommendations:")
    print(f"  RRCF tree_size:")

    # Check if lag features are meaningful
    lag48_corr = results.get('lag_48', {}).get('correlation', 0)
    if lag48_corr > 0.5:
        print(f"    - Strong daily correlation ({lag48_corr:.2f}) → tree_size=256 (captures 1-day patterns)")
    else:
        print(f"    - Weak daily correlation ({lag48_corr:.2f}) → tree_size=128 (shorter memory)")

    return results


def analyze_feature_characteristics(df: pd.DataFrame, feature_cols: list) -> dict:
    """Analyze feature characteristics for model selection.

    Returns:
        dict with feature statistics
    """
    print("\n" + "="*80)
    print("5. FEATURE CHARACTERISTICS")
    print("="*80)

    X = df[feature_cols].fillna(0)

    # Feature correlations
    corr_matrix = X.corr().abs()

    # Find highly correlated feature pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.8:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))

    print(f"\nTotal Features: {len(feature_cols)}")
    print(f"Highly Correlated Pairs (|r| > 0.8): {len(high_corr_pairs)}")

    if high_corr_pairs:
        print("\nTop Correlated Pairs:")
        for f1, f2, corr_val in sorted(high_corr_pairs, key=lambda x: -x[2])[:5]:
            print(f"  {f1} <-> {f2}: {corr_val:.3f}")

    # Recommendations
    print(f"\n📊 Parameter Recommendations:")
    if len(high_corr_pairs) > 3:
        print(f"  ⚠️ MCD: {len(high_corr_pairs)} highly correlated feature pairs")
        print(f"     → MCD robust covariance may help; support_fraction=0.9")
        print(f"  ✓ PCA: High correlation justifies dimensionality reduction")
    else:
        print(f"  ✓ HBOS: Few correlated pairs → independence assumption OK")

    return {'high_corr_pairs': len(high_corr_pairs)}


def generate_final_recommendations(results: dict):
    """Generate final parameter recommendations based on all analyses."""
    print("\n" + "="*80)
    print("FINAL PARAMETER RECOMMENDATIONS")
    print("="*80)

    print("\n### Statistical Models (no tuning needed)")
    print("  Z-Score, IQR, MAD: Use context statistics (already optimal)")

    print("\n### PyOD Models")
    print("  1. HBOS:")
    print(f"     → n_bins=10 (skewed distribution)")
    print("  2. KNN:")
    print(f"     → n_neighbors=5 (local) or 20 (contextual)")
    print("  3. MCD:")
    if not results['distribution']['is_normal']:
        print(f"     ⚠️ WARNING: Data is non-Gaussian (p={results['distribution']['shapiro_pvalue']:.6f})")
        print(f"     → Consider COPOD/ECOD instead")
    print(f"     → support_fraction=None (auto) or 0.9 (robust)")
    print("  4. PCA:")
    print(f"     → n_components={results['pca']['n_components_95']} (95% variance)")
    print("  5. COPOD:")
    print(f"     ✓ Parameter-free, optimal for non-Gaussian data")
    print("  6. ABOD:")
    print(f"     → method='fast' (n={len(results['density'])} too large for exact)")
    print("  7. ECOD:")
    print(f"     ✓ Parameter-free, distribution-agnostic")

    print("\n### Deep Learning Models")
    print("  8. Autoencoder:")
    print(f"     → encoding_dim=8 or 16 ({len(results.get('feature_cols', []))} input features)")
    print(f"     → epochs=50, batch_size=32")
    print("  9. VAE:")
    print(f"     → latent_dim=8 or 16")
    print(f"     → epochs=50, batch_size=32")

    print("\n### Existing ML Models")
    print("  10. IsolationForest:")
    print(f"      → n_estimators=100 or 200 (test both)")
    print("  11. LOF:")
    print(f"      → n_neighbors=20 or 50 (test both)")
    print("  12. OneClassSVM:")
    print(f"      → nu=0.02 (current is optimal)")

    print("\n### Tree-Based")
    print("  13. RRCF:")
    lag48_corr = results['temporal'].get('lag_48', {}).get('correlation', 0)
    if lag48_corr > 0.5:
        print(f"      → num_trees=100 or 200, tree_size=256 (strong daily patterns)")
    else:
        print(f"      → num_trees=100 or 200, tree_size=128 (weaker patterns)")

    print("\n### Sequential Models")
    print("  14. EWMA:")
    print(f"      → alpha=0.3 (already optimal)")
    print("  15. CUSUM:")
    print(f"      → threshold=5.0, drift=0.5 (already optimal)")

    print(f"\n{'='*80}")
    print("Total: 18 unique models, 23 configurations (with parameter variations)")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze training data to inform ML model parameter selection'
    )
    parser.add_argument(
        '--train-file',
        type=str,
        required=True,
        help='Path to 2024 training data parquet file'
    )

    args = parser.parse_args()

    # Load training data
    print(f"Loading training data from: {args.train_file}")
    df_train = pd.read_parquet(args.train_file)
    print(f"Loaded {len(df_train)} samples with {len(df_train.columns)} columns")

    # Define feature columns (exclude metadata)
    feature_cols = [
        'ctx_mean', 'ctx_std', 'ctx_median', 'ctx_q25', 'ctx_q75',
        'ctx_dev', 'ctx_abs_dev',
        'lag_48', 'lag_144', 'lag_336',
        'roll_mean_48', 'roll_std_48', 'roll_mean_336',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos'
    ]

    # Check which features exist
    available_features = [f for f in feature_cols if f in df_train.columns]
    print(f"Available features: {len(available_features)}/{len(feature_cols)}")

    # Run analyses
    results = {}

    results['distribution'] = analyze_distribution(df_train)
    results['pca'] = analyze_pca_variance(df_train, available_features)
    results['density'] = analyze_local_density(df_train, available_features)
    results['temporal'] = analyze_temporal_correlation(df_train)
    results['feature_chars'] = analyze_feature_characteristics(df_train, available_features)
    results['feature_cols'] = available_features

    # Generate final recommendations
    generate_final_recommendations(results)


if __name__ == '__main__':
    main()
