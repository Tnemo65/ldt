"""PyOD-based Machine Learning Anomaly Detection Models.

This module implements 7 unsupervised outlier detection models using the PyOD library,
each with full scientific rigor including paper citations, complexity analysis, and
best use cases.

All models are configured with contamination=0.01 (1% expected outliers in clean
training data) and return outlier probability scores via predict_proba().

Models implemented:
1. HBOS - Histogram-Based Outlier Score (Goldstein 2012)
2. KNN - k-Nearest Neighbors (Ramaswamy 2000)
3. MCD - Minimum Covariance Determinant (Rousseeuw 1999)
4. PCA - Principal Component Analysis (Shyu 2003)
5. COPOD - Copula-Based Outlier Detection (Li 2020)
6. ABOD - Angle-Based Outlier Detection (Kriegel 2008)
7. ECOD - Empirical Cumulative Distribution-Based Outlier Detection (Li 2022)

Usage:
    import pandas as pd
    from benchmark.anomalies.exp_ml_models_pyod import train_hbos

    X_train = pd.DataFrame(...)  # Clean training data
    model = train_hbos(X_train, contamination=0.01)

    X_test = pd.DataFrame(...)
    scores = model.predict_proba(X_test.values)[:, 1]  # Outlier probability
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import joblib
from pyod.models.hbos import HBOS
from pyod.models.knn import KNN
from pyod.models.mcd import MCD
from pyod.models.pca import PCA
from pyod.models.copod import COPOD
from pyod.models.abod import ABOD
from pyod.models.ecod import ECOD

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_hbos(X_train: pd.DataFrame, contamination: float = 0.01) -> HBOS:
    """Train Histogram-Based Outlier Score (HBOS) model.

    Scientific basis:
        Goldstein, M., & Dengel, A. (2012). Histogram-based outlier score (HBOS):
        A fast unsupervised anomaly detection algorithm. KI-2012: Poster and Demo Track.

    Algorithm:
        - Assumes feature independence
        - Builds histogram for each feature
        - Scores based on inverse probability density
        - Combines scores across features (multiplication or addition)

    Complexity:
        - Training: O(n * d) where n=samples, d=features
        - Prediction: O(d)
        - Memory: O(d * bins)

    Best use cases:
        - High-dimensional data with independent features
        - Fast training and prediction required
        - Linear complexity is essential
        - Features have varying distributions

    Args:
        X_train: Training features (clean data only), shape (n_samples, n_features)
        contamination: Expected outlier proportion in training data (default 0.01)

    Returns:
        Trained HBOS model with predict_proba() method

    Example:
        >>> X_train = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        >>> model = train_hbos(X_train, contamination=0.01)
        >>> scores = model.predict_proba(X_train.values)[:, 1]
    """
    logger.info(f"Training HBOS with {len(X_train)} samples, {X_train.shape[1]} features")
    model = HBOS(contamination=contamination, n_bins=10)
    model.fit(X_train.values)
    logger.info("HBOS training complete")
    return model


def train_knn(X_train: pd.DataFrame, contamination: float = 0.01) -> KNN:
    """Train k-Nearest Neighbors (KNN) outlier detection model.

    Scientific basis:
        Ramaswamy, S., Rastogi, R., & Shim, K. (2000). Efficient algorithms for
        mining outliers from large data sets. ACM SIGMOD Record, 29(2), 427-438.

    Algorithm:
        - Computes distance to k-th nearest neighbor for each point
        - Points with large distances to k-th neighbor are outliers
        - Uses 'largest' method: distance to k-th neighbor

    Complexity:
        - Training: O(1) (lazy learning)
        - Prediction: O(n * d) where n=training samples, d=features
        - Memory: O(n * d)

    Best use cases:
        - Small to medium datasets (n < 10,000)
        - Local density-based outliers
        - No assumptions about data distribution
        - Clusters with varying densities

    Args:
        X_train: Training features (clean data only), shape (n_samples, n_features)
        contamination: Expected outlier proportion in training data (default 0.01)

    Returns:
        Trained KNN model with predict_proba() method

    Example:
        >>> X_train = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        >>> model = train_knn(X_train, contamination=0.01)
        >>> scores = model.predict_proba(X_train.values)[:, 1]
    """
    logger.info(f"Training KNN with {len(X_train)} samples, {X_train.shape[1]} features")
    model = KNN(contamination=contamination, n_neighbors=5, method='largest')
    model.fit(X_train.values)
    logger.info("KNN training complete")
    return model


def train_mcd(X_train: pd.DataFrame, contamination: float = 0.01) -> MCD:
    """Train Minimum Covariance Determinant (MCD) outlier detection model.

    Scientific basis:
        Rousseeuw, P. J., & Driessen, K. V. (1999). A fast algorithm for the
        minimum covariance determinant estimator. Technometrics, 41(3), 212-223.

    Algorithm:
        - Finds subset of observations with minimum covariance determinant
        - Fits robust covariance estimate on this subset
        - Scores based on Mahalanobis distance from robust center
        - Assumes data follows multivariate Gaussian distribution

    Complexity:
        - Training: O(n * d²) where n=samples, d=features
        - Prediction: O(d²)
        - Memory: O(d²)

    Best use cases:
        - Gaussian or near-Gaussian data
        - Correlated features (multivariate dependencies)
        - Small to medium number of features (d < 50)
        - Detecting global outliers

    Args:
        X_train: Training features (clean data only), shape (n_samples, n_features)
        contamination: Expected outlier proportion in training data (default 0.01)

    Returns:
        Trained MCD model with predict_proba() method

    Example:
        >>> X_train = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        >>> model = train_mcd(X_train, contamination=0.01)
        >>> scores = model.predict_proba(X_train.values)[:, 1]
    """
    logger.info(f"Training MCD with {len(X_train)} samples, {X_train.shape[1]} features")
    model = MCD(contamination=contamination, support_fraction=None)
    model.fit(X_train.values)
    logger.info("MCD training complete")
    return model


def train_pca(X_train: pd.DataFrame, contamination: float = 0.01) -> PCA:
    """Train Principal Component Analysis (PCA) outlier detection model.

    Scientific basis:
        Shyu, M. L., Chen, S. C., Sarinnapakorn, K., & Chang, L. (2003).
        A novel anomaly detection scheme based on principal component classifier.
        ICDM Foundation and New Direction of Data Mining Workshop.

    Algorithm:
        - Performs PCA dimensionality reduction
        - Projects data to principal components
        - Scores based on reconstruction error
        - Outliers have high reconstruction error

    Complexity:
        - Training: O(n * d²) where n=samples, d=features
        - Prediction: O(d * k) where k=n_components
        - Memory: O(d * k)

    Best use cases:
        - High-dimensional data with linear relationships
        - Data lying on lower-dimensional manifold
        - Global outliers that don't fit linear structure
        - Dimensionality reduction beneficial

    Args:
        X_train: Training features (clean data only), shape (n_samples, n_features)
        contamination: Expected outlier proportion in training data (default 0.01)

    Returns:
        Trained PCA model with predict_proba() method

    Example:
        >>> X_train = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        >>> model = train_pca(X_train, contamination=0.01)
        >>> scores = model.predict_proba(X_train.values)[:, 1]
    """
    n_components = min(5, X_train.shape[1])
    logger.info(f"Training PCA with {len(X_train)} samples, {X_train.shape[1]} features, {n_components} components")
    model = PCA(contamination=contamination, n_components=n_components)
    model.fit(X_train.values)
    logger.info("PCA training complete")
    return model


def train_copod(X_train: pd.DataFrame, contamination: float = 0.01) -> COPOD:
    """Train Copula-Based Outlier Detection (COPOD) model.

    Scientific basis:
        Li, Z., Zhao, Y., Botta, N., Ionescu, C., & Hu, X. (2020).
        COPOD: Copula-Based Outlier Detection.
        IEEE International Conference on Data Mining (ICDM).
        ** BEST PAPER AWARD **

    Algorithm:
        - Models tail probability of each feature using empirical copula
        - Combines tail probabilities across features
        - Parameter-free (no hyperparameters to tune)
        - Handles feature dependencies via copula theory

    Complexity:
        - Training: O(n * d) where n=samples, d=features
        - Prediction: O(d)
        - Memory: O(n * d)

    Best use cases:
        - Any data distribution (non-parametric)
        - Features with complex dependencies
        - No hyperparameter tuning desired
        - Fast training and prediction required
        - Robust to different feature scales

    Args:
        X_train: Training features (clean data only), shape (n_samples, n_features)
        contamination: Expected outlier proportion in training data (default 0.01)

    Returns:
        Trained COPOD model with predict_proba() method

    Example:
        >>> X_train = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        >>> model = train_copod(X_train, contamination=0.01)
        >>> scores = model.predict_proba(X_train.values)[:, 1]
    """
    logger.info(f"Training COPOD with {len(X_train)} samples, {X_train.shape[1]} features")
    model = COPOD(contamination=contamination)
    model.fit(X_train.values)
    logger.info("COPOD training complete")
    return model


def train_abod(X_train: pd.DataFrame, contamination: float = 0.01) -> ABOD:
    """Train Angle-Based Outlier Detection (ABOD) model.

    Scientific basis:
        Kriegel, H. P., Schubert, M., & Zimek, A. (2008).
        Angle-based outlier detection in high-dimensional data.
        ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.

    Algorithm:
        - Computes variance of angles between point and all pairs of neighbors
        - Outliers have small angle variance (appear in sparse regions)
        - Uses 'fast' approximation for scalability
        - Samples random neighbors if n > 1000

    Complexity:
        - Training: O(n³) for full, O(n²) for fast approximation
        - Prediction: O(n²) for fast approximation
        - Memory: O(n²)

    Best use cases:
        - High-dimensional data where distance-based methods fail
        - Detecting outliers in sparse regions
        - Small datasets (n < 1000 for exact, n < 10000 for fast)
        - Angle relationships more meaningful than distances

    Args:
        X_train: Training features (clean data only), shape (n_samples, n_features)
        contamination: Expected outlier proportion in training data (default 0.01)

    Returns:
        Trained ABOD model with predict_proba() method

    Example:
        >>> X_train = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        >>> model = train_abod(X_train, contamination=0.01)
        >>> scores = model.predict_proba(X_train.values)[:, 1]
    """
    logger.info(f"Training ABOD with {len(X_train)} samples, {X_train.shape[1]} features")
    # Use 'fast' method and sample for large datasets (O(n³) is prohibitive)
    method = 'fast'
    logger.info(f"Using ABOD method='{method}'")
    model = ABOD(contamination=contamination, method=method)
    model.fit(X_train.values)
    logger.info("ABOD training complete")
    return model


def train_ecod(X_train: pd.DataFrame, contamination: float = 0.01) -> ECOD:
    """Train Empirical Cumulative Distribution-Based Outlier Detection (ECOD) model.

    Scientific basis:
        Li, Z., Zhao, Y., Hu, X., Botta, N., Ionescu, C., & Chen, G. H. (2022).
        ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions.
        IEEE Transactions on Knowledge and Data Engineering (TKDE).

    Algorithm:
        - Estimates empirical CDF for each feature
        - Computes tail probability for each feature value
        - Combines tail probabilities across features
        - Parameter-free (no hyperparameters to tune)

    Complexity:
        - Training: O(n * d * log n) where n=samples, d=features
        - Prediction: O(d * log n)
        - Memory: O(n * d)

    Best use cases:
        - Any data distribution (non-parametric)
        - Features with different distributions
        - No hyperparameter tuning desired
        - Fast and scalable to large datasets
        - Interpretable outlier scores (tail probabilities)

    Args:
        X_train: Training features (clean data only), shape (n_samples, n_features)
        contamination: Expected outlier proportion in training data (default 0.01)

    Returns:
        Trained ECOD model with predict_proba() method

    Example:
        >>> X_train = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        >>> model = train_ecod(X_train, contamination=0.01)
        >>> scores = model.predict_proba(X_train.values)[:, 1]
    """
    logger.info(f"Training ECOD with {len(X_train)} samples, {X_train.shape[1]} features")
    model = ECOD(contamination=contamination)
    model.fit(X_train.values)
    logger.info("ECOD training complete")
    return model


def main():
    """Train all 7 PyOD models on real data and save to disk.

    Usage:
        python exp_ml_models_pyod.py --input output_anomaly/train_2024.parquet \\
                                     --output-dir output_anomaly/models_pyod
    """
    parser = argparse.ArgumentParser(
        description='Train PyOD-based ML anomaly detection models'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to training data parquet file (clean data only)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory to save trained models'
    )
    parser.add_argument(
        '--contamination',
        type=float,
        default=0.01,
        help='Expected outlier proportion in training data (default: 0.01)'
    )

    args = parser.parse_args()

    # Load training data
    logger.info(f"Loading training data from {args.input}")
    df_train = pd.read_parquet(args.input)

    # Extract features (exclude date columns and identifiers)
    feature_cols = [col for col in df_train.columns
                   if col not in ['date', 'ticker', 'symbol', 'label', 'anomaly']]
    X_train = df_train[feature_cols]

    logger.info(f"Training on {len(X_train)} samples with {len(feature_cols)} features")
    logger.info(f"Features: {feature_cols}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving models to {output_dir}")

    # Train all 7 models
    models = {
        'hbos': train_hbos,
        'knn': train_knn,
        'mcd': train_mcd,
        'pca': train_pca,
        'copod': train_copod,
        'abod': train_abod,
        'ecod': train_ecod
    }

    for model_name, train_fn in models.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"Training {model_name.upper()} model")
        logger.info(f"{'='*80}")

        try:
            model = train_fn(X_train, contamination=args.contamination)

            # Save model
            model_path = output_dir / f"{model_name}_model.pkl"
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name.upper()} model to {model_path}")

            # Test prediction on first 10 samples
            test_scores = model.predict_proba(X_train.values[:10])[:, 1]
            logger.info(f"Test scores (first 10): {test_scores}")

        except Exception as e:
            logger.error(f"Failed to train {model_name.upper()}: {e}")
            raise

    logger.info("\n" + "="*80)
    logger.info("All models trained successfully!")
    logger.info("="*80)


if __name__ == '__main__':
    main()
