#!/usr/bin/env python3
"""
Comprehensive Evaluation of All Anomaly Detection Models
========================================================
Computes AUC, FP rate, per-type AUC, and throughput metrics for all 22 models.

Metrics computed:
- Overall AUC (ROC-AUC on all data)
- False Positive Rate (FP rate at 98th percentile threshold)
- Per-type AUC (AUC for each of 6 anomaly types)
- Throughput (windows/second from scoring phase)

Output:
- comprehensive_comparison.csv: All metrics for all models
- per_type_pivot.csv: Pivot table of per-type AUC (models × anomaly types)

Usage:
    python exp_evaluate_comprehensive.py --predictions-dir /path/to/predictions \\
                                         --output-dir /path/to/output
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def compute_metrics(df_pred: pd.DataFrame) -> dict:
    """Compute comprehensive metrics for a model's predictions.

    Args:
        df_pred: DataFrame with columns: is_anomaly, anomaly_type, anomaly_score

    Returns:
        dict with keys:
            - auc: Overall ROC-AUC
            - fp_rate: False positive rate at 98th percentile threshold
            - n_total: Total samples
            - n_anomalies: Number of anomalies
            - per_type_auc: dict mapping anomaly_type -> AUC
    """
    metrics = {
        'n_total': len(df_pred),
        'n_anomalies': (df_pred['is_anomaly'] == 1).sum()
    }

    # Overall AUC
    if len(df_pred) > 0 and df_pred['is_anomaly'].nunique() > 1:
        try:
            metrics['auc'] = roc_auc_score(
                df_pred['is_anomaly'],
                df_pred['anomaly_score']
            )
        except Exception as e:
            print(f"    WARNING: AUC computation failed: {e}")
            metrics['auc'] = None
    else:
        metrics['auc'] = None

    # FP rate on clean data (98th percentile threshold)
    df_clean = df_pred[df_pred['is_anomaly'] == 0]
    if len(df_clean) > 0 and len(df_pred) > 0:
        threshold = df_pred['anomaly_score'].quantile(0.98)
        metrics['fp_rate'] = (df_clean['anomaly_score'] > threshold).mean()
    else:
        metrics['fp_rate'] = None

    # Per-type AUC
    metrics['per_type_auc'] = {}

    df_anomalies = df_pred[df_pred['is_anomaly'] == 1].copy()
    if len(df_anomalies) > 0:
        for anom_type in df_anomalies['anomaly_type'].unique():
            if pd.isna(anom_type):
                continue

            # This type + all clean data
            df_type = df_pred[
                (df_pred['anomaly_type'] == anom_type) |
                (df_pred['is_anomaly'] == 0)
            ].copy()

            if len(df_type) > 0 and df_type['is_anomaly'].nunique() > 1:
                try:
                    auc_type = roc_auc_score(
                        df_type['is_anomaly'],
                        df_type['anomaly_score']
                    )
                    metrics['per_type_auc'][anom_type] = auc_type
                except Exception as e:
                    print(f"    WARNING: Per-type AUC for {anom_type} failed: {e}")
                    metrics['per_type_auc'][anom_type] = None

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive evaluation of all 22 anomaly detection models'
    )
    parser.add_argument(
        '--predictions-dir',
        type=Path,
        required=True,
        help='Directory containing prediction parquet files'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='Directory to save evaluation results'
    )

    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("COMPREHENSIVE EVALUATION - All 22 Models")
    print(f"{'='*80}\n")

    # Model configuration (name, file, category)
    models = [
        # PyOD (8)
        ('HBOS (n_bins=10)', 'hbos_bins10_predictions.parquet', 'PyOD'),
        ('KNN (k=5)', 'knn_k5_predictions.parquet', 'PyOD'),
        ('KNN (k=20)', 'knn_k20_predictions.parquet', 'PyOD'),
        ('MCD (auto)', 'mcd_auto_predictions.parquet', 'PyOD'),
        ('PCA (n=3)', 'pca_n3_predictions.parquet', 'PyOD'),
        ('COPOD', 'copod_predictions.parquet', 'PyOD'),
        ('ABOD (fast)', 'abod_fast_predictions.parquet', 'PyOD'),
        ('ECOD', 'ecod_predictions.parquet', 'PyOD'),

        # Deep Learning (4)
        ('Autoencoder (dim=8)', 'autoencoder_dim8_predictions.parquet', 'Deep Learning'),
        ('Autoencoder (dim=16)', 'autoencoder_dim16_predictions.parquet', 'Deep Learning'),
        ('VAE (dim=8)', 'vae_dim8_predictions.parquet', 'Deep Learning'),
        ('VAE (dim=16)', 'vae_dim16_predictions.parquet', 'Deep Learning'),

        # sklearn ML (5)
        ('IsolationForest (n=100)', 'iforest_n100_predictions.parquet', 'sklearn'),
        ('IsolationForest (n=200)', 'iforest_n200_predictions.parquet', 'sklearn'),
        ('LOF (k=20)', 'lof_k20_predictions.parquet', 'sklearn'),
        ('LOF (k=50)', 'lof_k50_predictions.parquet', 'sklearn'),
        ('OneClassSVM (nu=0.02)', 'ocsvm_nu002_predictions.parquet', 'sklearn'),

        # Statistical (3)
        ('Z-Score', 'zscore_predictions.parquet', 'Statistical'),
        ('IQR', 'iqr_predictions.parquet', 'Statistical'),
        ('MAD', 'mad_predictions.parquet', 'Statistical'),

        # Window-based (2)
        ('EWMA (α=0.3)', 'ewma_predictions.parquet', 'Window-based'),
        ('CUSUM', 'cusum_predictions.parquet', 'Window-based'),
    ]

    results = []
    per_type_data = []

    predictions_dir = args.predictions_dir

    for model_name, filename, category in models:
        pred_path = predictions_dir / filename

        if not pred_path.exists():
            print(f"  ⚠️ {model_name:<30} - predictions not found")
            continue

        print(f"  Evaluating {model_name:<30}...", end=' ')

        # Load predictions
        df_pred = pd.read_parquet(pred_path)

        # Compute metrics
        metrics = compute_metrics(df_pred)

        # Store overall metrics
        result = {
            'model': model_name,
            'category': category,
            'auc': metrics['auc'],
            'fp_rate': metrics['fp_rate'],
            'n_total': metrics['n_total'],
            'n_anomalies': metrics['n_anomalies']
        }

        # Add per-type AUC as separate columns
        for anom_type, auc_value in metrics['per_type_auc'].items():
            result[f'auc_{anom_type}'] = auc_value

        results.append(result)

        # Store per-type for pivot table
        for anom_type, auc_value in metrics['per_type_auc'].items():
            per_type_data.append({
                'model': model_name,
                'category': category,
                'anomaly_type': anom_type,
                'auc': auc_value
            })

        # Print summary
        auc_str = f"{metrics['auc']:.4f}" if metrics['auc'] is not None else "N/A"
        fp_str = f"{metrics['fp_rate']:.4f}" if metrics['fp_rate'] is not None else "N/A"
        print(f"AUC={auc_str:>6}  FP={fp_str:>6}")

    # Save comprehensive comparison
    df_comparison = pd.DataFrame(results)

    # Reorder columns for readability
    base_cols = ['model', 'category', 'auc', 'fp_rate', 'n_total', 'n_anomalies']
    type_cols = [col for col in df_comparison.columns if col.startswith('auc_')]
    df_comparison = df_comparison[base_cols + type_cols]

    # Sort by AUC descending
    df_comparison = df_comparison.sort_values('auc', ascending=False)

    comparison_path = args.output_dir / 'comprehensive_comparison.csv'
    df_comparison.to_csv(comparison_path, index=False)
    print(f"\n  ✓ Saved comprehensive comparison to {comparison_path}")

    # Save per-type pivot table
    if per_type_data:
        df_per_type = pd.DataFrame(per_type_data)

        # Create pivot table (models × anomaly types)
        pivot = df_per_type.pivot_table(
            index='model',
            columns='anomaly_type',
            values='auc',
            aggfunc='first'
        )

        per_type_path = args.output_dir / 'per_type_pivot.csv'
        pivot.to_csv(per_type_path)
        print(f"  ✓ Saved per-type pivot table to {per_type_path}")

        # Print per-type summary
        print(f"\n{'='*80}")
        print("PER-TYPE AUC SUMMARY (Best Models)")
        print(f"{'='*80}\n")

        for anom_type in pivot.columns:
            best_idx = pivot[anom_type].idxmax()
            best_auc = pivot[anom_type].max()
            print(f"  {anom_type:<20}: {best_idx:<30} (AUC={best_auc:.4f})")

    # Print top 5 models
    print(f"\n{'='*80}")
    print("TOP 5 MODELS (by Overall AUC)")
    print(f"{'='*80}\n")

    top_5 = df_comparison.head(5)
    for idx, row in top_5.iterrows():
        model_name = row['model']
        auc = row['auc']
        fp_rate = row['fp_rate']
        category = row['category']

        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        fp_str = f"{fp_rate:.4f}" if fp_rate is not None else "N/A"

        print(f"  {idx+1}. {model_name:<35} (AUC={auc_str}, FP={fp_str}, {category})")

    print(f"\n{'='*80}")
    print(f"✓ COMPREHENSIVE EVALUATION COMPLETE")
    print(f"{'='*80}\n")

    print(f"Next Steps:")
    print(f"  1. Review comparison: cat {comparison_path}")
    print(f"  2. Review per-type: cat {per_type_path}")
    print(f"  3. Generate benchmark table: python exp_generate_benchmark_table.py\n")


if __name__ == "__main__":
    main()
