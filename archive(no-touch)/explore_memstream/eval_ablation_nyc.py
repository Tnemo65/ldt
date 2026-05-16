#!/usr/bin/env python3
"""
Ablation Study: NYC Taxi-optimized MemStream hyperparameter sweep.

Sweeps: memory_len [128, 256, 512, 1024, 2048], gamma [0, 0.25, 0.5, 0.75, 1.0],
k [3, 5, 10, 20], beta [0.1, 0.5, 1.0, 5.0], hidden_dim [34, 68, 136], epochs [20, 50, 100, 500].

Usage:
    python eval_ablation_nyc.py --data /path/to/nyc_taxi.csv --output results/ablation/
"""

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score

# Add project root so 'from core.X import ...' and 'from scripts.X import ...' work
sys.path.insert(0, str(Path(__file__).parent.parent))
from memstream_src.core.memstream_core import MemStreamCore, MemStreamConfig, set_determinism
from memstream_src.core.feature_extractor import FeatureVectorizer
from memstream_src.scripts.inject_anomalies_multi import inject_anomalies


ABLATION_CONFIGS = [
    # Memory length sweep (primary)
    {'name': 'mem128',  'memory_len': 128},
    {'name': 'mem256',  'memory_len': 256},
    {'name': 'mem512',  'memory_len': 512},
    {'name': 'mem1024', 'memory_len': 1024},
    {'name': 'mem2048', 'memory_len': 2048},
    # gamma sweep (self-recovery)
    {'name': 'gamma_0',   'gamma': 0.0},
    {'name': 'gamma_025', 'gamma': 0.25},
    {'name': 'gamma_05',  'gamma': 0.5},
    {'name': 'gamma_1',   'gamma': 1.0},
    # k sweep
    {'name': 'k3',  'k': 3},
    {'name': 'k5',  'k': 5},
    {'name': 'k10', 'k': 10},
    {'name': 'k20', 'k': 20},
    # beta sweep
    {'name': 'beta_01', 'default_beta': 0.1},
    {'name': 'beta_05', 'default_beta': 0.5},
    {'name': 'beta_1',  'default_beta': 1.0},
    {'name': 'beta_5',  'default_beta': 5.0},
    # hidden_dim sweep
    {'name': 'hid34',  'hidden_dim': 34,  'latent_dim': 34},
    {'name': 'hid68',  'hidden_dim': 68,  'latent_dim': 60},
    {'name': 'hid136', 'hidden_dim': 136, 'latent_dim': 120},
    # epoch sweep
    {'name': 'ep20',  'warmup_epochs': 20},
    {'name': 'ep50',  'warmup_epochs': 50},
    {'name': 'ep100', 'warmup_epochs': 100},
    {'name': 'ep500', 'warmup_epochs': 500},
]


def build_config(overrides: Dict) -> MemStreamConfig:
    """Build MemStreamConfig with ablation overrides."""
    cfg = MemStreamConfig()
    for key, val in overrides.items():
        setattr(cfg, key, val)
    return cfg


def evaluate_config(cfg: MemStreamConfig, X_warmup: np.ndarray, X_test: np.ndarray,
                    labels_test: np.ndarray) -> Dict:
    """Train and evaluate one ablation config."""
    set_determinism(cfg.seed)
    model = MemStreamCore(cfg=cfg, device='cpu')
    model.warmup(X_warmup, epochs=cfg.warmup_epochs, verbose=False)

    scores = model.score_batch(X_test)
    preds = (scores >= 1.0).astype(int)

    tp = int(np.sum((preds == 1) & (labels_test == 1)))
    fp = int(np.sum((preds == 1) & (labels_test == 0)))
    tn = int(np.sum((preds == 0) & (labels_test == 0)))
    fn = int(np.sum((preds == 0) & (labels_test == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    try:
        auc_roc = roc_auc_score(labels_test, scores)
    except ValueError:
        auc_roc = 0.0

    prec_curve, rec_curve, _ = precision_recall_curve(labels_test, scores)
    auc_pr = auc(rec_curve, prec_curve)

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'fpr': float(fpr),
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'n_detected': int(np.sum(preds)),
    }


def main():
    parser = argparse.ArgumentParser(description='NYC Taxi MemStream Ablation Study')
    parser.add_argument('--data', type=str, required=True, help='NYC taxi CSV path')
    parser.add_argument('--output', type=str, default='results/ablation',
                        help='Output directory')
    parser.add_argument('--n-anomalies', type=int, default=2000,
                        help='Number of anomalies to inject')
    parser.add_argument('--warmup-frac', type=float, default=0.6,
                        help='Fraction of data for warmup')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', type=str, default=None,
                        help='JSON with specific configs to run')
    args = parser.parse_args()

    print("=" * 60)
    print("NYC Taxi MemStream Ablation Study")
    print("=" * 60)

    # Load and prepare data
    print("\n[1] Loading data...")
    df = pd.read_parquet(args.data) if args.data.endswith('.parquet') else pd.read_csv(args.data)
    print(f"  Records: {len(df):,}")

    # Extract features (use transform_batch for efficiency)
    print("\n[2] Extracting features...")
    vectorizer = FeatureVectorizer()
    X = vectorizer.transform_batch(df.to_dict('records'))
    print(f"  Features: {X.shape}")

    # Split
    n_warmup = int(len(X) * args.warmup_frac)
    X_warmup = X[:n_warmup]
    X_rest = X[n_warmup:]
    df_rest = df.iloc[n_warmup:].reset_index(drop=True)

    # Inject anomalies into test set
    print(f"\n[3] Injecting {args.n_anomalies} anomalies...")
    df_test, labels = inject_anomalies(df_rest, n_anomalies=args.n_anomalies, seed=args.seed)
    vectorizer2 = FeatureVectorizer()
    X_test = vectorizer2.transform_batch(df_test.to_dict('records'))
    print(f"  Test: {len(X_test):,} records, {int(labels.sum())} anomalies")

    # Determine configs to run
    if args.config:
        import json
        with open(args.config) as f:
            configs_to_run = json.load(f)
    else:
        configs_to_run = ABLATION_CONFIGS

    # Run ablation
    print(f"\n[4] Running {len(configs_to_run)} ablation configs...")
    results = []
    for i, cfg_overrides in enumerate(configs_to_run):
        name = cfg_overrides.get('name', f'config_{i}')
        print(f"  [{i+1}/{len(configs_to_run)}] {name}...", end='', flush=True)

        cfg = build_config(cfg_overrides)
        try:
            metrics = evaluate_config(cfg, X_warmup, X_test, labels)
            metrics['name'] = name
            metrics['config'] = cfg_overrides
            results.append(metrics)
            print(f" F1={metrics['f1']:.4f} AUC-PR={metrics['auc_pr']:.4f}")
        except Exception as e:
            print(f" ERROR: {e}")
            results.append({'name': name, 'config': cfg_overrides, 'error': str(e)})

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    with open(output_dir / f'ablation_results_{ts}.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'n_warmup': n_warmup,
            'n_test': len(X_test),
            'n_anomalies': int(labels.sum()),
            'results': results,
        }, f, indent=2, default=str)

    # Summary table
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY (sorted by F1)")
    print("=" * 60)
    valid = [r for r in results if 'error' not in r]
    valid.sort(key=lambda x: x['f1'], reverse=True)
    print(f"{'Name':<15} {'F1':>6} {'Recall':>7} {'FPR':>7} {'AUC-PR':>8} {'AUC-ROC':>9}")
    print("-" * 60)
    for r in valid:
        print(f"{r['name']:<15} {r['f1']:6.4f} {r['recall']:7.4f} "
              f"{r['fpr']:7.4f} {r['auc_pr']:8.4f} {r['auc_roc']:9.4f}")
    print(f"\nResults saved to {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
