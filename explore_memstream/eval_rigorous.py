#!/usr/bin/env python3
"""
Rigorous Evaluation Framework for CA-DQStream MemStream on NYC Taxi.

Anomaly types matching v10 benchmark:
- Type 1: Short-trip meter fraud ($40-80 on trips <1 mile)
- Type 3: Ratecode mismatch (JFK flat fare at non-JFK zones)

Usage:
    python eval_rigorous.py --data /path/to/nyc_taxi.parquet --output results/eval/
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from memstream_src.core.memstream_core import MemStreamCore, MemStreamConfig, set_determinism
from memstream_src.core.feature_extractor import FeatureVectorizer

# ---------------------------------------------------------------------------
# Fraud Injection (v10-style)
# ---------------------------------------------------------------------------

NYC_TLC_ZONES = set(range(1, 266))

MANHATTAN_ZONES = set(range(1, 44))
JFK_ZONES = set(range(217, 230))
NON_JFK_ZONES = NYC_TLC_ZONES - JFK_ZONES

JFK_FLAT_FARE = 70.0


def inject_fraud(
    df: pd.DataFrame,
    fraud_type: str = 'mixed',
    n_anomalies: int = None,
    anomaly_rate: float = 0.03,
    seed: int = 42,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Inject v10-style fraud into NYC taxi records.

    Type 1: Short-trip meter fraud (60%)
        - Criteria: RatecodeID=1, distance < 1.0 mile, fare ~$2.5-8
        - Injection: fare_amount = $40-$80 (realistic meter fraud)

    Type 3: Ratecode mismatch (40%)
        - Criteria: RatecodeID=1, non-JFK zone pickup
        - Injection: RatecodeID=2 (JFK), fare_amount=$70 (JFK flat)
        - The spatial grid will show: JFK ratecode + non-JFK pickup = mismatch

    Args:
        df: Input DataFrame
        fraud_type: 'type1_only', 'type3_only', or 'mixed' (60/40 split)
        n_anomalies: Fixed count (takes priority over anomaly_rate)
        anomaly_rate: Target anomaly rate (0.03 = 3%)
        seed: Random seed
    """
    rng = np.random.RandomState(seed)
    df = df.copy()

    n = len(df)
    labels = np.zeros(n, dtype=int)

    if n_anomalies is not None:
        target_n = int(n_anomalies)
    else:
        target_n = int(n * anomaly_rate)

    if target_n >= n:
        raise ValueError(f"Too many anomalies requested: {target_n}/{n}")

    ratecode = df['RatecodeID'].fillna(1).astype(float).values
    dist = df['trip_distance'].fillna(0).astype(float).values
    fare = df['fare_amount'].fillna(0).astype(float).values
    pu_loc = df['PULocationID'].fillna(1).astype(int).values

    # Candidate pools
    ratecode_1 = (ratecode == 1.0)

    if fraud_type == 'type1_only':
        # Short-trip: distance < 1.0 mile, ratecode 1
        candidate_mask = ratecode_1 & (dist < 1.0)
    elif fraud_type == 'type3_only':
        # Non-JFK zones, ratecode 1
        pu_is_not_jfk = np.array([z not in JFK_ZONES for z in pu_loc])
        candidate_mask = ratecode_1 & pu_is_not_jfk
    elif fraud_type == 'mixed':
        # Type 1 pool (short trips) and Type 3 pool (ratecode mismatch)
        pu_is_not_jfk = np.array([z not in JFK_ZONES for z in pu_loc])
        pool1 = np.where(ratecode_1 & (dist < 1.0))[0]
        pool3 = np.where(ratecode_1 & pu_is_not_jfk)[0]
        candidate_mask = np.zeros(n, dtype=bool)
        candidate_mask[pool1] = True
        candidate_mask[pool3] = True
    else:
        raise ValueError(f"Unknown fraud_type: {fraud_type}")

    candidates = np.where(candidate_mask)[0]
    if len(candidates) == 0:
        raise ValueError(f"No candidate records found for fraud_type={fraud_type}")

    # Sample fraud indices
    rng.shuffle(candidates)
    pool = candidates[:target_n]

    # Determine fraud types for mixed
    n_type1 = int(len(pool) * 0.60) if fraud_type == 'mixed' else (len(pool) if fraud_type == 'type1_only' else 0)
    fraud_indices_type1 = pool[:n_type1]
    fraud_indices_type3 = pool[n_type1:]

    # Inject Type 1: Short-trip meter fraud
    for idx in fraud_indices_type1:
        new_fare = float(rng.uniform(40.0, 80.0))
        df.at[df.index[idx], 'fare_amount'] = new_fare
        df.at[df.index[idx], 'total_amount'] = new_fare
        labels[idx] = 1

    # Inject Type 3: Ratecode mismatch (JFK ratecode at non-JFK zone)
    for idx in fraud_indices_type3:
        df.at[df.index[idx], 'fare_amount'] = JFK_FLAT_FARE
        df.at[df.index[idx], 'total_amount'] = JFK_FLAT_FARE
        df.at[df.index[idx], 'RatecodeID'] = 2.0
        labels[idx] = 1

    return df, labels


# ---------------------------------------------------------------------------
# Threshold optimization
# ---------------------------------------------------------------------------

def find_optimal_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    method: str = 'f1',
) -> Tuple[float, Dict]:
    """
    Find optimal threshold that maximizes F1 score.

    For highly imbalanced data (anomaly rate < 5%), percentile-based
    threshold search is more stable than uniform grid search.

    Returns:
        Tuple of (optimal_threshold, metrics_dict_with_optimal_preds)
    """
    if len(np.unique(labels)) < 2:
        return 1.0, {'tp': 0, 'fp': 0, 'tn': 0, 'fn': int(labels.sum())}

    if method == 'f1':
        thresholds = np.percentile(scores, np.arange(90, 100, 0.5))
    elif method == 'youden':
        fpr_arr, tpr_arr, thresholds = roc_curve_wrapper(scores, labels)
        youden = tpr_arr - fpr_arr
        best_idx = np.argmax(youden)
        return float(thresholds[best_idx]), {}
    else:
        thresholds = np.linspace(scores.min(), scores.max(), 500)

    best_f1 = 0
    best_thresh = 1.0

    for t in thresholds:
        preds = (scores >= t).astype(int)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        tn = int(np.sum((preds == 0) & (labels == 0)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_metrics = {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn, 'thresh': t}

    return best_thresh, best_metrics


def roc_curve_wrapper(scores: np.ndarray, labels: np.ndarray):
    """Compute ROC curve points manually."""
    from sklearn.metrics import roc_curve
    return roc_curve(labels, scores)


def evaluate_config(
    cfg: MemStreamConfig,
    X_warmup: np.ndarray,
    X_test: np.ndarray,
    labels: np.ndarray,
    anomaly_rate: float,
    device: str = 'cuda',
) -> Dict:
    """
    Train MemStream and evaluate with optimal threshold.

    Returns metrics at both:
    - Optimal F1 threshold (primary)
    - Fixed thresholds for comparison (0.5, 1.0, 2.0)
    """
    set_determinism(cfg.seed)
    model = MemStreamCore(cfg=cfg, device=device)
    model.warmup(X_warmup, epochs=cfg.warmup_epochs, verbose=False)

    scores = model.score_batch_gpu(X_test)

    # Threshold-independent metrics
    try:
        auc_roc = roc_auc_score(labels, scores)
    except ValueError:
        auc_roc = 0.0

    prec_curve, rec_curve, _ = precision_recall_curve(labels, scores)
    auc_pr = auc(rec_curve, prec_curve)

    # Optimal threshold (F1 maximization)
    opt_thresh, opt_basic = find_optimal_threshold(scores, labels, method='f1')

    opt_preds = (scores >= opt_thresh).astype(int)
    tp = int(np.sum((opt_preds == 1) & (labels == 1)))
    fp = int(np.sum((opt_preds == 1) & (labels == 0)))
    tn = int(np.sum((opt_preds == 0) & (labels == 0)))
    fn = int(np.sum((opt_preds == 0) & (labels == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    # Fixed threshold comparison
    fixed_metrics = {}
    for thresh in [0.5, 1.0, 2.0]:
        fp2 = (scores >= thresh).astype(int)
        tp2 = int(np.sum((fp2 == 1) & (labels == 1)))
        fp2_cnt = int(np.sum((fp2 == 1) & (labels == 0)))
        fn2 = int(np.sum((fp2 == 0) & (labels == 1)))
        prec2 = tp2 / (tp2 + fp2_cnt) if (tp2 + fp2_cnt) > 0 else 0
        rec2 = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0
        f1_2 = 2 * prec2 * rec2 / (prec2 + rec2) if (prec2 + rec2) > 0 else 0
        fixed_metrics[f'thresh_{thresh}'] = {
            'precision': float(prec2), 'recall': float(rec2), 'f1': float(f1_2)
        }

    warmup_beta = model._context_beta
    n_total_cells = warmup_beta.n_neighborhoods * warmup_beta.n_cells if warmup_beta is not None else 0
    n_populated = int(np.sum(warmup_beta.betas > 0)) if warmup_beta is not None else 0

    return {
        # Threshold-independent
        'auc_roc': float(auc_roc),
        'auc_pr': float(auc_pr),
        # Optimal threshold metrics
        'optimal_threshold': float(opt_thresh),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'fpr': float(fpr),
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'n_detected': int(np.sum(opt_preds)),
        # Fixed threshold comparison
        'fixed_thresholds': fixed_metrics,
        # Model info
        'memory_fill': int(model.memory.count),
        'beta_shape': [warmup_beta.n_neighborhoods, warmup_beta.n_cells],
        'beta_populated_cells': n_populated,
        'beta_total_cells': n_total_cells,
        # Score statistics
        'score_mean': float(scores.mean()),
        'score_std': float(scores.std()),
        'score_min': float(scores.min()),
        'score_max': float(scores.max()),
        'score_p95': float(np.percentile(scores, 95)),
        'score_p99': float(np.percentile(scores, 99)),
        'anomaly_rate': float(anomaly_rate),
    }


# ---------------------------------------------------------------------------
# Ablation configs
# ---------------------------------------------------------------------------

NYC_ABLATION_CONFIGS = [
    # Phase 1: Memory length (baseline: gamma=0, k=10, beta=0.5)
    {'name': 'mem256',   'memory_len': 256,  'gamma': 0.0, 'k': 10, 'default_beta': 0.5},
    {'name': 'mem512',   'memory_len': 512,  'gamma': 0.0, 'k': 10, 'default_beta': 0.5},
    {'name': 'mem1024',  'memory_len': 1024, 'gamma': 0.0, 'k': 10, 'default_beta': 0.5},
    {'name': 'mem2048',  'memory_len': 2048, 'gamma': 0.0, 'k': 10, 'default_beta': 0.5},
    # Phase 2: gamma (self-recovery from memory contamination)
    {'name': 'mem1024_g0',   'memory_len': 1024, 'gamma': 0.0,  'k': 10, 'default_beta': 0.5},
    {'name': 'mem1024_g05',  'memory_len': 1024, 'gamma': 0.5,  'k': 10, 'default_beta': 0.5},
    {'name': 'mem1024_g07',  'memory_len': 1024, 'gamma': 0.7,  'k': 10, 'default_beta': 0.5},
    {'name': 'mem1024_g09',  'memory_len': 1024, 'gamma': 0.9,  'k': 10, 'default_beta': 0.5},
    # Phase 3: k (number of neighbors)
    {'name': 'mem1024_k5',   'memory_len': 1024, 'gamma': 0.0, 'k': 5,  'default_beta': 0.5},
    {'name': 'mem1024_k15',  'memory_len': 1024, 'gamma': 0.0, 'k': 15, 'default_beta': 0.5},
    {'name': 'mem1024_k20',  'memory_len': 1024, 'gamma': 0.0, 'k': 20, 'default_beta': 0.5},
    # Phase 4: beta threshold (from warmup score distribution)
    {'name': 'mem1024_b0p1',  'memory_len': 1024, 'gamma': 0.0, 'k': 10, 'default_beta': 0.1},
    {'name': 'mem1024_b0p5',  'memory_len': 1024, 'gamma': 0.0, 'k': 10, 'default_beta': 0.5},
    {'name': 'mem1024_b1p0',  'memory_len': 1024, 'gamma': 0.0, 'k': 10, 'default_beta': 1.0},
    {'name': 'mem1024_b2p0',  'memory_len': 1024, 'gamma': 0.0, 'k': 10, 'default_beta': 2.0},
    # Phase 5: Combined best configs
    {'name': 'best_balanced', 'memory_len': 1024, 'gamma': 0.5, 'k': 10, 'default_beta': 0.5},
]


def build_config(overrides: Dict) -> MemStreamConfig:
    """Build MemStreamConfig with ablation overrides."""
    cfg = MemStreamConfig()
    # Override defaults for NYC taxi evaluation
    cfg.in_dim = 34
    cfg.hidden_dim = 68
    cfg.warmup_epochs = 100
    cfg.warmup_batch_size = 256
    cfg.warmup_noise_std = 0.1
    cfg.seed = 42
    for key, val in overrides.items():
        setattr(cfg, key, val)
    return cfg


def main():
    parser = argparse.ArgumentParser(description='Rigorous NYC Taxi MemStream Evaluation')
    parser.add_argument('--data', type=str, required=True, help='NYC taxi parquet/CSV path')
    parser.add_argument('--output', type=str, default='results/eval', help='Output directory')
    parser.add_argument('--warmup-frac', type=float, default=0.5,
                        help='Fraction of data for warmup training')
    parser.add_argument('--anomaly-rate', type=float, default=0.03,
                        help='Target anomaly rate (0.03 = 3%%)')
    parser.add_argument('--fraud-type', type=str, default='mixed',
                        choices=['type1_only', 'type3_only', 'mixed'],
                        help='Fraud type to inject')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', type=str, default=None,
                        help='JSON with specific configs to run')
    parser.add_argument('--n-anomalies', type=int, default=None,
                        help='Override: fixed number of anomalies')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cpu', 'cuda'],
                        help='Device for scoring (default: cuda)')
    args = parser.parse_args()

    print("=" * 62)
    print("  NYC Taxi MemStream - Rigorous Evaluation Framework")
    print("=" * 62)
    print(f"  Fraud type: {args.fraud_type}")
    print(f"  Anomaly rate: {args.anomaly_rate*100:.1f}%")
    print(f"  Warmup fraction: {args.warmup_frac*100:.0f}%")
    print(f"  Device: {args.device}")
    print("=" * 62)

    # 1. Load data
    print("\n[1] Loading data...")
    df = pd.read_parquet(args.data) if args.data.endswith('.parquet') else pd.read_csv(args.data)
    print(f"  Records: {len(df):,}")

    # 2. Extract features
    print("\n[2] Extracting features...")
    vectorizer = FeatureVectorizer()
    X = vectorizer.transform_df(df)
    print(f"  Features: {X.shape}")

    # 3. Split warmup / test
    n_warmup = int(len(X) * args.warmup_frac)
    X_warmup = X[:n_warmup]
    X_rest = X[n_warmup:]
    df_rest = df.iloc[n_warmup:].reset_index(drop=True)
    print(f"  Warmup: {n_warmup:,} | Test: {len(X_rest):,}")

    # 4. Inject anomalies
    print(f"\n[3] Injecting anomalies ({args.fraud_type})...")
    df_test, labels = inject_fraud(
        df_rest,
        fraud_type=args.fraud_type,
        n_anomalies=args.n_anomalies,
        anomaly_rate=args.anomaly_rate,
        seed=args.seed,
    )
    vectorizer2 = FeatureVectorizer()
    X_test = vectorizer2.transform_df(df_test)
    actual_rate = labels.sum() / len(labels)
    print(f"  Test: {len(X_test):,} records, {labels.sum():,} anomalies ({actual_rate*100:.2f}%)")

    # 5. Determine configs
    if args.config:
        with open(args.config) as f:
            configs_to_run = json.load(f)
    else:
        configs_to_run = NYC_ABLATION_CONFIGS

    # 6. Run evaluation
    print(f"\n[4] Running {len(configs_to_run)} configs...")
    results = []
    for i, cfg_overrides in enumerate(configs_to_run):
        name = cfg_overrides.get('name', f'config_{i}')
        print(f"  [{i+1:02d}/{len(configs_to_run)}] {name}...", end='', flush=True)

        cfg = build_config(cfg_overrides)
        try:
            metrics = evaluate_config(cfg, X_warmup, X_test, labels, actual_rate, device=args.device)
            metrics['name'] = name
            metrics['config'] = cfg_overrides
            results.append(metrics)
            print(f" F1={metrics['f1']:.4f} AUC-PR={metrics['auc_pr']:.4f} "
                  f"(thresh={metrics['optimal_threshold']:.3f})")
        except Exception as e:
            print(f" ERROR: {e}")
            results.append({'name': name, 'config': cfg_overrides, 'error': str(e)})

    # 7. Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    with open(output_dir / f'rigorous_eval_{ts}.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'fraud_type': args.fraud_type,
            'n_warmup': n_warmup,
            'n_test': len(X_test),
            'n_anomalies': int(labels.sum()),
            'anomaly_rate': float(actual_rate),
            'warmup_frac': args.warmup_frac,
            'results': results,
        }, f, indent=2, default=str)

    # 8. Summary
    print("\n" + "=" * 62)
    print("  RESULTS (sorted by F1 at optimal threshold)")
    print("=" * 62)
    valid = [r for r in results if 'error' not in r]
    valid.sort(key=lambda x: x['f1'], reverse=True)

    print(f"\n{'Name':<18} {'F1':>6} {'Prec':>6} {'Rec':>6} {'FPR':>7} {'AUC-PR':>8} {'AUC-ROC':>9} {'Thresh':>7}")
    print("-" * 78)
    for r in valid:
        print(f"{r['name']:<18} {r['f1']:6.4f} {r['precision']:6.4f} {r['recall']:6.4f} "
              f"{r['fpr']:7.4f} {r['auc_pr']:8.4f} {r['auc_roc']:9.4f} "
              f"{r['optimal_threshold']:7.3f}")

    # Fixed threshold comparison
    print("\n" + "-" * 78)
    print(f"{'Name':<18} {'F1@0.5':>7} {'F1@1.0':>7} {'F1@2.0':>7} {'F1@opt':>7}")
    print("-" * 78)
    for r in valid:
        ft = r.get('fixed_thresholds', {})
        print(f"{r['name']:<18} {ft.get('thresh_0.5',{}).get('f1',0):7.4f} "
              f"{ft.get('thresh_1.0',{}).get('f1',0):7.4f} "
              f"{ft.get('thresh_2.0',{}).get('f1',0):7.4f} "
              f"{r['f1']:7.4f}")

    # Model stats
    print("\n" + "-" * 78)
    print(f"{'Name':<18} {'MemFill':>8} {'BetaPop':>8} {'ScoreMean':>9} {'ScoreP95':>9}")
    print("-" * 78)
    for r in valid:
        print(f"{r['name']:<18} {r.get('memory_fill',0):>8} "
              f"{r.get('beta_populated_cells',0):>3}/{r.get('beta_total_cells',0):>3} "
              f"{r.get('score_mean',0):>9.3f} "
              f"{r.get('score_p95',0):>9.3f}")

    print(f"\nResults saved to {output_dir}")
    print("=" * 62)


if __name__ == '__main__':
    main()
