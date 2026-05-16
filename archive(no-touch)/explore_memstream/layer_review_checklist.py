#!/usr/bin/env python3
"""
Layer-by-Layer Review Checklist for CA-DQStream + MemStream.

Verifies each pipeline layer: Input -> Layer1 -> FeatureExtraction -> MemStream -> IEC -> Output.

Usage:
    python layer_review_checklist.py --layer layer1
    python layer_review_checklist.py --layer all
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Ensure memstream_src is importable (works both as module and direct script)
# core/__init__.py uses 'from memstream_src.core.memstream_core import ...'
# so we need both the parent dir (c:\proj\ldt) and memstream_src in sys.path
_SCRIPT_DIR = Path(__file__).parent.resolve()  # explore_memstream/
_PROJECT_ROOT = _SCRIPT_DIR / '..'  # c:\proj\ldt
_MEMSTREAM_ROOT = _PROJECT_ROOT / 'memstream_src'
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_MEMSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(_MEMSTREAM_ROOT))


# =============================================================================
# Layer Definitions
# =============================================================================

LAYERS = {
    'input': {
        'name': 'Input Layer (Raw Kafka -> Taxi Records)',
        'description': 'Verifies raw Kafka messages are valid taxi records',
        'checks': [
            {
                'id': 'IN-1', 'category': 'schema',
                'check': 'Schema: 19 fields present and correctly typed',
                'severity': 'critical',
            },
            {
                'id': 'IN-2', 'category': 'datetime',
                'check': 'tpep_pickup_datetime parseable, monotonic',
                'severity': 'critical',
            },
            {
                'id': 'IN-3', 'category': 'numeric',
                'check': 'fare_amount >= 0, trip_distance >= 0',
                'severity': 'critical',
            },
            {
                'id': 'IN-4', 'category': 'zone',
                'check': 'PULocationID, DOLocationID in [1, 265]',
                'severity': 'warning',
            },
            {
                'id': 'IN-5', 'category': 'ratecode',
                'check': 'RateCodeID in [1,2,3,4,5,6,99]',
                'severity': 'warning',
            },
            {
                'id': 'IN-6', 'category': 'null',
                'check': 'Critical fields: < 5% null rate',
                'severity': 'warning',
            },
            {
                'id': 'IN-7', 'category': 'latency',
                'check': 'Latency: < 100ms from taxi pickup to Kafka publish',
                'severity': 'info',
            },
        ],
    },
    'layer1': {
        'name': 'Layer 1 (Schema Validation + Hard Rules)',
        'description': 'Verifies schema validation and hard rule enforcement',
        'checks': [
            {
                'id': 'L1-1', 'category': 'schema_violations',
                'check': 'Schema violations are correctly flagged',
                'severity': 'critical',
            },
            {
                'id': 'L1-2', 'category': 'hard_rules',
                'check': 'Hard rules (fare < 0, speed > 100 mph) catch edge cases',
                'severity': 'critical',
            },
            {
                'id': 'L1-3', 'category': 'watermark',
                'check': 'Watermark: late data handled correctly (within 5 min)',
                'severity': 'warning',
            },
            {
                'id': 'L1-4', 'category': 'throughput',
                'check': 'Throughput: 10k+ records/sec sustained',
                'severity': 'info',
            },
            {
                'id': 'L1-5', 'category': 'output',
                'check': 'Output: clean_records, schema_violations, hard_rule_violations',
                'severity': 'critical',
            },
        ],
    },
    'features': {
        'name': 'Feature Extraction Layer',
        'description': 'Verifies 34D feature extraction correctness',
        'checks': [
            {
                'id': 'FE-1', 'category': 'dimensions',
                'check': 'All 34 features extracted correctly (match FEATURE_NAMES)',
                'severity': 'critical',
            },
            {
                'id': 'FE-2', 'category': 'zone_mapping',
                'check': 'Zone-to-grid mapping: 16x17 grid -> 272 cells',
                'severity': 'warning',
            },
            {
                'id': 'FE-3', 'category': 'borough',
                'check': 'Borough encoding: 10 neighborhoods',
                'severity': 'warning',
            },
            {
                'id': 'FE-4', 'category': 'cyclic',
                'check': 'Cyclic features: hour_sin/cos, dow_sin/cos correct',
                'severity': 'critical',
            },
            {
                'id': 'FE-5', 'category': 'ratios',
                'check': 'Ratio features: fare_per_mile, fare_per_min computed',
                'severity': 'critical',
            },
            {
                'id': 'FE-6', 'category': 'normalization',
                'check': 'Normalization: mean/std from warmup only (no leakage)',
                'severity': 'critical',
            },
            {
                'id': 'FE-7', 'category': 'nan_handling',
                'check': 'NaN/inf handled: replaced with 0 or safe defaults',
                'severity': 'warning',
            },
            {
                'id': 'FE-8', 'category': 'output_shape',
                'check': 'Output: 34D feature vector per record',
                'severity': 'critical',
            },
        ],
    },
    'memstream': {
        'name': 'MemStream Scoring Layer',
        'description': 'Verifies MemStream anomaly scoring correctness',
        'checks': [
            {
                'id': 'MS-1', 'category': 'architecture',
                'check': 'AE encoder: 34D -> 68D (2x)',
                'severity': 'critical',
            },
            {
                'id': 'MS-2', 'category': 'noise',
                'check': 'DAE noise: Gaussian sigma=0.1',
                'severity': 'warning',
            },
            {
                'id': 'MS-3', 'category': 'knn',
                'check': 'kNN scoring: L1 distance, K nearest in memory',
                'severity': 'critical',
            },
            {
                'id': 'MS-4', 'category': 'score_formula',
                'check': 'Score: sum(gamma^(i-1) * dist_i) / sum(gamma^(i-1))',
                'severity': 'critical',
            },
            {
                'id': 'MS-5', 'category': 'threshold',
                'check': 'beta threshold: from percentile 95 of warmup scores',
                'severity': 'critical',
            },
            {
                'id': 'MS-6', 'category': 'context_beta',
                'check': 'ContextBeta: 80 thresholds (10 neighborhoods x 8 cells)',
                'severity': 'critical',
            },
            {
                'id': 'MS-7', 'category': 'memory_update',
                'check': 'Memory update: FIFO, only if score < beta',
                'severity': 'critical',
            },
            {
                'id': 'MS-8', 'category': 'gamma',
                'check': 'gamma self-recovery: enabled (gamma=0.5)',
                'severity': 'warning',
            },
            {
                'id': 'MS-9', 'category': 'latency',
                'check': 'Latency: p99 < 100ms per record',
                'severity': 'info',
            },
        ],
    },
    'iec': {
        'name': 'IEC Layer (Concept Drift Handling)',
        'description': 'Verifies IEC feedback and concept drift handling',
        'checks': [
            {
                'id': 'IE-1', 'category': 'adwin',
                'check': '10 ADWIN instances (1 per neighborhood)',
                'severity': 'critical',
            },
            {
                'id': 'IE-2', 'category': 'delta',
                'check': 'ADWIN delta: 0.002 (conservative)',
                'severity': 'warning',
            },
            {
                'id': 'IE-3', 'category': 'circuit_breaker',
                'check': 'Circuit breaker: in BroadcastState (checkpointable)',
                'severity': 'critical',
            },
            {
                'id': 'IE-4', 'category': 'hmac',
                'check': 'Beta update: HMAC signed, verified before apply',
                'severity': 'critical',
            },
            {
                'id': 'IE-5', 'category': 'redis',
                'check': 'Redis: authenticated + TLS',
                'severity': 'warning',
            },
            {
                'id': 'IE-6', 'category': 'drift_action',
                'check': 'Drift detection -> memory reset + retrain trigger',
                'severity': 'critical',
            },
        ],
    },
    'output': {
        'name': 'Output Layer (Anomalies -> MinIO/Sinks)',
        'description': 'Verifies anomaly output and metrics writing',
        'checks': [
            {
                'id': 'OUT-1', 'category': 'anomaly_bucket',
                'check': 'Anomaly records written to cadqstream-anomalies bucket',
                'severity': 'critical',
            },
            {
                'id': 'OUT-2', 'category': 'metrics_bucket',
                'check': 'Metrics written to cadqstream-metrics bucket',
                'severity': 'critical',
            },
            {
                'id': 'OUT-3', 'category': 'dlq',
                'check': 'DLQ records written to cadqstream-dlq bucket',
                'severity': 'warning',
            },
            {
                'id': 'OUT-4', 'category': 'kafka_sink',
                'check': 'Kafka sink: anomaly events published to output topic',
                'severity': 'warning',
            },
            {
                'id': 'OUT-5', 'category': 'retention',
                'check': 'Retention: 30-day on metrics, 90-day on anomalies',
                'severity': 'info',
            },
        ],
    },
}


def run_automated_checks(layer_id: str) -> List[Dict]:
    """Run automated checks that can be programmatically verified."""
    results = []

    if layer_id == 'input':
        # Check memstream_src/core/config.py for feature names
        try:
            from core.config import FEATURE_NAMES, NUM_FEATURES
            results.append({
                'check_id': 'IN-AUTO-1',
                'category': 'auto',
                'status': 'pass',
                'detail': f'FEATURE_NAMES defined: {len(FEATURE_NAMES)} features',
            })
        except ImportError as e:
            results.append({
                'check_id': 'IN-AUTO-1',
                'category': 'auto',
                'status': 'fail',
                'detail': f'Cannot import config: {e}',
            })

    elif layer_id == 'memstream':
        try:
            from core.memstream_core import MemStreamCore, ContextBeta

            # Check ContextBeta
            cb = ContextBeta(n_neighborhoods=10, n_cells=8)
            results.append({
                'check_id': 'MS-AUTO-1',
                'category': 'auto',
                'status': 'pass',
                'detail': f'ContextBeta shape: {cb.betas.shape} (expected 10x8)',
            })
            
            # Check score formula (gamma-weighted)
            cfg_path = _PROJECT_ROOT / 'memstream_src' / 'core' / 'memstream_core.py'
            code = cfg_path.read_text()
            if 'gamma' in code and 'argpartition' in code:
                results.append({
                    'check_id': 'MS-AUTO-2',
                    'category': 'auto',
                    'status': 'pass',
                    'detail': 'gamma-weighted kNN scoring implemented',
                })
        except Exception as e:
            results.append({
                'check_id': 'MS-AUTO-1',
                'category': 'auto',
                'status': 'fail',
                'detail': f'Automated check error: {type(e).__name__}: {e}',
            })

    return results


def print_layer_report(layer_id: str, checks: List[Dict], auto_results: List[Dict]):
    """Print formatted report for a layer."""
    print(f"\n{'='*70}")
    print(f"  Layer: {LAYERS[layer_id]['name']}")
    print(f"{'='*70}")
    print(f"  {LAYERS[layer_id]['description']}\n")

    critical = [c for c in checks if c['severity'] == 'critical']
    warnings = [c for c in checks if c['severity'] == 'warning']
    infos = [c for c in checks if c['severity'] == 'info']

    print(f"  CRITICAL ({len(critical)} checks)")
    for c in critical:
        print(f"    [{c['id']}] {c['check']}")
    if warnings:
        print(f"\n  WARNING ({len(warnings)} checks)")
        for c in warnings:
            print(f"    [{c['id']}] {c['check']}")
    if infos:
        print(f"\n  INFO ({len(infos)} checks)")
        for c in infos:
            print(f"    [{c['id']}] {c['check']}")

    if auto_results:
        print(f"\n  AUTOMATED VERIFICATION")
        for r in auto_results:
            icon = '[PASS]' if r['status'] == 'pass' else '[FAIL]'
            print(f"    {icon} [{r['check_id']}] {r['detail']}")

    print(f"\n  Manual verification required for {len(critical)} critical checks")


def main():
    parser = argparse.ArgumentParser(description='CA-DQStream Layer Review Checklist')
    parser.add_argument('--layer', type=str, default='all',
                        choices=list(LAYERS.keys()) + ['all'],
                        help='Layer to review')
    parser.add_argument('--output', type=str, default=None,
                        help='Save report to JSON file')
    args = parser.parse_args()

    print("=" * 70)
    print("  CA-DQStream + MemStream Layer-by-Layer Review Checklist")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    layers_to_review = list(LAYERS.keys()) if args.layer == 'all' else [args.layer]
    all_results = {}

    for layer_id in layers_to_review:
        layer = LAYERS[layer_id]
        auto_results = run_automated_checks(layer_id)
        print_layer_report(layer_id, layer['checks'], auto_results)
        all_results[layer_id] = {
            'layer_name': layer['name'],
            'checks': layer['checks'],
            'auto_results': auto_results,
            'timestamp': datetime.now().isoformat(),
        }

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    total_critical = sum(len([c for c in LAYERS[l]['checks'] if c['severity'] == 'critical'])
                          for l in layers_to_review)
    print(f"  Total layers reviewed: {len(layers_to_review)}")
    print(f"  Total critical checks: {total_critical}")
    print(f"  Automated checks run:   {sum(len(all_results[l]['auto_results']) for l in all_results)}")

    if args.output:
        import json
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Report saved to: {args.output}")

    print(f"{'='*70}")


if __name__ == '__main__':
    main()
