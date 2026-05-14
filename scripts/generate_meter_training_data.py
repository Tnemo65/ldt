#!/usr/bin/env python3
"""
Generate METER Training Data
Task 3.4: Create synthetic drift scenarios for meta-learning

Generates meta-features + optimal strategy labels for METER training.

Meta-features (6D):
  - volume (events/min)
  - null_rate
  - violation_rate
  - anomaly_rate
  - avg_score
  - delta_score (change from baseline)

Strategy labels (4 classes):
  0: do_nothing
  1: adjust_threshold
  2: retrain_model
  3: switch_model

Usage:
  python scripts/generate_meter_training_data.py \
    --output data/meter_training_1000.parquet \
    --n-samples 1000
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class DriftScenario:
    """Drift scenario configuration."""
    name: str
    volume_range: tuple
    null_rate_range: tuple
    violation_rate_range: tuple
    anomaly_rate_range: tuple
    avg_score_range: tuple
    delta_score_range: tuple
    optimal_strategy: int  # 0=do_nothing, 1=adjust, 2=retrain, 3=switch


# Drift scenarios based on domain knowledge
DRIFT_SCENARIOS = [
    # Scenario 1: Normal operation → do_nothing
    DriftScenario(
        name='normal_operation',
        volume_range=(900, 1100),
        null_rate_range=(0.00, 0.02),
        violation_rate_range=(0.01, 0.05),
        anomaly_rate_range=(0.02, 0.06),
        avg_score_range=(0.35, 0.45),
        delta_score_range=(-0.05, 0.05),
        optimal_strategy=0  # do_nothing
    ),

    # Scenario 2: Slight threshold drift → adjust_threshold
    DriftScenario(
        name='threshold_drift',
        volume_range=(850, 1150),
        null_rate_range=(0.01, 0.03),
        violation_rate_range=(0.03, 0.07),
        anomaly_rate_range=(0.06, 0.12),  # Higher anomaly rate
        avg_score_range=(0.48, 0.58),     # Scores drifting up
        delta_score_range=(0.05, 0.15),   # Positive delta
        optimal_strategy=1  # adjust_threshold
    ),

    # Scenario 3: Significant data distribution shift → retrain_model
    DriftScenario(
        name='distribution_shift',
        volume_range=(700, 900),           # Volume drop
        null_rate_range=(0.03, 0.08),     # More nulls
        violation_rate_range=(0.08, 0.15), # More violations
        anomaly_rate_range=(0.15, 0.30),  # High anomaly rate
        avg_score_range=(0.55, 0.70),     # High scores
        delta_score_range=(0.15, 0.30),   # Large positive delta
        optimal_strategy=2  # retrain_model
    ),

    # Scenario 4: Concept drift (model invalid) → switch_model
    DriftScenario(
        name='concept_drift',
        volume_range=(600, 800),           # Significant volume drop
        null_rate_range=(0.05, 0.15),     # Many nulls
        violation_rate_range=(0.10, 0.25), # Many violations
        anomaly_rate_range=(0.25, 0.50),  # Very high anomaly rate
        avg_score_range=(0.65, 0.85),     # Very high scores
        delta_score_range=(0.25, 0.50),   # Very large delta
        optimal_strategy=3  # switch_model
    ),

    # Scenario 5: Transient spike → adjust_threshold (needs response even if temporary)
    DriftScenario(
        name='transient_spike',
        volume_range=(1100, 1300),         # Temporary volume spike
        null_rate_range=(0.00, 0.03),
        violation_rate_range=(0.02, 0.06),
        anomaly_rate_range=(0.08, 0.15),  # Moderate spike in anomaly rate
        avg_score_range=(0.45, 0.55),    # avg_score > 0.5 means model uncertain
        delta_score_range=(0.05, 0.12),
        optimal_strategy=1  # adjust_threshold — system should respond even to temporary spikes
    ),

    # Scenario 6: Gradual degradation → adjust_threshold
    DriftScenario(
        name='gradual_degradation',
        volume_range=(900, 1000),
        null_rate_range=(0.02, 0.05),
        violation_rate_range=(0.04, 0.09),
        anomaly_rate_range=(0.08, 0.14),
        avg_score_range=(0.48, 0.56),
        delta_score_range=(0.08, 0.18),
        optimal_strategy=1  # adjust_threshold
    ),

    # Scenario 7: Abrupt severe drift → switch_model
    # Extreme case: sudden change that makes model completely invalid
    DriftScenario(
        name='abrupt_severe_drift',
        volume_range=(400, 700),           # Sharp volume drop
        null_rate_range=(0.10, 0.25),    # Many nulls — data quality issue
        violation_rate_range=(0.20, 0.40), # Severe violations
        anomaly_rate_range=(0.40, 0.70),  # Very high anomaly rate
        avg_score_range=(0.80, 0.99),    # Scores near maximum — model overwhelmed
        delta_score_range=(0.40, 0.70),  # Very large delta
        optimal_strategy=3  # switch_model — model cannot recover
    ),
]


def generate_sample(scenario: DriftScenario) -> dict:
    """Generate one training sample from scenario.

    Returns:
        dict with meta-features (6D) and strategy label
    """
    # Generate base features
    violation_rate = np.random.uniform(*scenario.violation_rate_range)
    anomaly_rate = np.random.uniform(*scenario.anomaly_rate_range)
    avg_score = np.random.uniform(*scenario.avg_score_range)

    # Compute delta_score per thesis equation (5.18):
    # delta_score = |violation_rate - anomaly_rate| / (violation_rate + anomaly_rate + epsilon)
    epsilon = 1e-6
    delta_score = abs(violation_rate - anomaly_rate) / (violation_rate + anomaly_rate + epsilon)

    return {
        # Meta-features (6D)
        'volume': np.random.uniform(*scenario.volume_range),
        'null_rate': np.random.uniform(*scenario.null_rate_range),
        'violation_rate': violation_rate,
        'anomaly_rate': anomaly_rate,
        'avg_anomaly_score': avg_score,
        'delta_score': delta_score,  # Computed from violation_rate and anomaly_rate

        # Label
        'strategy': scenario.optimal_strategy,

        # Metadata (for debugging)
        'scenario_name': scenario.name
    }


def generate_meter_training_data(n_samples: int = 1000, output_path: Path = None):
    """Generate synthetic METER training data.

    Args:
        n_samples: Number of samples to generate
        output_path: Output parquet file path

    Returns:
        DataFrame with meta-features and strategy labels
    """
    print("="*80)
    print("GENERATING METER TRAINING DATA")
    print("="*80)

    print(f"\nTotal samples: {n_samples}")
    print(f"Scenarios: {len(DRIFT_SCENARIOS)}")

    # Generate samples (balanced across scenarios)
    samples_per_scenario = n_samples // len(DRIFT_SCENARIOS)
    remainder = n_samples % len(DRIFT_SCENARIOS)

    samples = []

    for scenario in DRIFT_SCENARIOS:
        # Base samples for this scenario
        scenario_samples = samples_per_scenario

        # Add remainder to first scenarios
        if remainder > 0:
            scenario_samples += 1
            remainder -= 1

        print(f"\n  {scenario.name:<30} {scenario_samples:>4} samples -> strategy={scenario.optimal_strategy}")

        for _ in range(scenario_samples):
            sample = generate_sample(scenario)
            samples.append(sample)

    # Create DataFrame
    df = pd.DataFrame(samples)

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Statistics
    print("\n" + "="*80)
    print("DATA STATISTICS")
    print("="*80)

    print("\nMeta-feature ranges:")
    for col in ['volume', 'null_rate', 'violation_rate', 'anomaly_rate', 'avg_anomaly_score', 'delta_score']:
        print(f"  {col:.<25} [{df[col].min():>8.4f}, {df[col].max():>8.4f}]")

    print("\nStrategy distribution:")
    strategy_names = ['do_nothing', 'adjust_threshold', 'retrain_model', 'switch_model']
    for strategy_id, name in enumerate(strategy_names):
        count = (df['strategy'] == strategy_id).sum()
        print(f"  {strategy_id}: {name:.<25} {count:>6} ({count/len(df)*100:>5.1f}%)")

    # Save as CSV (compatible with train_meter.py loading from data/)
    if output_path:
        print(f"\nSaving to: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path = output_path.with_suffix('.csv')
        df.to_csv(output_path, index=False)

        print("\n" + "="*80)
        print("[OK] METER TRAINING DATA GENERATED")
        print("="*80)
        print(f"Output: {output_path}")
        print(f"Samples: {len(df):,}")
        print(f"Features: 6D meta-features + 1 strategy label")
        print("="*80)

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Generate METER training data'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output parquet file path'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=1000,
        help='Number of samples to generate (default: 1000)'
    )

    args = parser.parse_args()

    output_path = Path(args.output)

    df = generate_meter_training_data(
        n_samples=args.n_samples,
        output_path=output_path
    )

    return 0


if __name__ == '__main__':
    exit(main())
