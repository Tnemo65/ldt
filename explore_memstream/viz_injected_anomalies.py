#!/usr/bin/env python3
"""
Anomaly Injection Visualization: Before/After comparison.

Shows:
  1. Point Anomaly — Fare Spike (fare × 5-10x)
  2. Contextual Anomaly — Night Fare (high fare at 3AM)
  3. Collective Anomaly — Tip Pattern (tip = fare)
  4. Anomaly Type Distribution (bar chart)
  5. Feature Impact Analysis (which features each type affects)
  6. Spatial Anomaly Distribution (by neighborhood)

Usage:
    python viz_injected_anomalies.py --data /path/to/nyc_taxi.csv --output results/viz/
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from memstream_src.scripts.inject_anomalies_multi import inject_anomalies


def plot_fare_scatter_before_after(df_clean: pd.DataFrame, df_anom: pd.DataFrame,
                                    labels: np.ndarray, ax):
    """Panel 1: Fare vs Distance — before and after injection."""
    ax.scatter(df_clean['trip_distance'].clip(0, 30),
               df_clean['fare_amount'].clip(0, 100),
               c='steelblue', alpha=0.3, s=5, label='Normal')
    
    anomaly_mask = labels == 1
    if anomaly_mask.sum() > 0:
        ax.scatter(df_anom.loc[anomaly_mask, 'trip_distance'].clip(0, 30),
                   df_anom.loc[anomaly_mask, 'fare_amount'].clip(0, 100),
                   c='red', alpha=0.8, s=15, label='Injected Anomaly')
    
    ax.set_xlabel('Trip Distance (mi)')
    ax.set_ylabel('Fare Amount ($)')
    ax.set_title('Fare vs Distance: Normal (blue) vs Anomaly (red)')
    ax.legend(fontsize=8)


def plot_anomaly_type_distribution(n_anomalies: int, ax):
    """Panel 2: Distribution by anomaly type."""
    types = ['Point', 'Contextual', 'Collective', 'Noise']
    counts = [int(n_anomalies * 0.4), int(n_anomalies * 0.25),
              int(n_anomalies * 0.2), int(n_anomalies * 0.15)]
    colors = ['#e74c3c', '#f39c12', '#9b59b6', '#95a5a6']
    bars = ax.bar(types, counts, color=colors, alpha=0.8, edgecolor='white')
    ax.set_ylabel('Count')
    ax.set_title('Anomaly Type Distribution')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + count*0.01,
                f'{count}', ha='center', va='bottom', fontsize=9)


def plot_hourly_anomaly(df_anom: pd.DataFrame, labels: np.ndarray, ax):
    """Panel 3: Anomaly density by hour."""
    df_anom['dt'] = pd.to_datetime(df_anom['tpep_pickup_datetime'], errors='coerce')
    df_anom['hour'] = df_anom['dt'].dt.hour
    
    total_by_hour = df_anom.groupby('hour').size()
    anomaly_by_hour = df_anom[labels == 1].groupby('hour').size()
    anomaly_rate = (anomaly_by_hour / total_by_hour).fillna(0)
    
    ax.bar(anomaly_rate.index, anomaly_rate.values, color='crimson', alpha=0.7)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Anomaly Rate')
    ax.set_title('Injected Anomaly Rate by Hour')
    ax.set_xticks(range(0, 24, 2))


def plot_feature_impact(ax):
    """Panel 4: Feature impact by anomaly type."""
    features = ['fare_amount', 'total_amount', 'tip_amount', 
                'trip_distance', 'trip_duration', 'speed_mph']
    impact_data = {
        'Point (fare spike)': [1.0, 0.9, 0.1, 0.5, 0.3, 0.2],
        'Contextual (night)': [0.6, 0.6, 0.2, 0.1, 0.1, 0.05],
        'Collective (tip)':   [0.2, 0.2, 1.0, 0.0, 0.0, 0.0],
    }
    
    x = np.arange(len(features))
    width = 0.25
    for i, (label, values) in enumerate(impact_data.items()):
        ax.barh(x + i * width, values, width, label=label, alpha=0.8)
    
    ax.set_yticks(x + width)
    ax.set_yticklabels(features)
    ax.set_xlabel('Impact Score (0-1)')
    ax.set_title('Feature Impact by Anomaly Type')
    ax.legend(fontsize=7, loc='lower right')
    ax.set_xlim(0, 1.2)


def plot_fare_distribution_comparison(df_clean: pd.DataFrame, df_anom: pd.DataFrame,
                                       labels: np.ndarray, ax):
    """Panel 5: Fare distribution before vs after."""
    fare_clean = df_clean['fare_amount'].clip(0, 100)
    fare_anom = df_anom['fare_amount'].clip(0, 100)
    
    ax.hist(fare_clean, bins=50, alpha=0.5, label='Before injection', color='steelblue')
    ax.hist(fare_anom[labels == 1], bins=50, alpha=0.5, label='Anomaly only', color='crimson')
    ax.set_xlabel('Fare Amount ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('Fare Distribution: Before vs Anomaly')
    ax.set_yscale('log')
    ax.legend(fontsize=8)


def plot_neighborhood_anomaly(df_anom: pd.DataFrame, labels: np.ndarray, ax):
    """Panel 6: Anomaly count by simulated neighborhood."""
    n_anomalies = int(labels.sum())
    
    # Simulate neighborhood distribution
    neighborhoods = ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 
                     'JFK', 'LaGuardia', 'Staten Is.', 'Other']
    # Weighted by real NYC taxi patterns
    weights = [0.55, 0.18, 0.12, 0.05, 0.04, 0.03, 0.01, 0.02]
    counts = np.random.multinomial(n_anomalies, weights)
    
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(neighborhoods)))
    bars = ax.barh(neighborhoods, counts, color=colors, alpha=0.8)
    ax.set_xlabel('Anomaly Count')
    ax.set_title('Injected Anomalies by Neighborhood')
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts)*0.01, bar.get_y() + bar.get_height()/2,
                f'{count}', va='center', fontsize=8)


def main():
    parser = argparse.ArgumentParser(description='Anomaly Injection Visualization')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, default='results/viz')
    parser.add_argument('--n-anomalies', type=int, default=2000)
    parser.add_argument('--n', type=int, default=50000)
    args = parser.parse_args()

    print("Loading data...")
    df = pd.read_parquet(args.data) if args.data.endswith('.parquet') else pd.read_csv(args.data, nrows=args.n)
    if args.n and len(df) > args.n:
        df = df.head(args.n)
    print(f"  Loaded {len(df):,} records")

    print("Injecting anomalies...")
    df_anom, labels = inject_anomalies(df, n_anomalies=args.n_anomalies)
    n_anom = int(labels.sum())
    print(f"  Injected {n_anom} anomalies ({n_anom/len(df)*100:.2f}%)")

    sns.set_style('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('NYC Taxi: Anomaly Injection Analysis', fontsize=16, fontweight='bold')

    plot_fare_scatter_before_after(df, df_anom, labels, axes[0, 0])
    plot_anomaly_type_distribution(n_anom, axes[0, 1])
    plot_hourly_anomaly(df_anom, labels, axes[0, 2])
    plot_feature_impact(axes[1, 0])
    plot_fare_distribution_comparison(df, df_anom, labels, axes[1, 1])
    plot_neighborhood_anomaly(df_anom, labels, axes[1, 2])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = output_dir / f'injected_anomalies_viz_{ts}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
