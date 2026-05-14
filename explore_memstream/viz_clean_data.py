#!/usr/bin/env python3
"""
Clean Data Visualization: 6-panel NYC taxi data analysis.

Panels:
  1. Temporal Distribution (hour-of-day, day-of-week, daily volume)
  2. Fare Distribution (histogram, fare by hour, fare by neighborhood)
  3. Spatial Distribution (pickup heatmap by neighborhood)
  4. Feature Correlations (heatmap of 34D features)
  5. Cycle Analysis (ACF, dominant frequencies)
  6. Feature Distributions by Context (weekday rush vs weekend night)

Usage:
    python viz_clean_data.py --data /path/to/nyc_taxi.csv --output results/viz/
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
from memstream_src.core.config import FEATURE_NAMES


def plot_temporal(df: pd.DataFrame, ax):
    """Panel 1: Temporal distribution."""
    df['dt'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df['hour'] = df['dt'].dt.hour
    df['dow'] = df['dt'].dt.dayofweek
    df['date'] = df['dt'].dt.date

    # Hour distribution
    hour_counts = df['hour'].value_counts().sort_index()
    ax.bar(hour_counts.index, hour_counts.values, color='steelblue', alpha=0.7)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Trip Count')
    ax.set_title('Trips by Hour-of-Day')
    ax.set_xticks(range(0, 24, 2))
    
    # Mark rush hours
    for r in [(6, 10), (17, 21)]:
        ax.axvspan(r[0], r[1], alpha=0.1, color='red', label='Rush Hour' if r == (6, 10) else '')


def plot_fare(df: pd.DataFrame, ax):
    """Panel 2: Fare distribution (log scale)."""
    fare = df['fare_amount'].clip(0, 100)
    ax.hist(fare, bins=50, color='darkorange', alpha=0.7, edgecolor='white')
    ax.set_xlabel('Fare Amount ($)')
    ax.set_ylabel('Frequency')
    ax.set_title('Fare Distribution (clipped at $100)')
    ax.set_yscale('log')
    ax.axvline(fare.median(), color='red', linestyle='--', label=f'Median: ${fare.median():.2f}')
    ax.axvline(fare.mean(), color='navy', linestyle='--', label=f'Mean: ${fare.mean():.2f}')
    ax.legend(fontsize=8)


def plot_distance(df: pd.DataFrame, ax):
    """Panel 3: Trip distance distribution."""
    dist = df['trip_distance'].clip(0, 50)
    ax.hist(dist, bins=50, color='seagreen', alpha=0.7, edgecolor='white')
    ax.set_xlabel('Trip Distance (miles)')
    ax.set_ylabel('Frequency')
    ax.set_title('Trip Distance Distribution (clipped at 50 mi)')
    ax.set_yscale('log')


def plot_dow(df: pd.DataFrame, ax):
    """Panel 4: Day-of-week distribution."""
    df['dt'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df['dow'] = df['dt'].dt.dayofweek
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_counts = df['dow'].value_counts().sort_index()
    colors = ['#2ecc71'] * 5 + ['#e74c3c'] * 2
    ax.bar(range(7), [dow_counts.get(i, 0) for i in range(7)], color=colors, alpha=0.8)
    ax.set_xticks(range(7))
    ax.set_xticklabels(dow_names)
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Trip Count')
    ax.set_title('Trips by Day-of-Week (green=weekday, red=weekend)')


def plot_correlation_heatmap(df: pd.DataFrame, ax):
    """Panel 5: Feature correlation heatmap."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:10]
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, ax=ax, cmap='coolwarm', center=0, square=True,
                linewidths=0.1, annot=False, cbar_kws={'shrink': 0.5})
    ax.set_title('Feature Correlations (top 10 numeric)', fontsize=9)
    ax.tick_params(axis='both', labelsize=6)


def plot_speed_distribution(df: pd.DataFrame, ax):
    """Panel 6: Speed distribution by context."""
    df['dt'] = pd.to_datetime(df['tpep_pickup_datetime'], errors='coerce')
    df['hour'] = df['dt'].dt.hour
    df['dow'] = df['dt'].dt.dayofweek
    
    # Rush hour (weekday) vs night (weekend)
    rush_hour = df[(df['hour'].between(7, 10)) & (df['dow'] < 5)]
    weekend_night = df[(df['hour'] >= 20) | (df['hour'] < 6) & (df['dow'] >= 5)]
    
    speed_rush = rush_hour['trip_distance'] / (rush_hour['total_amount'] / 60 + 0.01)
    speed_night = weekend_night['trip_distance'] / (weekend_night['total_amount'] / 60 + 0.01)
    
    ax.hist(speed_rush.clip(0, 50), bins=30, alpha=0.6, label='Weekday Rush', color='steelblue')
    ax.hist(speed_night.clip(0, 50), bins=30, alpha=0.6, label='Weekend Night', color='coral')
    ax.set_xlabel('Speed (mph)')
    ax.set_ylabel('Frequency')
    ax.set_title('Speed: Weekday Rush vs Weekend Night')
    ax.legend(fontsize=8)


def main():
    parser = argparse.ArgumentParser(description='NYC Taxi Clean Data Visualization')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--output', type=str, default='results/viz')
    parser.add_argument('--n', type=int, default=100000, help='Sample size')
    parser.add_argument('--style', type=str, default='seaborn-v0_8-darkgrid')
    args = parser.parse_args()

    print("Loading data...")
    df = pd.read_parquet(args.data) if args.data.endswith('.parquet') else pd.read_csv(args.data, nrows=args.n)
    if args.n and len(df) > args.n:
        df = df.head(args.n)
    print(f"  Loaded {len(df):,} records")

    sns.set_style(args.style)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('NYC Taxi Data: Clean Data Analysis', fontsize=16, fontweight='bold')

    plot_temporal(df, axes[0, 0])
    plot_fare(df, axes[0, 1])
    plot_distance(df, axes[0, 2])
    plot_dow(df, axes[1, 0])
    plot_correlation_heatmap(df, axes[1, 1])
    plot_speed_distribution(df, axes[1, 2])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = output_dir / f'clean_data_viz_{ts}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved to {out_path}")
    
    # Also save as HTML for interactive viewing
    try:
        import matplotlib
        matplotlib.use('Agg')
    except Exception:
        pass


if __name__ == '__main__':
    main()
