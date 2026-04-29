"""
Figure 2: Violation Breakdown
===============================
Horizontal bar chart of violation counts by rule for Jan 2024.
Bars are colored by severity (orange for frequent, blue for moderate).
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from _theme import apply_theme, PALETTE
from _data_loader import load_jan_sample

OUTPUT_DIR = Path('d:/final/visualization/figures')


def plot():
    df = load_jan_sample(sample_size=300_000)
    apply_theme()

    df['v_null_passenger'] = df['passenger_count'].isna()
    df['v_zero_distance']  = df['trip_distance'] <= 0
    df['v_neg_fare']       = df['fare_amount'] < 0
    df['v_null_ratecode']  = df['RatecodeID'].isna()
    df['v_null_pu']        = df['PULocationID'].isna()
    df['v_null_do']        = df['DOLocationID'].isna()

    df['v_dropoff_before_pickup'] = (
        df['tpep_dropoff_datetime'] < df['tpep_pickup_datetime']
    )

    dur_min = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
    speed = np.where(dur_min > 0, df['trip_distance'] / (dur_min / 60), 0)
    df['v_speed_high'] = speed > 100

    VIOLATIONS = {
        'v_null_passenger':        'NULL Passenger',
        'v_zero_distance':         'Zero Distance',
        'v_neg_fare':              'Negative Fare',
        'v_null_ratecode':         'NULL RatecodeID',
        'v_null_pu':              'NULL PU Location',
        'v_null_do':              'NULL DO Location',
        'v_dropoff_before_pickup': 'Dropoff < Pickup',
        'v_speed_high':            'Speed > 100 mph',
    }

    counts = {k: int(df[v].sum()) for k, v in VIOLATIONS.items()}
    sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    apply_theme()
    fig, ax = plt.subplots(figsize=(12, 6))

    labels_short = list(sorted_counts.keys())
    values = list(sorted_counts.values())

    bar_colors = [
        PALETTE[2] if v > 10_000 else PALETTE[0]
        for v in values
    ]

    bars = ax.barh(range(len(labels_short)), values, color=bar_colors, alpha=0.85, height=0.55)

    ax.set_yticks(range(len(labels_short)))
    ax.set_yticklabels(labels_short, fontsize=10)
    ax.invert_yaxis()

    ax.set_xlabel('Count', fontsize=11)
    ax.set_title('Violation Breakdown by Rule (Jan 2024)', fontsize=14, fontweight='bold', pad=12)

    for bar, val in zip(bars, values):
        ax.text(val + 200, bar.get_y() + bar.get_height() / 2,
                f'{val:,}', va='center', fontsize=9.5, color='#c9d1d9')

    ax.set_xlim(0, max(values) * 1.18)
    ax.grid(True, axis='x', alpha=0.4)

    fig.tight_layout()
    out = OUTPUT_DIR / '2_Violation_Breakdown.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117', edgecolor='none')
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    plot()
