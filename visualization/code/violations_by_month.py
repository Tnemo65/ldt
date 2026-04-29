"""
Figure T5: Violations by Month
===============================
Stacked area chart showing cumulative violation counts by type across all months.
Uses a diverging / grouped approach to show the relative contribution
of each violation type over time.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from _theme import apply_theme, PALETTE
from _data_loader import compute_violation_by_month

OUTPUT_DIR = Path('d:/final/visualization/figures')

VIOL_LABELS = [
    'NULL Passenger',
    'Zero Distance',
    'Negative Fare',
    'NULL Ratecode',
    'NULL PU Location',
    'NULL DO Location',
    'Dropoff < Pickup',
    'Speed > 100 mph',
]


def plot():
    df = compute_violation_by_month()
    apply_theme()

    cols = [
        'v_null_passenger', 'v_zero_distance', 'v_neg_fare',
        'v_null_ratecode', 'v_null_pu', 'v_null_do',
        'v_dropoff_before_pickup', 'v_speed_high',
    ]

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(13, 9),
                                          gridspec_kw={'height_ratios': [2, 1]},
                                          sharex=True)

    x = range(len(df))
    bottom = np.zeros(len(df))

    colors = PALETTE[:len(cols)]

    for col, label, color in zip(cols, VIOL_LABELS, colors):
        y = df[col].values.astype(float)
        ax_top.fill_between(x, bottom, bottom + y, label=label, color=color, alpha=0.8)
        bottom = bottom + y

    ax_top.set_ylabel('Cumulative Violation Count', fontsize=11)
    ax_top.set_title('Violation Counts Over Time (Jan 2024 – Feb 2026)',
                     fontsize=14, fontweight='bold', pad=10)
    ax_top.legend(loc='upper left', fontsize=8.5, ncol=2,
                  framealpha=0.9, bbox_to_anchor=(0.0, 1.0))
    ax_top.grid(True, alpha=0.3)

    # Bottom: violation rate trend line
    ax_bot.plot(x, df['v_null_passenger'].values, color=PALETTE[0],
                linewidth=2, marker='o', markersize=3, label='NULL Passenger')
    ax_bot.plot(x, df['v_zero_distance'].values, color=PALETTE[1],
                linewidth=2, marker='s', markersize=3, label='Zero Distance')
    ax_bot.plot(x, df['v_neg_fare'].values, color=PALETTE[2],
                linewidth=2, marker='^', markersize=3, label='Negative Fare')

    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(df['year_month'].tolist(), rotation=45, ha='right', fontsize=8)
    ax_bot.set_ylabel('Count', fontsize=11)
    ax_bot.set_xlabel('Month', fontsize=11)
    ax_bot.set_title('Key Violation Types Over Time', fontsize=12, pad=8)
    ax_bot.legend(fontsize=8.5, ncol=3)
    ax_bot.grid(True, alpha=0.3)

    fig.tight_layout()
    out = OUTPUT_DIR / 'T5_violations_by_month.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117', edgecolor='none')
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    plot()
