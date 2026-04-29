"""
Figure 5: Distance vs Fare
===========================
Hexbin scatter plot (with scatter fallback) of trip_distance vs fare_amount,
colored by VendorID. Violations highlighted with red dots.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from _theme import apply_theme, PALETTE
from _data_loader import load_jan_sample

OUTPUT_DIR = Path('d:/final/visualization/figures')

VENDOR_COLORS = {1: PALETTE[0], 2: PALETTE[1], 6: PALETTE[2]}
VENDOR_NAMES  = {1: 'V1 (CMT)', 2: 'V2 (VTS)', 6: 'V6 (Unknown)'}


def plot():
    df = load_jan_sample(sample_size=150_000)
    apply_theme()

    df['v_null_passenger'] = df['passenger_count'].isna()
    df['v_zero_distance']  = df['trip_distance'] <= 0
    df['v_neg_fare']       = df['fare_amount'] < 0
    any_viol = df['v_null_passenger'] | df['v_zero_distance'] | df['v_neg_fare']

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot normal points per vendor using scatter
    for vid in [1, 2, 6]:
        mask = (df['VendorID'] == vid) & (~any_viol)
        sub = df[mask]
        ax.scatter(sub['trip_distance'], sub['fare_amount'],
                   alpha=0.18, s=4, c=VENDOR_COLORS[vid],
                   label=VENDOR_NAMES[vid], rasterized=True)

    # Highlight violations
    viol_mask = any_viol & df['VendorID'].isin([1, 2, 6])
    if viol_mask.sum() > 0:
        ax.scatter(df.loc[viol_mask, 'trip_distance'],
                   df.loc[viol_mask, 'fare_amount'],
                   alpha=0.5, s=12, c=PALETTE[2],
                   label='Violation', marker='x', rasterized=True)

    # Filter for regression
    valid = df[
        (df['trip_distance'] > 0) & (df['fare_amount'] > 0) &
        (df['trip_distance'] < 50) & (df['fare_amount'] < 200) &
        (~any_viol)
    ]
    if len(valid) > 1000:
        z = np.polyfit(valid['trip_distance'], valid['fare_amount'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, 50, 200)
        ax.plot(x_line, p(x_line), color=PALETTE[3], linewidth=2.5,
                linestyle='--', label=f'Linear fit (slope={z[0]:.2f})', zorder=5)

    ax.set_xlim(0, 50)
    ax.set_ylim(0, 200)
    ax.set_xlabel('Trip Distance (miles)', fontsize=12)
    ax.set_ylabel('Fare Amount ($)', fontsize=12)
    ax.set_title('Distance vs Fare by VendorID (Jan 2024)', fontsize=14, fontweight='bold', pad=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = OUTPUT_DIR / '5_Distance_vs_Fare.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117', edgecolor='none')
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    plot()
