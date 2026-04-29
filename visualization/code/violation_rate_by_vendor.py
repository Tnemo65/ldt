"""
Figure 3: Violation Rate by VendorID
=====================================
Grouped bar chart comparing violation rates across vendors (V1, V2, V6)
over Jan 2024, Jul 2024, and Jan 2025.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from _theme import apply_theme, PALETTE
from _data_loader import load_month_sample

OUTPUT_DIR = Path('d:/final/visualization/figures')

VENDORS = [
    (1, 'V1 (CMT)', PALETTE[0]),
    (2, 'V2 (VTS)', PALETTE[1]),
    (6, 'V6 (Unknown)', PALETTE[2]),
]

MONTHS = [
    ('2024-01', 'Jan 2024'),
    ('2024-07', 'Jul 2024'),
    ('2025-01', 'Jan 2025'),
]


def compute_vendor_rates():
    data = {}
    for month, _ in MONTHS:
        df = load_month_sample(month=month, sample_size=300_000)
        df['v_null_passenger'] = df['passenger_count'].isna()
        df['v_zero_distance']  = df['trip_distance'] <= 0
        df['v_neg_fare']       = df['fare_amount'] < 0
        df['v_null_ratecode']  = df['RatecodeID'].isna()
        any_viol = (df['v_null_passenger'] | df['v_zero_distance'] |
                    df['v_neg_fare'] | df['v_null_ratecode'])

        rates = {}
        for vid, label, color in VENDORS:
            mask = df['VendorID'] == vid
            n = mask.sum()
            v = any_viol[mask].sum()
            rates[vid] = (v / n * 100) if n > 0 else 0
        data[month] = rates
    return data


def plot():
    rates = compute_vendor_rates()
    apply_theme()

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(MONTHS))
    width = 0.25

    for i, (vid, label, color) in enumerate(VENDORS):
        vals = [rates[m][vid] for m, _ in MONTHS]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                    f'{val:.2f}%', ha='center', fontsize=8.5, color='#c9d1d9')

    ax.set_xticks(x + width)
    ax.set_xticklabels([m[1] for m in MONTHS], fontsize=11)
    ax.set_ylabel('Violation Rate (%)', fontsize=11)
    ax.set_xlabel('Month', fontsize=11)
    ax.set_title('Violation Rate by VendorID', fontsize=14, fontweight='bold', pad=12)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim(0, max(
        rates[m][v[0]] for m, _ in MONTHS for v in VENDORS
    ) * 1.25)
    ax.grid(True, axis='y', alpha=0.4)

    fig.tight_layout()
    out = OUTPUT_DIR / '3_Violation_Rate_by_VendorID.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117', edgecolor='none')
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    plot()
