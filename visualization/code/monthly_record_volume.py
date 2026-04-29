"""
Figure 1: Monthly Record Volume
================================
Line chart showing total record count per month from Jan 2024 to Feb 2026.
Highlights Jan 2024 (training) and Jul 2024 (test) reference months.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from _theme import apply_theme, PALETTE
from _data_loader import compute_monthly_stats

OUTPUT_DIR = Path('d:/final/visualization/figures')


def plot():
    df = compute_monthly_stats()
    apply_theme()

    fig, ax = plt.subplots(figsize=(12, 5.5))

    x = range(len(df))
    y = df['total'].values

    ax.plot(x, y, color=PALETTE[0], linewidth=2.2, marker='o', markersize=5, zorder=3)
    ax.fill_between(x, y, alpha=0.12, color=PALETTE[0], zorder=2)

    mean_vol = y.mean()
    ax.axhline(mean_vol, color=PALETTE[1], linestyle='--', linewidth=1.5,
               label=f'Mean: {mean_vol:,.0f}', zorder=1)

    ax.set_xticks(x)
    labels = df['year_month_str'].tolist()
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)

    ax.set_ylabel('Total Records', fontsize=11)
    ax.set_xlabel('Month', fontsize=11)
    ax.set_title('Monthly Record Volume (Jan 2024 – Feb 2026)', fontsize=14, fontweight='bold', pad=12)

    # Annotate Jan 2024 and Jul 2024
    for i, label in enumerate(labels):
        if label == '2024-01':
            ax.axvline(i, color=PALETTE[2], linestyle=':', linewidth=1.5, alpha=0.9)
            ax.annotate('Training\n(Jan 2024)', xy=(i, y[i]),
                        xytext=(i + 0.8, y[i] + 120_000),
                        arrowprops=dict(arrowstyle='->', color=PALETTE[2]),
                        fontsize=9, color=PALETTE[2], ha='center')
        if label == '2024-07':
            ax.axvline(i, color=PALETTE[3], linestyle=':', linewidth=1.5, alpha=0.9)
            ax.annotate('Test Drift\n(Jul 2024)', xy=(i, y[i]),
                        xytext=(i + 0.8, y[i] - 120_000),
                        arrowprops=dict(arrowstyle='->', color=PALETTE[3]),
                        fontsize=9, color=PALETTE[3], ha='center')

    ax.legend(fontsize=10, loc='upper right')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    ax.grid(True, alpha=0.4)

    fig.tight_layout()
    out = OUTPUT_DIR / '1_Monthly_Record_Volume.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117', edgecolor='none')
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    plot()
