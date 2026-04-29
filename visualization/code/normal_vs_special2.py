"""
Figure T7: Normal vs Special — Time Bin Comparison
===================================================
Grouped bar chart comparing standard vs special ratecode groups
across the 4 time bins (night, morning, afternoon, evening),
showing record volume and average fare.
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from _theme import apply_theme, PALETTE
from _data_loader import load_jan_sample

OUTPUT_DIR = Path('d:/final/visualization/figures')

TIME_BINS = ['night', 'morning', 'afternoon', 'evening']
TIME_LABELS = ['Night\n(0–5h)', 'Morning\n(6–11h)', 'Afternoon\n(12–17h)', 'Evening\n(18–23h)']


def plot():
    df = load_jan_sample(sample_size=300_000)
    apply_theme()

    std = df[df['RatecodeID'] == 1]
    spc = df[df['RatecodeID'] != 1]

    std_counts = [std[std['time_bin'] == tb].shape[0] for tb in TIME_BINS]
    spc_counts = [spc[spc['time_bin'] == tb].shape[0] for tb in TIME_BINS]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9),
                                    gridspec_kw={'height_ratios': [1.2, 1]})

    x = np.arange(len(TIME_BINS))
    width = 0.35

    # ── Top: record count by time bin ────────────────────────────────────
    bars_std = ax1.bar(x - width/2, std_counts, width, label='Standard',
                       color=PALETTE[0], alpha=0.85)
    bars_spc = ax1.bar(x + width/2, spc_counts, width, label='Special',
                       color=PALETTE[2], alpha=0.85)

    for bar in bars_std:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                 f'{bar.get_height():,}', ha='center', fontsize=8, color='#c9d1d9')
    for bar in bars_spc:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                 f'{bar.get_height():,}', ha='center', fontsize=8, color='#c9d1d9')

    ax1.set_xticks(x)
    ax1.set_xticklabels(TIME_LABELS, fontsize=10)
    ax1.set_ylabel('Record Count', fontsize=11)
    ax1.set_title('Standard vs Special: Volume by Time Bin (Jan 2024)',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, axis='y', alpha=0.4)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:,.0f}'))

    # ── Bottom: average fare by time bin ─────────────────────────────────
    std_fares = [std[std['time_bin'] == tb]['fare_amount'].mean() for tb in TIME_BINS]
    spc_fares = [spc[spc['time_bin'] == tb]['fare_amount'].mean() for tb in TIME_BINS]

    bars_f_std = ax2.bar(x - width/2, std_fares, width, label='Standard',
                         color=PALETTE[0], alpha=0.85)
    bars_f_spc = ax2.bar(x + width/2, spc_fares, width, label='Special',
                         color=PALETTE[2], alpha=0.85)

    for bar, val in zip(bars_f_std, std_fares):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'${val:.1f}', ha='center', fontsize=8, color='#c9d1d9')
    for bar, val in zip(bars_f_spc, spc_fares):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'${val:.1f}', ha='center', fontsize=8, color='#c9d1d9')

    ax2.set_xticks(x)
    ax2.set_xticklabels(TIME_LABELS, fontsize=10)
    ax2.set_ylabel('Average Fare ($)', fontsize=11)
    ax2.set_xlabel('Time Bin', fontsize=11)
    ax2.set_title('Standard vs Special: Avg Fare by Time Bin',
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, axis='y', alpha=0.4)

    fig.tight_layout()
    out = OUTPUT_DIR / 'T7_normal_vs_special_copy.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117', edgecolor='none')
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    plot()
