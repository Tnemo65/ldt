"""
Figure F3: Normal vs Special Context — Fare & Distance Distributions
======================================================================
Side-by-side boxplots comparing standard vs special ratecode groups
across fare_amount and trip_distance distributions.
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

    std_mask = df['RatecodeID'] == 1
    spc_mask = df['RatecodeID'] != 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Normal (Standard) vs Special Ratecode: Jan 2024',
                 fontsize=14, fontweight='bold', y=1.01)

    # ── Left: fare_amount ──────────────────────────────────────────────
    ax = axes[0]
    fare_std = df[std_mask]['fare_amount'].clip(0, 200).dropna()
    fare_spc = df[spc_mask]['fare_amount'].clip(0, 200).dropna()

    bp1 = ax.boxplot(
        [fare_std, fare_spc],
        labels=['Standard\n(RatecodeID=1)', 'Special\n(RatecodeID≠1)'],
        patch_artist=True, widths=0.5, notch=False,
    )
    bp1['boxes'][0].set_facecolor(PALETTE[0])
    bp1['boxes'][0].set_alpha(0.7)
    bp1['boxes'][1].set_facecolor(PALETTE[2])
    bp1['boxes'][1].set_alpha(0.7)

    ax.set_ylabel('Fare Amount ($)', fontsize=11)
    ax.set_title('Fare Distribution: Standard vs Special', fontsize=12)
    ax.grid(True, axis='y', alpha=0.4)
    ax.set_ylim(0, 200)

    # Annotate means
    ax.axhline(fare_std.mean(), color=PALETTE[0], linestyle='--', linewidth=1.2,
               xmin=0.05, xmax=0.45, alpha=0.7)
    ax.axhline(fare_spc.mean(), color=PALETTE[2], linestyle='--', linewidth=1.2,
               xmin=0.55, xmax=0.95, alpha=0.7)
    ax.text(1, fare_std.mean() + 4, f'μ={fare_std.mean():.1f}',
            ha='center', fontsize=9, color=PALETTE[0])
    ax.text(2, fare_spc.mean() + 4, f'μ={fare_spc.mean():.1f}',
            ha='center', fontsize=9, color=PALETTE[2])

    # ── Right: trip_distance ────────────────────────────────────────────
    ax = axes[1]
    dist_std = df[std_mask]['trip_distance'].clip(0, 30).dropna()
    dist_spc = df[spc_mask]['trip_distance'].clip(0, 30).dropna()

    bp2 = ax.boxplot(
        [dist_std, dist_spc],
        labels=['Standard\n(RatecodeID=1)', 'Special\n(RatecodeID≠1)'],
        patch_artist=True, widths=0.5, notch=False,
    )
    bp2['boxes'][0].set_facecolor(PALETTE[0])
    bp2['boxes'][0].set_alpha(0.7)
    bp2['boxes'][1].set_facecolor(PALETTE[2])
    bp2['boxes'][1].set_alpha(0.7)

    ax.set_ylabel('Trip Distance (miles)', fontsize=11)
    ax.set_title('Distance Distribution: Standard vs Special', fontsize=12)
    ax.grid(True, axis='y', alpha=0.4)
    ax.set_ylim(0, 30)

    ax.axhline(dist_std.mean(), color=PALETTE[0], linestyle='--', linewidth=1.2,
               xmin=0.05, xmax=0.45, alpha=0.7)
    ax.axhline(dist_spc.mean(), color=PALETTE[2], linestyle='--', linewidth=1.2,
               xmin=0.55, xmax=0.95, alpha=0.7)
    ax.text(1, dist_std.mean() + 0.8, f'μ={dist_std.mean():.1f}',
            ha='center', fontsize=9, color=PALETTE[0])
    ax.text(2, dist_spc.mean() + 0.8, f'μ={dist_spc.mean():.1f}',
            ha='center', fontsize=9, color=PALETTE[2])

    fig.tight_layout()
    out = OUTPUT_DIR / 'F3_normal_vs_special.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117', edgecolor='none')
    plt.close()
    print(f"Saved: {out}")


if __name__ == '__main__':
    plot()
