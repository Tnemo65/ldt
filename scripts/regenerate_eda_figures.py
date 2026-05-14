#!/usr/bin/env python3
"""Regenerate EDA figures for thesis with clean academic styling (white bg, serif fonts).
Data: NYC Yellow Taxi 2024 only (12 months).
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).resolve().parent.parent / 'data' / 'raw'
OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'thesis' / 'figs'

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
})

MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']


def load_monthly_summaries():
    """Load per-month summaries without holding all data in memory."""
    records = []
    for m in range(1, 13):
        fp = DATA_DIR / 'yellow_tripdata_2024-{:02d}.parquet'.format(m)
        if not fp.exists():
            print('  SKIP {}'.format(fp.name))
            continue
        df = pd.read_parquet(fp)
        n = len(df)

        null_passenger = df['passenger_count'].isnull().sum()
        null_ratecode = df['RatecodeID'].isnull().sum()
        neg_fare = (df['fare_amount'] < 0).sum()
        neg_total = (df['total_amount'] < 0).sum()
        zero_dist = (df['trip_distance'] == 0).sum()
        zero_dur = False
        if 'tpep_pickup_datetime' in df.columns and 'tpep_dropoff_datetime' in df.columns:
            dur = (pd.to_datetime(df['tpep_dropoff_datetime']) -
                   pd.to_datetime(df['tpep_pickup_datetime'])).dt.total_seconds()
            zero_dur_count = (dur <= 0).sum()
        else:
            zero_dur_count = 0

        bad_vendor = (~df['VendorID'].isin([1, 2, 6])).sum() if 'VendorID' in df.columns else 0
        drop_before_pick = zero_dur_count

        # Normal vs special ratecode
        rc = df['RatecodeID'].dropna()
        normal_count = (rc == 1).sum()
        special_count = (rc != 1).sum()

        # Avg fare/distance/duration by ratecode
        df_clean = df.dropna(subset=['RatecodeID', 'fare_amount', 'trip_distance'])
        df_clean['duration_min'] = (
            pd.to_datetime(df_clean['tpep_dropoff_datetime']) -
            pd.to_datetime(df_clean['tpep_pickup_datetime'])
        ).dt.total_seconds() / 60.0

        normal_mask = df_clean['RatecodeID'] == 1
        special_mask = df_clean['RatecodeID'] != 1

        records.append({
            'month': m,
            'n': n,
            'null_passenger': null_passenger,
            'null_ratecode': null_ratecode,
            'neg_fare': neg_fare,
            'neg_total': neg_total,
            'zero_dist': zero_dist,
            'zero_dur': zero_dur_count,
            'bad_vendor': bad_vendor,
            'drop_before_pick': drop_before_pick,
            'normal_count': normal_count,
            'special_count': special_count,
            'normal_avg_fare': df_clean.loc[normal_mask, 'fare_amount'].mean(),
            'special_avg_fare': df_clean.loc[special_mask, 'fare_amount'].mean(),
            'normal_avg_dist': df_clean.loc[normal_mask, 'trip_distance'].mean(),
            'special_avg_dist': df_clean.loc[special_mask, 'trip_distance'].mean(),
            'normal_avg_dur': df_clean.loc[normal_mask, 'duration_min'].mean(),
            'special_avg_dur': df_clean.loc[special_mask, 'duration_min'].mean(),
        })
        print('  Loaded month {:02d}: {:,} records'.format(m, n))
        del df, df_clean

    return pd.DataFrame(records)


def fig1_monthly_volume(stats):
    """Figure 1: Monthly Record Volume - line chart."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(12)
    counts = stats['n'].values
    total = counts.sum()

    ax.plot(x, counts / 1e6, color='#3498db', linewidth=2.5, marker='o',
            markersize=7, markerfacecolor='#ffffff', markeredgecolor='#3498db',
            markeredgewidth=2, zorder=3)

    for xi, c in zip(x, counts):
        ax.annotate('{:.0f}K'.format(c / 1e3),
                   xy=(xi, c / 1e6), xytext=(0, 8),
                   textcoords='offset points', ha='center', va='bottom',
                   fontsize=8.5, fontweight='bold', color='#2c3e50')

    ax.set_xticks(x)
    ax.set_xticklabels(MONTHS)
    ax.set_xlabel('Month', fontweight='bold')
    ax.set_ylabel('Records (Million)', fontweight='bold')
    ax.set_title('Monthly Record Volume | NYC Yellow Taxi 2024\nTotal: {:,} records'.format(total),
                 fontweight='bold')
    ax.set_ylim(0, max(counts / 1e6) * 1.20)
    ax.set_xlim(-0.5, 11.5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1fM'))
    ax.fill_between(x, 0, counts / 1e6, alpha=0.08, color='#3498db')

    plt.tight_layout()
    out = OUTPUT_DIR / '1_Monthly_Record_Volume.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print('Created: {}'.format(out.name))


def fig2_violation_breakdown(stats):
    """Figure 2: Stacked violation breakdown by month."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(12)
    width = 0.65

    categories = [
        ('neg_fare', 'Negative Fare', '#e74c3c'),
        ('neg_total', 'Negative Total', '#9b59b6'),
        ('zero_dist', 'Zero Distance', '#27ae60'),
        ('null_ratecode', 'NULL Ratecode', '#3498db'),
        ('null_passenger', 'NULL Passenger', '#2980b9'),
    ]

    bottom = np.zeros(12)
    for col, label, color in categories:
        vals = (stats[col] / stats['n'] * 100).values
        ax.bar(x, vals, width, bottom=bottom, label=label, color=color, alpha=0.82,
               edgecolor='white', linewidth=0.4)
        bottom += vals

    total_vr = bottom
    ax2 = ax.twinx()
    ax2.plot(x, total_vr, 'D--', color='#e67e22', linewidth=1.8, markersize=5,
             label='Total Violation Rate', zorder=5)
    ax2.set_ylabel('Total Violation Rate (%)', color='#e67e22')
    ax2.tick_params(axis='y', labelcolor='#e67e22')
    ax2.spines['right'].set_visible(True)
    ax2.spines['top'].set_visible(False)

    ax.set_xticks(x)
    ax.set_xticklabels(MONTHS)
    ax.set_xlabel('Month')
    ax.set_ylabel('Violation Rate (%)')
    ax.set_title('Violation Breakdown by Month | NYC Yellow Taxi 2024',
                 fontweight='bold')

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper left', fontsize=8, ncol=2,
              framealpha=0.95)

    plt.tight_layout()
    out = OUTPUT_DIR / '2_Violation_Breakdown.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print('Created: {}'.format(out.name))


def fig_t5_violations_by_month(stats):
    """T5: Violation rate stacked bar (5 main rules)."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(12)
    width = 0.65

    categories = [
        ('neg_fare', 'Neg Fare', '#e74c3c'),
        ('neg_total', 'Neg Total', '#9b59b6'),
        ('zero_dist', 'Zero Dist', '#27ae60'),
        ('null_passenger', 'NULL Passenger', '#3498db'),
        ('null_ratecode', 'NULL Ratecode', '#85c1e9'),
    ]

    bottom = np.zeros(12)
    for col, label, color in categories:
        vals = (stats[col] / stats['n'] * 100).values
        ax.bar(x, vals, width, bottom=bottom, label=label, color=color, alpha=0.82,
               edgecolor='white', linewidth=0.4)
        bottom += vals

    sep_idx = 8
    peak = bottom[sep_idx]
    ax.annotate('Sep peak: {:.1f}%\n(NULL Passenger spike)'.format(peak),
                xy=(sep_idx, peak), xytext=(sep_idx + 1.5, peak + 1),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fdebd0', edgecolor='#e67e22'))

    ax.set_xticks(x)
    ax.set_xticklabels(MONTHS)
    ax.set_xlabel('Month')
    ax.set_ylabel('Violation Rate (%)')
    ax.set_title('Violation Rate by Month | NYC Yellow Taxi 2024\n5 rules: neg fare, neg total, zero dist, null passenger, null ratecode',
                 fontweight='bold')
    ax.legend(loc='upper left', fontsize=8.5, ncol=3, framealpha=0.95)

    plt.tight_layout()
    out = OUTPUT_DIR / 'T5_violations_by_month.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print('Created: {}'.format(out.name))


def fig_t7_normal_vs_special(stats):
    """T7: Normal vs Special ratecode duration profile (donut + bar)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # Left: Donut chart - annual share
    total_normal = stats['normal_count'].sum()
    total_special = stats['special_count'].sum()
    total_all = total_normal + total_special
    sizes = [total_normal / total_all * 100, total_special / total_all * 100]
    colors_pie = ['#27ae60', '#e67e22']

    wedges, texts, autotexts = ax1.pie(
        sizes, labels=['Normal\n(RC1)', 'Special\n(RC2-99)'],
        autopct='%1.1f%%', colors=colors_pie, startangle=90,
        pctdistance=0.75, wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2))
    for t in autotexts:
        t.set_fontsize(11)
        t.set_fontweight('bold')
    for t in texts:
        t.set_fontsize(10)
    ax1.set_title('Annual Duration Share\nNormal vs Special', fontweight='bold')

    # Right: Grouped bar - monthly avg duration
    x = np.arange(12)
    w = 0.35
    bars1 = ax2.bar(x - w/2, stats['normal_avg_dur'], w, label='Normal (RC1)',
                    color='#27ae60', alpha=0.82, edgecolor='#1e8449', linewidth=0.5)
    bars2 = ax2.bar(x + w/2, stats['special_avg_dur'], w, label='Special (RC2-99)',
                    color='#e67e22', alpha=0.82, edgecolor='#ca6f1e', linewidth=0.5)

    ratio = stats['special_avg_dur'].mean() / stats['normal_avg_dur'].mean()
    ax2.annotate('Special = {:.1f}x Normal'.format(ratio),
                 xy=(8, stats['special_avg_dur'].iloc[8]),
                 xytext=(6, stats['special_avg_dur'].max() + 2),
                 fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#fdebd0', edgecolor='#e67e22'),
                 arrowprops=dict(arrowstyle='->', color='#e67e22', lw=1.2))

    ax2.set_xticks(x)
    ax2.set_xticklabels(MONTHS, fontsize=9)
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Avg Duration (minutes)')
    ax2.set_title('Monthly Avg Duration\nNormal vs Special', fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9, framealpha=0.95)

    plt.tight_layout()
    out = OUTPUT_DIR / 'T7_normal_vs_special_copy.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print('Created: {}'.format(out.name))


def fig5_distance_vs_fare():
    """Figure 5: Distance vs Fare scatter (Jan 2024 sample)."""
    df = pd.read_parquet(DATA_DIR / 'yellow_tripdata_2024-01.parquet',
                         columns=['trip_distance', 'fare_amount', 'VendorID'])
    df = df.dropna(subset=['trip_distance', 'fare_amount'])
    df = df[(df['trip_distance'] > 0) & (df['trip_distance'] < 30) &
            (df['fare_amount'] > -150) & (df['fare_amount'] < 200)]

    sample = df.sample(min(40000, len(df)), random_state=42)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    ax.scatter(sample['trip_distance'], sample['fare_amount'],
               s=1.5, alpha=0.15, c='#3498db', rasterized=True)

    x_ref = np.linspace(0, 30, 100)
    ax.plot(x_ref, 3.0 + 2.5 * x_ref, '--', color='#e74c3c', linewidth=1.5,
            label='Reference: $3.0 + $2.5/mile', alpha=0.8)

    ax.set_xlabel('Trip Distance (miles)')
    ax.set_ylabel('Fare Amount ($)')
    ax.set_title('Distance vs Fare | NYC Yellow Taxi Jan 2024\n(sample: 40K records)',
                 fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.95)
    ax.set_xlim(0, 30)
    ax.set_ylim(-150, 200)

    plt.tight_layout()
    out = OUTPUT_DIR / '5_Distance_vs_Fare.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    del df, sample
    print('Created: {}'.format(out.name))


def fig6_distance_vs_duration():
    """Figure 6: Distance vs Duration scatter (Jan 2024 sample)."""
    df = pd.read_parquet(DATA_DIR / 'yellow_tripdata_2024-01.parquet',
                         columns=['trip_distance', 'tpep_pickup_datetime', 'tpep_dropoff_datetime'])
    df = df.dropna()
    df['duration_min'] = (
        pd.to_datetime(df['tpep_dropoff_datetime']) -
        pd.to_datetime(df['tpep_pickup_datetime'])
    ).dt.total_seconds() / 60.0
    df = df[(df['trip_distance'] > 0) & (df['trip_distance'] < 30) &
            (df['duration_min'] > 0) & (df['duration_min'] < 120)]

    sample = df.sample(min(40000, len(df)), random_state=42)

    fig, ax = plt.subplots(figsize=(7, 5.5))

    ax.scatter(sample['duration_min'], sample['trip_distance'],
               s=1.5, alpha=0.15, c='#8e44ad', rasterized=True)

    x_ref = np.linspace(0, 120, 100)
    ax.plot(x_ref, x_ref * 30 / 60, '--', color='#e74c3c', linewidth=1.2,
            label='30 mph ref', alpha=0.7)
    ax.plot(x_ref, x_ref * 60 / 60, '--', color='#e67e22', linewidth=1.2,
            label='60 mph ref', alpha=0.7)

    ax.set_xlabel('Duration (minutes)')
    ax.set_ylabel('Trip Distance (miles)')
    ax.set_title('Distance vs Duration | NYC Yellow Taxi Jan 2024\n(sample: 40K records)',
                 fontweight='bold')
    ax.legend(loc='upper left', fontsize=9, framealpha=0.95)
    ax.set_xlim(0, 120)
    ax.set_ylim(0, 30)

    plt.tight_layout()
    out = OUTPUT_DIR / '6_Distance_vs_Duration.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    del df, sample
    print('Created: {}'.format(out.name))


if __name__ == '__main__':
    print('=== Regenerating EDA Figures (2024 only) ===\n')

    print('Loading monthly summaries...')
    stats = load_monthly_summaries()

    print('\nGenerating figures...')
    fig1_monthly_volume(stats)
    fig2_violation_breakdown(stats)
    fig_t5_violations_by_month(stats)
    fig_t7_normal_vs_special(stats)
    fig5_distance_vs_fare()
    fig6_distance_vs_duration()

    print('\nAll 6 EDA figures regenerated in: {}'.format(OUTPUT_DIR))
