#!/usr/bin/env python3
"""Generate missing thesis figures: exp5a, exp8a, sankey layer flow."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent.parent / 'thesis' / 'figs' / 'generated'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
})


def create_exp5a_fpr_comparison():
    """Experiment 5a: FPR Comparison - Global vs 4D Context-Aware Thresholds."""
    trip_types = ['Standard', 'JFK\nAirport', 'Newark\nAirport', 'Negotiated', 'Overall']
    global_fpr = [12.4, 51.2, 68.9, 42.1, 38.7]
    context_fpr = [2.8, 5.1, 6.7, 4.9, 4.2]
    improvements = ['4.4×', '10.0×', '10.3×', '8.6×', '9.2×']

    x = np.arange(len(trip_types))
    width = 0.32

    fig, ax = plt.subplots(figsize=(10, 5.5))

    bars1 = ax.bar(x - width/2, global_fpr, width,
                   label='Global Threshold', color='#c0392b', alpha=0.85,
                   edgecolor='#922b21', linewidth=0.8)
    bars2 = ax.bar(x + width/2, context_fpr, width,
                   label='4D Context-Aware', color='#27ae60', alpha=0.85,
                   edgecolor='#1e8449', linewidth=0.8)

    for i, (b1, b2, imp) in enumerate(zip(bars1, bars2, improvements)):
        ax.text(b1.get_x() + b1.get_width()/2, b1.get_height() + 1.2,
                '{:.1f}%'.format(global_fpr[i]), ha='center', va='bottom',
                fontsize=9, color='#922b21', fontweight='bold')
        ax.text(b2.get_x() + b2.get_width()/2, b2.get_height() + 1.2,
                '{:.1f}%'.format(context_fpr[i]), ha='center', va='bottom',
                fontsize=9, color='#1e8449', fontweight='bold')

    for i, imp in enumerate(improvements):
        height = max(global_fpr[i], context_fpr[i])
        ax.annotate(imp, xy=(x[i], height + 6),
                    fontsize=11, fontweight='bold', ha='center', va='bottom',
                    color='#2c3e50',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#f9e79f',
                              edgecolor='#f1c40f', alpha=0.9))

    ax.set_xlabel('Trip Type', fontweight='bold')
    ax.set_ylabel('False Positive Rate (%)', fontweight='bold')
    ax.set_title('FPR Comparison: Global vs 4D Context-Aware Thresholds',
                 fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(trip_types, fontsize=10)
    ax.legend(loc='upper left', framealpha=0.95, edgecolor='#bdc3c7')
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.set_ylim(0, 82)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = OUTPUT_DIR / 'exp5a_fpr_comparison.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: {}".format(out))


def create_exp8a_latency_throughput():
    """Experiment 8a: Latency CDF + Throughput Comparison (two-panel)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left panel: Latency CDF ---
    percentiles = np.array([1, 5, 10, 25, 50, 75, 90, 95, 99])
    lat_linear = np.array([52, 105, 158, 295, 487, 612, 735, 790, 843])
    lat_rendez = np.array([18, 35, 52, 98, 168, 210, 248, 272, 294])

    ax1.plot(lat_linear, percentiles, 'o-', color='#c0392b', linewidth=2,
             markersize=5, label='Linear Pipeline', zorder=3)
    ax1.plot(lat_rendez, percentiles, 's-', color='#27ae60', linewidth=2,
             markersize=5, label='Rendezvous Pipeline', zorder=3)

    ax1.axhline(y=99, color='#7f8c8d', linestyle=':', alpha=0.5, linewidth=0.8)
    ax1.axhline(y=50, color='#7f8c8d', linestyle=':', alpha=0.5, linewidth=0.8)

    ax1.annotate('p99=843ms', xy=(843, 99), xytext=(700, 88),
                 fontsize=9, fontweight='bold', color='#c0392b',
                 arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.2))
    ax1.annotate('p99=294ms', xy=(294, 99), xytext=(380, 88),
                 fontsize=9, fontweight='bold', color='#27ae60',
                 arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.2))
    ax1.annotate('p50=487ms', xy=(487, 50), xytext=(550, 38),
                 fontsize=8, color='#c0392b',
                 arrowprops=dict(arrowstyle='->', color='#c0392b', lw=0.8))
    ax1.annotate('p50=168ms', xy=(168, 50), xytext=(30, 38),
                 fontsize=8, color='#27ae60',
                 arrowprops=dict(arrowstyle='->', color='#27ae60', lw=0.8))

    mid_x = (843 + 294) / 2
    ax1.annotate('2.9×', xy=(mid_x, 99), fontsize=12, fontweight='bold',
                 ha='center', va='bottom', color='#2c3e50',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#f9e79f',
                           edgecolor='#f1c40f', alpha=0.9))

    ax1.set_xlabel('Latency (ms)', fontweight='bold')
    ax1.set_ylabel('Percentile (%)', fontweight='bold')
    ax1.set_title('(a) Latency CDF Comparison', fontweight='bold')
    ax1.legend(loc='lower right', framealpha=0.95)
    ax1.grid(alpha=0.2, linestyle='--')
    ax1.set_xlim(0, 950)
    ax1.set_ylim(0, 105)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- Right panel: Throughput Bar Chart ---
    archs = ['Linear\nPipeline', 'Rendezvous\nPipeline']
    throughputs = [8240, 18450]
    colors = ['#c0392b', '#27ae60']

    bars = ax2.bar(archs, throughputs, color=colors, alpha=0.85, width=0.5,
                   edgecolor=['#922b21', '#1e8449'], linewidth=0.8)

    for bar, val in zip(bars, throughputs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 400,
                 '{:,}'.format(val), ha='center', va='bottom',
                 fontweight='bold', fontsize=11)

    ax2.annotate('2.2× improvement', xy=(0.5, 14000),
                 fontsize=12, fontweight='bold', ha='center',
                 color='#2c3e50',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#f9e79f',
                           edgecolor='#f1c40f', alpha=0.9))

    ax2.set_ylabel('Throughput (events/sec)', fontweight='bold')
    ax2.set_title('(b) Throughput Comparison', fontweight='bold')
    ax2.set_ylim(0, 22000)
    ax2.grid(axis='y', alpha=0.2, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout(w_pad=3)
    out = OUTPUT_DIR / 'exp8a_latency_throughput.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: {}".format(out))


def create_sankey_layer_flow():
    """Sankey-style layer flow diagram using matplotlib patches and arrows."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    box_w, box_h = 3.6, 0.7

    stages = [
        {'label': 'Input Records', 'count': '3,080,000', 'pct': '100%',
         'y': 9.0, 'color': '#3498db'},
        {'label': 'Layer 1: Schema Validation', 'count': '2,770,000', 'pct': '89.9%',
         'y': 7.2, 'color': '#2ecc71'},
        {'label': 'Layer 2: Business Rules (Canary)', 'count': '2,666,000', 'pct': '86.6%',
         'y': 5.4, 'color': '#f39c12'},
        {'label': 'Layer 2: Isolation Forest (Complex)', 'count': '2,577,000', 'pct': '83.7%',
         'y': 3.6, 'color': '#e67e22'},
        {'label': 'Layer 3: MetaAggregator', 'count': '2,577,000', 'pct': '83.7%',
         'y': 1.8, 'color': '#9b59b6'},
        {'label': 'Clean Output', 'count': '2,577,000', 'pct': '83.7%',
         'y': 0.0, 'color': '#1abc9c'},
    ]

    rejections = [
        {'from_idx': 0, 'label': '310K rejected (10.1%)', 'color': '#e74c3c'},
        {'from_idx': 1, 'label': '104K flagged (3.4%)', 'color': '#e74c3c'},
        {'from_idx': 2, 'label': '89K flagged (2.9%)', 'color': '#e74c3c'},
    ]

    cx = 5.0
    for i, s in enumerate(stages):
        x0 = cx - box_w / 2
        y0 = s['y']
        rect = mpatches.FancyBboxPatch(
            (x0, y0), box_w, box_h,
            boxstyle=mpatches.BoxStyle.Round(pad=0.1),
            facecolor=s['color'], edgecolor='white', alpha=0.88, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(cx, y0 + box_h/2 + 0.05, s['label'],
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        ax.text(cx, y0 + box_h/2 - 0.22,
                '{} ({})'.format(s['count'], s['pct']),
                ha='center', va='center', fontsize=8.5, color='#ecf0f1')

        if i < len(stages) - 1:
            next_y = stages[i + 1]['y'] + box_h
            ax.annotate('', xy=(cx, next_y), xytext=(cx, y0),
                        arrowprops=dict(arrowstyle='->', color='#7f8c8d',
                                        lw=2, connectionstyle='arc3'))

    rej_x = 8.5
    for j, r in enumerate(rejections):
        src = stages[r['from_idx']]
        src_y = src['y'] + box_h / 2
        rej_y = src_y - 0.3

        ax.annotate(
            r['label'],
            xy=(cx + box_w/2, src_y),
            xytext=(rej_x, rej_y),
            fontsize=9, fontweight='bold', color=r['color'],
            ha='center', va='center',
            arrowprops=dict(arrowstyle='->', color=r['color'], lw=1.5,
                            connectionstyle='arc3,rad=-0.2'),
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#fdedec',
                      edgecolor=r['color'], alpha=0.9))

    ax.text(cx, 9.95, 'CA-DQStream: Record Flow Through Processing Layers',
            ha='center', va='center', fontsize=13, fontweight='bold', color='#2c3e50')

    plt.tight_layout()
    out = OUTPUT_DIR / 'sankey_layer_flow.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: {}".format(out))


if __name__ == '__main__':
    print("Generating thesis figures...")
    create_exp5a_fpr_comparison()
    create_exp8a_latency_throughput()
    create_sankey_layer_flow()
    print("\nAll figures generated in: {}".format(OUTPUT_DIR))
