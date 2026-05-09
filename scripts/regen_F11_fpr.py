import matplotlib.pyplot as plt
import numpy as np

# Figure 6.5: FPR Comparison - Global vs 4D Context-Aware Thresholds
# Clear bars, readable labels, no overlap

# Data: Global FPR vs 4D Context-Aware FPR per context
contexts = [
    'Standard\n(All Day)',
    'JFK\nAirport',
    'Newark\nAirport',
    'Night\n(0-6h)',
    'Weekend',
    'Negotiated\nRate',
    'Overall\n(Weighted)',
]

global_fpr = [12.3, 51.2, 68.9, 28.4, 19.7, 45.3, 18.6]
ca4d_fpr = [4.8, 4.2, 5.1, 5.6, 5.2, 5.8, 3.8]

# Improvement ratios
improvement = [g / c for g, c in zip(global_fpr, ca4d_fpr)]

x = np.arange(len(contexts))
width = 0.35

fig, ax = plt.subplots(figsize=(15, 8))

# Bar colors
bars1 = ax.bar(x - width/2, global_fpr, width, label='Global Threshold',
               color='#e74c3c', alpha=0.85, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x + width/2, ca4d_fpr, width, label='4D Context-Aware',
               color='#27ae60', alpha=0.85, edgecolor='black', linewidth=0.5)

# Value labels on bars
for bar, val in zip(bars1, global_fpr):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0,
            f'{val}%', ha='center', va='bottom', fontsize=9, fontweight='bold',
            color='#c0392b')

for bar, val in zip(bars2, ca4d_fpr):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0,
            f'{val}%', ha='center', va='bottom', fontsize=9, fontweight='bold',
            color='#1e8449')

# Improvement annotation arrows
for i, (g, c, imp) in enumerate(zip(global_fpr, ca4d_fpr, improvement)):
    if i < len(x):
        ax.annotate('', xy=(x[i], c + 2.5), xytext=(x[i], g - 2.5),
                    arrowprops=dict(arrowstyle='->', color='navy', lw=1.5))
        ax.text(x[i], (g + c) / 2 + 1.5, f'{imp:.1f}x',
                ha='center', va='bottom', fontsize=8, color='navy',
                fontweight='bold')

# Reference line at 5% threshold
ax.axhline(y=5.0, color='orange', linestyle='--', linewidth=1.5, alpha=0.7,
           label='5% FPR Target')

# Formatting
ax.set_xlabel('Data Context', fontsize=12, fontweight='bold')
ax.set_ylabel('False Positive Rate (%)', fontsize=12, fontweight='bold')
ax.set_title('FPR Comparison: Global vs. 4D Context-Aware Thresholds',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(contexts, fontsize=10)
ax.set_ylim(0, 85)
ax.yaxis.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

# Summary text box
summary = ('Global thresholds catastrophically fail on airport trips\n'
           '(JFK: 51.2%, Newark: 68.9%), while 4D context-aware\n'
           'thresholds reduce FPR below 6% across all contexts.\n'
           'Overall improvement: 4.9x reduction in FPR.')
props = dict(boxstyle='round,pad=0.6', facecolor='lightyellow', alpha=0.9)
ax.text(0.02, 0.97, summary, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', horizontalalignment='left', bbox=props)

plt.tight_layout()
plt.savefig('thesis/figs/generated/F11_fpr.png', dpi=150, bbox_inches='tight')
plt.savefig('thesis/figs/generated/F11_fpr.pdf', bbox_inches='tight')
print('F11_fpr_comparison.png regenerated successfully')
