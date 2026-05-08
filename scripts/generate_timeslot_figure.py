#!/usr/bin/env python3
"""Generate trip volume by time slot visualization for Chapter 2."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent.parent / 'thesis' / 'figs' / 'generated'
OUTPUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linestyle': '--',
})

slots = ['Night\n(0-4h)', 'Morning\n(5-10h)', 'Afternoon\n(11-16h)', 'Evening\n(17-23h)']
counts = [230555, 660647, 1103177, 970245]
avg_fare = [16.42, 16.88, 17.54, 17.38]
special_pct = [15.5, 10.9, 8.5, 8.5]

x = np.arange(4)
fig, ax1 = plt.subplots(figsize=(9, 5))

colors = ['#2c3e50', '#e67e22', '#f39c12', '#3498db']
bars = ax1.bar(x, [c / 1e6 for c in counts], 0.55, color=colors, alpha=0.82,
               edgecolor='white', linewidth=1)
for bar, c, sp in zip(bars, counts, special_pct):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             '{:.0f}K'.format(c / 1e3), ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
             'Special: {:.1f}%'.format(sp), ha='center', va='center', fontsize=8,
             color='white', fontweight='bold')

ax1.set_xticks(x)
ax1.set_xticklabels(slots, fontsize=10)
ax1.set_ylabel('Record Count (Million)', fontweight='bold')
ax1.set_title('Trip Volume by Time Slot | NYC Yellow Taxi Jan 2024', fontweight='bold')
ax1.set_ylim(0, 1.3)

ax2 = ax1.twinx()
ax2.plot(x, avg_fare, 'D-', color='#e74c3c', linewidth=2, markersize=7, label='Avg Fare ($)', zorder=5)
ax2.set_ylabel('Average Fare ($)', color='#e74c3c', fontweight='bold')
ax2.tick_params(axis='y', labelcolor='#e74c3c')
ax2.set_ylim(15.5, 18.5)
ax2.spines['right'].set_visible(True)
ax2.spines['top'].set_visible(False)
ax2.legend(loc='upper right', fontsize=9)

plt.tight_layout()
out = OUTPUT / 'timeslot_volume.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.close()
print('Created: {}'.format(out))
