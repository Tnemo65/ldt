import matplotlib.pyplot as plt
import numpy as np

# Figure 3.4: Distance-Fare scatter plot with anomaly zones
# Clear labels, no text overlap, good whitespace

np.random.seed(42)

# Generate normal trips
n_normal = 800
distance_normal = np.random.exponential(3.0, n_normal)
distance_normal = np.clip(distance_normal, 0.1, 25)
fare_normal = 3.0 + 2.5 * distance_normal + np.random.normal(0, 2, n_normal)
fare_normal = np.clip(fare_normal, 2, 150)

# Anomaly types
# Type 1: Meter tampering (short trip, huge fare)
n1 = 30
d1 = np.random.uniform(0.5, 3.0, n1)
f1 = 40 + np.random.uniform(20, 80, n1)

# Type 2: Short-trip fraud (very short, high fare)
n2 = 25
d2 = np.random.uniform(0.1, 0.8, n2)
f2 = 30 + np.random.uniform(30, 50, n2)

# Type 3: GPS spoofing (impossible distance)
n3 = 20
d3 = np.random.uniform(20, 40, n3)
f3 = 20 + 1.5 * d3 + np.random.normal(0, 3, n3)

# Type 4: Passenger fraud (normal distance, inflated fare)
n4 = 25
d4 = np.random.uniform(2, 6, n4)
f4 = 80 + np.random.uniform(20, 40, n4)

# Type 5: Combined
n5 = 20
d5 = np.random.uniform(5, 12, n5)
f5 = 100 + np.random.uniform(30, 60, n5)

fig, ax = plt.subplots(figsize=(14, 9))

# Plot normal points
ax.scatter(distance_normal, fare_normal, c='lightgray', s=15, alpha=0.4,
           label='Normal trips', zorder=1)

# Plot anomaly zones with shaded regions
# Zone 1: High fare, short distance
ax.fill_between([0, 3.5], [35, 35], [130, 130], alpha=0.08, color='red', zorder=0)
ax.scatter(d1, f1, c='red', s=40, marker='x', linewidths=1.5, label='Meter tampering', zorder=3)

# Zone 2: Very short, high fare
ax.fill_between([0, 1.0], [28, 28], [90, 90], alpha=0.08, color='orange', zorder=0)
ax.scatter(d2, f2, c='orange', s=40, marker='^', linewidths=1.5, label='Short-trip fraud', zorder=3)

# Zone 3: Very long, normal-ish fare (GPS spoofing)
n3_half = n3 // 2
ax.fill_between([18, 42], [15, 15], [100, 100], alpha=0.08, color='purple', zorder=0)
ax.scatter(d3[:n3_half], f3[:n3_half], c='purple', s=40, marker='s',
           linewidths=1.5, label='GPS spoofing', zorder=3)

# Zone 4: Normal distance, very high fare
ax.fill_between([1.5, 7], [75, 75], [130, 130], alpha=0.08, color='brown', zorder=0)
ax.scatter(d4, f4, c='brown', s=40, marker='D', linewidths=1.5, label='Passenger fraud', zorder=3)

# Zone 5: Combined
ax.scatter(d5, f5, c='darkred', s=40, marker='v', linewidths=1.5, label='Combined anomaly', zorder=3)

# Expected fare line
x_line = np.linspace(0.1, 30, 100)
y_line = 3.0 + 2.5 * x_line
ax.plot(x_line, y_line, 'k--', linewidth=1.5, alpha=0.6, label='Expected fare line ($3 + $2.5/mile)', zorder=2)

# Labels and formatting
ax.set_xlabel('Trip Distance (miles)', fontsize=12)
ax.set_ylabel('Fare Amount ($)', fontsize=12)
ax.set_title('Distance-Fare Scatter Plot: Normal Trips vs. Anomaly Types',
             fontsize=13, fontweight='bold')
ax.set_xlim(0, 45)
ax.set_ylim(0, 135)
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)

# Annotation box
textstr = ('Business rules catch violations where single features fail thresholds.\n'
           'ML detects anomalies where feature COMBINATIONS are suspicious.')
props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.85)
ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', horizontalalignment='right', bbox=props)

plt.tight_layout()
plt.savefig('thesis/figs/generated/F2_anomaly_zones.png', dpi=150, bbox_inches='tight')
plt.savefig('thesis/figs/generated/F2_anomaly_zones.pdf', bbox_inches='tight')
print('F2_scatter_anomaly_zones.png regenerated successfully')
