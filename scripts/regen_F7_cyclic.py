import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Figure 4.3: Cyclic encoding diagram for temporal features
# Clear circular layout, no text overlap

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# ---- Left: Hour cyclic encoding ----
ax1 = axes[0]
ax1.set_aspect('equal')

# Draw unit circle
theta = np.linspace(0, 2 * np.pi, 500)
ax1.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)

# Hour markers and arrows
for hour in [0, 6, 12, 18]:
    angle = 2 * np.pi * hour / 24 - np.pi / 2
    x = np.cos(angle)
    y = np.sin(angle)
    # Tick mark
    ax1.plot([x * 0.88, x * 0.98], [y * 0.88, y * 0.98], 'k-', linewidth=2)
    # Label
    label_x = x * 1.15
    label_y = y * 1.15
    ax1.text(label_x, label_y, f'{hour}:00', ha='center', va='center',
             fontsize=11, fontweight='bold')

# Hour labels around circle
for h in range(0, 24, 3):
    if h not in [0, 6, 12, 18]:
        angle = 2 * np.pi * h / 24 - np.pi / 2
        x = np.cos(angle)
        y = np.sin(angle)
        ax1.text(x * 1.07, y * 1.07, str(h), ha='center', va='center', fontsize=8, color='gray')

# Draw sin and cos vectors for hour=9
hour_demo = 9
angle_demo = 2 * np.pi * hour_demo / 24 - np.pi / 2
hx = np.cos(angle_demo)
hy = np.sin(angle_demo)

# Projections
ax1.annotate('', xy=(hx, 0), xytext=(0, 0),
             arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax1.plot([hx, hx], [0, hy], 'b--', linewidth=1.5)
ax1.annotate('', xy=(0, hy), xytext=(0, 0),
             arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax1.plot([0, hx], [hy, hy], 'r--', linewidth=1.5)

# Point on circle
ax1.scatter([hx], [hy], c='green', s=100, zorder=5, edgecolors='black', linewidths=1.5)

# Projection labels
ax1.text(hx / 2, -0.12, r'$\cos(2\pi \cdot 9/24) = $' + f'{np.cos(2*np.pi*9/24):.2f}',
         ha='center', va='top', fontsize=9, color='blue')
ax1.text(-0.15, hy / 2, r'$\sin(2\pi \cdot 9/24) = $' + f'{np.sin(2*np.pi*9/24):.2f}',
         ha='right', va='center', fontsize=9, color='red')

# Legend for sin/cos
ax1.plot([0.5, 0.7], [0.95, 0.95], 'b-', linewidth=2)
ax1.text(0.72, 0.95, r'$\cos(2\pi h / 24)$', ha='left', va='center', fontsize=10, color='blue')
ax1.plot([0.5, 0.7], [0.90, 0.90], 'r-', linewidth=2)
ax1.text(0.72, 0.90, r'$\sin(2\pi h / 24)$', ha='left', va='center', fontsize=10, color='red')

ax1.set_title('Cyclic Encoding: Hour of Day (h=9)', fontsize=13, fontweight='bold')
ax1.set_xlim(-1.4, 1.4)
ax1.set_ylim(-1.4, 1.4)
ax1.axis('off')

# Add text explanation box
box1 = ax1.text(-1.35, -1.25,
    r'$\mathrm{hour\_sin} = \sin(2\pi h / 24)$' + '\n' +
    r'$\mathrm{hour\_cos} = \cos(2\pi h / 24)$' + '\n' +
    'Captures 23:59 $\\approx$ 00:01',
    fontsize=10, va='top', ha='left',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

# ---- Right: Day of week cyclic encoding ----
ax2 = axes[1]
ax2.set_aspect('equal')

# Draw unit circle
ax2.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)

# DOW markers (0=Mon, 6=Sun)
dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
dow_angles = [0, 90, 180, 270, 45, 135, 225]  # positions in degrees
for i, (label, deg) in enumerate(zip(dow_labels, dow_angles)):
    angle = np.radians(deg) - np.pi / 2
    x = np.cos(angle)
    y = np.sin(angle)
    ax2.plot([x * 0.88, x * 0.98], [y * 0.88, y * 0.98], 'k-', linewidth=2)
    ax2.text(x * 1.15, y * 1.15, label, ha='center', va='center',
             fontsize=10, fontweight='bold')

# Draw sin and cos vectors for DOW=4 (Friday)
dow_demo = 4
angle_demo2 = 2 * np.pi * dow_demo / 7 - np.pi / 2
dx = np.cos(angle_demo2)
dy = np.sin(angle_demo2)

ax2.annotate('', xy=(dx, 0), xytext=(0, 0),
             arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax2.plot([dx, dx], [0, dy], 'b--', linewidth=1.5)
ax2.annotate('', xy=(0, dy), xytext=(0, 0),
             arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax2.plot([0, dx], [dy, dy], 'r--', linewidth=1.5)
ax2.scatter([dx], [dy], c='green', s=100, zorder=5, edgecolors='black', linewidths=1.5)

# Projection labels
ax2.text(dx / 2, -0.12,
         r'$\cos(2\pi \cdot 4/7) = $' + f'{np.cos(2*np.pi*4/7):.2f}',
         ha='center', va='top', fontsize=9, color='blue')
ax2.text(-0.18, dy / 2,
         r'$\sin(2\pi \cdot 4/7) = $' + f'{np.sin(2*np.pi*4/7):.2f}',
         ha='right', va='center', fontsize=9, color='red')

# Legend
ax2.plot([0.5, 0.7], [0.95, 0.95], 'b-', linewidth=2)
ax2.text(0.72, 0.95, r'$\cos(2\pi \cdot \mathrm{dow} / 7)$', ha='left', va='center', fontsize=10, color='blue')
ax2.plot([0.5, 0.7], [0.90, 0.90], 'r-', linewidth=2)
ax2.text(0.72, 0.90, r'$\sin(2\pi \cdot \mathrm{dow} / 7)$', ha='left', va='center', fontsize=10, color='red')

ax2.set_title('Cyclic Encoding: Day of Week (Friday)', fontsize=13, fontweight='bold')
ax2.set_xlim(-1.4, 1.4)
ax2.set_ylim(-1.4, 1.4)
ax2.axis('off')

box2 = ax2.text(-1.35, -1.25,
    r'$\mathrm{dow\_sin} = \sin(2\pi \cdot \mathrm{dow} / 7)$' + '\n' +
    r'$\mathrm{dow\_cos} = \cos(2\pi \cdot \mathrm{dow} / 7)$' + '\n' +
    'Ensures Sunday $\\approx$ Monday proximity',
    fontsize=10, va='top', ha='left',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

# Main title
fig.suptitle('Cyclic Encoding for Temporal Features in CA-DQStream',
             fontsize=14, fontweight='bold', y=1.01)

plt.tight_layout()
plt.savefig('thesis/figs/generated/F7_cyclic.png', dpi=150, bbox_inches='tight')
plt.savefig('thesis/figs/generated/F7_cyclic.pdf', bbox_inches='tight')
print('F7_cyclic_encoding.png regenerated successfully')
