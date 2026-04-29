"""
Shared theme configuration for all visualization scripts.
Applies consistent dark GitHub-style theme across all charts.
"""

import matplotlib.pyplot as plt
import seaborn as sns

PALETTE = [
    '#58a6ff',  # blue
    '#3fb950',  # green
    '#f78166',  # orange-red
    '#d2a8ff',  # purple
    '#ffa657',  # orange
    '#79c0ff',  # light blue
    '#56d364',  # light green
    '#ff7b72',  # light red
]

THEME = {
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'axes.labelcolor':  '#c9d1d9',
    'text.color':       '#c9d1d9',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'grid.color':       '#21262d',
    'grid.linewidth':   0.5,
    'legend.facecolor': '#161b22',
    'legend.edgecolor': '#30363d',
    'legend.labelcolor': '#c9d1d9',
    'axes.titlesize':   13,
    'axes.titleweight': 'bold',
    'font.family':      'DejaVu Sans',
}


def apply_theme():
    plt.rcParams.update(THEME)
    sns.set_palette(PALETTE)
