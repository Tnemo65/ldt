"""
datasetv2/viz_anomaly_types.py
==============================
Visualize 7 MemStream-detectable anomaly types.
Each type: anomaly = RED, normal = BLUE. Key feature relationships highlighted.

Output: C:/proj/new/results/viz_anomaly_types/
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OUT_DIR = Path("C:/proj/new/datasetv2")
RESULT_DIR = Path("C:/proj/new/results/viz_anomaly_types")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(42)

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
df = pd.read_parquet(OUT_DIR / "valid_polluted.parquet")
gt_mask = np.load(OUT_DIR / "valid" / "ground_truth_mask.npy")
gt_types = json.load(open(OUT_DIR / "valid" / "ground_truth_per_type.json"))

# Stratified sampling: take ~50K normal + ~50K anomaly (balanced)
N_PER_CLASS = 40_000
normal_idx = np.where(~gt_mask)[0]
anomaly_idx = np.where(gt_mask)[0]

np.random.shuffle(normal_idx)
np.random.shuffle(anomaly_idx)

sample_normal = normal_idx[:N_PER_CLASS]
sample_anomaly = anomaly_idx[:N_PER_CLASS]
combined_idx = np.concatenate([sample_normal, sample_anomaly])

df_s = df.iloc[combined_idx].reset_index(drop=True)
gt_s = np.concatenate([np.zeros(N_PER_CLASS, dtype=bool),
                       np.ones(N_PER_CLASS, dtype=bool)])

normal_mask = ~gt_s
anomaly_mask = gt_s

# Derived features
df_s["dur_min"] = np.clip(df_s["duration_s"] / 60.0, 0, 1440)
df_s["speed_mph"] = np.clip(df_s["trip_distance"] / np.maximum(df_s["duration_s"] / 3600, 0.001), 0, 500)
df_s["fare_per_mile"] = np.where(
    df_s["trip_distance"] > 0,
    df_s["fare_amount"] / np.maximum(df_s["trip_distance"], 0.01),
    0
)
df_s["fare_per_min"] = np.where(
    df_s["dur_min"] > 0,
    df_s["fare_amount"] / np.maximum(df_s["dur_min"], 0.01),
    0
)

print(f"Samples: {len(df_s):,} (normal={normal_mask.sum():,}, anomaly={anomaly_mask.sum():,})")

# Per-type masks
type_masks = {}
for type_id, data in gt_types.items():
    if not isinstance(data, dict):
        continue
    indices = data.get("indices", [])
    if not indices:
        continue
    int_idx = set(int(i) for i in indices)
    mask = np.array([i in int_idx for i in combined_idx])
    type_masks[type_id] = mask

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
C_BLUE = "#2980B9"
C_RED = "#C0392B"
C_RED_DARK = "#8B0000"
C_GREEN = "#27AE60"
C_ORANGE = "#E67E22"
C_GRAY = "#7F8C8D"
C_BG = "#F9F9F9"

# ---------------------------------------------------------------------------
# Per-type configs: (x, y, xlabel, ylabel, xlim, ylim)
# ---------------------------------------------------------------------------
TYPE_CFG = {
    "1": dict(
        x="dur_min", y="fare_amount",
        xlabel="Duration (min)", ylabel="Fare ($)",
        xlim=(0, 200), ylim=(0, 300),
        title="Type 1: Short Expensive\nfare x 5-15",
        note="Short duration, VERY HIGH fare",
    ),
    "2": dict(
        x="fare_amount", y="total_amount",
        xlabel="Fare ($)", ylabel="Total ($)",
        xlim=(0, 150), ylim=(0, 400),
        title="Type 2: Tip Anomaly\ntip = fare x 10-20",
        note="Total >> Fare (tip inflated)",
    ),
    "3": dict(
        x="fare_amount", y="dur_min",
        xlabel="Fare ($)", ylabel="Duration (min)",
        xlim=(0, 100), ylim=(0, 400),
        title="Type 3: Slow Trip\nduration x 2-3",
        note="Same fare, VERY LONG duration",
    ),
    "4": dict(
        x="trip_distance", y="dur_min",
        xlabel="Trip Distance (mi)", ylabel="Duration (min)",
        xlim=(0, 15), ylim=(0, 300),
        title="Type 4: Combo Short+Long\ndist x 0.05-0.3, dur x 2-5",
        note="Tiny distance, HUGE duration",
    ),
    "5": dict(
        x="dur_min", y="speed_mph",
        xlabel="Duration (min)", ylabel="Speed (mph)",
        xlim=(0, 100), ylim=(0, 80),
        title="Type 5: Speed Anomaly\nduration / 2-4",
        note="Short duration, HIGH speed",
    ),
    "6": dict(
        x="dur_min", y="fare_amount",
        xlabel="Duration (min)", ylabel="Fare ($)",
        xlim=(0, 120), ylim=(0, 250),
        title="Type 6: Short Expensive v2\nfare x 3-6",
        note="Short duration, HIGH fare",
    ),
    "7": dict(
        x="fare_amount", y="dur_min",
        xlabel="Fare ($)", ylabel="Duration (min)",
        xlim=(0, 80), ylim=(0, 200),
        title="Type 7: Slow Trip v2\nduration x 1.5-2.5",
        note="Same fare, LONG duration",
    ),
}

TYPE_NAME = {
    "1": "short_expensive", "2": "tip_anomaly", "3": "slow_trip",
    "4": "combo_short_long", "5": "speed_anomaly",
    "6": "short_expensive_v2", "7": "slow_trip_v2",
}


# ---------------------------------------------------------------------------
# Figure 1: All 7 types in a 4x2 grid
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(4, 2, figsize=(18, 24))
fig.patch.set_facecolor(C_BG)

for i, type_id in enumerate(["1", "2", "3", "4", "5", "6", "7"]):
    ax = axes.flatten()[i]
    cfg = TYPE_CFG[type_id]
    type_m = type_masks.get(type_id)

    x_col = cfg["x"]
    y_col = cfg["y"]

    # Normal: blue
    x_n = df_s.loc[normal_mask, x_col].values
    y_n = df_s.loc[normal_mask, y_col].values
    ax.scatter(x_n, y_n, c=C_BLUE, alpha=0.15, s=4, rasterized=True,
               label=f"Normal (n={normal_mask.sum():,})")

    # This type: red
    if type_m is not None:
        x_t = df_s.loc[type_m, x_col].values
        y_t = df_s.loc[type_m, y_col].values
        ax.scatter(x_t, y_t, c=C_RED, alpha=0.4, s=10, rasterized=True,
                   label=f"Anomaly Type {type_id} (n={type_m.sum():,})")

    ax.set_xlabel(cfg["xlabel"], fontsize=9)
    ax.set_ylabel(cfg["ylabel"], fontsize=9)
    ax.set_xlim(cfg["xlim"])
    ax.set_ylim(cfg["ylim"])
    ax.set_title(cfg["title"], fontsize=10, fontweight="bold", pad=5)
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.set_facecolor(C_BG)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.85)

    # Annotation box
    ax.text(0.03, 0.97, cfg["note"],
            transform=ax.transAxes, fontsize=7.5,
            verticalalignment="top", horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow",
                     alpha=0.85, edgecolor="gray", linewidth=0.5))

axes.flatten()[7].axis("off")

p_blue = mpatches.Patch(color=C_BLUE, alpha=0.5, label="Normal")
p_red = mpatches.Patch(color=C_RED, alpha=0.5, label="Anomaly (all types)")
fig.legend(handles=[p_blue, p_red], loc="lower center", ncol=2,
           fontsize=11, framealpha=0.9,
           bbox_to_anchor=(0.5, 0.01))

fig.suptitle(
    "MemStream Anomaly Types — Validation Set (7 Types @ 8%)\n"
    "RED = Anomaly. BLUE = Normal. Anomalies deviate from normal patterns.",
    fontsize=14, fontweight="bold", y=0.995
)
fig.tight_layout(rect=[0, 0.03, 1, 0.97])

path1 = RESULT_DIR / "viz_anomaly_types_overview.png"
fig.savefig(path1, dpi=150, bbox_inches="tight", facecolor=C_BG)
print(f"Saved: {path1}")
plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Feature AUC heatmap per type
# ---------------------------------------------------------------------------
from sklearn.metrics import roc_auc_score

feat_cols = ["fare_amount", "total_amount", "trip_distance", "dur_min",
             "speed_mph", "fare_per_mile", "fare_per_min"]
feat_names = {
    "fare_amount": "Fare ($)", "total_amount": "Total ($)",
    "trip_distance": "Distance (mi)", "dur_min": "Duration (min)",
    "speed_mph": "Speed (mph)", "fare_per_mile": "Fare/Mi",
    "fare_per_min": "Fare/Min",
}

auc_matrix = []
for type_id in ["1", "2", "3", "4", "5", "6", "7"]:
    type_m = type_masks.get(type_id)
    if type_m is None:
        auc_matrix.append([0.5] * len(feat_cols))
        continue
    gt_bin = type_m.astype(int)
    row = []
    for feat in feat_cols:
        vals = df_s[feat].values
        auc = roc_auc_score(gt_bin, vals)
        auc = max(auc, 1 - auc)
        row.append(auc)
    auc_matrix.append(row)

auc_matrix = np.array(auc_matrix)

fig2, ax2 = plt.subplots(figsize=(14, 8))
fig2.patch.set_facecolor(C_BG)
ax2.set_facecolor(C_BG)

im = ax2.imshow(auc_matrix, cmap="RdYlGn", vmin=0.4, vmax=1.0, aspect="auto")

ax2.set_xticks(range(len(feat_cols)))
ax2.set_xticklabels([feat_names[f] for f in feat_cols], fontsize=10, rotation=30, ha="right")
ax2.set_yticks(range(7))
type_labels = [f"T{i}: {TYPE_NAME[str(i)].replace('_', ' ').title()}" for i in range(1, 8)]
ax2.set_yticklabels(type_labels, fontsize=9)

for i in range(7):
    for j in range(len(feat_cols)):
        val = auc_matrix[i, j]
        color = "white" if val > 0.75 else "black"
        ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                 fontsize=9, fontweight="bold", color=color)

plt.colorbar(im, ax=ax2, label="AUC-ROC (higher = more discriminative)", shrink=0.8)
ax2.set_title(
    "Feature AUC-ROC per Anomaly Type\n"
    "RED = detectable (AUC > 0.7), YELLOW = marginal, GREEN = not detectable",
    fontsize=13, fontweight="bold", pad=10
)

path2 = RESULT_DIR / "viz_feature_auc_heatmap.png"
fig2.savefig(path2, dpi=150, bbox_inches="tight", facecolor=C_BG)
print(f"Saved: {path2}")
plt.close(fig2)


# ---------------------------------------------------------------------------
# Figure 3: 4 key types large plots
# ---------------------------------------------------------------------------
KEY_TYPES = ["3", "1", "4", "6"]
fig3, axes3 = plt.subplots(2, 2, figsize=(18, 16))
fig3.patch.set_facecolor(C_BG)
axes3 = axes3.flatten()

for i, type_id in enumerate(KEY_TYPES):
    ax = axes3[i]
    cfg = TYPE_CFG[type_id]
    type_m = type_masks.get(type_id)

    x_col, y_col = cfg["x"], cfg["y"]

    # Normal: blue
    x_n = df_s.loc[normal_mask, x_col].values
    y_n = df_s.loc[normal_mask, y_col].values
    ax.scatter(x_n, y_n, c=C_BLUE, alpha=0.12, s=6, rasterized=True)

    # This type: red
    if type_m is not None:
        x_t = df_s.loc[type_m, x_col].values
        y_t = df_s.loc[type_m, y_col].values
        ax.scatter(x_t, y_t, c=C_RED_DARK, alpha=0.6, s=18, rasterized=True, zorder=10)

    ax.set_xlabel(cfg["xlabel"], fontsize=11)
    ax.set_ylabel(cfg["ylabel"], fontsize=11)
    ax.set_xlim(cfg["xlim"])
    ax.set_ylim(cfg["ylim"])
    ax.set_title(cfg["title"], fontsize=12, fontweight="bold", pad=6)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_facecolor(C_BG)

    n_ano = type_m.sum() if type_m is not None else 0
    n_norm = normal_mask.sum()

    # Stats
    x_n_med = np.median(x_n)
    y_n_med = np.median(y_n)
    x_t_med = np.median(x_t) if len(x_t) > 0 else 0
    y_t_med = np.median(y_t) if len(y_t) > 0 else 0

    stats = (f"Normal median: ({x_n_med:.1f}, {y_n_med:.1f})\n"
             f"Anomaly median: ({x_t_med:.1f}, {y_t_med:.1f})\n"
             f"n_normal={n_norm:,}, n_anomaly={n_ano:,}")
    ax.text(0.03, 0.97, stats,
            transform=ax.transAxes, fontsize=8.5,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                     alpha=0.9, edgecolor="gray"))

    # Legend
    p1 = mpatches.Patch(color=C_BLUE, alpha=0.5, label="Normal")
    p2 = mpatches.Patch(color=C_RED_DARK, alpha=0.7, label="Anomaly")
    ax.legend(handles=[p1, p2], fontsize=9, loc="upper right", framealpha=0.9)

p_blue2 = mpatches.Patch(color=C_BLUE, alpha=0.5, label="Normal (BLUE)")
p_red2 = mpatches.Patch(color=C_RED_DARK, alpha=0.7, label="Anomaly (RED)")
fig3.legend(handles=[p_blue2, p_red2], loc="lower center", ncol=2,
             fontsize=12, framealpha=0.9, bbox_to_anchor=(0.5, 0.01))

fig3.suptitle(
    "Key Anomaly Signal Patterns — Duration vs Fare/Distance\n"
    "RED = Anomaly. Normal BLUE clusters follow expected patterns. Anomalies deviate.",
    fontsize=14, fontweight="bold", y=0.995
)
fig3.tight_layout(rect=[0, 0.03, 1, 0.97])

path3 = RESULT_DIR / "viz_key_signals.png"
fig3.savefig(path3, dpi=150, bbox_inches="tight", facecolor=C_BG)
print(f"Saved: {path3}")
plt.close(fig3)


# ---------------------------------------------------------------------------
# Figure 4: Per-type scatter with fare_per_min (best signal)
# ---------------------------------------------------------------------------
fig4, axes4 = plt.subplots(2, 4, figsize=(20, 10))
fig4.patch.set_facecolor(C_BG)
axes4 = axes4.flatten()

for i, type_id in enumerate(["1", "2", "3", "4", "5", "6", "7"]):
    ax = axes4[i]
    type_m = type_masks.get(type_id)
    if type_m is None:
        ax.axis("off")
        continue

    # fare_per_min vs fare_amount
    x_col, y_col = "fare_amount", "fare_per_min"

    x_n = df_s.loc[normal_mask, x_col].values
    y_n = df_s.loc[normal_mask, y_col].values
    x_t = df_s.loc[type_m, x_col].values
    y_t = df_s.loc[type_m, y_col].values

    ax.scatter(x_n, y_n, c=C_BLUE, alpha=0.1, s=4, rasterized=True)
    ax.scatter(x_t, y_t, c=C_RED, alpha=0.5, s=12, rasterized=True)

    ax.set_xlabel("Fare ($)", fontsize=8)
    ax.set_ylabel("Fare/Min", fontsize=8)
    ax.set_xlim((0, 200))
    ax.set_ylim((0, 30))
    ax.set_title(f"Type {type_id}: {TYPE_NAME[type_id].replace('_', ' ').title()}",
                 fontsize=8, fontweight="bold")
    ax.grid(True, alpha=0.2)
    ax.set_facecolor(C_BG)

    n = type_m.sum()
    y_med_n = np.median(y_n)
    y_med_t = np.median(y_t)
    ax.text(0.97, 0.03, f"n={n:,}\nNorm.med={y_med_n:.2f}\nAnom.med={y_med_t:.2f}",
            transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

axes4[7].axis("off")

fig4.suptitle(
    "Fare/Min vs Fare — Best Discriminating Feature for MemStream\n"
    "RED = Anomaly. Most types show clear separation along Fare/Min axis.",
    fontsize=13, fontweight="bold", y=0.995
)
fig4.tight_layout(rect=[0, 0, 1, 0.95])

path4 = RESULT_DIR / "viz_fare_per_min.png"
fig4.savefig(path4, dpi=150, bbox_inches="tight", facecolor=C_BG)
print(f"Saved: {path4}")
plt.close(fig4)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Validation samples: {len(df_s):,}")
print(f"Normal: {normal_mask.sum():,} ({normal_mask.sum()/len(df_s)*100:.1f}%)")
print(f"Anomaly: {anomaly_mask.sum():,} ({anomaly_mask.sum()/len(df_s)*100:.1f}%)")
print()
print(f"{'Type':<6} {'Name':<25} {'Count':<10} {'Pct':<8} {'Best Feature':<20} {'AUC'}")
print("-"*90)
for type_id in ["1", "2", "3", "4", "5", "6", "7"]:
    type_m = type_masks.get(type_id)
    if type_m is None:
        continue
    n = type_m.sum()
    pct = n / len(df_s) * 100
    name = TYPE_NAME[type_id].replace("_", " ").title()

    # Find best feature
    gt_bin = type_m.astype(int)
    best_auc = 0
    best_feat = ""
    for feat in feat_cols:
        vals = df_s[feat].values
        auc = roc_auc_score(gt_bin, vals)
        auc = max(auc, 1 - auc)
        if auc > best_auc:
            best_auc = auc
            best_feat = feat_names.get(feat, feat)

    print(f"{type_id:<6} {name:<25} {n:<10,} {pct:<8.2f}% {best_feat:<20} {best_auc:.3f}")

print(f"\nAll charts saved to: {RESULT_DIR}")
