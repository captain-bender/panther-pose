"""
Figure for Section 3.4.2 — KPP model training and validation loss graphs
2x2 layout: Train Box Loss, Train Pose Loss, Val Box Loss, Val Pose Loss
A vertical dotted red line marks the best epoch (epoch 57).

Path to update:
  RESULTS_CSV : results.csv from the KPP training run
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from pathlib import Path

# =============================================================================
# STYLE CONFIG
# =============================================================================
FONT_FAMILY     = "serif"
FONT_SIZE_LABEL = 10
FONT_SIZE_TICK  = 8.5
FONT_SIZE_LEGEND= 8.5
FONT_SIZE_TITLE = 10

COLOR_RAW       = "#4878CF"
COLOR_SMOOTH    = "#E8800A"
ALPHA_RAW       = 0.45
SMOOTH_WINDOW   = 5
BEST_EPOCH      = 57

FIG_WIDTH       = 9.0
FIG_HEIGHT      = 5.5
DPI             = 300

# Path — update for local execution
RESULTS_CSV = Path("../runs/train/yolo11n-panther_v1-pose-v1/results.csv")

# =============================================================================
# DATA
# =============================================================================
rows      = list(csv.DictReader(RESULTS_CSV.read_text().splitlines()))
epochs    = np.array([int(float(r["epoch"])) + 1 for r in rows])

def get(col):
    return np.array([float(r[col]) for r in rows])

def smooth(y, w=SMOOTH_WINDOW):
    return uniform_filter1d(y, size=w)

train_box  = get("train/box_loss")
val_box    = get("val/box_loss")
train_pose = get("train/pose_loss")
val_pose   = get("val/pose_loss")

# =============================================================================
# FIGURE
# =============================================================================
plt.rcParams.update({
    "font.family":     FONT_FAMILY,
    "font.size":       FONT_SIZE_LABEL,
    "axes.labelsize":  FONT_SIZE_LABEL,
    "xtick.labelsize": FONT_SIZE_TICK,
    "ytick.labelsize": FONT_SIZE_TICK,
    "legend.fontsize": FONT_SIZE_LEGEND,
})

fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))
fig.subplots_adjust(hspace=0.42, wspace=0.32)

pairs = [
    (axes[0, 0], train_box,  "Train Box Loss"),
    (axes[0, 1], train_pose, "Train Pose Loss"),
    (axes[1, 0], val_box,    "Validation Box Loss"),
    (axes[1, 1], val_pose,   "Validation Pose Loss"),
]

for ax, data, title in pairs:
    ax.plot(epochs, data, color=COLOR_RAW, linewidth=0.9,
            alpha=ALPHA_RAW, label="Loss")
    ax.plot(epochs, smooth(data), color=COLOR_SMOOTH,
            linewidth=1.6, linestyle="--", label="Smoothed")
    ax.axvline(BEST_EPOCH, color="red", linestyle=":",
               linewidth=1.0, alpha=0.7)
    ax.set_title(title, fontsize=FONT_SIZE_TITLE)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right", fontsize=FONT_SIZE_LEGEND)

out = Path("figure_342_kpp_training_losses.pdf")
fig.savefig(out, dpi=DPI, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()
