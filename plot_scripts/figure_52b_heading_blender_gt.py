"""
Figure B for Section 5.2.2 — Heading accuracy against Blender GT
Per-prediction index plot with:
  - ±1 and ±2 std bands
  - Mean and median lines
  - Colour-coded individual points by error band
  - 3° and 5° threshold references

Path to update:
  GT_JSON : inference_with_ground_truth_final.json
"""
import json, numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# =============================================================================
# STYLE CONFIG
# =============================================================================
FONT_FAMILY      = "serif"
FONT_SIZE_LABEL  = 11
FONT_SIZE_TICK   = 9
FONT_SIZE_ANNOT  = 8.5
FONT_SIZE_LEGEND = 8.5

COLOR_LINE       = "#4878CF"
COLOR_MEAN       = "#2E7D32"
COLOR_MED        = "#E8800A"
COLOR_EXCELLENT  = "#2E7D32"
COLOR_VERYGOOD   = "#1565C0"

FIG_WIDTH        = 8.0
FIG_HEIGHT       = 4.2
DPI              = 300

# Path — update for local execution
GT_JSON = Path("../runs/test-orientation/yolo11n-panther_v1-pose-v1/inference_with_ground_truth_final.json")

# =============================================================================
# DATA
# =============================================================================
d          = json.loads(GT_JSON.read_text())
preds      = d["predictions"]
errors     = np.array([p["angular_error_deg"] for p in preds])
n          = len(errors)
idx        = np.arange(1, n + 1)
mean_err   = errors.mean()
median_err = np.median(errors)
std_err    = errors.std()

def pt_color(e):
    if e <= 3.0:   return COLOR_EXCELLENT
    elif e <= 5.0: return COLOR_VERYGOOD
    else:          return "#B71C1C"

pt_colors = [pt_color(e) for e in errors]

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

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

# ±2 std band
ax.fill_between(idx,
                np.maximum(0, mean_err - 2*std_err),
                mean_err + 2*std_err,
                color=COLOR_LINE, alpha=0.07, label="±2 Std band")

# ±1 std band
ax.fill_between(idx,
                np.maximum(0, mean_err - std_err),
                mean_err + std_err,
                color=COLOR_LINE, alpha=0.18, label="±1 Std band")

# Error line
ax.plot(idx, errors, color=COLOR_LINE, linewidth=0.9, alpha=0.6)

# Colour-coded points
for i, (e, col) in enumerate(zip(errors, pt_colors)):
    ax.scatter(i + 1, e, color=col, s=22, zorder=5)

# Mean and median
ax.axhline(mean_err,   color=COLOR_MEAN, linewidth=1.5,
           linestyle="--", label=f"Mean = {mean_err:.2f}°")
ax.axhline(median_err, color=COLOR_MED,  linewidth=1.5,
           linestyle="-.", label=f"Median = {median_err:.2f}°")

# Threshold references
ax.axhline(3.0, color=COLOR_EXCELLENT, linestyle=":", linewidth=1.0)
ax.text(n + 0.3, 3.0, "3°", color=COLOR_EXCELLENT, fontsize=FONT_SIZE_ANNOT, va="center")
ax.axhline(5.0, color=COLOR_VERYGOOD,  linestyle=":", linewidth=1.0)
ax.text(n + 0.3, 5.0, "5°", color=COLOR_VERYGOOD,  fontsize=FONT_SIZE_ANNOT, va="center")

# Stats box
# ax.text(0.97, 0.97,
#         f"Std = {std_err:.2f}°\nMin = {errors.min():.2f}°\nMax = {errors.max():.2f}°",
#         transform=ax.transAxes, fontsize=FONT_SIZE_ANNOT, va="top", ha="right",
#         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="grey", alpha=0.85))

# Combined legend
pt_legend = [
    Line2D([0],[0],marker="o",color="w",markerfacecolor=COLOR_EXCELLENT,
           markersize=6, label="Excellent (< 3°)"),
    Line2D([0],[0],marker="o",color="w",markerfacecolor=COLOR_VERYGOOD,
           markersize=6, label="Very good (3°–5°)"),
]
h, l = ax.get_legend_handles_labels()
ax.legend(h + pt_legend, l + [e.get_label() for e in pt_legend],
          loc="upper left", fontsize=FONT_SIZE_LEGEND, framealpha=0.85, ncol=2)

ax.set_xlabel("Prediction Index")
ax.set_ylabel("Heading Error (°)")
ax.set_xlim(1, n)
ax.set_ylim(bottom=0)

fig.tight_layout()
out = Path("figure_52b_heading_blender_gt.pdf")
fig.savefig(out, dpi=DPI, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()
