"""
Figure A for Section 5.2 — Annotation-based heading error comparison
Percentile rank x-axis with sorted error curves for all three cases:
  V1 (raw KPP), V2 bias-corrected (PCA), bolt-head

Paths to update:
  HINGE_JSON   : heading_debug.json
  BOLTHEAD_TXT : headind_debug_bolthead.txt
"""
import json, re, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# STYLE CONFIG
# =============================================================================
FONT_FAMILY     = "serif"
FONT_SIZE_LABEL = 11
FONT_SIZE_TICK  = 9
FONT_SIZE_ANNOT = 8.5
FONT_SIZE_LEGEND= 9

COLOR_V1        = "#4878CF"
COLOR_V2        = "#6ACC65"
COLOR_BH        = "#D65F5F"

FIG_WIDTH       = 7.5
FIG_HEIGHT      = 4.2
DPI             = 300

# Paths — update for local execution
HINGE_JSON   = Path("../runs/test-orientation/yolo11n-panther_v1-pose-v1/heading_debug.json")
BOLTHEAD_TXT = Path("../docs/heading_debug_bolthead.txt")

# =============================================================================
# DATA
# =============================================================================
d         = json.loads(HINGE_JSON.read_text())
records   = d["records"]
v1_errors = np.sort([r["v1_err_deg"]      for r in records])
v2_errors = np.sort([r["v2_corr_err_deg"] for r in records])
# Read with explicit encoding; fall back to latin-1 if utf-8 fails
try:
    text = BOLTHEAD_TXT.read_text(encoding="utf-8")
except UnicodeDecodeError:
    text = BOLTHEAD_TXT.read_text(encoding="latin-1")

# Match 'abs X.X°)' — degree symbol may vary, so match any non-digit char after number
bh_errors = np.sort([float(e) for e in re.findall(r'abs\s+([\d.]+)[^\d)]', text)])

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

for errors, col, lbl in [
    (v1_errors, COLOR_V1, "Raw KPP (V1)"),
    (v2_errors, COLOR_V2, "PCA refined (bias-corr.)"),
    (bh_errors, COLOR_BH, "Bolt-head"),
]:
    idx = np.linspace(0, 100, len(errors))
    ax.plot(idx, errors, color=col, linewidth=1.4, alpha=0.85, label=lbl)
    ax.fill_between(idx, 0, errors, color=col, alpha=0.08)
    ax.axhline(errors.mean(), color=col, linewidth=1.0, linestyle="--", alpha=0.7)
    ax.text(101, errors.mean(), f"  {errors.mean():.2f}°",
            color=col, fontsize=FONT_SIZE_ANNOT, va="center")

# 5° reference
ax.axhline(5.0, color="red", linestyle=":", linewidth=1.0)
ax.text(101, 5.05, "5°", color="red", fontsize=FONT_SIZE_ANNOT, va="bottom", ha="left")

ax.set_xlabel("Percentile Rank (%)")
ax.set_ylabel("Heading Error (°)")
ax.set_xlim(0, 100)
ax.set_ylim(bottom=0)
ax.legend(loc="upper left", fontsize=FONT_SIZE_LEGEND, framealpha=0.85)

fig.tight_layout()
out = Path("figure_52a_heading_annotation.pdf")
fig.savefig(out, dpi=DPI, bbox_inches="tight")
fig.savefig(out.with_suffix(".png"), dpi=DPI, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()