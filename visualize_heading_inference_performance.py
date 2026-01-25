import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Load the JSON file
json_path = Path("runs/test-orientation/yolo11n-panther_v1-pose-v1/inference_with_ground_truth_final.json")

with open(json_path, 'r') as f:
    data = json.load(f)

# Extract predictions
predictions = data['predictions']
metadata = data['metadata']['statistics']

# Create DataFrame for easier analysis
df = pd.DataFrame(predictions)
df['angular_error_deg'] = df['angular_error_deg'].astype(float)
df['heading_confidence'] = df['heading_confidence'].astype(float)

# Create output directory if it doesn't exist
output_dir = Path("runs/test-orientation/yolo11n-panther_v1-pose-v1/performance_metrics")
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Angular Error Distribution (Histogram)
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df['angular_error_deg'], bins=15, color='steelblue', edgecolor='black', alpha=0.7)
ax.axvline(metadata['mean_angular_error_deg'], color='red', linestyle='--', linewidth=2, label=f"Mean: {metadata['mean_angular_error_deg']:.2f}°")
ax.axvline(metadata['median_angular_error_deg'], color='green', linestyle='--', linewidth=2, label=f"Median: {metadata['median_angular_error_deg']:.2f}°")
ax.set_xlabel('Angular Error (degrees)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Angular Error Distribution', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / '01_angular_error_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_angular_error_distribution.png")
plt.close()

# 2. Quality Distribution (Donut Chart)
fig, ax = plt.subplots(figsize=(10, 8))
quality_dist = metadata['quality_distribution']
labels = [f"Excellent (0-3°)\n{quality_dist['excellent_0_to_3deg']} predictions", 
          f"Very Good (3-5°)\n{quality_dist['very_good_3_to_5deg']} predictions"]
sizes = [
    quality_dist['excellent_0_to_3deg'],
    quality_dist['very_good_3_to_5deg']
]
colors = ['#2ecc71', '#3498db']

wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
                                    colors=colors, textprops={'fontsize': 11})
# Draw circle for donut
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
ax.add_artist(centre_circle)

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(12)

ax.set_title('Quality Distribution', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(output_dir / '02_quality_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_quality_distribution.png")
plt.close()

# 3. Angular Error Over Predictions
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(range(len(df)), df['angular_error_deg'].values, marker='o', linestyle='-', 
        color='steelblue', markersize=5, alpha=0.7)
ax.axhline(metadata['mean_angular_error_deg'], color='red', linestyle='--', alpha=0.7, label='Mean', linewidth=2)
ax.fill_between(range(len(df)), 0, df['angular_error_deg'].values, alpha=0.2, color='steelblue')
ax.set_xlabel('Prediction Index', fontsize=12)
ax.set_ylabel('Angular Error (degrees)', fontsize=12)
ax.set_title('Angular Error Over All Predictions', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / '03_angular_error_over_predictions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_angular_error_over_predictions.png")
plt.close()

# 4. Inferred vs Ground Truth Heading
fig, ax = plt.subplots(figsize=(10, 10))
inferred = df['inferred_heading_robot_coords_deg'].values
ground_truth = df['ground_truth_heading_deg'].values
ax.scatter(ground_truth, inferred, alpha=0.6, s=100, edgecolors='black', color='steelblue')
# Add perfect prediction line
min_val = min(ground_truth.min(), inferred.min())
max_val = max(ground_truth.max(), inferred.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('Ground Truth Heading (degrees)', fontsize=12)
ax.set_ylabel('Inferred Heading (degrees)', fontsize=12)
ax.set_title('Inferred vs Ground Truth Heading', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig(output_dir / '04_inferred_vs_ground_truth.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 04_inferred_vs_ground_truth.png")
plt.close()

# 5. Statistics Summary
fig, ax = plt.subplots(figsize=(10, 8))
ax.axis('off')
stats_text = f"""
HEADING ESTIMATION PERFORMANCE SUMMARY

Total Predictions: {metadata['total_predictions']}

Mean Angular Error: {metadata['mean_angular_error_deg']:.2f}°
Median Angular Error: {metadata['median_angular_error_deg']:.2f}°
Std Deviation: {metadata['std_deviation_deg']:.2f}°

Min Error: {metadata['min_angular_error_deg']:.2f}°
Max Error: {metadata['max_angular_error_deg']:.2f}°

Quality Breakdown:
  • Excellent (0-3°): {quality_dist['excellent_0_to_3deg']} ({100*quality_dist['excellent_0_to_3deg']/metadata['total_predictions']:.1f}%)
  • Very Good (3-5°): {quality_dist['very_good_3_to_5deg']} ({100*quality_dist['very_good_3_to_5deg']/metadata['total_predictions']:.1f}%)
  • Good (5-10°): {quality_dist['good_5_to_10deg']} ({100*quality_dist['good_5_to_10deg']/metadata['total_predictions']:.1f}%)
  • Poor (>10°): {quality_dist['poor_above_10deg']} ({100*quality_dist['poor_above_10deg']/metadata['total_predictions']:.1f}%)
"""
ax.text(0.5, 0.5, stats_text, fontsize=12, verticalalignment='center', horizontalalignment='center',
        family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=1))
plt.tight_layout()
plt.savefig(output_dir / '05_statistics_summary.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 05_statistics_summary.png")
plt.close()

print(f"\n✓ All visualizations saved to: {output_dir}")
