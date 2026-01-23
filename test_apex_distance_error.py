"""
Script to compute apex (keypoint) distance error in pixels.

Compares predicted apex position (keypoint index 1 - 'front') 
against annotated apex positions and computes pixel-space distances.
"""

from ultralytics import YOLO
import cv2
import os
import argparse
from pathlib import Path
from datetime import datetime
import math
import statistics
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

KEYPOINT_NAMES = ["back-left", "front", "back-right"]


def _read_yolo_pose_labels(label_path, img_w, img_h):
    """Read YOLO pose labels from a text file."""
    gts = []
    if not os.path.exists(label_path):
        return gts
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            # denormalize bbox to pixels
            bw, bh = w * img_w, h * img_h
            bx, by = cx * img_w, cy * img_h
            x1, y1 = bx - bw / 2, by - bh / 2
            x2, y2 = bx + bw / 2, by + bh / 2
            # keypoints triplets follow
            kpt_vals = list(map(float, parts[5:]))
            kpts = []
            for i in range(0, len(kpt_vals), 3):
                if i + 2 >= len(kpt_vals):
                    break
                kx, ky, v = kpt_vals[i], kpt_vals[i+1], kpt_vals[i+2]
                kpts.append((kx * img_w, ky * img_h, int(v)))
            gts.append({
                'cls': cls,
                'bbox': (x1, y1, x2, y2),
                'kpts': kpts,
            })
    return gts


def _iou_xyxy(a, b):
    """Calculate IoU between two bounding boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _euclidean_distance(pt1, pt2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)


def _percentile(sorted_vals, p):
    """Linear-interpolated percentile on a pre-sorted list. p in [0, 100]."""
    if not sorted_vals:
        return 0.0
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    d0 = float(sorted_vals[f]) * (c - k)
    d1 = float(sorted_vals[c]) * (k - f)
    return d0 + d1


def _print_apex_distance_summary(distances):
    """Print summary statistics for apex distance errors."""
    if not distances:
        print("\nNo apex distance comparisons collected.")
        return

    sorted_distances = sorted(distances)
    n = len(distances)
    mean_dist = statistics.mean(distances)
    median_dist = statistics.median(distances)
    std_dist = statistics.pstdev(distances) if n > 1 else 0.0
    p90_dist = _percentile(sorted_distances, 90)
    p95_dist = _percentile(sorted_distances, 95)
    min_dist = sorted_distances[0]
    max_dist = sorted_distances[-1]

    def frac_within(th):
        return 100.0 * sum(1 for v in distances if v <= th) / n

    within_5px = frac_within(5.0)
    within_10px = frac_within(10.0)
    within_20px = frac_within(20.0)

    print("\n" + "=" * 72)
    print("APEX DISTANCE ERROR SUMMARY (pixels)")
    print("=" * 72)
    print(f"Samples (pose matches): {n}")
    print("\nMetric                         Value")
    print("--------------------------------------")
    print(f"Mean distance (px)              {mean_dist:8.2f}")
    print(f"Median distance (px)            {median_dist:8.2f}")
    print(f"Std distance (px)               {std_dist:8.2f}")
    print(f"P90 distance (px)               {p90_dist:8.2f}")
    print(f"P95 distance (px)               {p95_dist:8.2f}")
    print(f"Min / Max distance (px)         {min_dist:8.2f} / {max_dist:8.2f}")
    print(f"Fraction within 5 px            {within_5px:8.2f}%")
    print(f"Fraction within 10 px           {within_10px:8.2f}%")
    print(f"Fraction within 20 px           {within_20px:8.2f}%")
    print("=" * 72)


def _plot_apex_distance_analysis(distances, debug_info, output_dir):
    """Generate individual visualization plots for apex distance analysis."""
    if not distances:
        return
    
    # 1. Histogram with mean and median lines
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(distances, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(np.mean(distances), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(distances):.2f}px')
    ax.axvline(np.median(distances), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(distances):.2f}px')
    ax.set_xlabel('Distance (pixels)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Apex Distance Errors', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plot_file = os.path.join(output_dir, '01_histogram.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_file}")
    plt.close()
    
    # 2. Box plot
    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(distances, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    ax.set_ylabel('Distance (pixels)', fontsize=12)
    ax.set_title('Box Plot of Apex Distances', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    stats_text = (f"Mean: {np.mean(distances):.2f}px\n"
                  f"Median: {np.median(distances):.2f}px\n"
                  f"Std: {np.std(distances):.2f}px\n"
                  f"Min: {np.min(distances):.2f}px\n"
                  f"Max: {np.max(distances):.2f}px")
    ax.text(1.15, np.median(distances), stats_text, fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plot_file = os.path.join(output_dir, '02_boxplot.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_file}")
    plt.close()
    
    # 3. Cumulative Distribution Function
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_dist = np.sort(distances)
    cumulative = np.arange(1, len(sorted_dist) + 1) / len(sorted_dist) * 100
    ax.plot(sorted_dist, cumulative, marker='o', linestyle='-', linewidth=2, color='darkblue', markersize=4)
    ax.axhline(95, color='red', linestyle='--', alpha=0.5, label='95%')
    ax.axhline(90, color='orange', linestyle='--', alpha=0.5, label='90%')
    ax.set_xlabel('Distance (pixels)', fontsize=12)
    ax.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax.set_title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plot_file = os.path.join(output_dir, '03_cumulative_distribution.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_file}")
    plt.close()
    
    # 4. Per-sample distance bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    sample_indices = np.arange(len(distances))
    colors = ['green' if d <= 5 else 'orange' if d <= 10 else 'red' for d in distances]
    ax.bar(sample_indices, distances, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(5, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='5px threshold')
    ax.axhline(10, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='10px threshold')
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Distance (pixels)', fontsize=12)
    ax.set_title('Apex Distance per Sample', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plot_file = os.path.join(output_dir, '04_per_sample_distances.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_file}")
    plt.close()
    
    # 5. Predicted vs GT Apex Scatter
    fig, ax = plt.subplots(figsize=(10, 10))
    gt_x = []
    gt_y = []
    pred_x = []
    pred_y = []
    
    for info in debug_info[:100]:
        gt_apex = info['gt_apex']
        pred_apex = info['pred_apex']
        gt_x.append(gt_apex[0])
        gt_y.append(gt_apex[1])
        pred_x.append(pred_apex[0])
        pred_y.append(pred_apex[1])
    
    ax.scatter(gt_x, gt_y, alpha=0.6, s=100, c='blue', label='Ground Truth', edgecolors='darkblue', linewidths=1.5)
    ax.scatter(pred_x, pred_y, alpha=0.6, s=100, c='red', marker='^', label='Predicted', edgecolors='darkred', linewidths=1.5)
    
    # Draw lines connecting matched points
    for gx, gy, px, py in zip(gt_x, gt_y, pred_x, pred_y):
        ax.plot([gx, px], [gy, py], 'k-', alpha=0.1, linewidth=0.5)
    
    ax.set_xlabel('X (pixels)', fontsize=12)
    ax.set_ylabel('Y (pixels)', fontsize=12)
    ax.set_title('Predicted vs Ground Truth Apex Positions', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    plot_file = os.path.join(output_dir, '05_apex_positions_scatter.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_file}")
    plt.close()
    
    # 6. Error statistics summary
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    within_5 = sum(1 for d in distances if d <= 5) / len(distances) * 100
    within_10 = sum(1 for d in distances if d <= 10) / len(distances) * 100
    within_20 = sum(1 for d in distances if d <= 20) / len(distances) * 100
    
    stats_summary = (
        f"APEX DISTANCE ERROR STATISTICS\n"
        f"{'=' * 40}\n\n"
        f"Total Samples: {len(distances)}\n"
        f"Mean: {np.mean(distances):.2f} px\n"
        f"Median: {np.median(distances):.2f} px\n"
        f"Std Dev: {np.std(distances):.2f} px\n"
        f"Min: {np.min(distances):.2f} px\n"
        f"Max: {np.max(distances):.2f} px\n\n"
        f"Percentiles:\n"
        f"  P50: {np.percentile(distances, 50):.2f} px\n"
        f"  P75: {np.percentile(distances, 75):.2f} px\n"
        f"  P90: {np.percentile(distances, 90):.2f} px\n"
        f"  P95: {np.percentile(distances, 95):.2f} px\n\n"
        f"Within Thresholds:\n"
        f"  ≤ 5 px:  {within_5:6.2f}%\n"
        f"  ≤ 10 px: {within_10:6.2f}%\n"
        f"  ≤ 20 px: {within_20:6.2f}%"
    )
    
    ax.text(0.5, 0.5, stats_summary, transform=ax.transAxes, fontsize=12,
            verticalalignment='center', horizontalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=1))
    
    plot_file = os.path.join(output_dir, '06_statistics_summary.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_file}")
    plt.close()


def compute_apex_distance_errors(model_path, dataset_dir, model_version="v1", count=None):
    """
    Compute apex distance errors between predictions and ground truth.
    
    Args:
        model_path: Path to trained YOLO model
        dataset_dir: Path to dataset directory
        model_version: Version identifier (e.g., 'v1', 'v2')
        count: Limit number of samples to process (None = all)
    """
    model = YOLO(model_path)
    
    dataset_splits = ["test", "valid"]
    apex_distances = []
    debug_info = []
    total_processed = 0
    
    for split in dataset_splits:
        split_dir = os.path.join(dataset_dir, split)
        img_dir = os.path.join(split_dir, "images")
        label_dir = os.path.join(split_dir, "labels")
        
        if not os.path.exists(img_dir):
            continue
        
        # Check if we've already reached the count limit
        if count is not None and total_processed >= count:
            break
            
        img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        
        # Apply count limit across all splits combined
        if count is not None:
            remaining_count = count - total_processed
            img_files = img_files[:remaining_count]
        
        print(f"\nProcessing {split} split ({len(img_files)} images)...")
        
        for img_idx, img_file in enumerate(img_files):
            total_processed += 1
            img_path = os.path.join(img_dir, img_file)
            label_path = os.path.join(label_dir, Path(img_file).stem + ".txt")
            
            # Read image
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            img_h, img_w = img.shape[:2]
            
            # Get ground truth labels
            gts = _read_yolo_pose_labels(label_path, img_w, img_h)
            
            # Get predictions
            results = model.predict(img_path, verbose=False)
            
            if not results or not results[0].keypoints:
                continue
            
            preds = results[0]
            
            # Match predictions to ground truth by bounding box IoU
            for gt in gts:
                best_match = None
                best_iou = 0.0
                
                if preds.boxes is None or preds.keypoints is None:
                    continue
                
                num_dets = len(preds.boxes)
                
                for pred_idx in range(num_dets):
                    # Get predicted bbox
                    pred_bbox_tensor = preds.boxes[pred_idx].xyxy
                    if hasattr(pred_bbox_tensor, 'cpu'):
                        pred_bbox = pred_bbox_tensor.cpu().numpy().flatten()
                    else:
                        pred_bbox = np.array(pred_bbox_tensor).flatten()
                    
                    iou = _iou_xyxy(gt['bbox'], pred_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_match = (pred_idx, preds.keypoints.xy[pred_idx])
                
                # Only consider matches with reasonable IoU
                if best_iou < 0.5:
                    continue
                
                pred_idx, pred_kpts = best_match
                pred_kpts_np = pred_kpts.cpu().numpy() if hasattr(pred_kpts, 'cpu') else pred_kpts
                
                # Extract apex keypoint (index 1 = "front")
                apex_idx = 1  # front keypoint
                
                if len(gt['kpts']) > apex_idx and len(pred_kpts_np) > apex_idx:
                    gt_apex = gt['kpts'][apex_idx][:2]  # (x, y)
                    pred_apex = pred_kpts_np[apex_idx][:2]  # (x, y)
                    
                    # Compute distance
                    distance = _euclidean_distance(gt_apex, pred_apex)
                    apex_distances.append(distance)
                    
                    debug_info.append({
                        'image': img_file,
                        'gt_apex': (float(gt_apex[0]), float(gt_apex[1])),
                        'pred_apex': (float(pred_apex[0]), float(pred_apex[1])),
                        'distance_px': float(distance),
                        'bbox_iou': float(best_iou)
                    })
    
    # Create output directory
    output_dir = f"runs/test-apex-distance/yolo11n-panther_{model_version}-pose-{model_version}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Print summary
    _print_apex_distance_summary(apex_distances)
    
    # Generate plots
    _plot_apex_distance_analysis(apex_distances, debug_info, output_dir)
    
    # Save debug info to JSON
    
    debug_file = os.path.join(output_dir, "apex_distances_debug.json")
    with open(debug_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'model_version': model_version,
            'total_samples': len(apex_distances),
            'apex_distances': debug_info
        }, f, indent=2)
    
    print(f"\nDebug info saved to {debug_file}")
    
    return apex_distances, debug_info


def main():
    parser = argparse.ArgumentParser(description="Compute apex distance errors")
    parser.add_argument("--model", default="runs/train/yolo11n-panther_v1-pose-v1/weights/best.pt",
                       help="Path to trained YOLO model")
    parser.add_argument("--dataset", default="dataset/version-1",
                       help="Path to dataset directory")
    parser.add_argument("--version", default="v1",
                       help="Model version identifier")
    parser.add_argument("--count", type=int, default=None,
                       help="Limit number of samples to process")
    
    args = parser.parse_args()
    
    print(f"Starting apex distance error computation...")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Version: {args.version}")
    if args.count:
        print(f"Processing first {args.count} samples only")
    
    apex_distances, debug_info = compute_apex_distance_errors(
        args.model,
        args.dataset,
        model_version=args.version,
        count=args.count
    )
    
    print(f"\nProcessing complete. Total comparisons: {len(apex_distances)}")


if __name__ == "__main__":
    main()
