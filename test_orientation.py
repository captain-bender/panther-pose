from ultralytics import YOLO
import cv2
import os
import argparse
from pathlib import Path
from datetime import datetime
import random
import math
import statistics


def calculate_heading_from_base_perpendicular(kp0, kp1, kp2):
    """Heading from the perpendicular to the base (kp[0]-kp[2]) pointing toward the apex (kp[1]).

    Angle convention:
    - 0° is left
    - 90° is down
    - 180° is right
    - 270° is up
    """
    x0, y0 = float(kp0[0]), float(kp0[1])
    x1, y1 = float(kp1[0]), float(kp1[1])
    x2, y2 = float(kp2[0]), float(kp2[1])

    bx, by = (x2 - x0), (y2 - y0)
    if (bx * bx + by * by) < 1e-9:
        return None

    nx, ny = (-by, bx)
    mx, my = (x0 + x2) * 0.5, (y0 + y2) * 0.5
    ax, ay = (x1 - mx), (y1 - my)
    if (nx * ax + ny * ay) < 0:
        nx, ny = -nx, -ny

    ang = math.degrees(math.atan2(ny, -nx))
    return (ang + 360.0) % 360.0


def _angle_diff_signed_deg(pred_deg: float, gt_deg: float) -> float:
    """Signed smallest difference pred-gt in degrees in [-180, 180]."""
    return (float(pred_deg) - float(gt_deg) + 180.0) % 360.0 - 180.0


def _percentile(sorted_vals: list[float], p: float) -> float:
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


def _print_heading_error_summary(signed_diffs: list[float]):
    if not signed_diffs:
        print("\nNo GT heading comparisons collected.")
        return

    abs_diffs = [abs(d) for d in signed_diffs]
    abs_sorted = sorted(abs_diffs)
    signed_sorted = sorted(signed_diffs)

    n = len(abs_diffs)
    mean_abs = statistics.mean(abs_diffs)
    median_abs = statistics.median(abs_diffs)
    std_abs = statistics.pstdev(abs_diffs) if n > 1 else 0.0
    p90_abs = _percentile(abs_sorted, 90)
    p95_abs = _percentile(abs_sorted, 95)
    mn_abs = abs_sorted[0]
    mx_abs = abs_sorted[-1]

    mean_signed = statistics.mean(signed_diffs)
    median_signed = statistics.median(signed_diffs)

    def frac_within(th: float) -> float:
        return 100.0 * sum(1 for v in abs_diffs if v <= th) / n

    within_5 = frac_within(5.0)
    within_10 = frac_within(10.0)

    print("\n" + "=" * 72)
    print("HEADING ERROR SUMMARY (perpendicular-to-base; 0=left, 90=down)")
    print("=" * 72)
    print(f"Samples (pose matches): {n}")
    print("\nMetric                         Value")
    print("--------------------------------------")
    print(f"Mean abs error (°)              {mean_abs:8.2f}")
    print(f"Median abs error (°)            {median_abs:8.2f}")
    print(f"Std abs error (°)               {std_abs:8.2f}")
    print(f"P90 abs error (°)               {p90_abs:8.2f}")
    print(f"P95 abs error (°)               {p95_abs:8.2f}")
    print(f"Min / Max abs error (°)         {mn_abs:8.2f} / {mx_abs:8.2f}")
    print(f"Mean signed error (°)           {mean_signed:8.2f}")
    print(f"Median signed error (°)         {median_signed:8.2f}")
    print(f"Within 5° / 10° (%)              {within_5:7.1f} / {within_10:7.1f}")
    print("=" * 72)


def _find_label_file(dataset_root: Path, split: str, img_name: str) -> Path | None:
    label_dir = dataset_root / split / 'labels'
    if not label_dir.exists():
        return None
    stem = Path(img_name).stem
    candidates = list(label_dir.glob(f"{stem}*.txt"))
    return candidates[0] if candidates else None


def _load_gt_keypoints_pixels(dataset_root: Path, split: str, img_name: str):
    """Load YOLO pose labels for a single image.

    Returns (poses_kpts, (h, w))
    poses_kpts: list of list of (x_px, y_px, v)
    """
    img_path = dataset_root / split / 'images' / img_name
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None
    h, w = img.shape[:2]

    label_file = _find_label_file(dataset_root, split, img_name)
    if label_file is None:
        return None, (h, w)

    poses = []
    try:
        with open(label_file, 'r') as f:
            for line in f:
                vals = [float(x) for x in line.strip().split()]
                if len(vals) < 5:
                    continue
                kpt_vals = vals[5:]
                kpts = []
                for i in range(0, len(kpt_vals), 3):
                    if i + 2 >= len(kpt_vals):
                        break
                    x_n, y_n, v = kpt_vals[i], kpt_vals[i + 1], kpt_vals[i + 2]
                    kpts.append((x_n * w, y_n * h, v))
                if kpts:
                    poses.append(kpts)
    except Exception as e:
        print(f"   Warning: failed to load GT labels for {img_name}: {e}")
        return None, (h, w)

    return poses, (h, w)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='runs/train/yolo11n-panther_v1-pose-v1/weights/best.pt')
    parser.add_argument('--dataset', type=str, default='./dataset/version-1')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--count', type=int, default=2, help='Number of images to run')
    parser.add_argument('--tag', type=str, default='', help='Optional tag to include in tests output path')
    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)

    test_images_dir = Path(args.dataset) / args.split / 'images'

    print("Testing on individual images...")

    all_images = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not all_images:
        print(f"No images found in {test_images_dir}")
        return
    k = min(args.count, len(all_images))
    test_images = random.sample(all_images, k=k)

    heading_diffs_signed: list[float] = []

    for img_name in test_images:
        img_path = Path(test_images_dir) / img_name

        gt_poses, _ = _load_gt_keypoints_pixels(Path(args.dataset), args.split, img_name)
        if gt_poses:
            print(f"   Ground truth: {len(gt_poses)} pose(s) found")

        print(f"\n Processing: {img_name}")

        # Run inference
        results = model(str(img_path))

        # Access results
        for result in results:
            # Print detection (box) confidence(s) first, if available.
            try:
                if getattr(result, 'boxes', None) is not None and getattr(result.boxes, 'conf', None) is not None:
                    import torch
                    box_conf = result.boxes.conf
                    box_conf_np = box_conf.detach().cpu().numpy() if isinstance(box_conf, torch.Tensor) else box_conf
                    box_conf_list = [float(x) for x in box_conf_np.reshape(-1)]
                    if box_conf_list:
                        mean_box_conf = statistics.mean(box_conf_list)
                        min_box_conf = min(box_conf_list)
                        max_box_conf = max(box_conf_list)
                        print(f"   Detections: {len(box_conf_list)} | box conf mean={mean_box_conf:.3f}, min={min_box_conf:.3f}, max={max_box_conf:.3f}")
                        for di, dc in enumerate(box_conf_list):
                            print(f"     det[{di}] box conf: {dc:.3f}")
                    else:
                        print("   Detections: 0")
                else:
                    print("   Detections: (boxes/conf not available)")
            except Exception as e:
                print(f"   Warning: failed to print box confidences: {e}")

            # Plot first so we can access the palette used for keypoints
            annotated_frame = result.plot()

            # Get keypoints
            if result.keypoints is not None:
                xy = result.keypoints.xy  # x and y coordinates
                conf = result.keypoints.conf  # confidence scores
                print(f"   Detected {len(xy)} pose(s)")
                print(f"   Keypoints shape: {xy.shape}")

                try:
                    import torch
                    xy_np = xy.cpu().numpy() if isinstance(xy, torch.Tensor) else xy
                    conf_arr = None
                    if conf is not None:
                        conf_arr = conf
                        if isinstance(conf_arr, torch.Tensor):
                            conf_arr = conf_arr.squeeze(-1).cpu().numpy()

                    # Determine per-keypoint colors matching Ultralytics plotting
                    kp_colors = None
                    num_kpts = xy_np.shape[1]
                    try:
                        from ultralytics.utils.plotting import kpt_color as KP_COLORS, colors as ucolors
                        if KP_COLORS and len(KP_COLORS) > 0:
                            # Use provided keypoint palette; pad/trim to match current number of keypoints
                            kp_colors = [tuple(int(c) for c in KP_COLORS[i % len(KP_COLORS)]) for i in range(num_kpts)]
                        else:
                            kp_colors = [ucolors(i, True) for i in range(num_kpts)]
                    except Exception:
                        try:
                            from ultralytics.utils.plotting import colors as ucolors
                            kp_colors = [ucolors(i, True) for i in range(num_kpts)]
                        except Exception:
                            kp_colors = None

                    for i in range(xy_np.shape[0]):
                        print(f"   Pose {i} keypoints:")
                        if xy_np.shape[1] >= 3:
                            heading = calculate_heading_from_base_perpendicular(xy_np[i, 0], xy_np[i, 1], xy_np[i, 2])
                            if heading is not None:
                                print(f"     Heading: {heading:.1f}° (0=left, 90=down)")

                                # Compare against GT (same pose index) if available
                                if gt_poses and i < len(gt_poses) and len(gt_poses[i]) >= 3:
                                    g0, g1, g2 = gt_poses[i][0], gt_poses[i][1], gt_poses[i][2]
                                    if (g0[2] > 0) and (g1[2] > 0) and (g2[2] > 0):
                                        gt_heading = calculate_heading_from_base_perpendicular(g0, g1, g2)
                                        if gt_heading is not None:
                                            diff = _angle_diff_signed_deg(heading, gt_heading)
                                            heading_diffs_signed.append(float(diff))
                                            print(f"     GT heading: {gt_heading:.1f}° | diff: {diff:+.1f}° (abs {abs(diff):.1f}°)")
                        for k in range(xy_np.shape[1]):
                            xk, yk = float(xy_np[i, k, 0]), float(xy_np[i, k, 1])
                            color_str = ""
                            if kp_colors is not None and k < len(kp_colors):
                                col = kp_colors[k]
                                # Ensure tuple of ints as BGR
                                try:
                                    bgr = tuple(int(c) for c in col)
                                except Exception:
                                    bgr = col
                                color_str = f", color(BGR)={bgr}"
                            if conf_arr is not None:
                                ck = float(conf_arr[i, k])
                                print(f"     kp[{k}]: x={xk:.1f}, y={yk:.1f}, conf={ck:.3f}{color_str}")
                            else:
                                print(f"     kp[{k}]: x={xk:.1f}, y={yk:.1f}{color_str}")
                except Exception as e:
                    print(f"   Warning: failed to print per-keypoint confidences/colors: {e}")

            # Save results under runs/test-orientation/<model-name>/
            model_name = Path(args.model).parent.parent.name if 'weights' in args.model else Path(args.model).stem
            out_dir = Path('runs') / 'test-orientation' / model_name
            out_dir.mkdir(parents=True, exist_ok=True)

            output_path = out_dir / f"{img_name}"
            cv2.imwrite(str(output_path), annotated_frame)
            print(f" Saved result: {output_path}")

    print("\n Individual image testing complete!")
    _print_heading_error_summary(heading_diffs_signed)


if __name__ == '__main__':
    main()