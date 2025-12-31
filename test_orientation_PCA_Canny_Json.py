from ultralytics import YOLO
import cv2
import os
import argparse
from pathlib import Path
from datetime import datetime
import random
import math
import statistics
import json
import numpy as np


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
    parser.add_argument('--out_json', type=str, default='runs/test-orientation/yolo11n-panther_v1-pose-v1/heading_debug.json', help='Where to save per-image headings (gt, v1, v2, v2_bias_corrected).'
)
    return parser.parse_args()



def _angle_image_convention_from_vector(vx: float, vy: float) -> float:
    """
    Return angle in your convention:
    0° = left, 90° = down, 180° = right, 270° = up
    """
    ang = math.degrees(math.atan2(vy, -vx))
    return (ang + 360.0) % 360.0


def _pca_direction(points_xy: np.ndarray):
    """
    points_xy: (N,2) array of (x,y) in ROI pixel coordinates.
    Returns unit direction vector (dx,dy) for the dominant axis, or None.
    """
    if points_xy.shape[0] < 20:
        return None
    mean = points_xy.mean(axis=0)
    X = points_xy - mean
    # 2x2 covariance
    C = (X.T @ X) / max(1, (X.shape[0] - 1))
    eigvals, eigvecs = np.linalg.eigh(C)
    v = eigvecs[:, np.argmax(eigvals)]  # dominant eigenvector
    dx, dy = float(v[0]), float(v[1])
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return None
    return (dx / n, dy / n)

def _draw_poly(img, pts, color=(0, 255, 255), thickness=2):
    pts = np.asarray(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

def _draw_ray(img, x, y, angle_deg, length=120, color=(0, 255, 255), thickness=2):
    # Your angle convention: 0=left, 90=down, 180=right, 270=up
    # Convert to image dx,dy
    rad = math.radians(angle_deg)
    # In your convention, angle is atan2(vy, -vx) -> so invert accordingly:
    dx = -math.cos(rad)
    dy =  math.sin(rad)
    x2 = int(round(x + dx * length))
    y2 = int(round(y + dy * length))
    cv2.line(img, (int(round(x)), int(round(y))), (x2, y2), color, thickness, cv2.LINE_AA)

def calculate_heading_from_bracket(
    img_bgr: np.ndarray,
    kp0, kp1, kp2,
    bracket_thickness_px: float = 2.5,
    bracket_length_px: float = 60.0,
    offset_from_base_px: float = 2.0,
    # ROI tuning:
    roi_length_scale: float = 1.25,     # along the base direction
    roi_width_scale: float = 8.0,       # across the base direction (in thickness units)
    dark_thresh: int = 70,              # grayscale threshold for black bracket (tune)
    debug: bool = False):
    """
    Estimate heading using the black bracket inside a ROI derived from kp0-kp2.

    Returns:
      heading_deg (float) in your convention, or None if it fails.
      (optionally) a debug dict with ROI and masks.
    """
    x0, y0 = float(kp0[0]), float(kp0[1])
    x1, y1 = float(kp1[0]), float(kp1[1])
    x2, y2 = float(kp2[0]), float(kp2[1])

    bx, by = (x2 - x0), (y2 - y0)
    L = math.hypot(bx, by)
    if L < 1e-6:
        return (None, {}) if debug else None

    # Unit base direction
    ux, uy = bx / L, by / L

    # Perpendicular (two-sided). We'll pick orientation later (toward kp1).
    px, py = -uy, ux

    # Center around base midpoint. If bracket is slightly offset from the base line,
    # move ROI center by 'offset_from_base_px' along the perpendicular toward kp1 side.
    mx, my = (x0 + x2) * 0.5, (y0 + y2) * 0.5
    ax, ay = (x1 - mx), (y1 - my)
    # Choose perp direction that points toward apex (kp1)
    if (px * ax + py * ay) < 0:
        px, py = -px, -py

    # Shift ROI center toward bracket if needed (typically bracket is close to the base)
    cx, cy = mx + px * offset_from_base_px, my + py * offset_from_base_px

    # Define ROI size
    roi_len = max(bracket_length_px, L) * roi_length_scale
    roi_half_len = roi_len * 0.5
    roi_half_w = max(6.0, bracket_thickness_px * roi_width_scale) * 0.5

    # Build rotated rectangle corners in image coords
    # Local coords: s along base (u), t along perp (p)
    corners_local = np.array([
        [-roi_half_len, -roi_half_w],
        [ roi_half_len, -roi_half_w],
        [ roi_half_len,  roi_half_w],
        [-roi_half_len,  roi_half_w],
    ], dtype=np.float32)

    # Map local (s,t) -> image: c + s*u + t*p
    u = np.array([ux, uy], dtype=np.float32)
    p = np.array([px, py], dtype=np.float32)
    c = np.array([cx, cy], dtype=np.float32)
    corners_img = (c + corners_local[:, 0:1] * u + corners_local[:, 1:2] * p).astype(np.float32)

    # Warp ROI to an axis-aligned patch: width = roi_len, height = 2*roi_half_w
    out_w = int(max(40, round(roi_len)))
    out_h = int(max(20, round(2.0 * roi_half_w)))

    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners_img, dst)
    roi = cv2.warpPerspective(img_bgr, M, (out_w, out_h), flags=cv2.INTER_LINEAR)

    # Segment dark pixels (black bracket) in ROI
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # A simple threshold; robustify later with adaptive/HSV if needed
    mask = (gray < dark_thresh).astype(np.uint8) * 255

    # Morphology: connect thick line segments, remove pepper noise
    k_close = max(3, int(round(bracket_thickness_px * 2)))
    k_open  = max(3, int(round(bracket_thickness_px * 1.5)))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (k_close, k_close))
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (k_open, k_open))
    mask2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel_open)

    ys, xs = np.where(mask2 > 0)
    if xs.size < 30:
        return ((None, {"roi": roi, "mask": mask2}) if debug else None)

    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    d = _pca_direction(pts)
    if d is None:
        return ((None, {"roi": roi, "mask": mask2}) if debug else None)

    # PCA direction is bracket direction in ROI coordinates.
    # Convert that direction back to image coordinates.
    # For warps: a direction in ROI corresponds to applying inverse-homography to two points and subtracting.
    dx_roi, dy_roi = d
    p0_roi = np.array([[out_w * 0.5, out_h * 0.5, 1.0]], dtype=np.float64).T
    p1_roi = np.array([[out_w * 0.5 + dx_roi * 50.0, out_h * 0.5 + dy_roi * 50.0, 1.0]], dtype=np.float64).T

    Minv = np.linalg.inv(M)
    p0_img = Minv @ p0_roi
    p1_img = Minv @ p1_roi
    p0_img /= p0_img[2, 0]
    p1_img /= p1_img[2, 0]
    vx = float(p1_img[0, 0] - p0_img[0, 0])
    vy = float(p1_img[1, 0] - p0_img[1, 0])
    nv = math.hypot(vx, vy)
    if nv < 1e-9:
        return ((None, {"roi": roi, "mask": mask2}) if debug else None)
    vx, vy = vx / nv, vy / nv

    # Bracket is parallel to base. Heading is perpendicular to bracket.
    hx, hy = -vy, vx

    # Disambiguate heading direction using kp1 (same idea as your v1)
    if (hx * ax + hy * ay) < 0:
        hx, hy = -hx, -hy

    heading = _angle_image_convention_from_vector(hx, hy)

    if debug:
        dbg = {"roi": roi, "mask": mask2, "corners_img": corners_img, "M": M, "heading": heading}
        return heading, dbg
    return heading


def _safe_float(x):
    """Convert tensor/np scalars/arrays/tuples to float if possible."""
    try:
        if x is None:
            return None
        if isinstance(x, (tuple, list)) and len(x) == 1:
            x = x[0]
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return None

def _draw_v2_overlay(img_bgr, dbg, save_path: Path):
    if not dbg or "corners_img" not in dbg:
        return
    out = img_bgr.copy()
    corners = dbg["corners_img"].astype(int)

    # ROI quad
    cv2.polylines(out, [corners.reshape(-1, 1, 2)], True, (0, 255, 255), 2)

    # If we have heading, draw it from ROI center
    if "heading" in dbg and dbg["heading"] is not None:
        # use ROI quad center as anchor
        cx = int(corners[:, 0].mean())
        cy = int(corners[:, 1].mean())

        ang = math.radians(float(dbg["heading"]))
        # invert your convention back to image dx,dy:
        # heading convention: 0=left,90=down => vector (vx,vy) ~ (-cos, sin)
        vx = -math.cos(ang)
        vy =  math.sin(ang)

        L = 80
        x2 = int(cx + vx * L)
        y2 = int(cy + vy * L)
        cv2.arrowedLine(out, (cx, cy), (x2, y2), (0, 255, 255), 2, tipLength=0.2)

    cv2.imwrite(str(save_path), out)

def _eigen_ratio_line_score(points_xy: np.ndarray) -> float:
    """How line-like is the point cloud? Higher is more line-ish."""
    if points_xy.shape[0] < 20:
        return 0.0
    mean = points_xy.mean(axis=0)
    X = points_xy - mean
    C = (X.T @ X) / max(1, (X.shape[0] - 1))
    eigvals, _ = np.linalg.eigh(C)
    l1, l2 = float(np.max(eigvals)), float(np.min(eigvals))
    return l1 / (l2 + 1e-9)

def _mask_from_edges(gray: np.ndarray, canny1=40, canny2=120) -> np.ndarray:
    edges = cv2.Canny(gray, canny1, canny2)
    # connect broken segments a bit
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=1)
    return edges

def _wrap_360(deg: float) -> float:
    return (float(deg) + 360.0) % 360.0


def estimate_bias_circular_deg(signed_diffs_deg: list[float]) -> float:
    """
    Estimate a constant angular bias (in degrees) from signed diffs (pred-gt in [-180,180]).
    Uses circular mean so wrap-around is handled correctly.

    Returns: bias_deg such that corrected_pred = pred - bias_deg
    """
    if not signed_diffs_deg:
        return 0.0

    # Convert diffs to radians on the circle
    ang = [math.radians(d) for d in signed_diffs_deg]
    s = sum(math.sin(a) for a in ang) / len(ang)
    c = sum(math.cos(a) for a in ang) / len(ang)

    # Circular mean angle
    bias_rad = math.atan2(s, c)
    bias_deg = math.degrees(bias_rad)

    # Keep it in [-180,180] (not strictly necessary, but nice)
    bias_deg = (bias_deg + 180.0) % 360.0 - 180.0
    return bias_deg


def apply_bias_correction_deg(pred_heading_deg: float, bias_deg: float) -> float:
    """
    If signed_diff = pred - gt, and bias ≈ mean(signed_diff),
    then corrected_pred = pred - bias.
    """
    return _wrap_360(pred_heading_deg - bias_deg)

def _wrap_angle_0_360_deg(a: float) -> float:
    """Wrap angle to [0, 360)."""
    return float(a % 360.0)

def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def main():
    args = parse_args()

    model = YOLO(args.model)

    test_images_dir = Path(args.dataset) / args.split / 'images'

    heading_diffs_signed_v1: list[float] = []
    heading_diffs_signed_v2: list[float] = []
    heading_records = []  # we will dump this to JSON later

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

        img_bgr = None
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"   Warning: could not read image: {img_path}")
            continue

        # Save results under runs/test-orientation/<model-name>/
        model_name = Path(args.model).parent.parent.name if 'weights' in args.model else Path(args.model).stem
        out_dir = Path('runs') / 'test-orientation' / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

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
                            # heading = calculate_heading_from_base_perpendicular(xy_np[i, 0], xy_np[i, 1], xy_np[i, 2])
                            # You already have annotated_frame, but you should load the original image for ROI work:
                            # img_bgr = cv2.imread(str(img_path))

                            heading_v2 = None
                            dbg = None


                            heading_v1 = calculate_heading_from_base_perpendicular(xy_np[i, 0], xy_np[i, 1], xy_np[i, 2])  # existing :contentReference[oaicite:3]{index=3}
                            out = calculate_heading_from_bracket(
                                img_bgr,
                                xy_np[i, 0], xy_np[i, 1], xy_np[i, 2],
                                bracket_thickness_px=2.5,
                                bracket_length_px=60.0,
                                offset_from_base_px=3.0,
                                dark_thresh=80,
                                roi_width_scale=5.0,
                                roi_length_scale=1.10,
                                debug=False
                            )

                            if isinstance(out, tuple):
                                heading_v2, dbg = out
                            else:
                                heading_v2 = out
                            
                            # Always produce a debug overlay image for v2 (success or fail)
                            overlay = img_bgr.copy()

                            if dbg:
                                cv2.imwrite(str(out_dir / f"{img_name}_roi.jpg"), dbg.get("roi"))
                                cv2.imwrite(str(out_dir / f"{img_name}_mask.png"), dbg.get("mask"))
                                _draw_v2_overlay(img_bgr, dbg, out_dir / f"{img_name}_v2_overlay.jpg")


                            # Draw v1 ray (optional) and v2 ray (if available)
                            mx = float((xy_np[i, 0, 0] + xy_np[i, 2, 0]) * 0.5)
                            my = float((xy_np[i, 0, 1] + xy_np[i, 2, 1]) * 0.5)

                            if heading_v2 is not None:
                                _draw_ray(overlay, mx, my, heading_v2, color=(0, 255, 255), thickness=2)

                            # Save v2 overlay + (optional) roi/mask tiles
                            debug_dir = out_dir / "v2_debug"
                            debug_dir.mkdir(parents=True, exist_ok=True)

                            cv2.imwrite(str(debug_dir / f"{img_name.replace('.jpg','')}_v2_overlay.jpg"), overlay)

                            if dbg is not None and dbg.get("roi") is not None:
                                cv2.imwrite(str(debug_dir / f"{img_name.replace('.jpg','')}_roi.jpg"), dbg["roi"])
                            if dbg is not None and dbg.get("mask") is not None:
                                cv2.imwrite(str(debug_dir / f"{img_name.replace('.jpg','')}_mask.png"), dbg["mask"])

                            print(f"     Heading v1: {heading_v1:.1f}° (kp base ⟂)")
                            if heading_v2 is not None:
                                print(f"     Heading v2: {heading_v2:.1f}° (bracket PCA ⟂)")
                            else:
                                print("     Heading v2: (failed; fallback to v1)")
                                if dbg is not None:
                                    # Print *something* useful every time it fails
                                    roi = dbg.get("roi", None)
                                    mask = dbg.get("mask", None)
                                    roi_nz = int(np.count_nonzero(mask)) if mask is not None else -1
                                    print(f"       v2 debug: mask_nonzero={roi_nz}, roi_shape={None if roi is None else roi.shape}")

                            # Compare against GT (same pose index) if available
                            if gt_poses and i < len(gt_poses) and len(gt_poses[i]) >= 3:
                                g0, g1, g2 = gt_poses[i][0], gt_poses[i][1], gt_poses[i][2]
                                if (g0[2] > 0) and (g1[2] > 0) and (g2[2] > 0):
                                    gt_heading_v1 = calculate_heading_from_base_perpendicular(g0, g1, g2)

                                    # V1 diff vs GT
                                    if gt_heading_v1 is not None and heading_v1 is not None:
                                        diff1 = _angle_diff_signed_deg(heading_v1, gt_heading_v1)
                                        heading_diffs_signed_v1.append(float(diff1))
                                        print(f"     GT v1: {gt_heading_v1:.1f}° | v1 diff: {diff1:+.1f}° (abs {abs(diff1):.1f}°)")

                                    # V2 diff vs SAME GT (triangle-based GT)
                                    if gt_heading_v1 is not None and heading_v2 is not None:
                                        diff2 = _angle_diff_signed_deg(heading_v2, gt_heading_v1)
                                        heading_diffs_signed_v2.append(float(diff2))
                                        print(f"     v2 diff vs GT v1: {diff2:+.1f}° (abs {abs(diff2):.1f}°)")
                                    
                                    # --- store per-image/per-pose record for JSON ---
                                    rec = {
                                        "image": str(img_path),
                                        "image_name": str(img_name),
                                        "pose_index": int(i),
                                        "gt_deg": float(gt_heading_v1) if gt_heading_v1 is not None else None,
                                        "v1_deg": float(heading_v1) if heading_v1 is not None else None,
                                        "v2_deg": float(heading_v2) if heading_v2 is not None else None,
                                    }
                                    # Optional: store raw signed errors (before bias correction)
                                    if gt_heading_v1 is not None and heading_v1 is not None:
                                        rec["v1_err_deg"] = float(_angle_diff_signed_deg(heading_v1, gt_heading_v1))
                                    else:
                                        rec["v1_err_deg"] = None

                                    if gt_heading_v1 is not None and heading_v2 is not None:
                                        rec["v2_err_deg"] = float(_angle_diff_signed_deg(heading_v2, gt_heading_v1))
                                    else:
                                        rec["v2_err_deg"] = None

                                    heading_records.append(rec)

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
                                ck = _safe_float(conf_arr[i, k]) if conf_arr is not None else None
                                if ck is not None:
                                    print(f"     kp[{k}]: x={xk:.1f}, y={yk:.1f}, conf={ck:.3f}{color_str}")
                                else:
                                    print(f"     kp[{k}]: x={xk:.1f}, y={yk:.1f}{color_str}")

                            
                except Exception as e:
                    print(f"   Warning: failed to print per-keypoint confidences/colors: {e}")

            

            output_path = out_dir / f"{img_name}"
            cv2.imwrite(str(output_path), annotated_frame)
            print(f" Saved result: {output_path}")

    print("\n Individual image testing complete!")

    print("\nV1 summary:")
    _print_heading_error_summary(heading_diffs_signed_v1)

    print("\nV2 summary:")
    _print_heading_error_summary(heading_diffs_signed_v2)

    # After processing all images:
    bias_v2 = estimate_bias_circular_deg(heading_diffs_signed_v2)
    print(f"\nEstimated V2 heading bias (pred-gt): {bias_v2:+.3f}°")

    heading_diffs_signed_v2_corrected = []

    for d in heading_diffs_signed_v2:
        # If corrected_pred = pred - bias, then corrected_diff = (pred - bias) - gt = d - bias
        heading_diffs_signed_v2_corrected.append(d - bias_v2)
    
    for rec in heading_records:
        gt = rec.get("gt_deg", None)
        v2 = rec.get("v2_deg", None)

        if bias_v2 is None or gt is None or v2 is None:
            rec["v2_bias_corrected_deg"] = None
            rec["v2_corr_err_deg"] = None
            continue

        v2_corr = _wrap_angle_0_360_deg(v2 - bias_v2)
        rec["v2_bias_corrected_deg"] = float(v2_corr)

        diff_corr = _angle_diff_signed_deg(v2_corr, gt)
        rec["v2_corr_err_deg"] = float(diff_corr)


    print("\nV2 summary (bias-corrected):")
    _print_heading_error_summary(heading_diffs_signed_v2_corrected)

    # -------------------------
    # Write JSON
    # -------------------------
    out_payload = {
        "angle_convention": "0=left, 90=down (image coordinates), degrees",
        "bias_deg": float(bias_v2) if bias_v2 is not None else None,
        "counts": {
            "n_records": len(heading_records),
            "n_v2_used_for_bias": len(heading_diffs_signed_v2),
            "n_v2_corr_errors": len(heading_diffs_signed_v2_corrected),
        },
        "records": heading_records,
    }

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    print(f"\nWrote per-image headings JSON: {out_path}")


if __name__ == '__main__':
    main()