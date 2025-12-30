from ultralytics import YOLO
import numpy as np
import cv2
import os
from math import sqrt
from datetime import datetime
from multiprocessing import freeze_support
import json

KEYPOINT_NAMES = ["back-left", "front", "back-right"]

# OKS sigmas
OKS_SIGMAS = np.array([0.07, 0.06, 0.07], dtype=float)


def _read_yolo_pose_labels(label_path, img_w, img_h):
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


def _oks(gt_kpts, pred_kpts, gt_bbox, sigmas=None):
    # gt_kpts: list of (x,y,v)
    # pred_kpts: (K,2) ndarray
    # gt_bbox: (x1,y1,x2,y2)
    # sigmas: (K,) ndarray; will be broadcast/truncated to min K
    x1, y1, x2, y2 = gt_bbox
    area = max(1.0, float((x2 - x1) * (y2 - y1)))

    K = min(len(gt_kpts), pred_kpts.shape[0])
    if K <= 0:
        return 0.0

    g = np.array(gt_kpts[:K], dtype=float)  # (K,3)
    p = np.array(pred_kpts[:K], dtype=float)  # (K,2)

    vis = (g[:, 2] > 0).astype(float)  # only visible/labeled keypoints
    if vis.sum() == 0:
        return 0.0

    dx = p[:, 0] - g[:, 0]
    dy = p[:, 1] - g[:, 1]
    d2 = dx * dx + dy * dy

    if sigmas is None or len(sigmas) < K:
        s = np.full(K, 0.05, dtype=float)
    else:
        s = np.array(sigmas[:K], dtype=float)

    vars_ = (s * 2) ** 2  # follow COCO convention
    e = np.exp(-d2 / (2 * vars_ * area + 1e-9))
    oks = (e * vis).sum() / (vis.sum() + 1e-9)
    return float(oks)


def _ap_from_pr(rec, prec):
    # 101-point interpolated AP
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    x = np.linspace(0, 1, 101)
    p = np.interp(x, mrec, mpre)
    return float(p.mean())


def _kpt_name(idx, names):
    if isinstance(names, (list, tuple)) and idx < len(names):
        return str(names[idx])
    return f"kpt {idx}"


def calculate_pose_metrics(model, test_dir, labels_dir):
    """Calculate custom pose estimation metrics"""
    
    # Get test images
    test_images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    total_images = 0
    total_keypoints_detected = 0
    total_keypoints_gt = 0
    correct_keypoints = 0
    pck_distances = []  # For PCK calculation
    
    print(f"   Analyzing {len(test_images)} test images...")
    
    for img_name in test_images[:10]:  # Limit to first 10 for speed
        img_path = os.path.join(test_dir, img_name)
        
        # Get predictions
        results = model(img_path, verbose=False)
        
        # Load ground truth (simplified - would need actual label parsing)
        total_images += 1
        
        for result in results:
            if result.keypoints is not None:
                keypoints = result.keypoints.xy.cpu().numpy()
                confidences = result.keypoints.conf.cpu().numpy()
                
                # Count detected keypoints
                for person_kpts in keypoints:
                    for kpt_conf in confidences:
                        for conf in kpt_conf:
                            if conf > 0.5:  # Confidence threshold
                                total_keypoints_detected += 1
    
    return {
        'total_images': total_images,
        'avg_keypoints_per_image': total_keypoints_detected / max(total_images, 1),
        'detection_rate': total_keypoints_detected / max(total_images * 3, 1)  # Assuming 3 keypoints max
    }


def calculate_pck(model, test_dir, labels_dir, alphas=(0.1, 0.2), iou_thresh=0.5, limit=None):
    images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images.sort()
    if limit:
        images = images[:limit]

    totals = {alpha: 0 for alpha in alphas}
    corrects = {alpha: 0 for alpha in alphas}
    per_kpt_totals = {}
    per_kpt_corrects = {}
    matched_pairs = 0
    total_gts = 0

    for img_name in images:
        img_path = os.path.join(test_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')
        gts = _read_yolo_pose_labels(label_path, w, h)
        if not gts:
            continue

        total_gts += len(gts)

        preds = model(img_path, verbose=False)[0]
        if preds is None or preds.boxes is None or preds.keypoints is None:
            continue

        pred_boxes = preds.boxes.xyxy.cpu().numpy()
        pred_kpts = preds.keypoints.xy.cpu().numpy()  # shape: (N, K, 2)

        # Build IoU matrix
        iou_mat = np.zeros((len(pred_boxes), len(gts)), dtype=float)
        for i, pb in enumerate(pred_boxes):
            for j, gt in enumerate(gts):
                iou_mat[i, j] = _iou_xyxy(tuple(pb.tolist()), gt['bbox'])

        # Greedy matching by IoU
        pred_used = set()
        gt_used = set()
        while True:
            max_idx = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
            max_iou = iou_mat[max_idx]
            if max_iou < iou_thresh:
                break
            pi, gi = int(max_idx[0]), int(max_idx[1])
            if pi in pred_used or gi in gt_used:
                iou_mat[pi, gi] = -1.0
                continue
            pred_used.add(pi)
            gt_used.add(gi)
            iou_mat[pi, :] = -1.0
            iou_mat[:, gi] = -1.0

            matched_pairs += 1

            # Compute PCK for this pair
            x1, y1, x2, y2 = gts[gi]['bbox']
            ref = max(x2 - x1, y2 - y1)
            if ref <= 0:
                continue
            gt_k = gts[gi]['kpts']
            pred_k = pred_kpts[pi]
            k_len = min(len(gt_k), pred_k.shape[0])
            for k in range(k_len):
                gx, gy, gv = gt_k[k]
                if gv <= 0:  # skip unlabeled keypoints
                    continue
                px, py = pred_k[k]
                dist = sqrt((px - gx) ** 2 + (py - gy) ** 2)
                for alpha in alphas:
                    totals[alpha] += 1
                    per_kpt_totals.setdefault(alpha, {}).setdefault(k, 0)
                    per_kpt_totals[alpha][k] += 1
                    if dist <= alpha * ref:
                        corrects[alpha] += 1
                        per_kpt_corrects.setdefault(alpha, {}).setdefault(k, 0)
                        per_kpt_corrects[alpha][k] += 1

    pck = {alpha: (corrects[alpha] / totals[alpha]) if totals[alpha] > 0 else 0.0 for alpha in alphas}
    return {
        'pck': pck,
        'totals': totals,
        'corrects': corrects,
        'per_keypoint_pck': {alpha: {k: (per_kpt_corrects.get(alpha, {}).get(k, 0) / per_kpt_totals.get(alpha, {}).get(k, 1)) for k in per_kpt_totals.get(alpha, {})} for alpha in alphas},
        'per_keypoint_totals': per_kpt_totals,
        'per_keypoint_corrects': per_kpt_corrects,
        'matched_pairs': matched_pairs,
        'total_gts': total_gts,
    }


def plot_per_keypoint_pck(pck_metrics, keypoint_names=None, out_dir="runs/pose-metrics/<model-name>/pck", alphas=(0.1, 0.2)):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"  Skipping PCK plot (matplotlib not available): {e}")
        return None

    os.makedirs(out_dir, exist_ok=True)

    # Union of keypoint indices across alphas
    kpt_set = set()
    for a in alphas:
        kpt_set.update(pck_metrics['per_keypoint_totals'].get(a, {}).keys())
    kpt_ids = sorted(kpt_set)
    if not kpt_ids:
        print("  No per-keypoint data to plot.")
        return None

    labels = [_kpt_name(k, keypoint_names) for k in kpt_ids]
    x = np.arange(len(kpt_ids))
    n = max(1, len(alphas))
    width = min(0.8 / n, 0.35)

    fig, ax = plt.subplots(figsize=(max(6, len(kpt_ids) * 0.6), 4.5))
    for i, a in enumerate(alphas):
        vals = [pck_metrics['per_keypoint_pck'].get(a, {}).get(k, 0.0) for k in kpt_ids]
        ax.bar(x + (i - (n - 1) / 2) * width, vals, width, label=f"PCK@{a:.1f}")

    ax.set_ylabel('PCK')
    ax.set_title('Per-Keypoint PCK')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend()
    plt.tight_layout()

    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    out_path = os.path.join(out_dir, f"pck_per_keypoint_{ts}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Per-keypoint PCK plot saved: {out_path}")
    return out_path


def calculate_oks_map(model, test_dir, labels_dir, oks_thresholds=None, limit=None, use_iou_gate=False, iou_thresh=0.5):
    if oks_thresholds is None:
        # ensure Python floats (not numpy scalars) and stable rounding
        oks_thresholds = [float(f"{t:.2f}") for t in np.arange(0.50, 0.96, 0.05)]

    images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images.sort()
    if limit:
        images = images[:limit]

    # Prepare accumulators
    ap_by_thr = {}
    npos_total = 0  # number of GT persons

    # Pre-collect per-image data to avoid recompute
    per_image = []
    for img_name in images:
        img_path = os.path.join(test_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt')
        gts = _read_yolo_pose_labels(label_path, w, h)
        if not gts:
            per_image.append(([], [], []))
            continue

        preds = model(img_path, verbose=False)[0]
        if preds is None or preds.boxes is None or preds.keypoints is None:
            per_image.append((gts, [], []))
            npos_total += len(gts)
            continue

        pred_boxes = preds.boxes.xyxy.cpu().numpy()  # (N,4)
        pred_scores = preds.boxes.conf.cpu().numpy() if preds.boxes.conf is not None else np.ones((pred_boxes.shape[0],), dtype=float)
        pred_kpts = preds.keypoints.xy.cpu().numpy()  # (N,K,2)

        per_image.append((gts, list(zip(pred_boxes, pred_kpts, pred_scores)), pred_boxes))
        npos_total += len(gts)

    if npos_total == 0:
        return {'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0, 'AP_by_threshold': {}, 'npos': 0}

    # Evaluate for each OKS threshold
    for thr in oks_thresholds:
        thr = float(thr)
        scores_all = []
        tps_all = []
        fps_all = []

        for (gts, preds, _) in per_image:
            # Sort predictions by confidence descending
            preds_sorted = sorted(preds, key=lambda x: float(x[2]), reverse=True)

            gt_matched = np.zeros((len(gts),), dtype=bool)

            for pb, pk, ps in preds_sorted:
                # choose best GT by OKS (optionally gate by IoU)
                best_i = -1
                best_oks = -1.0
                for gi, gt in enumerate(gts):
                    if gt_matched[gi]:
                        continue
                    if use_iou_gate:
                        if _iou_xyxy(tuple(pb.tolist()), gt['bbox']) < iou_thresh:
                            continue
                    oks = _oks(gt['kpts'], pk, gt['bbox'], sigmas=OKS_SIGMAS)
                    if oks > best_oks:
                        best_oks, best_i = oks, gi

                scores_all.append(float(ps))
                if best_i >= 0 and best_oks >= thr:
                    tps_all.append(1.0)
                    fps_all.append(0.0)
                    gt_matched[best_i] = True
                else:
                    tps_all.append(0.0)
                    fps_all.append(1.0)

        if len(scores_all) == 0:
            ap_by_thr[thr] = 0.0
            continue

        # Sort globally by confidence
        order = np.argsort(-np.array(scores_all))
        tps = np.array(tps_all)[order]
        fps = np.array(fps_all)[order]

        cum_tp = np.cumsum(tps)
        cum_fp = np.cumsum(fps)

        recalls = cum_tp / max(1, npos_total)
        precisions = cum_tp / np.maximum(1, cum_tp + cum_fp)

        ap_by_thr[thr] = _ap_from_pr(recalls, precisions)

    def _ap_lookup(ap_dict, target, tol=1e-6):
        # robustly fetch AP for a target threshold
        if not ap_dict:
            return 0.0
        keys = np.array([float(k) for k in ap_dict.keys()], dtype=float)
        idx = int(np.argmin(np.abs(keys - float(target))))
        return float(list(ap_dict.values())[idx]) if abs(keys[idx] - float(target)) <= tol else 0.0

    ap_vals = list(ap_by_thr.values())
    AP = float(np.mean(ap_vals)) if ap_vals else 0.0
    AP50 = _ap_lookup(ap_by_thr, 0.50)
    AP75 = _ap_lookup(ap_by_thr, 0.75)

    return {
        'AP': AP,
        'AP50': AP50,
        'AP75': AP75,
        'AP_by_threshold': ap_by_thr,
        'npos': npos_total,
    }


def summarize_oks_distribution(model, test_dir, labels_dir, thresholds=(0.50, 0.60, 0.70, 0.75, 0.80, 0.90), limit=None, use_iou_gate=False, iou_thresh=0.5):
    images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images.sort()
    if limit:
        images = images[:limit]

    oks_samples = []

    for img_name in images:
        img_path = os.path.join(test_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        gts = _read_yolo_pose_labels(os.path.join(labels_dir, os.path.splitext(img_name)[0] + '.txt'), w, h)
        if not gts:
            continue

        pred = model(img_path, verbose=False)[0]
        if pred is None or pred.boxes is None or pred.keypoints is None:
            continue

        pred_boxes = pred.boxes.xyxy.cpu().numpy()
        pred_kpts = pred.keypoints.xy.cpu().numpy()  # (P,K,2)

        if len(pred_boxes) == 0 or len(gts) == 0:
            continue

        # Compute OKS matrix (P x G)
        P, G = len(pred_boxes), len(gts)
        oks_mat = np.zeros((P, G), dtype=float)
        for i in range(P):
            for j in range(G):
                if use_iou_gate and _iou_xyxy(tuple(pred_boxes[i].tolist()), gts[j]['bbox']) < iou_thresh:
                    oks_mat[i, j] = -1.0
                else:
                    oks_mat[i, j] = _oks(gts[j]['kpts'], pred_kpts[i], gts[j]['bbox'], sigmas=OKS_SIGMAS)

        # Greedy match by OKS and collect matched OKS
        used_p, used_g = set(), set()
        while True:
            idx = np.unravel_index(np.argmax(oks_mat), oks_mat.shape)
            best = oks_mat[idx]
            if best < 0:  # no valid pairs left
                break
            i, j = int(idx[0]), int(idx[1])
            oks_samples.append(float(best))
            used_p.add(i); used_g.add(j)
            oks_mat[i, :] = -1.0
            oks_mat[:, j] = -1.0

    if not oks_samples:
        print("  No OKS pairs to summarize.")
        return

    arr = np.array(oks_samples, dtype=float)
    print("\n Distribution (matched pairs):")
    print(f"   count: {arr.size}, mean: {arr.mean():.3f}, median: {np.median(arr):.3f}, max: {arr.max():.3f}")
    for t in thresholds:
        share = (arr >= t).mean()
        print(f"   share >= {t:.2f}: {share:.3f}")


def calculate_keypoint_confidence(model, test_dir, limit=None, thresholds=(0.3, 0.5, 0.7)):
    """Aggregate predicted keypoint confidence statistics over the test set."""
    images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    images.sort()
    if limit:
        images = images[:limit]

    all_confs = []
    per_kpt_confs = {}  # k -> list[float]
    per_det_avg_confs = []  # mean confidence per detected instance

    for img_name in images:
        img_path = os.path.join(test_dir, img_name)
        preds = model(img_path, verbose=False)[0]
        if preds is None or preds.keypoints is None:
            continue

        # confidences: shape (N, K)
        conf = preds.keypoints.conf
        if conf is None:
            continue
        conf = conf.cpu().numpy()
        if conf.size == 0:
            continue

        # Per-detection average confidence
        per_det_avg_confs.extend(np.nanmean(conf, axis=1).tolist())

        # Collect overall and per-kpt confidences
        flat = conf.reshape(-1)
        flat = flat[np.isfinite(flat)]
        all_confs.extend(flat.tolist())

        K = conf.shape[1]
        for k in range(K):
            vals = conf[:, k]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            per_kpt_confs.setdefault(k, []).extend(vals.tolist())

    if len(all_confs) == 0:
        return {
            'count': 0,
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'shares_above': {t: 0.0 for t in thresholds},
            'per_keypoint_mean': {},
            'per_keypoint_median': {},
            'per_keypoint_count': {},
            'per_keypoint_shares_above': {t: {} for t in thresholds},
            'per_detection_avg_mean': 0.0,
            'per_detection_avg_median': 0.0,
        }
    
    arr = np.array(all_confs, dtype=float)
    shares = {t: float((arr >= t).mean()) for t in thresholds}

    per_kpt_mean = {k: float(np.mean(v)) for k, v in per_kpt_confs.items()}
    per_kpt_median = {k: float(np.median(v)) for k, v in per_kpt_confs.items()}
    per_kpt_count = {k: int(len(v)) for k, v in per_kpt_confs.items()}
    per_kpt_shares = {t: {k: float((np.array(v) >= t).mean()) for k, v in per_kpt_confs.items()} for t in thresholds}

    det_avg = np.array(per_det_avg_confs, dtype=float) if per_det_avg_confs else np.array([], dtype=float)

    return {
        'count': int(arr.size),
        'mean': float(arr.mean()),
        'median': float(np.median(arr)),
        'std': float(arr.std()),
        'shares_above': shares,
        'per_keypoint_mean': per_kpt_mean,
        'per_keypoint_median': per_kpt_median,
        'per_keypoint_count': per_kpt_count,
        'per_keypoint_shares_above': per_kpt_shares,
        'per_detection_avg_mean': float(det_avg.mean()) if det_avg.size > 0 else 0.0,
        'per_detection_avg_median': float(np.median(det_avg)) if det_avg.size > 0 else 0.0,
    }


if __name__ == '__main__':
    freeze_support()
    
    # Load your trained model
    weight_path = 'runs/train/yolo11n-panther_v1-pose-v1/weights/best.pt'
    model_name = os.path.basename(os.path.dirname(os.path.dirname(weight_path)))
    model = YOLO(weight_path)

    # Prepare model-specific output directories
    out_root = os.path.join("runs", "pose-metrics", model_name)
    os.makedirs(out_root, exist_ok=True)
    # Do NOT pre-create the yolo_eval folder: Ultralytics will create
    # `yolo_eval`, `yolo_eval2`, ... automatically. If we pre-create an
    # empty `yolo_eval` dir, Ultralytics will create `yolo_eval2` instead.
    # We'll detect which folder was actually created after evaluation.
    yolo_eval_dir = None

    print("POSE ESTIMATION QUALITY METRICS")
    print("=" * 50)

    # 1. Standard YOLO validation metrics
    print("\nSTANDARD YOLO METRICS:")
    # Direct Ultralytics evaluation outputs into the model-specific folder
    # Force evaluation outputs into our project/name folder to override defaults
    results = model.val(
        data="./dataset/version-1/data.yaml",
        split='test',
        imgsz=1920,
        batch=4,
        verbose=True,
        save=True,
        project=out_root,
        name='yolo_eval'
    )

    # Detect the actual evaluation folder created by Ultralytics (yolo_eval, yolo_eval2, ...)
    try:
        eval_candidates = [d for d in os.listdir(out_root) if d.startswith('yolo_eval')]
        eval_candidates = sorted(eval_candidates)
        actual_eval_dir = None
        for d in eval_candidates:
            full = os.path.join(out_root, d)
            # prefer directory that contains files
            try:
                if os.path.isdir(full) and os.listdir(full):
                    actual_eval_dir = full
                    break
            except Exception:
                continue

        # If none have files, pick the newest candidate directory if any
        if actual_eval_dir is None and eval_candidates:
            latest = max(eval_candidates, key=lambda x: os.path.getmtime(os.path.join(out_root, x)))
            actual_eval_dir = os.path.join(out_root, latest)

        if actual_eval_dir:
            print(f"  YOLO evaluation outputs written to: {actual_eval_dir}")
        else:
            print(f"  No YOLO evaluation folder found under {out_root}")

        # Use detected folder for any future references if needed
        yolo_eval_dir = actual_eval_dir
    except Exception as e:
        print(f"  Warning: failed to detect YOLO eval folder: {e}")

    # Extract pose-specific metrics
    if hasattr(results, 'pose') and results.pose is not None:
        print(f" Pose mAP50:     {results.pose.map50:.4f}")
        print(f" Pose mAP50-95:  {results.pose.map:.4f}")
        print(f" Pose mAP75:     {results.pose.map75:.4f}")
    else:
        print("  Pose metrics not available in results object")

    # Print box metrics for person detection
    if hasattr(results, 'box') and results.box is not None:
        print(f" Box mAP50:      {results.box.map50:.4f}")
        print(f" Box mAP50-95:   {results.box.map:.4f}")

    print("\nPOSE-SPECIFIC QUALITY METRICS:")

    # Calculate custom metrics (fix dataset path)
    test_dir = "./dataset/version-1/test/images"
    labels_dir = "./dataset/version-1/test/labels"

    if os.path.exists(test_dir):
        custom_metrics = calculate_pose_metrics(model, test_dir, labels_dir)
        
        print(f" Images analyzed:           {custom_metrics['total_images']}")
        print(f" Avg keypoints per image:   {custom_metrics['avg_keypoints_per_image']:.2f}")
        print(f" Keypoint detection rate:   {custom_metrics['detection_rate']:.2f}")
    else:
        print("  Test directory not found")

    # Compute PCK metrics
    if os.path.exists(test_dir) and os.path.exists(labels_dir):
        pck_metrics = calculate_pck(model, test_dir, labels_dir, alphas=(0.1, 0.2), iou_thresh=0.5, limit=None)
        print("\n PCK METRICS:")
        for alpha, val in pck_metrics['pck'].items():
            print(f"   PCK@{alpha:.1f}: {val:.3f}  ({pck_metrics['corrects'][alpha]}/{pck_metrics['totals'][alpha]} keypoints)")
        print(f"   Matched predictions/GT pairs: {pck_metrics['matched_pairs']}/{pck_metrics['total_gts']}")
        for alpha, per_k in pck_metrics['per_keypoint_pck'].items():
            if not per_k:
                continue
            print(f"   Per-keypoint PCK@{alpha:.1f}:")
            for k_idx in sorted(per_k.keys()):
                corr = pck_metrics['per_keypoint_corrects'].get(alpha, {}).get(k_idx, 0)
                tot = pck_metrics['per_keypoint_totals'].get(alpha, {}).get(k_idx, 0)
                val = per_k[k_idx]
                print(f"     - {_kpt_name(k_idx, KEYPOINT_NAMES)}: {val:.3f} ({corr}/{tot})")

        try:
            out_dir = os.path.join("runs", "pose-metrics", model_name, "pck")
            plot_per_keypoint_pck(pck_metrics, keypoint_names=KEYPOINT_NAMES, out_dir=out_dir, alphas=(0.1, 0.2))
        except Exception as e:
            print(f"  Failed to plot per-keypoint PCK: {e}")

    # ---------------------------
    # OKS mAP metrics
    # ---------------------------
    oks_metrics = calculate_oks_map(model, test_dir, labels_dir, oks_thresholds=np.arange(0.50, 0.96, 0.05), limit=None, use_iou_gate=False)
    print("\nOKS METRICS:")
    print(f"   OKS AP@[.50:.95]: {oks_metrics['AP']:.3f}")
    print(f"   OKS AP@0.50:      {oks_metrics['AP50']:.3f}")
    print(f"   OKS AP@0.75:      {oks_metrics['AP75']:.3f}")

    # Optional: inspect where scores land
    summarize_oks_distribution(model, test_dir, labels_dir, use_iou_gate=False)

    # ---------------------------
    # Keypoint confidence statistics
    # ---------------------------
    if os.path.exists(test_dir):
        conf_metrics = calculate_keypoint_confidence(model, test_dir, limit=None, thresholds=(0.3, 0.5, 0.7))
        print("\nKEYPOINT CONFIDENCE")
        print(f"   samples: {conf_metrics['count']}")
        print(f"   mean / median / std: {conf_metrics['mean']:.3f} / {conf_metrics['median']:.3f} / {conf_metrics['std']:.3f}")
        for t, s in conf_metrics['shares_above'].items():
            print(f"   share >= {t:.1f}: {s:.3f}")
        print(f"   per-detection avg confidence (mean/median): {conf_metrics['per_detection_avg_mean']:.3f} / {conf_metrics['per_detection_avg_median']:.3f}")

        if conf_metrics['per_keypoint_mean']:
            print("   Per-keypoint mean confidence:")
            for k in sorted(conf_metrics['per_keypoint_mean'].keys()):
                name = _kpt_name(k, KEYPOINT_NAMES)
                mean_v = conf_metrics['per_keypoint_mean'][k]
                cnt = conf_metrics['per_keypoint_count'].get(k, 0)
                s50 = conf_metrics['per_keypoint_shares_above'][0.5].get(k, 0.0)
                print(f"     - {name}: {mean_v:.3f} (n={cnt}, share>=0.5: {s50:.3f})")

