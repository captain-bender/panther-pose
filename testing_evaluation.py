from ultralytics import YOLO
import cv2
import argparse
import shutil
from pathlib import Path
import yaml
import subprocess
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='runs/train/yolo11n-panther_v1-pose-v1/weights/best.pt',
                        help='Path to trained model .pt')
    parser.add_argument('--data', type=str, default='./dataset/version-1/data.yaml', help='Path to data.yaml')
    parser.add_argument('--save-dir', type=str, default='./runs/test-evaluation/yolo11n-panther_v1-pose-v1/',
                        help='Directory to save evaluation results. Defaults to ./runs/test-evaluation/<model-name>/')
    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model)

    print("Evaluating model on test dataset...")

    data_yaml = Path(args.data)

    # Determine save directory
    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        # Extract model name from path (without .pt extension)
        model_name = Path(args.model).stem
        save_dir = Path('./runs/test-evaluation') / model_name

    # Evaluate on test set
    results = model.val(
        data=str(data_yaml),
        split='test',
        imgsz=1920,
        batch=8,
        save_json=True,
        save_hybrid=True,
        plots=True,
        project=str(save_dir.parent),
        name=save_dir.name,
        exist_ok=True,
        verbose=True
    )

    if results is None:
        print("Evaluation failed or was interrupted; no results to show.")
        return

    print("\n Test Results:")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")

    # If you have pose metrics
    if hasattr(results, 'pose'):
        print(f"Pose mAP50: {results.pose.map50:.4f}")
        print(f"Pose mAP50-95: {results.pose.map:.4f}")

    # Get save directory from results
    if hasattr(results, 'save_dir'):
        print(f"\n Results saved to: {results.save_dir}")
    else:
        print(f"\n Results saved to: runs/detect/ (default location)")

    # Per-image timing and throughput (Hz)
    try:
        data_yaml_path = Path(data_to_use)
        with open(data_yaml_path, 'r') as f:
            data_cfg = yaml.safe_load(f)
        test_entry = data_cfg.get('test')
        if test_entry is None:
            print("No 'test' entry found in data yaml; skipping per-image timing.")
        else:
            # Resolve the test images directory robustly
            raw = str(test_entry)
            candidates = []
            # As given (CWD relative or absolute)
            try:
                candidates.append(Path(raw).resolve())
            except Exception:
                pass
            # Relative to YAML parent
            try:
                candidates.append((data_yaml_path.parent / raw).resolve())
            except Exception:
                pass
            # If YAML defines a 'path' root, try relative to it
            ds_root = data_cfg.get('path')
            if ds_root:
                try:
                    candidates.append((Path(ds_root) / raw).resolve())
                except Exception:
                    pass
            # If starts with ./ or ../, also try stripped version
            if raw.startswith('../') or raw.startswith('./'):
                stripped = raw.lstrip('./')
                try:
                    candidates.append((data_yaml_path.parent / stripped).resolve())
                except Exception:
                    pass
                if ds_root:
                    try:
                        candidates.append((Path(ds_root) / stripped).resolve())
                    except Exception:
                        pass
                try:
                    candidates.append(Path(stripped).resolve())
                except Exception:
                    pass

            # Pick the first existing dir
            test_path = None
            for c in candidates:
                if isinstance(c, Path) and c.is_dir():
                    test_path = c
                    break

            if test_path and test_path.is_dir():
                # Collect images
                img_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
                images = sorted([p for p in test_path.iterdir() if p.suffix.lower() in img_exts])
                if not images:
                    print(f"No images found under {test_path}; skipping per-image timing.")
                else:
                    print("\n Per-image timing (single-image inference):")
                    times = []
                    for img in images:
                        t0 = time.perf_counter()
                        _ = model(str(img), imgsz=1280)
                        t1 = time.perf_counter()
                        dt = t1 - t0
                        hz = (1.0 / dt) if dt > 0 else float('inf')
                        times.append(dt)
                        print(f"  {img.name}: {dt*1000:.1f} ms | {hz:.2f} Hz")

                    # Summary
                    if times:
                        n = len(times)
                        total = sum(times)
                        avg = total / n
                        mn = min(times)
                        mx = max(times)
                        print("\n Timing summary:")
                        print(f"  Images: {n}")
                        print(f"  Avg: {avg*1000:.1f} ms | {1.0/avg if avg>0 else float('inf'):.2f} Hz")
                        print(f"  Min: {mn*1000:.1f} ms | {1.0/mn if mn>0 else float('inf'):.2f} Hz")
                        print(f"  Max: {mx*1000:.1f} ms | {1.0/mx if mx>0 else float('inf'):.2f} Hz")
            else:
                print("Unable to resolve a valid 'test' images directory; skipping per-image timing.")
    except Exception as e:
        print(f"Warning: failed per-image timing: {e}")



if __name__ == '__main__':
    main()