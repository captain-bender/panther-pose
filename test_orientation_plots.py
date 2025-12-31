import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def wrap_deg(deg: float) -> float:
    return (deg % 360.0 + 360.0) % 360.0


def angle_diff_deg(pred_deg: float, gt_deg: float) -> float:
    """Signed smallest difference pred - gt in degrees, in [-180, 180)."""
    return (pred_deg - gt_deg + 180.0) % 360.0 - 180.0


def get_field(r: dict, key: str):
    v = r.get(key, None)
    return None if v is None else float(v)


def safe_abs_list(vals):
    return [abs(v) for v in vals if v is not None]


def cdf_xy(abs_errors, n_points=200):
    if not abs_errors:
        return [], []
    xs = sorted(abs_errors)
    ys = [(i + 1) / len(xs) for i in range(len(xs))]
    return xs, ys


def summarize(name, errs_abs):
    if not errs_abs:
        print(f"{name}: no samples")
        return
    xs = sorted(errs_abs)
    n = len(xs)
    mean = sum(xs) / n
    med = xs[n // 2] if n % 2 == 1 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])
    p90 = xs[max(0, math.ceil(0.90 * n) - 1)]
    p95 = xs[max(0, math.ceil(0.95 * n) - 1)]
    print(f"{name}: n={n} | mean={mean:.3f}° | median={med:.3f}° | p90={p90:.3f}° | p95={p95:.3f}° | max={xs[-1]:.3f}°")


def main():
    in_path = Path("heading_debug.json")
    if not in_path.exists():
        raise FileNotFoundError(f"Cannot find {in_path.resolve()}")

    payload = json.loads(in_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    bias = payload.get("bias_deg", None)

    # Collect errors (prefer using explicit *_err_deg if present, else compute from angles)
    v1_err = []
    v2_err = []
    v2c_err = []

    gt_list = []
    name_list = []

    for r in records:
        gt = get_field(r, "gt_deg")
        if gt is None:
            continue

        name = r.get("image_name") or r.get("image") or r.get("path") or "unknown"
        name_list.append(name)
        gt_list.append(gt)

        # v1
        e1 = r.get("v1_err_deg", None)
        if e1 is None:
            v1 = get_field(r, "v1_deg")
            e1 = None if v1 is None else angle_diff_deg(v1, gt)
        else:
            e1 = float(e1)
        v1_err.append(e1)

        # v2
        e2 = r.get("v2_err_deg", None)
        if e2 is None:
            v2 = get_field(r, "v2_deg")
            e2 = None if v2 is None else angle_diff_deg(v2, gt)
        else:
            e2 = float(e2)
        v2_err.append(e2)

        # v2 bias corrected
        e2c = r.get("v2_corr_err_deg", None)
        if e2c is None:
            v2c = get_field(r, "v2_bias_corrected_deg")
            e2c = None if v2c is None else angle_diff_deg(v2c, gt)
        else:
            e2c = float(e2c)
        v2c_err.append(e2c)

    v1_abs = safe_abs_list(v1_err)
    v2_abs = safe_abs_list(v2_err)
    v2c_abs = safe_abs_list(v2c_err)

    print("=== SUMMARY (abs error) ===")
    if bias is not None:
        print(f"Bias used (from JSON): {float(bias):.3f}°")
    summarize("V1 abs", v1_abs)
    summarize("V2 abs", v2_abs)
    summarize("V2 corrected abs", v2c_abs)
    print()

    # Print worst cases for v2 corrected
    worst = []
    for i, e in enumerate(v2c_err):
        if e is None:
            continue
        worst.append((abs(e), e, gt_list[i], name_list[i]))
    worst.sort(reverse=True, key=lambda x: x[0])

    print("=== TOP 10 WORST (V2 corrected) ===")
    for a, e, gt, nm in worst[:10]:
        print(f"{a:6.2f}°  (signed {e:+6.2f}°)  GT={gt:7.2f}°  {nm}")
    print()

    # 1) Histogram of absolute errors
    plt.figure()
    bins = 30
    if v1_abs:
        plt.hist(v1_abs, bins=bins, alpha=0.5, label="V1 |err|")
    if v2_abs:
        plt.hist(v2_abs, bins=bins, alpha=0.5, label="V2 |err|")
    if v2c_abs:
        plt.hist(v2c_abs, bins=bins, alpha=0.5, label="V2 corrected |err|")
    plt.xlabel("Absolute heading error (degrees)")
    plt.ylabel("Count")
    plt.title("Heading absolute error distribution")
    plt.legend()
    plt.tight_layout()

    # 2) CDF of absolute errors
    plt.figure()
    for xs, label in [
        (v1_abs, "V1"),
        (v2_abs, "V2"),
        (v2c_abs, "V2 corrected"),
    ]:
        x, y = cdf_xy(xs)
        if x:
            plt.plot(x, y, label=label)
    plt.xlabel("Absolute heading error (degrees)")
    plt.ylabel("CDF")
    plt.title("CDF of heading absolute error")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # 3) Signed error histogram (shows bias / skew)
    plt.figure()
    bins = 40
    if any(e is not None for e in v1_err):
        plt.hist([e for e in v1_err if e is not None], bins=bins, alpha=0.5, label="V1 err")
    if any(e is not None for e in v2_err):
        plt.hist([e for e in v2_err if e is not None], bins=bins, alpha=0.5, label="V2 err")
    if any(e is not None for e in v2c_err):
        plt.hist([e for e in v2c_err if e is not None], bins=bins, alpha=0.5, label="V2 corrected err")
    plt.xlabel("Signed heading error (deg): pred - GT")
    plt.ylabel("Count")
    plt.title("Signed heading error distribution")
    plt.legend()
    plt.tight_layout()

    # 4) Error vs GT angle (spot angle-dependent effects)
    plt.figure()
    gx = []
    ey = []
    for gt, e in zip(gt_list, v2c_err):
        if e is None:
            continue
        gx.append(wrap_deg(gt))
        ey.append(e)
    if gx:
        plt.scatter(gx, ey, s=12)
    plt.xlabel("GT heading (degrees)")
    plt.ylabel("Signed error (deg)  [V2 corrected]")
    plt.title("V2 corrected error vs GT heading")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plots too (handy for reports)
    out_dir = Path("heading_plots")
    out_dir.mkdir(exist_ok=True)
    for i, fig_num in enumerate(plt.get_fignums(), start=1):
        plt.figure(fig_num)
        plt.savefig(out_dir / f"plot_{i:02d}.png", dpi=200)

    print(f"Saved plots to: {out_dir.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
