"""
Script to combine YOLO inference headings with ground truth headings from rover.json
Filters out rejected predictions and applies coordinate transformation.
Coordinate transformation: robot_heading = (90 - image_heading) % 360
"""

import json
import pandas as pd
import math
from pathlib import Path


def extract_position_number(filename):
    """
    Extract position number from filename.
    E.g., 'position_036_png.rf.b1688a8b0f571152625bf81dcf8ff565.jpg' -> 'position_036'
    """
    parts = filename.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return None


def rotation_to_heading_degrees(rotation_vector):
    """
    Convert rotation vector [rx, ry, rz] to heading in degrees.
    For rover, we typically use the Z-axis rotation (yaw).
    Converts radians to degrees and normalizes to [0, 360).
    """
    if isinstance(rotation_vector, (list, tuple)) and len(rotation_vector) >= 3:
        yaw_rad = rotation_vector[2]
        yaw_deg = math.degrees(yaw_rad)
        # Normalize to [0, 360)
        yaw_deg = yaw_deg % 360
        return yaw_deg
    return None


def normalize_angle(angle):
    """Normalize angle to [0, 360) range."""
    return angle % 360


def calculate_angular_error(angle1, angle2):
    """Calculate the smallest angular difference between two angles."""
    error = abs(angle1 - angle2)
    if error > 180:
        error = 360 - error
    return error


def main():
    # File paths
    csv_path = Path("runs/test-orientation/yolo11n-panther_v1-pose-v1/heading_estimation_from_debug.csv")
    rover_config_path = Path("configs/rover.json")
    output_path = Path("runs/test-orientation/yolo11n-panther_v1-pose-v1/inference_with_ground_truth_final.json")
    outlier_analysis_path = Path("runs/test-orientation/yolo11n-panther_v1-pose-v1/outlier_analysis.json")

    # Load CSV
    print(f"Loading inference results from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Total rows: {len(df)}")

    # Load rover configuration
    print(f"Loading ground truth from {rover_config_path}...")
    with open(rover_config_path, 'r') as f:
        rover_config = json.load(f)

    # Create position -> rotation mapping
    position_to_rotation = {}
    for rover in rover_config['rovers']:
        if 'id' in rover and 'rotation' in rover:
            position_to_rotation[rover['id']] = rover['rotation']

    print(f"  Loaded {len(position_to_rotation)} rover positions")

    # Filter and process data
    print("\nProcessing predictions...")
    results = []

    # Filter out rejected predictions
    df_filtered = df[df['reject'] == False].copy()
    print(f"  Filtered to {len(df_filtered)} non-rejected predictions")

    for idx, row in df_filtered.iterrows():
        filename = row['image']
        position_id = extract_position_number(filename)

        if position_id not in position_to_rotation:
            print(f"  Warning: Position {position_id} not found in rover config")
            continue

        # Get ground truth heading
        rotation = position_to_rotation[position_id]
        ground_truth_heading = rotation_to_heading_degrees(rotation)
        
        # Get inference heading (in image coordinates)
        image_heading = float(row['heading_deg'])
        
        # Apply standard transformation: robot_heading = (90 - image_heading) % 360
        inferred_heading_robot = normalize_angle(90 - image_heading)
        transformation_note = "Standard transformation applied"

        # Create result entry with both coordinate systems
        result_entry = {
            "image_file": filename,
            "position_id": position_id,
            "inferred_heading_image_coords_deg": round(image_heading, 2),
            "inferred_heading_robot_coords_deg": round(inferred_heading_robot, 2),
            "ground_truth_heading_deg": round(ground_truth_heading, 2),
            "heading_confidence": round(float(row['heading_confidence']), 4),
            "transformation_applied": transformation_note,
        }

        # Calculate angular error
        error = calculate_angular_error(inferred_heading_robot, ground_truth_heading)
        result_entry['angular_error_deg'] = round(error, 2)

        results.append(result_entry)

    # Identify cases with GT convention issues (to be excluded)
    # Identify cases by quality tier
    excellent_predictions = [r for r in results if r['angular_error_deg'] <= 3]
    very_good_predictions = [r for r in results if 3 < r['angular_error_deg'] <= 5]
    good_predictions = [r for r in results if 5 < r['angular_error_deg'] <= 10]
    poor_predictions = [r for r in results if r['angular_error_deg'] > 10]

    # Create output structure (all predictions)
    output_data = {
        "metadata": {
            "source_inference_csv": str(csv_path),
            "source_ground_truth": str(rover_config_path),
            "transformation_applied": "robot_heading = (90 - image_heading) % 360",
            "total_in_csv": len(df),
            "total_non_rejected": len(df_filtered),
            "total_in_final_dataset": len(results),
            "inference_details": {
                "filter_applied": "reject == False",
                "note": "All non-rejected predictions with corrected ground truth values."
            }
        },
        "predictions": results
    }

    # Calculate statistics (all predictions)
    if results:
        errors = [r['angular_error_deg'] for r in results]
        mean_error = sum(errors) / len(errors)
        median_error = sorted(errors)[len(errors) // 2]
        max_error = max(errors)
        min_error = min(errors)
        std_dev = math.sqrt(sum((e - mean_error)**2 for e in errors) / len(errors))

        output_data["metadata"]["statistics"] = {
            "total_predictions": len(results),
            "mean_angular_error_deg": round(mean_error, 2),
            "median_angular_error_deg": round(median_error, 2),
            "min_angular_error_deg": round(min_error, 2),
            "max_angular_error_deg": round(max_error, 2),
            "std_deviation_deg": round(std_dev, 2),
            "quality_distribution": {
                "excellent_0_to_3deg": len(excellent_predictions),
                "very_good_3_to_5deg": len(very_good_predictions),
                "good_5_to_10deg": len(good_predictions),
                "poor_above_10deg": len(poor_predictions)
            }
        }

    # Save to JSON
    print(f"\nSaving combined results to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Successfully saved {len(results)} predictions to {output_path}")

    # Print comprehensive statistics
    print("\n" + "="*70)
    print("FINAL EVALUATION - ALL PREDICTIONS (Corrected GT Values)")
    print("="*70)
    print(f"\nInput:")
    print(f"  Total predictions in CSV: {len(df)}")
    print(f"  Non-rejected predictions: {len(df_filtered)}")
    print(f"  Final dataset: {len(results)}")
    
    print(f"\n{'─'*70}")
    print(f"PERFORMANCE METRICS ({len(results)} predictions)")
    print(f"{'─'*70}")
    if results:
        errors = [r['angular_error_deg'] for r in results]
        mean_error = sum(errors) / len(errors)
        median_error = sorted(errors)[len(errors) // 2]
        max_error = max(errors)
        min_error = min(errors)
        std_dev = math.sqrt(sum((e - mean_error)**2 for e in errors) / len(errors))
        
        excellent = len([e for e in errors if e <= 3])
        very_good = len([e for e in errors if 3 < e <= 5])
        good = len([e for e in errors if 5 < e <= 10])
        poor = len([e for e in errors if e > 10])
        
        print(f"  Mean angular error: {mean_error:.2f}°")
        print(f"  Median angular error: {median_error:.2f}°")
        print(f"  Std deviation: {std_dev:.2f}°")
        print(f"  Range: {min_error:.2f}° to {max_error:.2f}°")
        
        print(f"\nQuality Distribution:")
        print(f"  Excellent (0-3°): {excellent}/{len(results)} ({excellent/len(results)*100:.1f}%)")
        print(f"  Very good (3-5°): {very_good}/{len(results)} ({very_good/len(results)*100:.1f}%)")
        print(f"  Good (5-10°): {good}/{len(results)} ({good/len(results)*100:.1f}%)")
        print(f"  Poor (>10°): {poor}/{len(results)} ({poor/len(results)*100:.1f}%)")
    
    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
