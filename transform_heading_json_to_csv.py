import json
import csv
from pathlib import Path

# Input and output file paths
json_file = Path("runs/test-orientation/yolo11n-panther_v1-pose-v1/heading_debug.json")
output_csv = Path("runs/test-orientation/yolo11n-panther_v1-pose-v1/heading_estimation_from_debug.csv")

# Read the JSON file
with open(json_file, 'r') as f:
    data = json.load(f)

# Extract records and create CSV data
csv_rows = []
for record in data.get("records", []):
    csv_rows.append({
        "image": record["image_name"],
        "heading_deg": record["v3_deg"],
        "reject": False,
        "heading_confidence": 0.1
    })

# Write to CSV file
with open(output_csv, 'w', newline='') as f:
    fieldnames = ["image", "heading_deg", "reject", "heading_confidence"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"✓ Successfully transformed {len(csv_rows)} records")
print(f"✓ Output saved to: {output_csv}")
