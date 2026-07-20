import json
import os
import argparse
import numpy as np

def classify_crowds(labels_path, count_threshold):
    if not os.path.exists(labels_path):
        print(f"Error: Labels file {labels_path} does not exist. Run labeler script first.")
        return

    print(f"Loading grid annotations from {labels_path}...")
    with open(labels_path, "r") as f:
        data = json.load(f)

    frames = data.get("frames", {})
    total_grids = 0
    crowd_grids = 0
    non_crowd_grids = 0

    counts = []

    for fid, grid_anns in frames.items():
        for ann in grid_anns:
            total_grids += 1
            cnt = ann["count_estimate"]
            counts.append(cnt)

            # Apply absolute headcount threshold to classify as crowd
            is_crowd = cnt >= count_threshold

            if is_crowd:
                crowd_grids += 1
                ann["crowd_present"] = True
            else:
                non_crowd_grids += 1
                ann["crowd_present"] = False

    counts = np.array(counts)
    crowd_ratio = crowd_grids / total_grids if total_grids > 0 else 0.0

    print("\n==========================================================")
    print("             CROWD CLASSIFICATION REPORT                  ")
    print("==========================================================")
    print(f"Sequence ID:              {data.get('sequence_id', 'Unknown')}")
    print(f"Total Grid Cells Audited: {total_grids}")
    print(f"Headcount Threshold:      {count_threshold} people/grid")
    print(f"Classified as CROWD:      {crowd_grids} ({crowd_ratio:.2%})")
    print(f"Classified as NO CROWD:   {non_crowd_grids} ({1.0-crowd_ratio:.2%})")
    print(f"Grid Count statistics:")
    print(f"  Max Count:   {np.max(counts):.1f} people")
    print(f"  Mean Count:  {np.mean(counts):.2f} people")
    print(f"  Median Count:{np.median(counts):.2f} people")

    # Save updated annotations file with new crowd classifications
    updated_path = labels_path.replace(".json", "_classified.json")
    with open(updated_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved updated classifications to: {updated_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify ShanghaiTech grids as crowd or not based on head counts")
    parser.add_argument("--labels", default="configs/labels/shanghaitech_part_a_labels.json", help="Path to labels JSON")
    parser.add_argument("--threshold", type=int, default=5, help="Headcount threshold to classify a grid cell as crowd (default: 5)")

    args = parser.parse_args()
    classify_crowds(args.labels, args.threshold)
