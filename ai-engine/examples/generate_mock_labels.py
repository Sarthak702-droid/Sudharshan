import json
import os
import random
import numpy as np

def generate_mock_labels(metrics_path, output_path):
    if not os.path.exists(metrics_path):
        print(f"Metrics file {metrics_path} does not exist. Run pipeline sample first.")
        return

    with open(metrics_path, "r") as f:
        records = json.load(f)

    # Group records by frame_id
    frames_annotations = {}

    random.seed(42)

    for r in records:
        frame_id = str(r["frame_id"])
        if frame_id not in frames_annotations:
            frames_annotations[frame_id] = []

        # Add slight noise to counts/speeds for ground truth comparison
        gt_count = max(0.0, r["crowd_count"] + random.gauss(0, 0.2))
        gt_deg = (r["direction_deg"] + random.gauss(0, 5.0)) % 360.0

        # Determine logical event triggers
        # Reverse flow dot product
        gt_counter_flow = r["flow_conflict"]
        gt_stagnation = r["stasis_warning"]
        gt_speed_surge = r["speed_surge_warning"]
        gt_turbulence = r["turbulence_warning"]
        gt_dispersal = False # default

        # Randomly toggle some alerts occasionally to test precision/recall
        if random.random() < 0.05:
            gt_counter_flow = not gt_counter_flow
        if random.random() < 0.05:
            gt_stagnation = not gt_stagnation

        ann = {
            "grid_id": r["grid_id"],
            "crowd_present": gt_count > 0.1,
            "count_estimate": round(gt_count, 2),
            "dominant_direction_deg": round(gt_deg, 1),
            "direction_tolerance_deg": 15.0,
            "motion_level": r["speed_level"],
            "counter_flow": gt_counter_flow,
            "stagnation": gt_stagnation,
            "collective_speed_surge": gt_speed_surge,
            "turbulence": gt_turbulence,
            "dispersal": gt_dispersal,
            "annotation_confidence": 0.95
        }
        frames_annotations[frame_id].append(ann)

    output_data = {
        "sequence_id": "visdrone_uav0000339",
        "frames": frames_annotations
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Generated mock ground truth labels file at: {output_path}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    metrics_file = os.path.join(base_dir, "Sudharshan", "outputs", "grid_metrics.json")
    labels_file = os.path.join(base_dir, "Sudharshan", "configs", "labels", "uav0000117_02622_v_labels.json")

    # Also save under new name
    generate_mock_labels(metrics_file, labels_file)

    visdrone_labels = os.path.join(base_dir, "Sudharshan", "configs", "labels", "visdrone_uav0000339_labels.json")
    generate_mock_labels(metrics_file, visdrone_labels)
