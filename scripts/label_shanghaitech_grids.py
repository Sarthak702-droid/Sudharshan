import os
import sys
import json
import yaml
import re
import argparse
import numpy as np
import scipy.io as sio
from tqdm import tqdm

# Add directories to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "ai-engine"))

from grid.generator import GridGenerator

def label_shanghaitech_dataset(dataset_dir, config_path, output_json_path):
    print(f"Loading configuration from {config_path}...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    grid_cfg = config.get("grid", {})
    grid_size = grid_cfg.get("grid_size_px", 100)
    overlap_ratio = grid_cfg.get("overlap_ratio", 0.20)
    max_overlap_ratio = grid_cfg.get("max_overlap_ratio", 0.25)

    img_dir = os.path.join(dataset_dir, "images")
    mat_dir = os.path.join(dataset_dir, "ground_truth")

    if not os.path.exists(img_dir) or not os.path.exists(mat_dir):
        print("Error: Images or ground_truth folder not found.")
        sys.exit(1)

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    def natural_keys(text):
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(text))]

    img_names = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)], key=natural_keys)

    print(f"Found {len(img_names)} images to label.")

    # Grid generator
    grid_gen = GridGenerator(grid_size=grid_size, overlap_ratio=overlap_ratio, max_overlap_ratio=max_overlap_ratio)

    frames_annotations = {}

    for idx, img_name in enumerate(tqdm(img_names, desc="Labeling Grids")):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        height, width = img.shape[:2]

        # Generate grids for this frame's dimensions
        grids = grid_gen.generate_grids(width=width, height=height)

        # Load MAT annotations
        base_name, _ = os.path.splitext(img_name)
        num_match = re.search(r'\d+', base_name)
        if not num_match:
            continue

        img_num = num_match.group()
        mat_name = f"GT_IMG_{img_num}.mat"
        mat_path = os.path.join(mat_dir, mat_name)

        if not os.path.exists(mat_path):
            continue

        try:
            mat = sio.loadmat(mat_path)
            # Coordinates of heads: shape (N, 2) -> (x, y)
            head_locations = mat['image_info'][0,0]['location'][0,0]
        except Exception as e:
            print(f"Warning: Could not read {mat_name}: {e}")
            continue

        frame_id = str(idx)
        frames_annotations[frame_id] = []

        # Group head locations into grids
        grid_counts = {g.grid_id: 0 for g in grids}

        for pt in head_locations:
            x, y = pt[0], pt[1]
            for g in grids:
                # Check if point falls inside grid boundaries
                if g.x1 <= x <= g.x2 and g.y1 <= y <= g.y2:
                    grid_counts[g.grid_id] += 1

        for g in grids:
            count = grid_counts[g.grid_id]
            ann = {
                "grid_id": g.grid_id,
                "crowd_present": count > 0,
                "count_estimate": int(count),
                "dominant_direction_deg": 0.0, # static image has no direction
                "direction_tolerance_deg": 15.0,
                "motion_level": "STATIONARY",
                "counter_flow": False,
                "stagnation": True,
                "collective_speed_surge": False,
                "dispersal": False,
                "annotation_confidence": 1.0 # direct point annotations
            }
            frames_annotations[frame_id].append(ann)

    output_data = {
        "sequence_id": os.path.basename(os.path.dirname(dataset_dir)),
        "frames": frames_annotations
    }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSuccessfully generated {len(frames_annotations)} frame-grid annotations.")
    print(f"Saved JSON labels to: {output_json_path}")

if __name__ == "__main__":
    import cv2  # import here to verify cv2 is loaded

    parser = argparse.ArgumentParser(description="Convert ShanghaiTech point annotations to grid-level JSON format")
    parser.add_argument("--dataset_dir", default="ShanghaiTech/part_A_final/test_data", help="Path to ShanghaiTech dataset part folder")
    parser.add_argument("--config", default="configs/grid_config.yaml", help="Path to configuration YAML")
    parser.add_argument("--output_json", default=None, help="Path to save output JSON annotations (auto-detected if None)")

    args = parser.parse_args()

    # Auto-resolve output path if not specified
    if not args.output_json:
        if "part_B" in args.dataset_dir or "part_b" in args.dataset_dir:
            args.output_json = "configs/labels/shanghaitech_part_b_labels.json"
        else:
            args.output_json = "configs/labels/shanghaitech_part_a_labels.json"

    label_shanghaitech_dataset(args.dataset_dir, args.config, args.output_json)
