import os
import sys
import argparse
import json
import csv
import yaml
import cv2
import math
import numpy as np
from tqdm import tqdm

# Add root folder and ai-engine to path for import safety
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "ai-engine"))
sys.path.append(os.path.join(BASE_DIR, "src"))

# Import our custom modules from the new ai-engine
from density.scalnet_adapter import SCALNetAdapter
from grid.generator import GridGenerator
from flow.engine import OpticalFlowEngine
from fusion.aggregator import FusionAggregator
from fusion.tracker import TemporalTracker

try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

def run_shanghaitech(args):
    # 1. Load Configurations
    print(f"Loading configuration from {args.config}...")
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    grid_cfg = config.get("grid", {})
    flow_cfg = config.get("flow", {})
    risk_cfg = config.get("risk", {})

    grid_size = grid_cfg.get("grid_size_px", 100)
    overlap_ratio = grid_cfg.get("overlap_ratio", 0.20)
    max_overlap_ratio = grid_cfg.get("max_overlap_ratio", 0.25)

    flow_method = flow_cfg.get("method", "dis")
    density_red = risk_cfg.get("density_red", 4.0)
    expected_dir = risk_cfg.get("expected_direction", "EAST")
    normal_speed = float(flow_cfg.get("normal_speed_px", 3.0))

    # 2. Get list of images
    img_dir = os.path.join(args.dataset_dir, "images")
    mat_dir = os.path.join(args.dataset_dir, "ground_truth")

    if not os.path.exists(img_dir):
        print(f"Error: Images directory not found at {img_dir}")
        sys.exit(1)

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    import re
    def natural_keys(text):
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(text))]

    img_names = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)], key=natural_keys)

    if not img_names:
        print(f"Error: No images found in {img_dir}")
        sys.exit(1)

    if args.max_images:
        img_names = img_names[:args.max_images]

    print(f"Found {len(img_names)} images in dataset directory.")

    # Get dimensions from first image
    first_img_path = os.path.join(img_dir, img_names[0])
    first_img = cv2.imread(first_img_path)
    if first_img is None:
        print(f"Error: Could not read first image: {first_img_path}")
        sys.exit(1)

    height, width = first_img.shape[:2]

    # Target resize width to accelerate
    if args.resize_width and width > args.resize_width:
        aspect_ratio = height / width
        width = args.resize_width
        height = int(width * aspect_ratio)

    # 3. Initialize AI Engine Components
    print("Initializing Sudharshan AI Core components...")

    # 3.1 Overlapping Grid Generator
    grid_gen = GridGenerator(grid_size=grid_size, overlap_ratio=overlap_ratio, max_overlap_ratio=max_overlap_ratio)
    grids = grid_gen.generate_grids(width=width, height=height)
    overlap_weights = grid_gen.compute_overlap_weights(width=width, height=height, grids=grids)
    adjacency_graph = grid_gen.build_adjacency_graph(grids, expected_direction=expected_dir)
    print(f"Generated {len(grids)} overlapping grids ({grid_size}px, {int(overlap_ratio*100)}% overlap).")

    # 3.2 SCALNet Density Estimator
    checkpoint_path = os.path.join(BASE_DIR, "SCALNet", "checkpoints", "model.pth")
    scalnet_root = os.path.join(BASE_DIR, "SCALNet")
    adapter = SCALNetAdapter(
        scalnet_root=scalnet_root,
        checkpoint_path=checkpoint_path,
        device="auto",
        use_onnx=args.use_onnx
    )
    adapter.load()

    # 3.3 DIS/Farneback Optical Flow Engine
    flow_engine = OpticalFlowEngine(method=flow_method)

    # 3.4 Fusion Aggregator (Scale density_red to match pixel grid counts)
    density_red_scaled = 5.0 / (grid_size * grid_size)
    aggregator = FusionAggregator(
        density_red=density_red_scaled,
        speed_normal=normal_speed,
        expected_direction=expected_dir,
    )

    # 3.5 Temporal Tracker
    tracker = TemporalTracker(alpha=0.25, fps=5.0, predictor_type=args.predictor_type)

    # 4. Processing Loop
    telemetry_records = []
    prev_frame = None

    # Accuracy evaluation metrics
    count_differences = []
    gt_counts = []
    pred_counts = []

    print("\nProcessing ShanghaiTech image sequence...")
    pbar = tqdm(total=len(img_names), desc="ShanghaiTech Evaluator")

    for idx, img_name in enumerate(img_names):
        img_path = os.path.join(img_dir, img_name)
        curr_frame = cv2.imread(img_path)
        if curr_frame is None:
            continue

        if args.resize_width:
            curr_frame = cv2.resize(curr_frame, (width, height))

        # 1. SCALNet Density Inference
        density_res = adapter.infer(curr_frame)
        pred_cnt = density_res.estimated_count
        pred_counts.append(pred_cnt)

        # Load Ground Truth count if available
        gt_cnt = None
        base_name, _ = os.path.splitext(img_name)
        # ST mat files usually named GT_IMG_1.mat matching IMG_1.jpg
        num_match = re.search(r'\d+', base_name)
        if num_match:
            img_num = num_match.group()
            mat_name = f"GT_IMG_{img_num}.mat"
            mat_path = os.path.join(mat_dir, mat_name)

            if os.path.exists(mat_path) and HAS_SCIPY:
                try:
                    mat = sio.loadmat(mat_path)
                    gt_cnt = float(mat['image_info'][0,0]['number'][0,0][0,0])
                    gt_counts.append(gt_cnt)
                    count_differences.append(pred_cnt - gt_cnt)
                except Exception as mat_err:
                    pass

        # 2. Optical Flow (if previous frame exists)
        if prev_frame is None:
            prev_frame = curr_frame.copy()
            pbar.update(1)
            continue

        flow_res = flow_engine.calculate_flow(prev_frame, curr_frame, crowd_mask=density_res.crowd_mask)

        # 3. Spatial Fusion
        raw_metrics = aggregator.fuse(grids, density_res, flow_res, overlap_weights=overlap_weights)

        # 4. Temporal Tracker
        stabilized_metrics = tracker.track(raw_metrics, adjacency_graph=adjacency_graph)

        # 5. Populate Grid Telemetry
        timestamp_sec = idx / 5.0
        for g in grids:
            m = stabilized_metrics[g.grid_id]
            telemetry_records.append({
                "image_name": img_name,
                "frame_id": idx,
                "timestamp_sec": round(timestamp_sec, 2),
                "grid_id": g.grid_id,
                "crowd_count": round(m.count, 2),
                "density": round(m.density, 6),
                "mean_dx": m.flow_x,
                "mean_dy": m.flow_y,
                "motion_magnitude": m.speed,
                "direction": m.direction_label,
                "direction_deg": m.direction_deg,
                "speed_level": "FAST" if m.speed > 2.0 else ("MODERATE" if m.speed > 0.5 else "SLOW"),
                "flow_conflict": m.reverse_score > 0.5,
                "coherence": 1.0 - m.flow_conflict_score,
                "risk_score": m.congestion_score,
                "risk_level": m.risk_level,
                "turbulence_score": round(m.turbulence_score, 4),
                "speed_surge_warning": m.speed_surge_warning,
                "stasis_warning": m.stasis_warning,
                "turbulence_warning": m.turbulence_warning,
                "crowd_present": m.crowd_present,
                "predicted_count": round(m.predicted_count, 2),
                "predicted_congestion_score": round(m.predicted_congestion_score, 2),
                "predicted_risk_level": m.predicted_risk_level,
                "trend_slope": round(m.trend_slope, 4),
                "ground_truth_total_count": gt_cnt if gt_cnt is not None else -1.0
            })

        prev_frame = curr_frame.copy()
        pbar.update(1)

    pbar.close()

    # Save Output JSON/CSV
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "shanghaitech_grid_metrics.json")
    with open(json_path, "w") as f:
        json.dump(telemetry_records, f, indent=2)
    print(f"Saved JSON metrics to: {json_path}")

    csv_path = os.path.join(args.output_dir, "shanghaitech_grid_metrics.csv")
    if telemetry_records:
        keys = telemetry_records[0].keys()
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(telemetry_records)
        print(f"Saved CSV metrics to: {csv_path}")

    # Print Accuracy Metrics
    print("\n==========================================================")
    print("             SHANGHAITECH DENSITY-FLOW EVALUATION         ")
    print("==========================================================")
    print(f"Total processed frames: {len(img_names)}")
    if count_differences:
        count_differences = np.array(count_differences)
        mae = float(np.mean(np.abs(count_differences)))
        rmse = float(math.sqrt(np.mean(count_differences ** 2)))
        avg_gt = float(np.mean(gt_counts))
        avg_pred = float(np.mean(pred_counts))

        print(f"Ground Truth Count Range: {min(gt_counts):.1f} to {max(gt_counts):.1f} people")
        print(f"Average Ground Truth Count: {avg_gt:.2f} people")
        print(f"Average Predicted Count:    {avg_pred:.2f} people")
        print(f"Head Counting Accuracy:")
        print(f"  Mean Absolute Error (MAE): {mae:.4f} people")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.4f} people")

        # Write validation report
        doc_path = os.path.join(BASE_DIR, "docs", "SHANGHAITECH_EVALUATION.md")
        doc_content = f"""# ShanghaiTech Dataset Hardened Crowd-Flow Evaluation

This report documents the performance of the **Sudharshan** crowd flow detection engine evaluated over the dense crowd scenes in the **ShanghaiTech** dataset.

## 1. Sequence Details
* **Source Folder**: `{args.dataset_dir}`
* **Total Images Processed**: {len(img_names)}
* **Resolution**: {width}x{height}

## 2. Density Verification (SCALNet Counting)
* **Average Ground Truth Count**: {avg_gt:.2f} people
* **Average Predicted Count**: {avg_pred:.2f} people
* **Head Counting MAE**: {mae:.4f} people
* **Head Counting RMSE**: {rmse:.4f} people

## 3. Combined Flow Telemetry Output
* Output saved to `outputs/shanghaitech_grid_metrics.json`
* Output includes local grid-wise motion speeds, directions, risk scores, stasis/turbulence warnings, and linear trend forecasts.

---
*Report generated automatically.*
"""
        with open(doc_path, "w") as f:
            f.write(doc_content)
        print(f"Saved evaluation document to: {doc_path}")
    else:
        print("Note: Scipy not available or ground truth mat files not matched. MAE/RMSE skipped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sudharshan ShanghaiTech Dataset Ingestion and Fusion Pipeline")
    parser.add_argument("--dataset_dir", default="ShanghaiTech/part_A_final/test_data", help="Path to ShanghaiTech dataset part folder")
    parser.add_argument("--config", default="configs/grid_config.yaml", help="Path to configuration YAML")
    parser.add_argument("--output_dir", default="outputs/shanghaitech", help="Directory to save output files")
    parser.add_argument("--max_images", type=int, default=5, help="Limit number of images to evaluate (default: 5)")
    parser.add_argument("--resize_width", type=int, default=960, help="Resize width (default: 960)")
    parser.add_argument("--predictor_type", default="linear", choices=["linear", "gru", "gnn"], help="Predictor type")
    parser.add_argument("--use_onnx", action="store_true", help="Use ONNX model for density inference")

    args = parser.parse_args()
    run_shanghaitech(args)
