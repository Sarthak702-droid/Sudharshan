import os
import sys
import argparse
import json
import csv
import yaml
import cv2
import numpy as np
from tqdm import tqdm

# Add root folder and ai-engine to path for import safety
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "ai-engine"))
sys.path.append(os.path.join(BASE_DIR, "src"))

# Import our custom modules from the new ai-engine
from density.scalnet_adapter import SCALNetAdapter
from ingest.reader import FrameReader
from grid.generator import GridGenerator
from flow.engine import OpticalFlowEngine
from fusion.aggregator import FusionAggregator
from fusion.tracker import TemporalTracker

# Import the visualization renderer from legacy src
from visualization.renderer import VisualizationRenderer


def load_yaml_config(config_path):
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    # Check under the repository root.
    alt_path = os.path.join(BASE_DIR, config_path)
    if os.path.exists(alt_path):
        with open(alt_path, "r") as f:
            return yaml.safe_load(f)
    # Check absolute location under BASE_DIR
    alt_path2 = os.path.join(BASE_DIR, "configs", "grid_config.yaml")
    if os.path.exists(alt_path2):
        with open(alt_path2, "r") as f:
            return yaml.safe_load(f)
    raise FileNotFoundError(f"Configuration file not found at {config_path}")


def run_pipeline(args):
    # 1. Load Configurations
    print(f"Loading configuration from {args.config}...")
    config = load_yaml_config(args.config)

    grid_cfg = config.get("grid", {})
    flow_cfg = config.get("flow", {})
    risk_cfg = config.get("risk", {})

    grid_size = grid_cfg.get("grid_size_px", 100)
    overlap_ratio = grid_cfg.get("overlap_ratio", 0.20)
    max_overlap_ratio = grid_cfg.get("max_overlap_ratio", 0.25)

    flow_method = flow_cfg.get("method", "dis")

    density_red = risk_cfg.get("density_red", 4.0)
    expected_dir = args.expected_direction or risk_cfg.get("expected_direction", "EAST")
    normal_speed = float(flow_cfg.get("normal_speed_px", 3.0))

    # 2. Resolve target dimensions
    # Probe frame dimension from input to determine resizing factor
    width, height = 960, 540  # Defaults
    if os.path.isdir(args.input):
        valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
        frame_names = sorted([f for f in os.listdir(args.input) if f.lower().endswith(valid_extensions)])
        if not frame_names:
            raise ValueError(f"No image frames found in directory: {args.input}")
        first_frame_path = os.path.join(args.input, frame_names[0])
        first_frame = cv2.imread(first_frame_path)
        if first_frame is not None:
            height, width = first_frame.shape[:2]
            total_frames = len(frame_names)
    else:
        cap_probe = cv2.VideoCapture(args.input)
        if cap_probe.isOpened():
            width = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap_probe.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_probe.release()
        else:
            raise ValueError(f"Could not open input video source: {args.input}")

    # Set Resize Scaling Factor
    if args.resize_width and width > args.resize_width:
        aspect_ratio = height / width
        new_width = args.resize_width
        new_height = int(new_width * aspect_ratio)
        scale_factor = new_width / width
        print(f"Resizing active frame resolution from {width}x{height} to {new_width}x{new_height} (Scale Factor: {scale_factor:.3f})")
        width = new_width
        height = new_height
    else:
        scale_factor = 1.0

    # 3. Initialize AI Engine Components
    print("Initializing Sudharshan AI Core components...")

    # 3.1 Overlapping Grid Generator
    boundary_polygon = None  #monitored corridor defaults to None (full frame)
    grid_gen = GridGenerator(grid_size=grid_size, overlap_ratio=overlap_ratio, max_overlap_ratio=max_overlap_ratio)
    grids = grid_gen.generate_grids(width=width, height=height, boundary_polygon=boundary_polygon)
    overlap_weights = grid_gen.compute_overlap_weights(width=width, height=height, grids=grids)
    adjacency_graph = grid_gen.build_adjacency_graph(grids, expected_direction=expected_dir)
    print(f"Generated {len(grids)} overlapping grids ({grid_size}px size, {int(overlap_ratio*100)}% overlap). Precomputed overlap weights and adjacency graph.")

    # 3.2 SCALNet Density Estimator
    trained_checkpoint = os.path.join(BASE_DIR, "SCALNet", "outputs", "shanghaitech_scalnet", "best.h5")
    original_checkpoint = os.path.join(BASE_DIR, "SCALNet", "checkpoints", "model.pth")
    checkpoint_path = args.checkpoint or (original_checkpoint if args.use_onnx else (
        trained_checkpoint if os.path.exists(trained_checkpoint) else original_checkpoint
    ))
    print(f"Using SCALNet checkpoint: {checkpoint_path}")
    scalnet_root = os.path.join(BASE_DIR, "SCALNet")
    adapter = SCALNetAdapter(
        scalnet_root=scalnet_root,
        checkpoint_path=checkpoint_path,
        device="auto",
        use_onnx=args.use_onnx
    )
    adapter.load()

    # 3.3 DIS/Farneback Optical Flow Engine
    flow_engine = OpticalFlowEngine(
        method=flow_method,
        min_motion_px=float(flow_cfg.get("min_motion_px", 0.05)),
        max_motion_px=float(flow_cfg.get("max_motion_px", 40.0)),
        scene_cut_threshold=float(flow_cfg.get("scene_cut_threshold", 45.0)),
        min_blur_score=float(flow_cfg.get("min_blur_score", 12.0)),
    )

    # 3.4 Fusion Aggregator (Scale density_red to match pixel grid counts)
    # Approx 5 people in grid is critical -> 5 / area
    density_red_scaled = 5.0 / (grid_size * grid_size)
    fps = args.fps if args.fps else 15
    aggregator = FusionAggregator(
        density_red=density_red_scaled,
        speed_normal=normal_speed,
        expected_direction=expected_dir,
        min_confidence=float(risk_cfg.get("min_confidence", 0.60)),
        crowd_presence_count=float(risk_cfg.get("crowd_presence_count", 1.0)),
        crowd_class_thresholds=tuple(risk_cfg.get("crowd_class_thresholds", [1.0, 3.0, 6.0, 10.0])),
        crowd_density_thresholds=tuple(risk_cfg.get("crowd_density_thresholds", [0.5, 2.0, 4.0, 6.0])),
        fps=fps,
        meters_per_pixel=flow_cfg.get("meters_per_pixel"),
    )

    # 3.5 Temporal Tracker (EMA and Alert Hysteresis)
    tracker = TemporalTracker(alpha=0.25, fps=fps, predictor_type=args.predictor_type)

    # 3.6 Legacy Visual Renderer (opacity translucent grid overlay)
    renderer = VisualizationRenderer(opacity=0.30)

    # 4. Prepare Video Writer and Rendered Frames Directory
    os.makedirs(os.path.dirname(os.path.abspath(args.output_video)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(args.output_video, fourcc, fps, (width, height))

    # Clean and recreate outputs/rendered_frames for browser playback
    frames_out_dir = os.path.join(os.path.dirname(args.output_video), "rendered_frames")
    if os.path.exists(frames_out_dir):
        import shutil
        try:
            shutil.rmtree(frames_out_dir)
        except Exception as e:
            print(f"Warning: Could not clear frames directory: {e}")
    os.makedirs(frames_out_dir, exist_ok=True)

    # Keep track of telemetry outputs
    telemetry_records = []

    # 5. Execute Frame Ingestion Reader Loop
    reader = FrameReader(source=args.input, realtime=False)
    reader.start()

    prev_frame_obj = reader.read()
    if prev_frame_obj is None:
        raise ValueError(f"Could not read any frames from input: {args.input}")

    prev_frame = prev_frame_obj.frame
    if scale_factor != 1.0:
        prev_frame = cv2.resize(prev_frame, (width, height))

    # Adjust tqdm total if max_frames is set
    display_total = total_frames - 1 if not args.max_frames else args.max_frames
    pbar = tqdm(total=display_total, desc="Processing Frames")

    while True:
        curr_frame_obj = reader.read()
        if curr_frame_obj is None:
            break

        frame_idx = curr_frame_obj.frame_index
        curr_frame = curr_frame_obj.frame
        if scale_factor != 1.0:
            curr_frame = cv2.resize(curr_frame, (width, height))

        # Check frame limit early exit
        if args.max_frames and frame_idx > args.max_frames:
            break

        # 1. SCALNet Density Inference (applying cadence optimization)
        if frame_idx % args.density_cadence == 0 or 'density_res' not in locals():
            density_res = adapter.infer(prev_frame)

        # 2. Dense Optical Flow Calculation
        flow_res = flow_engine.calculate_flow(prev_frame, curr_frame, crowd_mask=density_res.crowd_mask)

        # 3. Spatial Fusion
        raw_metrics = aggregator.fuse(grids, density_res, flow_res, overlap_weights=overlap_weights)

        # 4. Temporal Stability Tracker (passing adjacency graph)
        stabilized_metrics = tracker.track(raw_metrics, adjacency_graph=adjacency_graph)

        # 5. Populate Grid Telemetry Records (maintaining backward-compatible properties)
        active_alerts_count = 0
        grid_combined_metrics = {}
        timestamp_sec = frame_idx / fps

        for g in grids:
            grid_id = g.grid_id
            m = stabilized_metrics[grid_id]

            if m.alert_eligible and (m.risk_level in ("ORANGE", "RED") or m.reverse_score > 0.5):
                active_alerts_count += 1

            grid_combined_metrics[grid_id] = {
                "frame_id": frame_idx,
                "timestamp_sec": round(timestamp_sec, 2),
                "grid_id": grid_id,
                "row": g.row,
                "col": g.col,
                "x1": g.x1,
                "y1": g.y1,
                "x2": g.x2,
                "y2": g.y2,
                "mean_dx": m.flow_x,
                "mean_dy": m.flow_y,
                "motion_magnitude": m.speed,
                "direction": m.direction_label,
                "direction_deg": m.direction_deg,
                "speed_level": "FAST" if m.speed > 2.0 else ("MODERATE" if m.speed > 0.5 else "SLOW"),
                "crowd_count": round(m.count, 2),
                "density": round(m.density, 6),
                "flow_conflict": m.reverse_score > 0.5,
                "coherence": 1.0 - m.flow_conflict_score,
                "reverse_score": m.reverse_score,
                "risk_score": m.congestion_score,
                "risk_level": m.risk_level,
                "confidence": m.confidence,
                "flow_quality": m.flow_quality,
                "valid_flow_ratio": m.valid_flow_ratio,
                "alert_eligible": m.alert_eligible,
                "turbulence_score": round(m.turbulence_score, 4),
                "speed_surge_warning": m.speed_surge_warning,
                "stasis_warning": m.stasis_warning,
                "turbulence_warning": m.turbulence_warning,
                "crowd_present": m.crowd_present,
                "crowd_class": m.crowd_class,
                "crowd_probability": round(m.crowd_probability, 4),
                "physical_calibrated": m.physical_calibrated,
                "speed_mps": m.speed_mps,
                "density_people_m2": m.density_people_m2,
                "divergence": m.divergence,
                "acceleration": m.acceleration,
                "predicted_count": round(m.predicted_count, 2),
                "predicted_congestion_score": round(m.predicted_congestion_score, 2),
                "predicted_risk_level": m.predicted_risk_level,
                "trend_slope": round(m.trend_slope, 4)
            }

            telemetry_records.append(grid_combined_metrics[grid_id])

        # 6. Render Visualization Overlay using legacy renderer
        legacy_grids = [{
            "grid_id": g.grid_id,
            "row": g.row,
            "col": g.col,
            "x1": g.x1,
            "y1": g.y1,
            "x2": g.x2,
            "y2": g.y2,
            "area": g.area
        } for g in grids]

        visualized_frame = renderer.render_overlay(curr_frame, legacy_grids, grid_combined_metrics, active_alerts_count)

        # Write frame to video
        video_writer.write(visualized_frame)

        # Save frame as JPEG for browser frame-perfect playback
        try:
            cv2.imwrite(os.path.join(frames_out_dir, f"frame_{frame_idx:06d}.jpg"), visualized_frame)
        except Exception as e:
            print(f"Warning: Failed to write frame {frame_idx}: {e}")

        # Rotate frames
        prev_frame = curr_frame.copy()
        pbar.update(1)

    # Cleanup
    pbar.close()
    reader.stop()
    video_writer.release()
    print(f"Processed video written to {args.output_video}")

    # 6. Save JSON Metrics
    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(telemetry_records, f, indent=2)
    print(f"Grid metrics JSON written to {args.output_json}")

    # 7. Save CSV Metrics
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    if telemetry_records:
        keys = telemetry_records[0].keys()
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(telemetry_records)
        print(f"Grid metrics CSV written to {args.output_csv}")

    # 8. Optionally POST to Go Backend (Offline Telemetry Endpoint)
    backend_cfg = config.get("backend", {})
    if backend_cfg.get("enabled", False) or args.post_to_backend:
        base_url = backend_cfg.get("base_url", "http://localhost:8080")
        path = backend_cfg.get("telemetry_path", "/api/v1/telemetry/grid-metrics")
        url = f"{base_url}{path}"
        print(f"POSTing telemetry metrics to Go Backend: {url}...")

        # Prepare Go backend data contract format
        go_payload = []
        import time
        now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for r in telemetry_records:
            go_payload.append({
                "timestamp": now_str,
                "camera_id": "CAMERA_01",
                "zone_id": "ZONE_01",
                "frame_id": r["frame_id"],
                "grid_id": r["grid_id"],
                "count": r["crowd_count"],
                "density": r["density"],
                "flow_x": r["mean_dx"],
                "flow_y": r["mean_dy"],
                "direction_deg": r["direction_deg"],
                "direction_label": r["direction"],
                "relative_speed": r["motion_magnitude"],
                "speed_level": r["speed_level"],
                "coherence": r["coherence"],
                "reverse_score": r["reverse_score"],
                "conflict_score": 1.0 - r["coherence"],
                "congestion_score": r["risk_score"],
                "risk_level": r["risk_level"],
                "confidence": r["confidence"],
                "flow_quality": r.get("flow_quality", 0.0),
                "valid_flow_ratio": r.get("valid_flow_ratio", 0.0),
                "alert_eligible": r.get("alert_eligible", False),
                "turbulence_score": r.get("turbulence_score", 0.0),
                "speed_surge_warning": r.get("speed_surge_warning", False),
                "stasis_warning": r.get("stasis_warning", False),
                "turbulence_warning": r.get("turbulence_warning", False),
                "crowd_present": r.get("crowd_present", False),
                "crowd_class": r.get("crowd_class", "EMPTY"),
                "crowd_probability": r.get("crowd_probability", 0.0),
                "physical_calibrated": r.get("physical_calibrated", False),
                "speed_mps": r.get("speed_mps", 0.0),
                "density_people_m2": r.get("density_people_m2", 0.0),
                "divergence": r.get("divergence", 0.0),
                "acceleration": r.get("acceleration", 0.0),
                "predicted_count": r.get("predicted_count", 0.0),
                "predicted_congestion_score": r.get("predicted_congestion_score", 0.0),
                "predicted_risk_level": r.get("predicted_risk_level", "GREEN"),
                "trend_slope": r.get("trend_slope", 0.0)
            })

        import urllib.request
        req = urllib.request.Request(
            url,
            data=json.dumps(go_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status in (200, 201):
                    print("[+] Successfully posted telemetry to Go Backend!")
                else:
                    print(f"[-] Go Backend returned error status: {response.status}")
        except Exception as e:
            print(f"[-] Could not connect/post to Go Backend: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sudharshan AI Crowd Flow Detection Pipeline")
    parser.add_argument("--input", required=True, help="Path to input video file or image sequence directory")
    parser.add_argument("--labels", help="Path to JSON file containing custom point annotations")
    parser.add_argument("--config", default="configs/grid_config.yaml", help="Path to configuration YAML file")
    parser.add_argument("--output_video", default="outputs/flow_overlay.mp4", help="Path to save processed video file")
    parser.add_argument("--output_json", default="outputs/grid_metrics.json", help="Path to save metrics JSON file")
    parser.add_argument("--output_csv", default="outputs/grid_metrics.csv", help="Path to save metrics CSV file")
    parser.add_argument("--fps", type=int, help="Override frame rate for output video")
    parser.add_argument("--expected_direction", help="Override expected flow direction label (e.g. EAST, WEST)")
    parser.add_argument("--max_frames", type=int, help="Limit number of frames to process")
    parser.add_argument("--resize_width", type=int, default=960, help="Resize input frames to this width to accelerate processing (default: 960)")
    parser.add_argument("--post_to_backend", action="store_true", help="Post telemetry directly to Go backend")
    parser.add_argument("--predictor_type", default="linear", choices=["linear", "gru", "gnn"], help="Type of predictor to use (default: linear)")
    parser.add_argument("--density_cadence", type=int, default=4, help="Run density inference every N frames (default: 4)")
    parser.add_argument("--use_onnx", action="store_true", help="Run density inference using exported ONNX model instead of PyTorch")
    parser.add_argument("--checkpoint", help="Override SCALNet HDF5 checkpoint path")

    args = parser.parse_args()
    run_pipeline(args)
