import argparse
import sys
import json
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / "ai-engine") not in sys.path:
    sys.path.insert(0, str(project_root / "ai-engine"))

from ingest.reader import FrameReader
from grid.generator import GridGenerator
from density.scalnet_adapter import SCALNetAdapter
from flow.engine import OpticalFlowEngine
from fusion.aggregator import FusionAggregator


def main():
    parser = argparse.ArgumentParser(description="End-to-end verification script for Step 5 SCALNet + Flow Fusion.")
    parser.add_argument(
        "--source",
        type=str,
        default=str(project_root / "Sudharshan" / "outputs" / "rendered_frames"),
        help="Path to directory of consecutive image frames",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(project_root / "SCALNet" / "checkpoints" / "model.pth"),
        help="Path to SCALNet weights checkpoint file",
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=120,
        help="Grid window size in pixels",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.20,
        help="Grid overlap ratio",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(project_root / "outputs" / "step_5_debug"),
        help="Directory to save fusion debug artifacts",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Monitored corridor polygon boundary
    boundary = [
        (100, 500),  # Bottom Left
        (350, 100),  # Top Left
        (650, 100),  # Top Right
        (900, 500),  # Bottom Right
    ]

    print("==================================================")
    print("STARTING SUDHARSHAN END-TO-END FUSION PIPELINE")
    print("==================================================")

    # 1. Read frames
    print("Step 1: Ingesting consecutive frames using FrameReader...")
    try:
        with FrameReader(source=args.source, realtime=False) as reader:
            frame_obj1 = reader.read()
            frame_obj2 = reader.read()
    except Exception as e:
        print(f"Error reading frames: {e}", file=sys.stderr)
        sys.exit(1)

    if frame_obj1 is None or frame_obj2 is None:
        print("Error: Could not read two consecutive frames.", file=sys.stderr)
        sys.exit(1)

    frame1 = frame_obj1.frame
    frame2 = frame_obj2.frame
    h, w = frame1.shape[:2]
    print(f"Loaded Frame 1: {w}x{h}")
    print(f"Loaded Frame 2: {w}x{h}")

    # 2. Grid segmentation
    print("\nStep 2: Segmenting regions into overlapping grids...")
    grid_gen = GridGenerator(grid_size=args.grid_size, overlap_ratio=args.overlap)
    grids = grid_gen.generate_grids(width=w, height=h, boundary_polygon=boundary)
    print(f"Generated {len(grids)} active grids.")

    # 3. Density Estimation
    print("\nStep 3: Initializing SCALNet adapter and executing density mapping...")
    adapter = SCALNetAdapter(scalnet_root=project_root / "SCALNet", checkpoint_path=args.checkpoint, device="auto")
    adapter.load()
    density_result = adapter.infer(frame1)
    print(f"SCALNet crowd count: {density_result.estimated_count:.3f} people.")
    print(f"SCALNet latency: {density_result.inference_time_ms:.2f} ms.")

    # 4. Dense Optical Flow
    print("\nStep 4: Executing DIS dense optical flow...")
    flow_engine = OpticalFlowEngine(method="dis")
    flow_result = flow_engine.calculate_flow(frame1, frame2)
    print(f"Flow calculation latency: {flow_result.inference_time_ms:.2f} ms.")

    # 5. Fusion Aggregator
    print("\nStep 5: Aggregating and fusing density & flow fields...")
    # Scale density_red to match pixel counts:
    # A density of 4.0 people per m2 is high. Since SCALNet returns values in pixel counts,
    # let's set a density threshold in pixel coordinates.
    # For a 120x120 grid (area = 14400), a count of 5 people is high.
    # density = count / area = 5 / 14400 ~ 0.00035.
    # So we set density_red = 0.00035 (representing 5 people inside a grid box).
    aggregator = FusionAggregator(
        density_red=0.00035,
        speed_normal=3.0,
        expected_direction="EAST",
        min_confidence=0.60,
    )

    grid_metrics = aggregator.fuse(grids, density_result, flow_result)

    # 6. Print summary
    print("\n" + "=" * 90)
    print(f"{'Grid ID':<8} | {'Count':<7} | {'Speed':<7} | {'Direction':<12} | {'Conflict':<8} | {'Congestion':<10} | {'Risk Level':<10}")
    print("=" * 90)

    metrics_list = []

    for idx, (gid, m) in enumerate(grid_metrics.items()):
        metrics_list.append({
            "grid_id": m.grid_id,
            "count": m.count,
            "density": m.density,
            "flow_x": m.flow_x,
            "flow_y": m.flow_y,
            "speed": m.speed,
            "direction_deg": m.direction_deg,
            "direction_label": m.direction_label,
            "density_score": m.density_score,
            "slow_score": m.slow_score,
            "stagnation_score": m.stagnation_score,
            "flow_conflict_score": m.flow_conflict_score,
            "reverse_score": m.reverse_score,
            "congestion_score": m.congestion_score,
            "risk_level": m.risk_level,
            "confidence": m.confidence,
        })

        if idx < 15:
            print(f"{m.grid_id:<8} | {m.count:<7.2f} | {m.speed:<7.2f} | {m.direction_label:<12} | {m.flow_conflict_score:<8.2f} | {m.congestion_score:<10.1f} | {m.risk_level:<10}")

    if len(grid_metrics) > 15:
        print(f"... and {len(grid_metrics) - 15} more grids.")

    print("=" * 90 + "\n")

    # Save JSON log
    json_path = output_dir / "grid_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics_list, f, indent=4)
    print(f"JSON metrics log saved to: {json_path.resolve()}")

    # 7. Render visualization overlay
    canvas = frame2.copy()

    # Draw boundary polygon
    poly_pts = np.array(boundary, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [poly_pts], isClosed=True, color=(0, 0, 150), thickness=2)

    # Color map for risk levels
    risk_colors = {
        "GREEN": (0, 180, 0),       # Translucent Green
        "YELLOW": (0, 220, 220),    # Translucent Yellow
        "ORANGE": (0, 120, 255),    # Translucent Orange
        "RED": (0, 0, 220),         # Translucent Red
    }

    # Translucent overlays
    overlay_risks = canvas.copy()

    for g in grids:
        m = grid_metrics[g.grid_id]
        color = risk_colors.get(m.risk_level, (0, 255, 0))

        # Fill grid rectangle with risk color
        cv2.rectangle(overlay_risks, (g.x1, g.y1), (g.x2, g.y2), color, -1)

        # Draw grid box outline
        cv2.rectangle(canvas, (g.x1, g.y1), (g.x2, g.y2), color, thickness=1)

        # Draw vector arrows for flow
        cx, cy = int((g.x1 + g.x2) / 2), int((g.y1 + g.y2) / 2)
        scale_arrow = 10.0
        arrow_dx = int(m.flow_x * scale_arrow)
        arrow_dy = int(m.flow_y * scale_arrow)

        target_pt = (cx + arrow_dx, cy + arrow_dy)

        if m.speed > 0.1:
            cv2.arrowedLine(canvas, (cx, cy), target_pt, (255, 255, 255), thickness=1, tipLength=0.3)

        # Grid ID label
        label_pos = (g.x1 + 5, g.y1 + 18)
        cv2.putText(canvas, f"{g.grid_id}", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), thickness=1)

        # Risk score label
        score_pos = (g.x1 + 5, g.y1 + 32)
        cv2.putText(canvas, f"C:{m.congestion_score:.0f}", score_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), thickness=1)

    # Blend translucent overlays
    cv2.addWeighted(overlay_risks, 0.25, canvas, 0.75, 0, dst=canvas)

    # Save output visualization
    img_path = output_dir / "fusion_overlay.png"
    cv2.imwrite(str(img_path), canvas)
    print(f"End-to-end Fusion risk overlay visualization saved to: {img_path.resolve()}")
    print("Overlay shows grid blocks colored by risk levels (Green, Yellow, Orange, Red) and flow arrows.")
    print("=" * 60)


if __name__ == "__main__":
    main()
