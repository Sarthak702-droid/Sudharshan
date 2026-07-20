import argparse
import sys
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
from flow.engine import OpticalFlowEngine


def main():
    parser = argparse.ArgumentParser(description="Verification script for Step 4 Optical-Flow Engine.")
    parser.add_argument(
        "--source",
        type=str,
        default=str(project_root / "Sudharshan" / "outputs" / "rendered_frames"),
        help="Path to directory of consecutive image frames",
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
        default=str(project_root / "outputs" / "step_4_debug"),
        help="Directory to save flow debug artifacts",
    )
    args = parser.parse_args()

    # Define the monitored corridor boundary
    boundary = [
        (100, 500),  # Bottom Left
        (350, 100),  # Top Left
        (650, 100),  # Top Right
        (900, 500),  # Bottom Right
    ]

    print("Step 1: Reading consecutive frames using FrameReader...")
    try:
        with FrameReader(source=args.source, realtime=False) as reader:
            frame_obj1 = reader.read()
            frame_obj2 = reader.read()
    except Exception as e:
        print(f"Error reading frames: {e}", file=sys.stderr)
        sys.exit(1)

    if frame_obj1 is None or frame_obj2 is None:
        print("Error: Could not read two consecutive frames from the source directory.", file=sys.stderr)
        sys.exit(1)

    frame1 = frame_obj1.frame
    frame2 = frame_obj2.frame
    h, w = frame1.shape[:2]
    print(f"Loaded Frame 1: Index {frame_obj1.frame_index}, Size: {w}x{h}")
    print(f"Loaded Frame 2: Index {frame_obj2.frame_index}, Size: {w}x{h}")

    print("\nStep 2: Generating grids inside corridor boundary...")
    grid_gen = GridGenerator(grid_size=args.grid_size, overlap_ratio=args.overlap)
    grids = grid_gen.generate_grids(width=w, height=h, boundary_polygon=boundary)
    print(f"Generated {len(grids)} active grids.")

    print("\nStep 3: Calculating dense optical flow (DIS method)...")
    flow_engine = OpticalFlowEngine(method="dis")
    flow_result = flow_engine.calculate_flow(frame1, frame2)
    print(f"Dense flow calculated in {flow_result.inference_time_ms:.2f} ms.")

    print("\nStep 4: Aggregates dense flow into grid-wise movement vectors...")
    grid_flows = flow_engine.aggregate_grid_flow(flow_result.flow_x, flow_result.flow_y, grids)

    # Print summary of first 10 grids
    print("\n" + "-" * 80)
    print(f"{'Grid ID':<10} | {'u_bar (flow_x)':<15} | {'v_bar (flow_y)':<15} | {'Magnitude':<12} | {'Direction':<12}")
    print("-" * 80)
    for idx, (gid, f) in enumerate(grid_flows.items()):
        if idx >= 10:
            print(f"... and {len(grid_flows) - 10} more grids.")
            break
        print(f"{gid:<10} | {f['flow_x']:15.4f} | {f['flow_y']:15.4f} | {f['magnitude']:12.4f} | {f['direction_label']:<12} ({f['direction_deg']:.1f}°)")
    print("-" * 80 + "\n")

    # Step 5: Render flow overlay visualization
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a canvas starting from the second frame BGR
    canvas = frame2.copy()

    # Draw boundary corridor
    poly_pts = np.array(boundary, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [poly_pts], isClosed=True, color=(0, 0, 200), thickness=2)

    # Draw grid boxes and flow arrows
    for g in grids:
        f = grid_flows[g.grid_id]

        # Draw grid bounding box in faint green
        cv2.rectangle(canvas, (g.x1, g.y1), (g.x2, g.y2), (0, 80, 0), thickness=1)

        # Draw a flow vector arrow from the center of the grid box
        cx, cy = int((g.x1 + g.x2) / 2), int((g.y1 + g.y2) / 2)

        # Scale flow vectors to be visible (e.g. multiply displacement by 10)
        scale_arrow = 10.0
        arrow_dx = int(f["flow_x"] * scale_arrow)
        arrow_dy = int(f["flow_y"] * scale_arrow)

        target_pt = (cx + arrow_dx, cy + arrow_dy)

        # Draw a line and arrow if there is meaningful motion (magnitude > 0.1 pixel)
        if f["magnitude"] > 0.1:
            # Vector arrow in Cyan/Green
            cv2.arrowedLine(canvas, (cx, cy), target_pt, (255, 255, 0), thickness=2, tipLength=0.3)
            # Center marker dot
            cv2.circle(canvas, (cx, cy), 2, (0, 255, 0), -1)
        else:
            # Minimal/no motion: draw a red dot at grid center
            cv2.circle(canvas, (cx, cy), 2, (0, 0, 255), -1)

    # Save output overlay
    output_path = output_dir / "flow_overlay.png"
    cv2.imwrite(str(output_path), canvas)
    print(f"Optical Flow vectors overlay saved to: {output_path.resolve()}")
    print("Yellow/Cyan arrows represent scaled grid-wise flow vectors (displacement direction).")
    print("Red dots represent grids with negligible/no movement.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
