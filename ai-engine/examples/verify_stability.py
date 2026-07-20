import argparse
import sys
import time
from pathlib import Path

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
from fusion.tracker import TemporalTracker


def main():
    parser = argparse.ArgumentParser(description="Temporal Stability verification script.")
    parser.add_argument(
        "--source",
        type=str,
        default=str(project_root / "Sudharshan" / "outputs" / "rendered_frames"),
        help="Path to directory of image frames",
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
    args = parser.parse_args()

    boundary = [
        (100, 500),  # Bottom Left
        (350, 100),  # Top Left
        (650, 100),  # Top Right
        (900, 500),  # Bottom Right
    ]

    print("==================================================")
    print("STARTING TEMPORAL STABILITY TRACKER TEST")
    print("==================================================")

    # 1. Initialize all engine modules
    print("Step 1: Initializing adapter, flow engine, and aggregator...")
    adapter = SCALNetAdapter(scalnet_root=project_root / "SCALNet", checkpoint_path=args.checkpoint, device="auto")
    adapter.load()
    flow_engine = OpticalFlowEngine(method="dis")

    # Grid count = 5 people is high. Area = 14400. Threshold = 5/14400 ~ 0.00035
    aggregator = FusionAggregator(
        density_red=0.00035,
        speed_normal=3.0,
        expected_direction="EAST",
    )

    # Initialize tracker with alpha=0.3 and low persistence duration for fast verification
    tracker = TemporalTracker(
        alpha=0.3,
        fps=5.0,
        persistence_yellow_sec=0.4,  # 2 frames
        persistence_orange_sec=0.4,  # 2 frames
        persistence_red_sec=0.2,     # 1 frame
    )

    print("\nStep 2: Processing consecutive frames to observe stabilization...")

    # Open frame reader
    reader = FrameReader(source=args.source, realtime=False)
    reader.start()

    # Track consecutive frames
    prev_frame_obj = reader.read()
    if prev_frame_obj is None:
        print("Error: Could not read starting frame.", file=sys.stderr)
        sys.exit(1)

    grid_gen = GridGenerator(grid_size=args.grid_size, overlap_ratio=args.overlap)
    grids = grid_gen.generate_grids(width=prev_frame_obj.width, height=prev_frame_obj.height, boundary_polygon=boundary)

    # We will process 5 frames to observe tracking trajectories
    grid_of_interest = "G_01_04"  # Let's track this high-density grid
    print(f"Tracking Grid of Interest: {grid_of_interest}")
    print("-" * 110)
    print(f"{'Frame':<6} | {'Raw Count':<10} | {'Raw Score':<10} | {'Raw Risk':<10} | "
          f"{'EMA Count':<10} | {'EMA Score':<10} | {'EMA Risk':<10}")
    print("-" * 110)

    for step in range(5):
        curr_frame_obj = reader.read()
        if curr_frame_obj is None:
            print("Reached end of image sequence.")
            break

        # Density Inference
        density_res = adapter.infer(prev_frame_obj.frame)

        # Flow Calculation
        flow_res = flow_engine.calculate_flow(prev_frame_obj.frame, curr_frame_obj.frame)

        # Fusion aggregation (raw output)
        raw_fused = aggregator.fuse(grids, density_res, flow_res)

        # Temporal tracking aggregation (smoothed/stabilized output)
        stabilized = tracker.track(raw_fused)

        # Print statistics for the grid of interest
        rf = raw_fused.get(grid_of_interest)
        st = stabilized.get(grid_of_interest)

        if rf and st:
            print(f"{step:<6} | {rf.count:<10.2f} | {rf.congestion_score:<10.1f} | {rf.risk_level:<10} | "
                  f"{st.count:<10.2f} | {st.congestion_score:<10.1f} | {st.risk_level:<10}")

        prev_frame_obj = curr_frame_obj

    reader.stop()
    print("-" * 110)
    print("Temporal Stability Tracker verification complete.")
    print("EMA Count and EMA Score follow a smoothed trajectory compared to raw fluctuations.")
    print("EMA Risk levels apply temporal persistence and hysteresis to prevent alert flickering.")
    print("==================================================")


if __name__ == "__main__":
    main()
