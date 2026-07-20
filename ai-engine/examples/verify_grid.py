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

from grid.generator import GridGenerator


def main():
    parser = argparse.ArgumentParser(description="Verification script for Step 3 Overlapping Grid Engine.")
    parser.add_argument(
        "--width",
        type=int,
        default=960,
        help="Monitored width in pixels",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=540,
        help="Monitored height in pixels",
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
        help="Grid overlap ratio (0.0 to 1.0)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(project_root / "outputs" / "step_3_debug"),
        help="Directory to save grid debug overlays",
    )
    args = parser.parse_args()

    # Define a sample boundary polygon corridor representing the monitored street path
    # A trapezoid/corridor shape inside the 960x540 frame
    boundary = [
        (100, 500),  # Bottom Left
        (350, 100),  # Top Left
        (650, 100),  # Top Right
        (900, 500),  # Bottom Right
    ]

    print(f"Initializing GridGenerator with size={args.grid_size}, overlap_ratio={args.overlap}")
    gen = GridGenerator(grid_size=args.grid_size, overlap_ratio=args.overlap)

    print(f"Dividing events region ({args.width}x{args.height}) with perspective corridor boundaries...")
    grids = gen.generate_grids(width=args.width, height=args.height, boundary_polygon=boundary)

    print(f"Successfully generated {len(grids)} active grids clipping to the boundary polygon.")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Render a clean visualization of the grid layout and boundaries
    # Create a dark canvas for rich visual feedback
    canvas = np.zeros((args.height, args.width, 3), dtype=np.uint8)
    canvas.fill(20)  # Dark gray background

    # Draw boundary polygon in Red
    pts = np.array(boundary, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(canvas, [pts], isClosed=True, color=(0, 0, 180), thickness=3)
    # Fill translucent red for the boundary region
    overlay_poly = canvas.copy()
    cv2.fillPoly(overlay_poly, [pts], color=(0, 0, 50))
    cv2.addWeighted(overlay_poly, 0.4, canvas, 0.6, 0, dst=canvas)

    print("\n" + "-" * 80)
    print(f"{'Grid ID':<10} | {'Bounds (x1, y1) -> (x2, y2)':<30} | {'Full Area':<10} | {'Effective Area':<15} | {'Coverage %':<10}")
    print("-" * 80)

    # Draw grids
    for g in grids:
        coverage_pct = (g.effective_area / g.area) * 100.0
        print(f"{g.grid_id:<10} | ({g.x1:3d}, {g.y1:3d}) -> ({g.x2:3d}, {g.y2:3d}) | {g.area:<10.0f} | {g.effective_area:<15.1f} | {coverage_pct:<10.1f}%")

        # Choose color based on boundary coverage
        if coverage_pct > 99.0:
            color = (0, 200, 0)  # Solid Green for fully inside grids
        else:
            color = (200, 200, 0)  # Cyan/Yellow-Green for clipped edge grids

        # Draw box
        cv2.rectangle(canvas, (g.x1, g.y1), (g.x2, g.y2), color, thickness=1)

        # Label grid ID
        # Offset slightly for overlap visibility
        label_pos = (g.x1 + 10, g.y1 + 25)
        cv2.putText(canvas, g.grid_id, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, thickness=1)

    print("-" * 80 + "\n")

    # Save debug overlay
    output_path = output_dir / "grids_overlay.png"
    cv2.imwrite(str(output_path), canvas)
    print(f"Grid Layout visualization saved to: {output_path.resolve()}")
    print("Green boxes represent fully active grids inside boundaries.")
    print("Yellow/Cyan boxes represent clipped/boundary boundary grids.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
