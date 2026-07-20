import argparse
import json
import sys
from pathlib import Path
import time
import cv2
import numpy as np

# Ensure ai-engine is in path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root / "ai-engine") not in sys.path:
    sys.path.insert(0, str(project_root / "ai-engine"))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from density.scalnet_adapter import SCALNetAdapter
from density.postprocessing import normalize_density_for_debug


def main():
    parser = argparse.ArgumentParser(description="Developer verification script for SCALNet density foundation module.")
    parser.add_argument(
        "--image",
        type=str,
        default=str(project_root / "Sudharshan" / "outputs" / "rendered_frames" / "frame_000018.jpg"),
        help="Path to the test frame image",
    )
    parser.add_argument(
        "--scalnet_root",
        type=str,
        default=str(project_root / "SCALNet"),
        help="Path to SCALNet repository root",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(project_root / "SCALNet" / "checkpoints" / "model.pth"),
        help="Path to trained weights (.pth file)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(project_root / "outputs" / "step_1_debug"),
        help="Directory to save debug outputs",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Crowd presence mask threshold",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device (cuda, cpu, auto)",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Test image not found at {image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading test image from: {image_path}")
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"Error: Failed to decode image from {image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Initializing SCALNetAdapter...")
    adapter = SCALNetAdapter(
        scalnet_root=Path(args.scalnet_root),
        checkpoint_path=Path(args.checkpoint),
        device=args.device,
        mask_threshold=args.threshold,
    )

    print(f"Loading SCALNet model weights onto device '{adapter.device}'...")
    t0 = time.perf_counter()
    adapter.load()
    t1 = time.perf_counter()
    print(f"Model loaded successfully in {(t1 - t0) * 1000.0:.2f} ms.")

    print(f"Running crowd density inference...")
    result = adapter.infer(frame)
    print(f"Inference completed.")

    # Create outputs directory
    outputs_dir = Path(args.output_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save raw density map
    raw_path = outputs_dir / "density_raw.npy"
    np.save(raw_path, result.density_map)

    # 2. Save density heatmap
    norm_density = normalize_density_for_debug(result.density_map)
    heatmap = cv2.applyColorMap(norm_density, cv2.COLORMAP_JET)
    cv2.imwrite(str(outputs_dir / "density_heatmap.png"), heatmap)

    # 3. Save crowd presence mask (scaled to 0-255 for viewing)
    cv2.imwrite(str(outputs_dir / "crowd_mask.png"), result.crowd_mask * 255)

    # 4. Save overlay image
    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(str(outputs_dir / "overlay.png"), overlay)

    # 5. Save metadata result.json
    result_dict = {
        "estimated_count": round(result.estimated_count, 3),
        "inference_time_ms": round(result.inference_time_ms, 2),
        "device": result.device,
        "input_width": result.input_width,
        "input_height": result.input_height,
        "model_name": result.model_name,
        "checkpoint_path": str(result.checkpoint_path),
    }

    with open(outputs_dir / "result.json", "w") as f:
        json.dump(result_dict, f, indent=4)

    print("\n" + "=" * 50)
    print("DEVELOPMENT VERIFICATION SUMMARY")
    print("=" * 50)
    print(f"Input Frame Size     : {result.input_width}x{result.input_height}")
    print(f"Estimated Crowd Count: {result.estimated_count:.3f} people")
    print(f"Inference Latency    : {result.inference_time_ms:.2f} ms")
    print(f"Running Device       : {result.device}")
    print(f"Model Name           : {result.model_name}")
    print(f"Checkpoint Loaded    : {result.checkpoint_path}")
    print(f"Debug Artifacts Saved To: {outputs_dir.resolve()}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
