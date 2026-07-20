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


def main():
    parser = argparse.ArgumentParser(description="Verification script for Step 2 FrameReader abstraction.")
    parser.add_argument(
        "--source",
        type=str,
        default=str(project_root / "Sudharshan" / "outputs" / "rendered_frames"),
        help="Path to video file, directory of images, RTSP stream, or camera index",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Optional fps limit for offline source simulation",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum frames to read for verification",
    )
    args = parser.parse_args()

    print(f"Initializing FrameReader for source: {args.source}")
    print(f"Settings: fps_limit={args.fps}, loop=False")

    t_start = time.perf_counter()

    try:
        with FrameReader(source=args.source, fps_limit=args.fps, realtime=False) as reader:
            print(f"Detected Source Type: {reader.source_type}")
            print(f"Reader is running: {reader._running}")
            print("-" * 60)

            read_times = []

            for i in range(args.limit):
                t_frame_start = time.perf_counter()
                frame_obj = reader.read()
                t_frame_end = time.perf_counter()

                if frame_obj is None:
                    print("Reached EOF / End of stream.")
                    break

                latency_ms = (t_frame_end - t_frame_start) * 1000.0
                read_times.append(latency_ms)

                print(f"Frame {frame_obj.frame_index:03d} | "
                      f"Shape: {frame_obj.height}x{frame_obj.width}x{frame_obj.frame.shape[2]} | "
                      f"Timestamp: {frame_obj.timestamp:.4f} | "
                      f"Read Latency: {latency_ms:.2f} ms")

            print("-" * 60)
            if read_times:
                avg_latency = sum(read_times) / len(read_times)
                print(f"Successfully verified FrameReader.")
                print(f"Read {len(read_times)} frames.")
                print(f"Average Frame Ingestion Latency: {avg_latency:.2f} ms")
            else:
                print("No frames were read.")

    except Exception as e:
        print(f"Error during FrameReader verification: {e}", file=sys.stderr)
        sys.exit(1)

    t_end = time.perf_counter()
    print(f"Verification script complete. Total run time: {(t_end - t_start):.2f} seconds.")


if __name__ == "__main__":
    main()
