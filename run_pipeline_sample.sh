#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INPUT_PATH="${1:?Usage: ./run_pipeline_sample.sh /path/to/video-or-image-directory}"
export PYTHONPATH="$PROJECT_DIR/SCALNet:$PROJECT_DIR/SCALNet/src:$PROJECT_DIR/ai-engine"
"$PYTHON_BIN" "$PROJECT_DIR/scripts/run_pipeline.py" \
  --input "$INPUT_PATH" \
  --max_frames 5 \
  --output_video "$PROJECT_DIR/outputs/flow_overlay.mp4" \
  --output_json "$PROJECT_DIR/outputs/grid_metrics.json" \
  --output_csv "$PROJECT_DIR/outputs/grid_metrics.csv" \
  --config "$PROJECT_DIR/configs/grid_config.yaml"
