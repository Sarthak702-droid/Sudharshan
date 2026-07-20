#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="$PROJECT_DIR/SCALNet:$PROJECT_DIR/SCALNet/src:$PROJECT_DIR/ai-engine"
"$PYTHON_BIN" -m unittest discover -s "$PROJECT_DIR/ai-engine/tests" -p "test_*.py"
