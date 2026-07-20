#!/bin/bash
# Sudharshan AI - Offline Operations Dashboard Launcher

set -e

# Get script directory
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$PROJECT_DIR"

echo "=========================================================="
echo "      SUDHARSHAN AI - CROWD FLOW OPERATIONS DASHBOARD"
echo "=========================================================="

# Check if python virtual environment exists
if [ ! -d ".venv" ]; then
    echo "[*] Creating Python Virtual Environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    echo "[*] Installing dependencies..."
    pip install --upgrade pip
    pip install numpy opencv-python-headless pyyaml flask tqdm
else
    source .venv/bin/activate
fi

# Print instructions
echo ""
echo "[+] Starting local offline web server..."
echo "[+] Exposing Dashboard UI on: http://localhost:5000"
echo "[+] Exposing REST APIs for scanning and custom labeling."
echo ""
echo "Press Ctrl+C to terminate the server."
echo "=========================================================="
echo ""

# Run Flask server
python src/server.py
