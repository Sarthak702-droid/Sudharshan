#!/bin/bash
# Sudharshan AI - Go Backend Engine Launcher

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Go to the backend folder inside this repository.
cd "$SCRIPT_DIR/backend"

echo "=========================================================="
echo "         SUDHARSHAN AI - GO BACKEND DATABASE & WS"
echo "=========================================================="
echo ""
echo "[*] Checking Go dependencies..."
go mod tidy

# Default port
PORT=${PORT:-8080}
echo "[+] Starting local offline Go backend on http://localhost:$PORT..."
echo "[+] Telemetry contract exposed at: http://localhost:$PORT/api/v1/telemetry/grid-metrics"
echo "[+] WebSocket broadcast route: ws://localhost:$PORT/api/v1/live/ws"
echo ""
echo "Press Ctrl+C to terminate the server."
echo "=========================================================="
echo ""

PORT=$PORT go run cmd/server/main.go
