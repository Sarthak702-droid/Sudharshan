# Step 8: Go Orchestration Backend Module

This document details the implementation, verification, and benchmarking of Step 8 for **Sudharshan** (Live Go Backend & Database Orchestrator).

---

## 1. Executive Summary

Step 8 has successfully updated the Go Orchestration Backend (`backend/cmd/server/main.go`) to store cameras, zones, and grid definitions, ingest the real-time Python AI worker telemetry feed, persist historical metrics in local SQLite databases, handle operator alert lifecycle state transitions, and provide WebSocket updates to connected clients.

The Go backend was enhanced to dynamically parse the new robust AI warning fields and turbulence scores, perform automatic SQLite schema column migrations, and trigger database alerts for anomalous crowd behavior (speed surge, stasis, and turbulence).

---

## 2. Deliverables & Technical Architectures

### A. Database Auto-Migrations
* **Location:** `backend/cmd/server/main.go` (inside `initDB`)
* **Logic:** Checks the existing table structure of `grid_metrics` using `PRAGMA table_info`. If the database file `sudharshan.db` already exists from a previous session and is missing the new robust columns, it executes `ALTER TABLE` statements dynamically to add:
  - `turbulence_score REAL`
  - `speed_surge_warning INTEGER`
  - `stasis_warning INTEGER`
  - `turbulence_warning INTEGER`
* **Benefit:** Upgrades existing databases in place without throwing SQL execution errors or requiring data deletion.

### B. Automated Crowd Safety Alert Triggers
* **Location:** `backend/cmd/server/main.go` (inside `handleTelemetry`)
* **Triggers:** Automatically generates corresponding entries in the database `alerts` table based on telemetry flags:
  1. **Speed Surge:** Creates severity="RED", type="SPEED_SURGE" alert (stampede hazard).
  2. **Stasis / Crowd Crush:** Creates severity="RED", type="CROWD_CRUSH_HAZARD" alert (crush hazard).
  3. **Turbulence:** Creates severity="ORANGE", type="FLOW_TURBULENCE" alert (conflicting multidirectional crowd streams).

### C. Live Query & WebSocket Broadcasts
* **Location:** `/api/v1/live/grid-state` and `/api/v1/live/ws`
* **Features:** Parses and broadcasts new AI indicators to WebSocket connection groups for real-time visualization dashboards.

---

## 3. Verification & Testing

### How to Build and Run the Go Backend
1. Make sure you are in the parent directory `/home/sarthaktripathy/Documents/Sudharsan/`.
2. Start the backend:
```bash
./Sudharshan/start_backend.sh
```
* **Expected Output:**
```text
==========================================================
         SUDHARSHAN AI - GO BACKEND DATABASE & WS
==========================================================
[*] Checking Go dependencies...
[+] Starting local offline Go backend on http://localhost:8080...
[+] Telemetry contract exposed at: http://localhost:8080/api/v1/telemetry/grid-metrics
[+] WebSocket broadcast route: ws://localhost:8080/api/v1/live/ws
...
Go server successfully listening on :8080
```

### Verification telemetry POST payload example:
You can verify backend consumption by POSTing grid metrics:
```json
[
  {
    "frame_id": 1,
    "timestamp_sec": 0.07,
    "grid_id": "G_00_00",
    "row": 0,
    "col": 0,
    "x1": 0,
    "y1": 0,
    "x2": 100,
    "y2": 100,
    "mean_dx": -0.063,
    "mean_dy": 0.257,
    "motion_magnitude": 0.264,
    "direction": "SOUTH",
    "direction_deg": 103.8,
    "speed_level": "SLOW",
    "crowd_count": 0.03,
    "density": 3e-06,
    "flow_conflict": false,
    "coherence": 0.978,
    "risk_score": 21.2,
    "risk_level": "GREEN",
    "turbulence_score": 0.0242,
    "speed_surge_warning": false,
    "stasis_warning": false,
    "turbulence_warning": false
  }
]
```
* **Backend log response:**
`Migrating database: Adding column turbulence_score to grid_metrics table...` (Runs auto-migrations successfully on start).
