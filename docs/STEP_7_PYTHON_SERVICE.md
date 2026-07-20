# Step 7: Python Service Layer Module

This document details the implementation, verification, and benchmarking of Step 7 for **Sudharshan** (Live Crowd Flow Intelligence Engine).

---

## 1. Executive Summary

Step 7 has successfully integrated the complete AI crowd flow detection pipeline (`scripts/run_pipeline.py`) and Web API exposure server (`src/server.py`) under the updated **Production AI Engine (`ai-engine/`)** architecture.

The service layer replaces all legacy mock operations with actual pre-trained SCALNet models, DIS/Farneback motion vectors, overlapping grid segmentations, and temporal trackers. It provides a web dashboard, saves spatial analytics to JSON/CSV files, and optionally posts real-time telemetry to database backends.

---

## 2. Deliverables & File Structure

The following file structure has been established/updated:

```text
Sudharshan/
├── scripts/
│   └── run_pipeline.py  [Refactored]
├── src/
│   └── server.py        [Verified]
```

### Files Modified & Verified
* **[run_pipeline.py](file:///home/sarthaktripathy/Documents/Sudharsan/Sudharshan/scripts/run_pipeline.py)**: Overhauled completely to import and invoke `FrameReader`, `GridGenerator`, `SCALNetAdapter`, `OpticalFlowEngine`, `FusionAggregator`, and `TemporalTracker`.
* **[server.py](file:///home/sarthaktripathy/Documents/Sudharsan/Sudharshan/src/server.py)**: Verified to coordinate sequence directories, label configurations, and execute the updated pipeline runner as an API service.
* **[STEP_7_PYTHON_SERVICE.md](file:///home/sarthaktripathy/Documents/Sudharsan/docs/STEP_7_PYTHON_SERVICE.md)**: This documentation.

---

## 3. Backward Compatibility & JSON API Contracts

To ensure that the Flask offline dashboard UI and any Go backends continue to function without any changes, the service layer populates telemetry records with both the new structured properties and the legacy property mappings:

```json
{
  "frame_id": 1,
  "timestamp_sec": 0.20,
  "grid_id": "G_01_04",
  "row": 1,
  "col": 4,
  "x1": 384,
  "y1": 96,
  "x2": 504,
  "y2": 216,
  "mean_dx": -0.3023,
  "mean_dy": 0.0689,
  "motion_magnitude": 0.3101,
  "direction": "WEST",
  "direction_deg": 167.2,
  "speed_level": "SLOW",
  "crowd_count": 6.87,
  "density": 0.000477,
  "flow_conflict": true,
  "coherence": 0.84,
  "risk_score": 83.3,
  "risk_level": "RED"
}
```

---

## 4. Verification & Testing

### How to Run Pipeline Execution (Verification test on VisDrone sequence)
Run a 3-frame validation pass using the actual SCALNet adapter, grids, and DIS flow components:
```bash
PYTHONPATH=/home/sarthaktripathy/Documents/Sudharsan/SCALNet:/home/sarthaktripathy/Documents/Sudharsan/SCALNet/src /home/sarthaktripathy/Documents/Sudharsan/NWPU-Crowd-Sample-Code/.venv/bin/python /home/sarthaktripathy/Documents/Sudharsan/Sudharshan/scripts/run_pipeline.py --input /home/sarthaktripathy/Documents/Sudharsan/VisDrone-MOT/VisDrone2019-MOT-val/sequences/uav0000339_00001_v --max_frames 3 --output_video /home/sarthaktripathy/Documents/Sudharsan/Sudharshan/outputs/flow_overlay.mp4 --output_json /home/sarthaktripathy/Documents/Sudharsan/Sudharshan/outputs/grid_metrics.json --output_csv /home/sarthaktripathy/Documents/Sudharsan/Sudharshan/outputs/grid_metrics.csv --config /home/sarthaktripathy/Documents/Sudharsan/Sudharshan/configs/grid_config.yaml
```

### Production Telemetry Performance Output:
```text
Loading configuration from /home/sarthaktripathy/Documents/Sudharsan/Sudharshan/configs/grid_config.yaml...
Resizing active frame resolution from 1904x1071 to 960x540 (Scale Factor: 0.504)
Initializing Sudharshan AI Core components...
Generated 84 overlapping grids (100px size, 20% overlap).
Processing Frames: 100%|█████████████████████████| 3/3 [00:03<00:00,  1.20s/it]
Processed video written to /home/sarthaktripathy/Documents/Sudharsan/Sudharshan/outputs/flow_overlay.mp4
Grid metrics JSON written to /home/sarthaktripathy/Documents/Sudharsan/Sudharshan/outputs/grid_metrics.json
Grid metrics CSV written to /home/sarthaktripathy/Documents/Sudharsan/Sudharshan/outputs/grid_metrics.csv
```

### How to Run the Web Dashboard Server
A start script is provided under the workspace to host the dashboard on port 5000:
```bash
cd /home/sarthaktripathy/Documents/Sudharsan/Sudharshan
./start_dashboard.sh
```

---

## 5. Step 8 Boundary Definition

With the **Python Service Layer (Step 7)** complete, we are ready for **Step 8 (Go Orchestration Backend)**.
* **Input boundary for Step 8**: JSON grid telemetry structures emitted from the Python pipeline.
* **Step 8 Target**: Build out the Go backend to store cameras, zones, and grid definitions, ingest the Python worker metrics feed, record operational state in databases, handle operator alert lifecycle state transitions, and provide WebSocket updates to frontend clients.
