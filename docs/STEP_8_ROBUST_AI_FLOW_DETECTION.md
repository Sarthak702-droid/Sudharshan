# Step 8: Robust AI Flow Detection & Anomalies Module

This document details the implementation, verification, and benchmarking of the advanced **Robust AI Flow Detection and Anomalies Module** for **Sudharshan** (Live Crowd Flow Intelligence Engine).

---

## 1. Executive Summary

To make the AI crowd flow logic extremely robust under challenging operational conditions (low light/dust, perspective projection distortion, camera vibration, and chaotic/conflicting crowd streams), we have successfully implemented and verified several advanced mathematical and image processing models inside the core `ai-engine/` packages.

We have integrated these models directly into the pipeline runner and successfully exported these new telemetry indicators to both JSON and CSV output targets.

---

## 2. Deliverables & Technical Architectures

### A. Preprocessing: CLAHE Adaptive Histogram Equalization
* **Location:** `ai-engine/flow/engine.py` (inside `calculate_flow`)
* **Method:** Grayscale conversions of subsequent video frames are processed using OpenCV's `createCLAHE(clipLimit=2.0, tileGridSize=(8,8))` prior to DIS flow vector calculations.
* **Benefit:** Local contrast, edge definitions, and crowd textures are enhanced, stabilizing vector tracking under adverse weather, twilight, shadows, or dust.

### B. Math: Perspective Scale Calibration
* **Location:** `ai-engine/fusion/aggregator.py`
* **Formulation:** For each grid, the vertical center $y_{\text{center}}$ is evaluated:
  $$\text{scale} = 1.0 + 1.5 \times \left(1.0 - \frac{y_{\text{center}}}{H_{\text{frame}}}\right)$$
* **Usage:** Compensates for standard perspective view sizing. Counts, density, flow vectors, and speed magnitudes are dynamically scaled up for far-away zones (top of the image) to match physical metrics.

### C. Indicators: Circular Variance & Turbulence Score
* **Location:** `ai-engine/fusion/aggregator.py`
* **Formulation:** Local circular variance `local_cv` (angular variance of vectors in a grid) is computed and stored as `turbulence_score`. This identifies chaotic/conflicting crowd packing where opposite vectors might cancel the net average.

### D. Warnings: Abnormal Behavior Detections
* **Location:** `ai-engine/fusion/tracker.py`
* **Logic:** Emits three distinct operational safety alarms based on temporal state analysis:
  1. **Speed Surge (`speed_surge_warning`):** True if a sudden speed surge exceeds $2\times$ the historical moving average of the last 5 frames.
  2. **Stasis / Crowd Crush buidup (`stasis_warning`):** True if high density ($> 0.0002$) and slow speed ($< 0.3$) is sustained continuously for at least 2.0 seconds.
  3. **Turbulence Warning (`turbulence_warning`):** True if high local circular variance ($> 0.6$) is detected under dense packing ($> 0.0002$).

---

## 3. Backward Compatibility & JSON Schema

The output JSON telemetry records are fully backward compatible, appending the new indicators to the schema:

```json
{
  "frame_id": 1,
  "timestamp_sec": 0.07,
  "grid_id": "G_00_00",
  ...
  "risk_score": 21.22,
  "risk_level": "GREEN",
  "turbulence_score": 0.0242,
  "speed_surge_warning": false,
  "stasis_warning": false,
  "turbulence_warning": false
}
```

---

## 4. Verification & Testing

### How to Run the Unit Tests
All unit tests (including the updated perspective scale assertions) compile and pass successfully:
```bash
PYTHONPATH=/home/sarthaktripathy/Documents/Sudharsan/SCALNet:/home/sarthaktripathy/Documents/Sudharsan/SCALNet/src /home/sarthaktripathy/Documents/Sudharsan/NWPU-Crowd-Sample-Code/.venv/bin/python -m unittest discover -s /home/sarthaktripathy/Documents/Sudharsan/ai-engine/tests -p "test_*.py"
```
* **Status:** `OK` (All 31 test cases pass).

### How to Run Pipeline Verification
Run a 5-frame test over the VisDrone sequence:
```bash
PYTHONPATH=/home/sarthaktripathy/Documents/Sudharsan/SCALNet:/home/sarthaktripathy/Documents/Sudharsan/SCALNet/src /home/sarthaktripathy/Documents/Sudharsan/NWPU-Crowd-Sample-Code/.venv/bin/python /home/sarthaktripathy/Documents/Sudharsan/Sudharshan/scripts/run_pipeline.py --input /home/sarthaktripathy/Documents/Sudharsan/VisDrone-MOT/VisDrone2019-MOT-val/sequences/uav0000339_00001_v --max_frames 5 --output_video /home/sarthaktripathy/Documents/Sudharsan/Sudharshan/outputs/flow_overlay.mp4 --output_json /home/sarthaktripathy/Documents/Sudharsan/Sudharshan/outputs/grid_metrics.json --output_csv /home/sarthaktripathy/Documents/Sudharsan/Sudharshan/outputs/grid_metrics.csv --config /home/sarthaktripathy/Documents/Sudharsan/Sudharshan/configs/grid_config.yaml
```
* **Status:** `Success` (Grid metrics JSON/CSV successfully saved with new warning fields).
