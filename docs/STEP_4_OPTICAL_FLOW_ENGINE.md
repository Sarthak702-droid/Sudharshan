# Step 4: Optical-Flow Engine Module

This document details the implementation, verification, and benchmarking of Step 4 for **Sudharshan** (Live Crowd Flow Intelligence Engine).

---

## 1. Executive Summary

Step 4 has successfully introduced the Optical-Flow Engine (`ai-engine/flow/`). The module wraps dense optical flow algorithms (OpenCV DIS and Farneback) under a unified class. It aggregates dense displacement fields $(u, v)$ inside localized `GridBox` coordinates generated in Step 3.

It computes grid-wise horizontal displacement, vertical displacement, movement magnitude, rotation angle, and cardinal direction labels (EAST, WEST, SOUTH, etc.). It supports both simple arithmetic mean and density-weighted mean aggregation (where motion in high-crowd pixels is heavily prioritized).

---

## 2. Deliverables & File Structure

The following file structure has been established under the project workspace:

```text
ai-engine/
├── flow/
│   ├── __init__.py
│   ├── types.py
│   └── engine.py
└── tests/
    └── test_flow.py
```

### Files Created
* **[types.py](file:///home/sarthaktripathy/Documents/Sudharsan/ai-engine/flow/types.py)**: Holds the `FlowResult` dataclass (storing 2D dense float32 arrays of horizontal displacements `flow_x` and vertical displacements `flow_y`, alongside inference timing).
* **[engine.py](file:///home/sarthaktripathy/Documents/Sudharsan/ai-engine/flow/engine.py)**: The `OpticalFlowEngine` wrapper executing flow calculation (DIS or Farneback), grid bounds slicing, cardinal labeling, and density-weighted aggregation mathematics.
* **[test_flow.py](file:///home/sarthaktripathy/Documents/Sudharsan/ai-engine/tests/test_flow.py)**: Unit tests verifying algorithm initialization, angle-to-label cardinal mappings, grid flow arithmetic averaging, and density-weighted average prioritization.
* **[verify_flow.py](file:///home/sarthaktripathy/Documents/Sudharsan/ai-engine/examples/verify_flow.py)**: Verification script extracting dense flow on consecutive Rath Yatra frames and rendering vector overlays.

---

## 3. Optical Flow & Aggregation Mathematics

### Dense Flow Calculation
Consecutive frames $I_t$ and $I_{t+1}$ are converted to grayscale and passed to OpenCV:
* **DIS Flow** (`cv2.DISOpticalFlow`): Selected as the default algorithm due to its high efficiency and near-real-time speed.
* **Farneback Flow** (`cv2.calcOpticalFlowFarneback`): Supported as a fallback option.

### Grid Flow Aggregation
For each grid window $g$ (bounds: $x_1, y_1 \to x_2, y_2$):
1. **Arithmetic Mean** (Standard):
   $$u_g = \frac{1}{N} \sum_{i \in g} u_i$$
   $$v_g = \frac{1}{N} \sum_{i \in g} v_i$$
2. **Density-Weighted Mean** (Preferred for Crowd Tracking):
   Priority is given to motion in dense crowd zones to filter out background noise (e.g., flags waving, camera jitter):
   $$u_g = \frac{\sum_{i \in g} D_i \times u_i}{\sum_{i \in g} D_i}$$
   $$v_g = \frac{\sum_{i \in g} D_i \times v_i}{\sum_{i \in g} D_i}$$
   *If the grid contains no density ($\sum D_i \approx 0$), it falls back to the arithmetic mean.*

### Magnitude & Direction Mapping
For each grid, displacement magnitude $m_g$ (in pixels per frame) is computed:
$$m_g = \sqrt{u_g^2 + v_g^2}$$
The motion direction angle $\theta$ is extracted:
$$\theta = \text{atan2}(v_g, u_g)$$
$$\theta_{\text{deg}} = (\theta \times \frac{180.0}{\pi} + 360.0) \pmod{360}$$
The angle $\theta_{\text{deg}}$ is mapped to one of the 8 cardinal direction labels (EAST, SOUTH-EAST, SOUTH, etc.).

---

## 4. Verification & Testing

### How to Run Flow Verification
To run the optical flow and draw grid-wise motion vector arrows, execute:
```bash
PYTHONPATH=SCALNet:SCALNet/src NWPU-Crowd-Sample-Code/.venv/bin/python ai-engine/examples/verify_flow.py
```

* **Visual Output:** A canvas visualization is rendered at [outputs/step_4_debug/flow_overlay.png](file:///home/sarthaktripathy/Documents/Sudharsan/outputs/step_4_debug/flow_overlay.png).
  * Yellow/Cyan arrows represent scaled grid-wise flow vectors (displacement direction).
  * Red dots represent grids with negligible/no movement.

### How to Run Flow Tests
Execute unit tests for all modules (Step 1-4):
```bash
PYTHONPATH=SCALNet:SCALNet/src NWPU-Crowd-Sample-Code/.venv/bin/python -m unittest discover -s ai-engine/tests -p "test_*.py"
```
* **Status:** `OK` (All 25 tests passing successfully).

---

## 5. Step 5 Boundary Definition

With the **Density Adapter (Step 1)**, **Frame Ingest (Step 2)**, **Grid division (Step 3)**, and **Optical Flow calculation (Step 4)** complete, we are ready for **Step 5 (SCALNet + Flow Fusion)**.
* **Input boundary for Step 5**: Frame-wise `DensityInferenceResult` and `FlowResult` metrics mapped per grid box.
* **Step 5 Target**: Build the fusion aggregator module that combines density and flow fields to compute crowd direction, crowd speed, flow conflict scores, stagnation indicators, and congestion risks per grid box.
