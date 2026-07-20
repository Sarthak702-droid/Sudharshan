# Step 5: SCALNet + Flow Fusion Module

This document details the implementation, verification, and benchmarking of Step 5 for **Sudharshan** (Live Crowd Flow Intelligence Engine).

---

## 1. Executive Summary

Step 5 has successfully introduced the Fusion Aggregator module (`ai-engine/fusion/`). This module integrates **Step 1 (SCALNet density)** and **Step 4 (DIS optical flow)** inside the **Step 3 (overlapping grids)** to generate complete crowd analytics.

It calculates density scores, slow movement scores, localized stagnation indices, neighbor conflict metrics, expected direction reverse flow dot-products, congestion ratings, risk classifications, and operational confidence scores. All outputs are exported as structured JSON logs and colored visual overlays.

---

## 2. Deliverables & File Structure

The following file structure has been established under the project workspace:

```text
ai-engine/
├── fusion/
│   ├── __init__.py
│   ├── types.py
│   └── aggregator.py
└── tests/
    └── test_fusion.py
```

### Files Created
* **[types.py](file:///home/sarthaktripathy/Documents/Sudharsan/ai-engine/fusion/types.py)**: Holds the `GridMetrics` data model storing fused crowd attributes (count, density, flow vectors, speed, angles, scores, risk levels, and confidence).
* **[aggregator.py](file:///home/sarthaktripathy/Documents/Sudharsan/ai-engine/fusion/aggregator.py)**: The `FusionAggregator` core carrying out spatial alignment, circular variance flow calculations, 8-neighbor conflict evaluations, expected-observed vector dot products, and nonlinear congestion risk scoring.
* **[test_fusion.py](file:///home/sarthaktripathy/Documents/Sudharsan/ai-engine/tests/test_fusion.py)**: Tests verifying default direction vector routing, free-flow safety calculations (Green risk), and reverse-flow high-density stagnation warnings (Red risk).
* **[verify_fusion.py](file:///home/sarthaktripathy/Documents/Sudharsan/ai-engine/examples/verify_fusion.py)**: The end-to-end integration pipeline script running the full suite of modules (Ingest $\to$ Grids $\to$ SCALNet $\to$ DIS Flow $\to$ Fusion) on test sequence frames.

---

## 3. Core Fusion & Risk Mathematics

### Crowd Density Scoring
For each grid window $g$:
$$\text{grid\_density}_g = \frac{\text{grid\_count}_g}{\text{effective\_area}_g}$$
$$\text{density\_score}_g = \text{clamp}\left(\frac{\text{grid\_density}_g}{\text{density\_red}}, 0, 1\right)$$

### Slow Movement Scoring
$$\text{slow\_score}_g = 1.0 - \text{clamp}\left(\frac{\text{speed}_g}{\text{speed\_normal}}, 0, 1\right)$$

### Stagnation
Captures high-density stagnant crowds (high danger for crush events):
$$\text{stagnation}_g = \text{density\_score}_g \times \text{slow\_score}_g$$

### Neighbor Flow Conflict
Combines local pixel circular variance (internal turbulence) and neighbor motion disagreements:
$$\text{flow\_conflict}_g = 0.70 \times \text{local\_circular\_variance} + 0.30 \times \text{mean\_neighbor\_conflict}$$
$$\text{neighbor\_conflict}_{g,h} = \frac{1 - \cos(\theta_g - \theta_h)}{2}$$

### Expected Direction Reverse Flow
Let expected route direction unit vector be $E_g$ and observed flow unit vector be $O_g$:
$$s = E_g \cdot O_g$$
$$\text{reverse\_score}_g = \max(0, -s)$$

### Nonlinear Congestion Score & Risks
$$\text{congestion\_score}_g = 100 \times \text{clamp}\left(\begin{array}{l}0.35 \times \text{density\_score}_g + 0.20 \times \text{slow\_score}_g \\ + 0.20 \times \text{stagnation}_g + 0.15 \times \text{flow\_conflict}_g \\ + 0.10 \times \text{reverse\_score}_g\end{array}, 0, 1\right)$$

* **Risk Levels:**
  * **GREEN:** $0$ to $39$
  * **YELLOW:** $40$ to $59$
  * **ORANGE:** $60$ to $79$
  * **RED:** $80$ to $100$

---

## 4. Verification & Testing

### How to Run End-to-End Fusion
Execute the entire integration pipeline on consecutive test frames:
```bash
PYTHONPATH=SCALNet:SCALNet/src NWPU-Crowd-Sample-Code/.venv/bin/python ai-engine/examples/verify_fusion.py
```

### Outputs Generated:
* **JSON Metrics Log:** Saved at [outputs/step_5_debug/grid_metrics.json](file:///home/sarthaktripathy/Documents/Sudharsan/outputs/step_5_debug/grid_metrics.json) containing full telemetry.
* **Risk Overlay Map:** Saved at [outputs/step_5_debug/fusion_overlay.png](file:///home/sarthaktripathy/Documents/Sudharsan/outputs/step_5_debug/fusion_overlay.png). The overlay blends semi-translucent grid panels representing risk severity (Green, Yellow, Orange, Red) and prints white displacement arrow indicators.

### How to Run Tests
Execute unit tests for all modules (Step 1-5):
```bash
PYTHONPATH=SCALNet:SCALNet/src NWPU-Crowd-Sample-Code/.venv/bin/python -m unittest discover -s ai-engine/tests -p "test_*.py"
```
* **Status:** `OK` (All 28 tests passing successfully).

---

## 5. Step 6 Boundary Definition

With the core **SCALNet + Flow Fusion Engine (Step 5)** complete, we are ready for **Step 6 (Temporal Stability and Abnormal-Flow Features)**.
* **Input boundary for Step 6**: Time-series list of `GridMetrics` over consecutive frames.
* **Step 6 Target**: Implement exponential moving average (EMA) smoothing for congestion scores to prevent rapid risk level flickering (hysteresis), and build alarms for sudden speed surges or stasis buildups.
