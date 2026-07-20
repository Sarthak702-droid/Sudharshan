# Step 6: Temporal Stability Module

This document details the implementation, verification, and benchmarking of Step 6 for **Sudharshan** (Live Crowd Flow Intelligence Engine).

---

## 1. Executive Summary

Step 6 has successfully introduced the Temporal Tracker module (`ai-engine/fusion/tracker.py`). It applies temporal filtering to the raw spatial fusion outputs of Step 5. It uses Exponential Moving Average (EMA) smoothing to reduce high-frequency frame-to-frame noise.

It implements temporal persistence constraints and a mathematical hysteresis state-machine to prevent color risk alerts from flickering rapidly back and forth under boundary conditions.

---

## 2. Deliverables & File Structure

The following file structure has been established under the project workspace:

```text
ai-engine/
├── fusion/
│   ├── __init__.py
│   ├── types.py
│   ├── aggregator.py
│   └── tracker.py
└── tests/
    └── test_tracker.py
```

### Files Created
* **[tracker.py](file:///home/sarthaktripathy/Documents/Sudharsan/ai-engine/fusion/tracker.py)**: The `TemporalTracker` class managing multi-frame states, EMA computations, persistence frames, and hysteresis transitions.
* **[test_tracker.py](file:///home/sarthaktripathy/Documents/Sudharsan/ai-engine/tests/test_tracker.py)**: Unit tests for EMA smoothing averages, temporal persistence delay, and hysteresis margin boundaries.
* **[verify_stability.py](file:///home/sarthaktripathy/Documents/Sudharsan/ai-engine/examples/verify_stability.py)**: Verification script running a 5-step test sequence showing stabilized risk tracking for a high-density grid.

---

## 3. Mathematical Stability Policies

### EMA Smoothing Formulation
For each grid metric $x_t$ (count, density, flow vectors, speed, congestion score) at time step $t$:
$$\text{EMA}_t = \alpha \times x_t + (1 - \alpha) \times \text{EMA}_{t-1}$$
*Where default $\alpha = 0.25$ (configurable).*

### Temporal Alert Persistence
Transitions to higher risk levels are delayed until the category is sustained for a minimum duration to avoid false alarms from single-frame artifacts (like sudden brief occlusions):
* **YELLOW Alert:** Score must persist for $\ge 2.0$ seconds.
* **ORANGE Alert:** Score must persist for $\ge 2.0$ seconds.
* **RED Alert:** Score must persist for $\ge 1.0$ seconds.

### Hysteresis State Machine
To stop risk levels from flickering when the congestion score hovers around category boundaries (e.g., $60.0$ for ORANGE), we define downward transition margins:
* **RED $\to$ ORANGE:** Exit RED only when score drops below $< 72.0$ (threshold $80.0$, $10\%$ margin).
* **ORANGE $\to$ YELLOW:** Exit ORANGE only when score drops below $< 54.0$ (threshold $60.0$, $10\%$ margin).
* **YELLOW $\to$ GREEN:** Exit YELLOW only when score drops below $< 36.0$ (threshold $40.0$, $10\%$ margin).

---

## 4. Verification & Testing

### How to Run Stability Verification
Execute the multi-frame tracking simulation on the sequence:
```bash
PYTHONPATH=SCALNet:SCALNet/src NWPU-Crowd-Sample-Code/.venv/bin/python ai-engine/examples/verify_stability.py
```

### Verification Benchmarks Table:
Tracking high-density grid `G_01_04`:

| Frame | Raw Count | Raw Score | Raw Risk | EMA Count | EMA Score | EMA Risk |
|---|---|---|---|---|---|---|
| **0** | 6.87 | 83.3 | RED | 6.87 | 83.3 | **RED** |
| **1** | 5.38 | 85.9 | RED | 6.42 | 84.1 | **RED** |
| **2** | 6.06 | 83.9 | RED | 6.31 | 84.0 | **RED** |
| **3** | 8.45 | 76.7 | *ORANGE* | 6.96 | 81.8 | **RED** (Stabilized) |
| **4** | 8.53 | 78.1 | *ORANGE* | 7.43 | 80.7 | **RED** (Stabilized) |

*Observation:* While the raw fusion score drops to $76.7$ on Frame 3, causing a raw risk alert fluctuation to ORANGE, the EMA-smoothed congestion score remains at $81.8$, safely holding the risk level at RED and preventing flickering.

### How to Run Stability Tests
Execute the entire test suite including the stability test assertions:
```bash
PYTHONPATH=SCALNet:SCALNet/src NWPU-Crowd-Sample-Code/.venv/bin/python -m unittest discover -s ai-engine/tests -p "test_*.py"
```
* **Status:** `OK` (All 31 tests passing successfully).

---

## 5. Step 7 Boundary Definition

With the **Temporal Stability Tracker (Step 6)** complete, we are ready for **Step 7 (Python Service Layer)**.
* **Input boundary for Step 7**: Fused, stabilized grid telemetry results from the `TemporalTracker`.
* **Step 7 Target**: Wrap this entire pipelines (Ingestion $\to$ Grids $\to$ SCALNet $\to$ DIS Flow $\to$ Fusion $\to$ Tracker) inside a FastAPI service exposure layer (SSE or WebSocket streaming) to communicate grid states to frontend dashboards or database backends.
