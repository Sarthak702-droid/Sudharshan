# Step 10: Crowd Flow & Congestion Prediction Module

This document details the implementation, verification, and benchmarking of Step 10 (Crowd Flow & Congestion Prediction Engine) for **Sudharshan**.

---

## 1. Executive Summary

Step 10 has successfully designed, implemented, and verified a mathematical forecasting engine (`ai-engine/prediction/forecaster.py`) integrated directly into the `TemporalTracker`.

Using least-squares linear trend regression over a sliding historical frame window, the prediction module forecasts the crowd count, congestion score, risk level, and trend direction slope $K$ frames into the future. It exports these predictive safety values directly to JSON and CSV metrics targets.

---

## 2. Deliverables & File Structure

The following file structure has been established:

```text
ai-engine/
├── prediction/
│   ├── __init__.py
│   └── forecaster.py    [Created]
└── tests/
    └── test_prediction.py [Created]
```

### Files Created & Verified
* **[forecaster.py](file:///home/sarthaktripathy/Documents/Sudharsan/ai-engine/prediction/forecaster.py)**: The `CrowdFlowPredictor` class computing linear least-squares regression slopes and projecting counts/congestion scores into the future.
* **[test_prediction.py](file:///home/sarthaktripathy/Documents/Sudharsan/ai-engine/tests/test_prediction.py)**: Unit tests for linear forecasting trends and fallback limits.
* **[STEP_10_PREDICTION.md](file:///home/sarthaktripathy/Documents/Sudharsan/docs/STEP_10_PREDICTION.md)**: This documentation.

---

## 3. Mathematical Forecasting Formulation

For a sliding history window of $N$ frames, we map frame offsets $t \in [0, N-1]$ to metrics $y_t$ (count or congestion score):
1. **Slope ($m$):**
   $$m = \frac{N \sum_{t=0}^{N-1} (t \cdot y_t) - \sum_{t=0}^{N-1} t \sum_{t=0}^{N-1} y_t}{N \sum_{t=0}^{N-1} t^2 - \left(\sum_{t=0}^{N-1} t\right)^2}$$
2. **Intercept ($c$):**
   $$c = \frac{\sum_{t=0}^{N-1} y_t - m \sum_{t=0}^{N-1} t}{N}$$
3. **Forecast ($K$ frames ahead):**
   $$\hat{y}_{N-1+K} = m \cdot (N - 1 + K) + c$$

*Note: Default window size $N=15$ frames and prediction horizon $K=15$ frames (~1 second ahead at 15 FPS).*

---

## 4. Verification & Testing

### How to Run the Unit Tests
All unit tests (including the new linear forecasting trend assertions) compile and pass successfully:
```bash
./run_all_tests.sh
```
* **Status:** `OK` (All 33 test cases pass).

### How to Run Pipeline Verification
Run the 5-frame test over the VisDrone sequence:
```bash
./run_pipeline_sample.sh
```
* **Status:** `Success` (Grid metrics JSON/CSV successfully saved with new prediction fields: `predicted_count`, `predicted_congestion_score`, `predicted_risk_level`, and `trend_slope`).

---

## 5. System Integration Summary

With **Step 10 (Prediction)** complete, all core pipeline components of Sudharshan are fully finished, integrated, and verified!

The system provides:
1. **SCALNet Density Inference** under perspective projection corrections.
2. **DIS Dense Optical Flow** under CLAHE contrast adjustments.
3. **Overlapping Grid Segmentation** with GIS mask clipping.
4. **Spatial Fusion & Temporal EMA Stability** with alert hysteresis.
5. **Abnormal Behavior Warnings** (Panic Speed Surge, Crush Hazard Stasis, Flow Turbulence Conflict).
6. **Least-squares linear trend predictions** of crowd variables.
7. **Production Go API backend** persisting data and firing automated safety alerts.
