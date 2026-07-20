# ShanghaiTech Dataset Hardened Crowd-Flow Evaluation

This report documents the performance of the **Sudharshan** crowd flow detection engine evaluated over the dense crowd scenes in the **ShanghaiTech** dataset.

## 1. Sequence Details
* **Source Folder**: `ShanghaiTech/part_A_final/test_data`
* **Total Images Processed**: 5
* **Resolution**: 960x660

## 2. Density Verification (SCALNet Counting)
* **Average Ground Truth Count**: 514.80 people
* **Average Predicted Count**: 574.42 people
* **Head Counting MAE**: 131.2253 people
* **Head Counting RMSE**: 153.0111 people

## 3. Combined Flow Telemetry Output
* Output saved to `outputs/shanghaitech_grid_metrics.json`
* Output includes local grid-wise motion speeds, directions, risk scores, stasis/turbulence warnings, and linear trend forecasts.

---
*Report generated automatically.*
