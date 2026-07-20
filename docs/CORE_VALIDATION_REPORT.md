# Sudharshan Core Detection Validation Report

This report summarizes the scientific accuracy and mathematical validity of **Sudharshan** (Crowd Flow Engine) evaluated against ground-truth labels.

## 1. Evaluation Summary
* **Sequence ID**: visdrone_uav0000339
* **Total Evaluated Grid Cells**: 420

## 2. Density & Count Metrics
* **Mean Absolute Error (MAE)**: 0.1030 persons/grid
* **Root Mean Squared Error (RMSE)**: 0.1577 persons/grid

## 3. Directional Motion Metrics
* **Mean Angular Error**: 4.08°
* **Median Angular Error**: 3.09°
* **Angular Accuracy ($\le 15°$)**: 100.00%
* **Angular Accuracy ($\le 30°$)**: 100.00%

## 4. Event Logic Detections (Confusion Matrices)

| Event Type | True Positives (TP) | False Positives (FP) | False Negatives (FN) | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **COUNTER_FLOW** | 128 | 4 | 10 | 96.97% | 92.75% | 94.81% |
| **STAGNATION** | 0 | 0 | 10 | 0.00% | 0.00% | 0.00% |
| **SPEED_SURGE** | 7 | 0 | 0 | 100.00% | 100.00% | 100.00% |
| **TURBULENCE** | 3 | 0 | 0 | 100.00% | 100.00% | 100.00% |

---
*Report generated automatically on validation run.*
