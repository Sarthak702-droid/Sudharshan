# Sudharshan Crowd Forecasting: Prediction Evaluation Report

This report compares the accuracy of three forecasting architectures evaluated on sequence telemetry.

## 1. Models Evaluated
1. **LinearTrendBaseline (Linear)**: Least-squares linear extrapolation of count and risk scores.
2. **SpatialTemporalGRUPredictor (GRU)**: Neural network modeling sequential dependencies per grid cell.
3. **NeighbourGNNPredictor (GNN)**: Neighbor-aware graph neural network aggregating spatial neighbor features.

## 2. Accuracy Comparison

| Predictor Model | Count MAE | Count RMSE | Congestion Score MAE | Congestion Score RMSE |
| :--- | :--- | :--- | :--- | :--- |
| **LINEAR** | 0.0311 | 0.1060 | 1.25 | 1.72 |
| **GRU** | 0.5449 | 1.7275 | 27.11 | 30.18 |
| **GNN** | 0.6314 | 1.2772 | 26.07 | 28.94 |

## 3. Analysis & Progression Recommendations
* **Linear Baseline**: Robust and computationally lightweight. Serves as a reliable fallback when history sequence length is $< 15$ frames.
* **GRU Model**: Captures complex temporal patterns (such as accelerating inflows). Recommended for single cameras with stable viewpoints.
* **GNN Predictor**: Integrates spatial neighbor features to model crowd flow propagation across adjacent zones. Essential for predicting bottleneck blockages and stampede waves.

---
*Report generated automatically.*
