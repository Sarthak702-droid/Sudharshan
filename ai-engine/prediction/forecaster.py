import numpy as np
import torch
import torch.nn as nn
import math
from typing import List, Tuple, Dict, Optional, Union


class LinearTrendBaseline:
    def __init__(self, history_window_size: int = 15, forecast_horizon_frames: int = 15) -> None:
        self.history_window_size = history_window_size
        self.forecast_horizon_frames = forecast_horizon_frames

    def predict_next(
        self,
        grid_id: str,
        historical_metrics: List['GridMetrics'],
        adjacency_graph: Optional[Dict[str, List]] = None,
    ) -> Tuple[float, float, str, float]:
        """Predicts the future count, congestion score, risk level, and trend slope using linear trend baseline."""
        if not historical_metrics:
            return 0.0, 0.0, "GREEN", 0.0

        latest = historical_metrics[-1]
        n = len(historical_metrics)

        if n < 3:
            return latest.count, latest.congestion_score, latest.risk_level, 0.0

        # Truncate history to window size
        window = historical_metrics[-self.history_window_size:]
        n_window = len(window)

        # X represents frame index offsets, Y represents counts or congestion scores
        t = np.arange(n_window, dtype=np.float64)
        counts = np.array([m.count for m in window], dtype=np.float64)
        scores = np.array([m.congestion_score for m in window], dtype=np.float64)

        # Fit linear regression: y = slope * t + intercept
        sum_t = np.sum(t)
        sum_y_count = np.sum(counts)
        sum_t2 = np.sum(t**2)
        sum_ty_count = np.sum(t * counts)

        denom = n_window * sum_t2 - (sum_t**2)
        if abs(denom) < 1e-6:
            slope_count = 0.0
            intercept_count = latest.count
        else:
            slope_count = (n_window * sum_ty_count - sum_t * sum_y_count) / denom
            intercept_count = (sum_y_count - slope_count * sum_t) / n_window

        # For Congestion Score
        sum_y_score = np.sum(scores)
        sum_ty_score = np.sum(t * scores)
        if abs(denom) < 1e-6:
            slope_score = 0.0
            intercept_score = latest.congestion_score
        else:
            slope_score = (n_window * sum_ty_score - sum_t * sum_y_score) / denom
            intercept_score = (sum_y_score - slope_score * sum_t) / n_window

        # Calculate forecast at (n_window + forecast_horizon_frames - 1)
        future_t = (n_window - 1) + self.forecast_horizon_frames
        pred_count = max(0.0, float(slope_count * future_t + intercept_count))
        pred_score = float(np.clip(slope_score * future_t + intercept_score, 0.0, 100.0))

        # Classify future risk level based on predicted score
        if pred_score < 40.0:
            pred_risk = "GREEN"
        elif pred_score < 60.0:
            pred_risk = "YELLOW"
        elif pred_score < 80.0:
            pred_risk = "ORANGE"
        else:
            pred_risk = "RED"

        return pred_count, pred_score, pred_risk, float(slope_score)


class SpatialTemporalGRUPredictor(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 16, num_layers: int = 1) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        # Linear layer outputs future count and future congestion score
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, sequence_length, input_dim]
        out, _ = self.gru(x)
        # Take the output of the last sequence step
        last_out = out[:, -1, :]
        preds = self.fc(last_out)
        return preds


class NeighbourGNNPredictor(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 16) -> None:
        super().__init__()
        # Simple spatial neighbor aggregation + fully connected layers
        self.spatial_layer = nn.Linear(input_dim * 2, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, target_features: torch.Tensor, neighbor_features: torch.Tensor) -> torch.Tensor:
        # target_features shape: [batch_size, input_dim]
        # neighbor_features shape: [batch_size, input_dim]
        combined = torch.cat([target_features, neighbor_features], dim=-1)
        hidden = torch.relu(self.spatial_layer(combined))
        preds = self.fc(hidden)
        return preds


class CrowdFlowPredictor:
    def __init__(
        self,
        history_window_size: int = 15,
        forecast_horizon_frames: int = 15,
        model_type: str = "linear",
        weights_path: Optional[str] = None
    ) -> None:
        self.history_window_size = history_window_size
        self.forecast_horizon_frames = forecast_horizon_frames
        self.model_type = model_type.lower()

        # Instantiate baseline
        self.baseline = LinearTrendBaseline(history_window_size, forecast_horizon_frames)

        # Neural predictors setup
        self.gru_model: Optional[SpatialTemporalGRUPredictor] = None
        self.gnn_model: Optional[NeighbourGNNPredictor] = None
        self.has_weights = False

        # Features dim: count, density, flow_x, flow_y, speed, coherence, congestion_score, risk_score
        self.input_dim = 8

        if self.model_type == "gru":
            self.gru_model = SpatialTemporalGRUPredictor(input_dim=self.input_dim)
            if weights_path and torch.os.path.exists(weights_path):
                try:
                    self.gru_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
                    self.has_weights = True
                except Exception as e:
                    import warnings
                    warnings.warn(f"Failed to load GRU weights from {weights_path}, using random initialization: {e}")
            self.gru_model.eval()

        elif self.model_type == "gnn":
            self.gnn_model = NeighbourGNNPredictor(input_dim=self.input_dim)
            if weights_path and torch.os.path.exists(weights_path):
                try:
                    self.gnn_model.load_state_dict(torch.load(weights_path, map_location="cpu"))
                    self.has_weights = True
                except Exception as e:
                    import warnings
                    warnings.warn(f"Failed to load GNN weights from {weights_path}, using random initialization: {e}")
            self.gnn_model.eval()

    def _extract_feature_vector(self, m: 'GridMetrics') -> np.ndarray:
        # Extract 8 key features per time step
        return np.array([
            m.count,
            m.density,
            m.flow_x,
            m.flow_y,
            m.speed,
            1.0 - m.flow_conflict_score, # coherence
            m.congestion_score,
            m.confidence
        ], dtype=np.float32)

    def predict_next(
        self,
        grid_id: str,
        historical_metrics: List['GridMetrics'],
        adjacency_graph: Optional[Dict[str, List]] = None,
        all_grids_metrics: Optional[Dict[str, 'GridMetrics']] = None,
    ) -> Tuple[float, float, str, float]:
        """Predicts the future count, congestion score, risk level, and trend slope.
        Dispatches to appropriate model based on configuration, falling back to linear baseline if needed.
        """
        if not historical_metrics:
            return 0.0, 0.0, "GREEN", 0.0

        latest = historical_metrics[-1]

        # Calculate trend slope from linear regression regardless of model type for consistency
        _, _, _, linear_slope = self.baseline.predict_next(grid_id, historical_metrics)

        # Fallback to linear if insufficient window size or model is set to linear
        if self.model_type == "linear" or len(historical_metrics) < self.history_window_size:
            return self.baseline.predict_next(grid_id, historical_metrics)

        # Slice history to window size
        window = historical_metrics[-self.history_window_size:]

        # --- GRU FORECASTER ---
        if self.model_type == "gru" and self.gru_model is not None:
            try:
                features = np.stack([self._extract_feature_vector(m) for m in window], axis=0) # [seq_len, dim]
                tensor_input = torch.tensor(features).unsqueeze(0) # [1, seq_len, dim]

                with torch.no_grad():
                    preds = self.gru_model(tensor_input).squeeze(0).numpy() # [2]

                pred_count = max(0.0, float(preds[0]))
                pred_score = float(np.clip(preds[1], 0.0, 100.0))

                # Classify risk level
                if pred_score < 40.0:
                    pred_risk = "GREEN"
                elif pred_score < 60.0:
                    pred_risk = "YELLOW"
                elif pred_score < 80.0:
                    pred_risk = "ORANGE"
                else:
                    pred_risk = "RED"

                return pred_count, pred_score, pred_risk, linear_slope
            except Exception:
                # Fallback on any runtime prediction error
                return self.baseline.predict_next(grid_id, historical_metrics)

        # --- GNN FORECASTER ---
        elif self.model_type == "gnn" and self.gnn_model is not None:
            try:
                # Extract features for the target grid cell at the latest time step
                target_feats = self._extract_feature_vector(latest)

                # Extract and aggregate neighbor features at the latest time step
                neighbor_feats_list = []
                if adjacency_graph and grid_id in adjacency_graph and all_grids_metrics:
                    edges = adjacency_graph[grid_id]
                    for edge in edges:
                        neigh_id = edge.target_id
                        if neigh_id in all_grids_metrics:
                            neighbor_feats_list.append(self._extract_feature_vector(all_grids_metrics[neigh_id]))

                if neighbor_feats_list:
                    neighbor_feats = np.mean(neighbor_feats_list, axis=0)
                else:
                    neighbor_feats = np.zeros_like(target_feats)

                t_target = torch.tensor(target_feats).unsqueeze(0) # [1, dim]
                t_neighbor = torch.tensor(neighbor_feats).unsqueeze(0) # [1, dim]

                with torch.no_grad():
                    preds = self.gnn_model(t_target, t_neighbor).squeeze(0).numpy() # [2]

                pred_count = max(0.0, float(preds[0]))
                pred_score = float(np.clip(preds[1], 0.0, 100.0))

                if pred_score < 40.0:
                    pred_risk = "GREEN"
                elif pred_score < 60.0:
                    pred_risk = "YELLOW"
                elif pred_score < 80.0:
                    pred_risk = "ORANGE"
                else:
                    pred_risk = "RED"

                return pred_count, pred_score, pred_risk, linear_slope
            except Exception:
                return self.baseline.predict_next(grid_id, historical_metrics)

        return self.baseline.predict_next(grid_id, historical_metrics)
