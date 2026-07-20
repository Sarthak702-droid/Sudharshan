import math
import time
from typing import Dict, List, Tuple, Optional

from .types import GridMetrics
from flow.engine import get_direction_label
from prediction.forecaster import CrowdFlowPredictor


class TemporalTracker:
    def __init__(
        self,
        alpha: float = 0.25,
        fps: float = 5.0,  # Expected pipeline execution frame rate
        persistence_yellow_sec: float = 2.0,
        persistence_orange_sec: float = 2.0,
        persistence_red_sec: float = 1.0,
        predictor_type: str = "linear",
    ) -> None:
        """Initializes the TemporalTracker.

        Args:
            alpha: Exponential moving average smoothing coefficient (0.0 to 1.0).
            fps: Pipeline FPS to convert time duration into frame counts.
            persistence_yellow_sec: Duration in seconds a YELLOW score must persist to trigger YELLOW risk.
            persistence_orange_sec: Duration in seconds an ORANGE score must persist to trigger ORANGE risk.
            persistence_red_sec: Duration in seconds a RED score must persist to trigger RED risk.
            predictor_type: Type of prediction model to use ("linear", "gru", "gnn").
        """
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.alpha = alpha
        self.fps = fps

        # Convert seconds to frames
        self.persistence_frames = {
            "YELLOW": max(1, int(persistence_yellow_sec * fps)),
            "ORANGE": max(1, int(persistence_orange_sec * fps)),
            "RED": max(1, int(persistence_red_sec * fps)),
        }

        # Grid histories: grid_id -> state dict
        self._history: Dict[str, dict] = {}
        self._predictor = CrowdFlowPredictor(
            history_window_size=15,
            forecast_horizon_frames=15,
            model_type=predictor_type
        )

    def track(
        self,
        current_metrics: Dict[str, GridMetrics],
        adjacency_graph: Optional[Dict[str, List]] = None,
    ) -> Dict[str, GridMetrics]:
        """Applies temporal smoothing, persistence counters, and hysteresis to grid metrics.

        Args:
            current_metrics: Dict mapping grid_id to GridMetrics for the current frame.
            adjacency_graph: Optional adjacency graph mapping grid_id to its neighbors.

        Returns:
            Dict mapping grid_id to updated/stabilized GridMetrics.
        """
        updated_metrics: Dict[str, GridMetrics] = {}

        for grid_id, m in current_metrics.items():
            # Retrieve or initialize history
            if grid_id not in self._history:
                self._history[grid_id] = {
                    "ema_count": m.count,
                    "ema_density": m.density,
                    "ema_flow_x": m.flow_x,
                    "ema_flow_y": m.flow_y,
                    "ema_speed": m.speed,
                    "ema_speed_mps": m.speed_mps,
                    "ema_crowd_probability": m.crowd_probability,
                    "crowd_present": m.crowd_present,
                    "crowd_class": m.crowd_class,
                    "candidate_crowd_class": m.crowd_class,
                    "candidate_crowd_frames": 0,
                    "ema_congestion_score": m.congestion_score,
                    "risk_level": "GREEN",
                    "consecutive_category_frames": {
                        "GREEN": 0,
                        "YELLOW": 0,
                        "ORANGE": 0,
                        "RED": 0,
                    },
                    "speed_history": [m.speed],  # for surge detection
                    "consecutive_stasis_frames": 0,
                    "metrics_history": [],
                }

            h = self._history[grid_id]
            previous_speed = float(h["ema_speed"])

            # 1. Apply EMA Smoothing: EMA = alpha * x_t + (1 - alpha) * EMA_prev
            h["ema_count"] = self.alpha * m.count + (1 - self.alpha) * h["ema_count"]
            h["ema_density"] = self.alpha * m.density + (1 - self.alpha) * h["ema_density"]
            h["ema_flow_x"] = self.alpha * m.flow_x + (1 - self.alpha) * h["ema_flow_x"]
            h["ema_flow_y"] = self.alpha * m.flow_y + (1 - self.alpha) * h["ema_flow_y"]

            # Recalculate speed from smoothed flow vector to preserve vector integrity
            smoothed_speed = math.sqrt(h["ema_flow_x"]**2 + h["ema_flow_y"]**2)
            h["ema_speed"] = self.alpha * m.speed + (1 - self.alpha) * h["ema_speed"]
            h["ema_speed_mps"] = self.alpha * m.speed_mps + (1 - self.alpha) * h["ema_speed_mps"]
            h["ema_crowd_probability"] = (
                self.alpha * m.crowd_probability
                + (1 - self.alpha) * h["ema_crowd_probability"]
            )
            acceleration = (smoothed_speed - previous_speed) * self.fps

            # Recalculate direction label
            smoothed_theta = math.atan2(h["ema_flow_y"], h["ema_flow_x"])
            smoothed_theta_deg = (smoothed_theta * 180.0 / math.pi + 360.0) % 360.0
            smoothed_dir_label = get_direction_label(smoothed_theta_deg) if smoothed_speed > 1e-6 else "UNKNOWN"

            h["ema_congestion_score"] = (
                self.alpha * m.congestion_score + (1 - self.alpha) * h["ema_congestion_score"]
            )

            # Update speed history window (max 10 frames) for surge analysis
            h["speed_history"].append(m.speed)
            if len(h["speed_history"]) > 10:
                h["speed_history"].pop(0)

            # 2. Hysteresis & Persistence Logic
            # Identify the score-based risk category for the current frame
            score = h["ema_congestion_score"]

            # Default thresholds: GREEN < 40 <= YELLOW < 60 <= ORANGE < 80 <= RED
            raw_category = "GREEN"
            if m.alert_eligible:
                if score >= 80.0:
                    raw_category = "RED"
                elif score >= 60.0:
                    raw_category = "ORANGE"
                elif score >= 40.0:
                    raw_category = "YELLOW"

            # Increment consecutive frame counters for the category matches
            for cat in ["GREEN", "YELLOW", "ORANGE", "RED"]:
                if cat == raw_category:
                    h["consecutive_category_frames"][cat] += 1
                else:
                    h["consecutive_category_frames"][cat] = 0

            # Hysteresis transitions to stop rapid flickers:
            # - Leave ORANGE only when score < 54.
            # - Leave YELLOW only when score < 36.
            # - Leave RED only when score < 72.
            prev_risk = h["risk_level"]
            candidate_risk = prev_risk

            # Upward risk triggers (requires persistence check)
            if raw_category == "RED" and prev_risk != "RED":
                if h["consecutive_category_frames"]["RED"] >= self.persistence_frames["RED"]:
                    candidate_risk = "RED"
            elif raw_category == "ORANGE" and prev_risk not in ("ORANGE", "RED"):
                if h["consecutive_category_frames"]["ORANGE"] >= self.persistence_frames["ORANGE"]:
                    candidate_risk = "ORANGE"
            elif raw_category == "YELLOW" and prev_risk == "GREEN":
                if h["consecutive_category_frames"]["YELLOW"] >= self.persistence_frames["YELLOW"]:
                    candidate_risk = "YELLOW"

            # Downward risk transitions with hysteresis margins
            if prev_risk == "RED" and score < 72.0:  # 80 - 8 (10% hysteresis margin)
                candidate_risk = "ORANGE"
            if candidate_risk == "ORANGE" and score < 54.0:  # 60 - 6 (10% hysteresis margin)
                candidate_risk = "YELLOW"
            if candidate_risk == "YELLOW" and score < 36.0:  # 40 - 4 (10% hysteresis margin)
                candidate_risk = "GREEN"

            # If downward transition is triggered, reset category counters
            if candidate_risk != prev_risk:
                h["risk_level"] = candidate_risk

            # 3. Abnormal Flow Feature Detections
            # Sudden Speed Surge: Speed is twice the average of the previous 5 frames
            is_surge = False
            if len(h["speed_history"]) >= 5:
                prev_avg_speed = sum(h["speed_history"][:-1]) / (len(h["speed_history"]) - 1)
                # Panic surge threshold: twice the average, and current speed must be significant
                if m.alert_eligible and prev_avg_speed > 0.5 and m.speed > 2.0 * prev_avg_speed:
                    is_surge = True

            # Stasis / Crowd Crush buildup hazard: sustained high density and low speed
            # density_red_scaled is approx 0.00035. So density > 0.0002 is high. Speed < 0.3 is slow.
            if m.physical_calibrated and m.density_people_m2 > 0:
                stasis_condition = m.density_people_m2 >= 3.5 and h["ema_speed_mps"] < 0.35
            else:
                stasis_condition = h["ema_density"] > 0.0002 and smoothed_speed < 0.3
            if m.alert_eligible and stasis_condition:
                h["consecutive_stasis_frames"] += 1
            else:
                h["consecutive_stasis_frames"] = 0

            stasis_warning = h["consecutive_stasis_frames"] >= max(1, int(2.0 * self.fps))  # Sustained for 2 seconds

            # Turbulence warning: high conflict score in dense crowds
            turbulence_warning = (
                m.alert_eligible
                and m.flow_conflict_score > 0.6
                and (m.density_people_m2 >= 2.0 if m.physical_calibrated and m.density_people_m2 > 0
                     else h["ema_density"] > 0.0002)
            )

            # Crowd-presence hysteresis avoids class flicker at the decision
            # boundary. Class changes also require three consistent frames.
            if h["crowd_present"]:
                h["crowd_present"] = h["ema_crowd_probability"] >= 0.35
            else:
                h["crowd_present"] = h["ema_crowd_probability"] >= 0.55
            if m.crowd_class == h["candidate_crowd_class"]:
                h["candidate_crowd_frames"] += 1
            else:
                h["candidate_crowd_class"] = m.crowd_class
                h["candidate_crowd_frames"] = 1
            if h["candidate_crowd_frames"] >= 3:
                h["crowd_class"] = h["candidate_crowd_class"]

            # Populate stabilized GridMetrics object
            updated_metrics[grid_id] = GridMetrics(
                grid_id=m.grid_id,
                count=float(h["ema_count"]),
                density=float(h["ema_density"]),
                flow_x=float(h["ema_flow_x"]),
                flow_y=float(h["ema_flow_y"]),
                speed=float(smoothed_speed),
                direction_deg=float(smoothed_theta_deg),
                direction_label=smoothed_dir_label,
                density_score=m.density_score,
                slow_score=m.slow_score,
                stagnation_score=m.stagnation_score,
                flow_conflict_score=m.flow_conflict_score,
                reverse_score=m.reverse_score,
                congestion_score=float(score),
                risk_level=h["risk_level"],
                confidence=m.confidence,
                # New parameters:
                turbulence_score=m.turbulence_score,
                speed_surge_warning=bool(is_surge),
                stasis_warning=bool(stasis_warning),
                turbulence_warning=bool(turbulence_warning),
                crowd_present=bool(h["crowd_present"]),
                crowd_class=h["crowd_class"],
                crowd_probability=float(h["ema_crowd_probability"]),
                flow_quality=m.flow_quality,
                valid_flow_ratio=m.valid_flow_ratio,
                alert_eligible=m.alert_eligible,
                physical_calibrated=m.physical_calibrated,
                speed_mps=float(h["ema_speed_mps"]),
                density_people_m2=m.density_people_m2,
                divergence=m.divergence,
                acceleration=float(acceleration),
            )

            # Store in metrics history
            h["metrics_history"].append(updated_metrics[grid_id])
            if len(h["metrics_history"]) > 15:
                h["metrics_history"].pop(0)

            # Perform prediction passing adjacency information
            pred_count, pred_score, pred_risk, slope = self._predictor.predict_next(
                grid_id=grid_id,
                historical_metrics=h["metrics_history"],
                adjacency_graph=adjacency_graph,
                all_grids_metrics=current_metrics,
            )
            updated_metrics[grid_id].predicted_count = pred_count
            updated_metrics[grid_id].predicted_congestion_score = pred_score
            updated_metrics[grid_id].predicted_risk_level = pred_risk
            updated_metrics[grid_id].trend_slope = slope

        return updated_metrics
