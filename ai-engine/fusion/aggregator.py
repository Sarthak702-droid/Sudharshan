import numpy as np
import cv2
import math
from typing import List, Dict, Union, Tuple, Optional

from .types import GridMetrics
from density.types import DensityInferenceResult
from flow.types import FlowResult
from grid.types import GridBox
from flow.engine import get_direction_label


# Add helper if needed
def label_to_vector(label: str) -> Tuple[float, float]:
    label = label.upper().strip()
    if label == "EAST":
        return (1.0, 0.0)
    elif label == "WEST":
        return (-1.0, 0.0)
    elif label == "SOUTH":
        return (0.0, 1.0)
    elif label == "NORTH":
        return (0.0, -1.0)
    elif label == "SOUTH-EAST":
        inv_sqrt2 = 1.0 / math.sqrt(2)
        return (inv_sqrt2, inv_sqrt2)
    elif label == "SOUTH-WEST":
        inv_sqrt2 = 1.0 / math.sqrt(2)
        return (-inv_sqrt2, inv_sqrt2)
    elif label == "NORTH-WEST":
        inv_sqrt2 = 1.0 / math.sqrt(2)
        return (-inv_sqrt2, -inv_sqrt2)
    elif label == "NORTH-EAST":
        inv_sqrt2 = 1.0 / math.sqrt(2)
        return (inv_sqrt2, -inv_sqrt2)
    else:
        return (0.0, 0.0)


class FusionAggregator:
    def __init__(
        self,
        density_red: float = 4.0,       # persons per m2 or normalized density scale
        speed_normal: float = 3.0,      # pixels per frame or normal speed scale
        expected_direction: Union[str, Dict[str, str]] = "EAST",
        min_confidence: float = 0.60,
        use_heuristic_perspective_scaling: bool = False,
        huber_delta: float = 1.5,
        crowd_presence_count: float = 1.0,
        crowd_class_thresholds: Tuple[float, float, float, float] = (1.0, 3.0, 6.0, 10.0),
        crowd_density_thresholds: Tuple[float, float, float, float] = (0.5, 2.0, 4.0, 6.0),
        fps: float = 15.0,
        meters_per_pixel: Optional[float] = None,
    ) -> None:
        """Initializes the FusionAggregator.

        Args:
            density_red: Configurable density threshold indicating high danger.
            speed_normal: Configurable speed threshold indicating normal free flow.
            expected_direction: Default direction string or dictionary mapping grid_id to expected direction label.
            min_confidence: Configurable confidence threshold below which alerts are suppressed.
            use_heuristic_perspective_scaling: If True, uses the legacy arbitrary perspective scaling. Defaults to False.
            huber_delta: Residual threshold for the Huber M-estimator outlier rejection.
        """
        self.density_red = density_red
        self.speed_normal = speed_normal
        self.expected_direction = expected_direction
        self.min_confidence = min_confidence
        self.huber_delta = huber_delta
        self.crowd_presence_count = float(crowd_presence_count)
        self.crowd_class_thresholds = tuple(float(v) for v in crowd_class_thresholds)
        self.crowd_density_thresholds = tuple(float(v) for v in crowd_density_thresholds)
        if tuple(sorted(self.crowd_class_thresholds)) != self.crowd_class_thresholds:
            raise ValueError("crowd_class_thresholds must be sorted")
        if tuple(sorted(self.crowd_density_thresholds)) != self.crowd_density_thresholds:
            raise ValueError("crowd_density_thresholds must be sorted")
        if fps <= 0 or (meters_per_pixel is not None and meters_per_pixel <= 0):
            raise ValueError("Physical calibration parameters must be positive")
        self.fps = float(fps)
        self.meters_per_pixel = meters_per_pixel
        if use_heuristic_perspective_scaling:
            raise ValueError(
                "Arbitrary perspective scaling has been disabled in Sprint 1 Audit and Freeze. "
                "Do not use arbitrary y-position scaling as physical calibration."
            )
        self.use_heuristic_perspective_scaling = False

    def _robust_flow(
        self,
        local_u: np.ndarray,
        local_v: np.ndarray,
        local_d: np.ndarray,
        local_valid: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """Return robust u/v, circular variance, and usable-vector ratio."""
        finite = np.isfinite(local_u) & np.isfinite(local_v) & local_valid
        valid_ratio = float(np.mean(finite)) if finite.size else 0.0
        if not np.any(finite):
            return 0.0, 0.0, 0.0, valid_ratio

        u = local_u[finite].astype(np.float64)
        v = local_v[finite].astype(np.float64)
        density_weights = np.clip(local_d[finite].astype(np.float64), 0.0, None)
        if density_weights.sum() <= 1e-8:
            base_weights = np.ones_like(u)
        else:
            # A small floor prevents a single hot density pixel from owning the
            # estimate while still focusing the vector on occupied pixels.
            floor = float(np.percentile(density_weights, 50)) * 0.05
            base_weights = density_weights + floor

        # Median initialization gives the M-estimator a 50% breakdown point;
        # starting at the arithmetic mean lets a small extreme cluster pull the
        # first Huber iteration too far away from the dominant crowd motion.
        u_est = float(np.median(u))
        v_est = float(np.median(v))
        weights = base_weights
        for _ in range(3):
            residual = np.hypot(u - u_est, v - v_est)
            positive = residual[residual > 1e-6]
            scale = float(np.median(positive)) if positive.size else 0.0
            if scale <= 1e-6:
                break
            cutoff = self.huber_delta * 1.4826 * scale
            robust = np.ones_like(residual)
            outliers = residual > cutoff
            robust[outliers] = cutoff / np.maximum(residual[outliers], 1e-8)
            weights = base_weights * robust
            weight_sum = float(weights.sum())
            if weight_sum <= 1e-8:
                break
            u_est = float(np.sum(weights * u) / weight_sum)
            v_est = float(np.sum(weights * v) / weight_sum)

        magnitudes = np.hypot(u, v)
        angle_weights = weights * np.maximum(magnitudes, 1e-3)
        angles = np.arctan2(v, u)
        weight_sum = float(angle_weights.sum())
        if weight_sum <= 1e-8:
            circular_variance = 0.0
        else:
            mean_cos = float(np.sum(angle_weights * np.cos(angles)) / weight_sum)
            mean_sin = float(np.sum(angle_weights * np.sin(angles)) / weight_sum)
            circular_variance = float(np.clip(1.0 - math.hypot(mean_cos, mean_sin), 0.0, 1.0))
        return u_est, v_est, circular_variance, valid_ratio

    def _crowd_class(self, value: float, physical: bool = False) -> str:
        sparse, moderate, dense, critical = (
            self.crowd_density_thresholds if physical else self.crowd_class_thresholds
        )
        # Density maps are float32; tolerate summation error exactly at a class boundary.
        value += 1e-6
        if value < sparse:
            return "EMPTY"
        if value < moderate:
            return "SPARSE"
        if value < dense:
            return "MODERATE"
        if value < critical:
            return "DENSE"
        return "CRITICAL"

    @staticmethod
    def _polygon_area(points: Optional[List[Tuple[float, float]]]) -> float:
        if not points or len(points) < 3:
            return 0.0
        coords = np.asarray(points, dtype=np.float64)
        return float(abs(np.dot(coords[:, 0], np.roll(coords[:, 1], 1))
                         - np.dot(coords[:, 1], np.roll(coords[:, 0], 1))) * 0.5)

    def _get_expected_vector(self, grid_id: str) -> Tuple[float, float]:
        """Resolves expected direction unit vector for a specific grid."""
        if isinstance(self.expected_direction, dict):
            label = self.expected_direction.get(grid_id, "EAST")
        else:
            label = self.expected_direction
        return label_to_vector(label)

    def fuse(
        self,
        grids: List[GridBox],
        density_result: DensityInferenceResult,
        flow_result: FlowResult,
        overlap_weights: Optional[np.ndarray] = None,
    ) -> Dict[str, GridMetrics]:
        """Fuses density and flow maps to generate grid-level crowd analytics and risks.

        Args:
            grids: List of active GridBox structures.
            density_result: Output from SCALNetAdapter density inference.
            flow_result: Output from OpticalFlowEngine displacement mapping.
            overlap_weights: Optional precomputed overlap correction matrix.

        Returns:
            Dict mapping grid_id to GridMetrics.
        """
        grid_metrics_dict: Dict[str, GridMetrics] = {}

        h_flow, w_flow = flow_result.flow_x.shape[:2]

        # 1. Resize/Align density map to flow spatial resolution
        h_den, w_den = density_result.density_map.shape[:2]
        if h_den != h_flow or w_den != w_flow:
            resized_density = cv2.resize(
                density_result.density_map,
                (w_flow, h_flow),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            resized_density = density_result.density_map

        # Initialize overlap weights map if not provided
        if overlap_weights is None:
            overlap_weights = np.ones((h_flow, w_flow), dtype=np.float32)
        elif overlap_weights.shape != (h_flow, w_flow):
            raise ValueError("overlap_weights must match flow resolution")
        weighted_density = resized_density * overlap_weights
        valid_map = flow_result.valid_mask
        if valid_map is None:
            valid_map = np.isfinite(flow_result.flow_x) & np.isfinite(flow_result.flow_y)
        elif valid_map.shape != (h_flow, w_flow):
            valid_map = cv2.resize(valid_map.astype(np.uint8), (w_flow, h_flow),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)

        # Helper lookups to extract neighbor directions
        grid_by_id = {g.grid_id: g for g in grids}

        # Precompute individual grid observed flow variables
        observed_flows: Dict[str, Tuple[float, float, float]] = {}
        local_circular_variances: Dict[str, float] = {}
        valid_flow_ratios: Dict[str, float] = {}
        grid_masks: Dict[str, Optional[np.ndarray]] = {}

        for g in grids:
            # Handle ground-plane projected vs. image-relative grids
            if not g.is_relative and g.image_polygon is not None:
                mask = np.zeros((h_flow, w_flow), dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(g.image_polygon, dtype=np.int32)], 1)

                local_u = flow_result.flow_x[mask == 1]
                local_v = flow_result.flow_y[mask == 1]
                local_d = resized_density[mask == 1]
                local_valid = valid_map[mask == 1]
                grid_masks[g.grid_id] = mask
            else:
                x1, y1, x2, y2 = g.x1, g.y1, g.x2, g.y2
                local_u = flow_result.flow_x[y1:y2, x1:x2]
                local_v = flow_result.flow_y[y1:y2, x1:x2]
                local_d = resized_density[y1:y2, x1:x2]
                local_valid = valid_map[y1:y2, x1:x2]
                grid_masks[g.grid_id] = None

            u_bar, v_bar, local_cv, valid_ratio = self._robust_flow(
                local_u, local_v, local_d, local_valid
            )
            theta = math.atan2(v_bar, u_bar)

            observed_flows[g.grid_id] = (u_bar, v_bar, theta)
            local_circular_variances[g.grid_id] = local_cv
            valid_flow_ratios[g.grid_id] = valid_ratio

        # 2. Perform second-pass neighbor conflict & fusion calculation
        for g in grids:
            x1, y1, x2, y2 = g.x1, g.y1, g.x2, g.y2

            # Calculate vertical center of the grid cell
            cy = (y1 + y2) / 2.0

            if self.use_heuristic_perspective_scaling:
                # Legacy arbitrary perspective scale factor (gamma=1.5). Near y=0 (top / far), scale is 2.5. Near y=h_flow (bottom / close), scale is 1.0.
                # WARNING: This is an uncalibrated heuristic scaling, not a camera calibration model.
                perspective_scale = 1.0 + 1.5 * (1.0 - cy / h_flow)
            else:
                # Production default is disabled (no scaling) until actual homography / metric camera calibration is loaded.
                perspective_scale = 1.0

            # Density metrics (with perspective scaling and overlap weight correction)
            if not g.is_relative and g.image_polygon is not None:
                mask = grid_masks[g.grid_id]
                grid_count = float(np.sum(weighted_density[mask == 1]))
            else:
                local_d = resized_density[y1:y2, x1:x2]
                local_w = overlap_weights[y1:y2, x1:x2]
                grid_count = float(np.sum(local_d * local_w))

            grid_count_corrected = grid_count * perspective_scale
            visible_area_ratio = float(g.effective_area / g.area) if g.area > 0 else 0.0
            ground_area = self._polygon_area(g.ground_polygon)
            if ground_area > 0:
                ground_area *= visible_area_ratio
                grid_density = grid_count_corrected / max(ground_area, 1e-6)
                density_people_m2 = grid_density
            else:
                grid_density = grid_count_corrected / g.effective_area if g.effective_area > 0 else 0.0
                density_people_m2 = 0.0

            # Flow metrics (with perspective scaling)
            u_bar, v_bar, theta_g = observed_flows[g.grid_id]
            u_bar_corrected = u_bar * perspective_scale
            v_bar_corrected = v_bar * perspective_scale
            magnitude_corrected = math.sqrt(u_bar_corrected**2 + v_bar_corrected**2)

            theta_deg = (theta_g * 180.0 / math.pi + 360.0) % 360.0
            dir_label = get_direction_label(theta_deg) if magnitude_corrected > 1e-6 else "UNKNOWN"

            # Circular variance
            local_cv = local_circular_variances[g.grid_id]

            # Neighbor conflict calculation
            neighbor_conflicts = []
            row_g, col_g = g.row, g.col

            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    neighbor_id = f"G_{row_g+dr:02d}_{col_g+dc:02d}"
                    if neighbor_id in observed_flows:
                        _, _, theta_h = observed_flows[neighbor_id]
                        # Disagreement: (1 - cos(theta_g - theta_h)) / 2
                        conflict_val = (1.0 - math.cos(theta_g - theta_h)) / 2.0
                        neighbor_conflicts.append(conflict_val)

            mean_neighbor_conflict = np.mean(neighbor_conflicts) if neighbor_conflicts else 0.0

            # Combine local & neighbor conflicts: 70% local circular variance + 30% neighbor conflict
            flow_conflict = float(0.70 * local_cv + 0.30 * mean_neighbor_conflict)

            # Expected Direction Dot Product
            expected_u, expected_v = self._get_expected_vector(g.grid_id)

            if magnitude_corrected > 1e-6:
                obs_u_unit = u_bar_corrected / magnitude_corrected
                obs_v_unit = v_bar_corrected / magnitude_corrected
                dot_product = obs_u_unit * expected_u + obs_v_unit * expected_v
                reverse_score = max(0.0, -dot_product)
            else:
                reverse_score = 0.0

            # Normalization and Congestion Score (using corrected values)
            density_score = float(np.clip(grid_density / self.density_red, 0.0, 1.0))
            slow_score = float(1.0 - np.clip(magnitude_corrected / self.speed_normal, 0.0, 1.0))
            stagnation = density_score * slow_score

            # Congestion Score (Nonlinear Formula):
            raw_congestion = (
                0.35 * density_score
                + 0.20 * slow_score
                + 0.20 * stagnation
                + 0.15 * flow_conflict
                + 0.10 * reverse_score
            )
            congestion_score = float(np.clip(raw_congestion, 0.0, 1.0) * 100.0)

            # Risk Classification
            if congestion_score < 40.0:
                risk_level = "GREEN"
            elif congestion_score < 60.0:
                risk_level = "YELLOW"
            elif congestion_score < 80.0:
                risk_level = "ORANGE"
            else:
                risk_level = "RED"

            # Confidence calculation
            density_model_confidence = 1.0
            flow_coherence = float(1.0 - local_cv)
            frame_flow_quality = float(np.clip(flow_result.flow_quality, 0.0, 1.0))
            local_vector_quality = float(np.clip(valid_flow_ratios[g.grid_id] / 0.15, 0.0, 1.0))
            effective_flow_quality = frame_flow_quality * local_vector_quality

            confidence = (
                0.35 * density_model_confidence
                + 0.30 * flow_coherence
                + 0.20 * visible_area_ratio
                + 0.15 * effective_flow_quality
            )

            # Smooth probabilistic presence avoids flipping around a hard count
            # boundary. Confidence lowers certainty without erasing count data.
            physical_density_available = ground_area > 0
            presence_value = grid_density if physical_density_available else grid_count_corrected
            presence_threshold = self.crowd_density_thresholds[0] if physical_density_available else self.crowd_presence_count
            softness = max(0.15 if physical_density_available else 0.35, presence_threshold * 0.35)
            logit = (presence_value - presence_threshold) / softness
            crowd_probability = float(1.0 / (1.0 + math.exp(-np.clip(logit, -30.0, 30.0))))
            crowd_probability *= float(np.clip(0.5 + 0.5 * confidence, 0.0, 1.0))
            crowd_class = self._crowd_class(presence_value, physical=physical_density_available)

            physical_calibrated = False
            speed_mps = 0.0
            if ground_area > 0:
                physical_calibrated = True
            local_meters_per_pixel = self.meters_per_pixel
            full_ground_area = self._polygon_area(g.ground_polygon)
            if local_meters_per_pixel is None and full_ground_area > 0 and g.area > 0:
                # Local area-Jacobian approximation for a calibrated projected
                # ground cell. This adapts scale across perspective-distorted grids.
                local_meters_per_pixel = math.sqrt(full_ground_area / g.area)
            if local_meters_per_pixel is not None:
                speed_mps = magnitude_corrected * local_meters_per_pixel * self.fps
                physical_calibrated = True

            # Low-confidence frames retain telemetry but are not allowed to
            # autonomously raise operational alerts.
            alert_eligible = bool(confidence >= self.min_confidence and not flow_result.scene_cut_detected)

            grid_metrics_dict[g.grid_id] = GridMetrics(
                grid_id=g.grid_id,
                count=grid_count_corrected,
                density=grid_density,
                flow_x=u_bar_corrected,
                flow_y=v_bar_corrected,
                speed=magnitude_corrected,
                direction_deg=theta_deg,
                direction_label=dir_label,
                density_score=density_score,
                slow_score=slow_score,
                stagnation_score=stagnation,
                flow_conflict_score=flow_conflict,
                reverse_score=reverse_score,
                congestion_score=congestion_score,
                risk_level=risk_level,
                confidence=confidence,
                # New parameters:
                turbulence_score=local_cv,
                speed_surge_warning=False,
                stasis_warning=False,
                turbulence_warning=False,
                crowd_present=bool(crowd_probability >= 0.5),
                crowd_class=crowd_class,
                crowd_probability=crowd_probability,
                flow_quality=effective_flow_quality,
                valid_flow_ratio=valid_flow_ratios[g.grid_id],
                alert_eligible=alert_eligible,
                physical_calibrated=physical_calibrated,
                speed_mps=speed_mps,
                density_people_m2=density_people_m2,
            )

        # Spatial divergence is evaluated after every grid has a robust vector.
        # It is image-space per grid step unless physical calibration is present.
        by_position = {(g.row, g.col): g for g in grids}
        for g in grids:
            left = by_position.get((g.row, g.col - 1))
            right = by_position.get((g.row, g.col + 1))
            up = by_position.get((g.row - 1, g.col))
            down = by_position.get((g.row + 1, g.col))
            du_dx = 0.0
            dv_dy = 0.0
            if left and right:
                du_dx = (observed_flows[right.grid_id][0] - observed_flows[left.grid_id][0]) / 2.0
            if up and down:
                dv_dy = (observed_flows[down.grid_id][1] - observed_flows[up.grid_id][1]) / 2.0
            grid_metrics_dict[g.grid_id].divergence = float(du_dx + dv_dy)

        return grid_metrics_dict
