import cv2
import numpy as np
import time
import math
from typing import List, Tuple, Optional, Dict

from .types import FlowResult
from grid.types import GridBox


def get_direction_label(deg: float) -> str:
    """Converts a flow direction angle in degrees to a cardinal label."""
    deg = deg % 360
    if deg >= 337.5 or deg < 22.5:
        return "EAST"
    elif 22.5 <= deg < 67.5:
        return "SOUTH-EAST"
    elif 67.5 <= deg < 112.5:
        return "SOUTH"
    elif 112.5 <= deg < 157.5:
        return "SOUTH-WEST"
    elif 157.5 <= deg < 202.5:
        return "WEST"
    elif 202.5 <= deg < 247.5:
        return "NORTH-WEST"
    elif 247.5 <= deg < 292.5:
        return "NORTH"
    else:
        return "NORTH-EAST"


class OpticalFlowEngine:
    def __init__(
        self,
        method: str = "dis",
        preset: str = "fast",
        min_motion_px: float = 0.05,
        max_motion_px: float = 40.0,
        scene_cut_threshold: float = 45.0,
        min_blur_score: float = 12.0,
    ) -> None:
        """Initializes the OpticalFlowEngine.

        Args:
            method: Optical flow algorithm to use ('dis' or 'farneback').
            preset: Preset for DIS flow ('fast', 'medium', 'ultrafast').
        """
        self.method = method.lower()
        if min_motion_px < 0 or max_motion_px <= min_motion_px:
            raise ValueError("Flow motion thresholds are invalid")
        self.min_motion_px = float(min_motion_px)
        self.max_motion_px = float(max_motion_px)
        self.scene_cut_threshold = float(scene_cut_threshold)
        self.min_blur_score = float(min_blur_score)

        if self.method == "dis":
            # Initialize OpenCV DIS flow estimator
            if preset == "ultrafast":
                dis_preset = cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST
            elif preset == "medium":
                dis_preset = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
            else:
                dis_preset = cv2.DISOPTICAL_FLOW_PRESET_FAST

            self._dis_flow = cv2.DISOpticalFlow_create(dis_preset)
        elif self.method == "farneback":
            # Store configuration for Farneback
            self.farneback_params = dict(
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )
        else:
            raise ValueError(f"Unsupported optical flow method: {method}")

    def compensate_camera_motion(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        crowd_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float, float, int, float, float, bool]:
        """Estimates global camera motion using feature tracking and RANSAC registration.

        Returns:
            Tuple containing (flow_camera_x, flow_camera_y, inlier_ratio,
                             reprojection_error, tracked_features, scene_cut_score,
                             blur_score, camera_motion_unreliable)
        """
        height, width = prev_gray.shape[:2]

        # 1. Blur score: variance of Laplacian
        blur_score = float(cv2.Laplacian(curr_gray, cv2.CV_64F).var())

        # 2. Scene cut score: mean absolute difference (MAD)
        scene_cut_score = float(np.mean(np.abs(curr_gray.astype(np.float32) - prev_gray.astype(np.float32))))

        # Determine background mask (exclude crowd)
        background_mask = np.ones((height, width), dtype=np.uint8)
        if crowd_mask is not None:
            if crowd_mask.shape[:2] != (height, width):
                crowd_mask_resized = cv2.resize(crowd_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            else:
                crowd_mask_resized = crowd_mask
            background_mask = cv2.bitwise_not(crowd_mask_resized)

        # 3. Detect background corners using Shi-Tomasi
        feature_params = dict(
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7
        )

        features = cv2.goodFeaturesToTrack(prev_gray, mask=background_mask, **feature_params)

        # Fallbacks if tracking cannot be performed
        flow_camera_x = np.zeros((height, width), dtype=np.float32)
        flow_camera_y = np.zeros((height, width), dtype=np.float32)
        inlier_ratio = 1.0
        reprojection_error = 0.0
        tracked_features = 0
        camera_motion_unreliable = False

        if features is None or len(features) < 40:
            return (flow_camera_x, flow_camera_y, inlier_ratio, reprojection_error,
                    len(features) if features is not None else 0, scene_cut_score, blur_score, True)

        # 4. Track corners using Lucas-Kanade
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, features, None, **lk_params)

        if status is None:
            return (flow_camera_x, flow_camera_y, inlier_ratio, reprojection_error,
                    0, scene_cut_score, blur_score, True)

        good_prev = features[status == 1]
        good_next = next_pts[status == 1]
        tracked_features = len(good_prev)

        if tracked_features < 40:
            return (flow_camera_x, flow_camera_y, inlier_ratio, reprojection_error,
                    tracked_features, scene_cut_score, blur_score, True)

        # 5. Estimate transform with RANSAC
        M, inliers = cv2.estimateAffinePartial2D(good_prev, good_next, method=cv2.RANSAC, ransacReprojThreshold=2.5)

        if M is None or inliers is None:
            return (flow_camera_x, flow_camera_y, inlier_ratio, reprojection_error,
                    tracked_features, scene_cut_score, blur_score, True)

        inlier_count = int(np.sum(inliers))
        inlier_ratio = float(inlier_count) / len(inliers) if len(inliers) > 0 else 0.0

        # Calculate median reprojection error
        transformed = cv2.transform(good_prev[:, np.newaxis, :], M)
        errors = np.linalg.norm(transformed[:, 0, :] - good_next, axis=1)
        reprojection_error = float(np.median(errors))

        if inlier_ratio < 0.55 or reprojection_error > 2.5:
            camera_motion_unreliable = True

        # 6. Generate dense global flow field
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        coords = np.stack([xx, yy], axis=-1).astype(np.float32).reshape((-1, 2))
        coords_transformed = cv2.transform(coords[np.newaxis, ...], M)[0]
        coords_transformed = coords_transformed.reshape((height, width, 2))

        flow_camera_x = coords_transformed[..., 0] - xx
        flow_camera_y = coords_transformed[..., 1] - yy

        return (flow_camera_x, flow_camera_y, inlier_ratio, reprojection_error,
                tracked_features, scene_cut_score, blur_score, camera_motion_unreliable)

    def calculate_flow(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
        crowd_mask: Optional[np.ndarray] = None,
    ) -> FlowResult:
        """Calculates dense optical flow between two consecutive frames with camera-motion compensation."""
        if prev_frame.shape != curr_frame.shape:
            raise ValueError("Input frames must have matching spatial dimensions.")

        t_start = time.perf_counter()

        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE to improve contrast under difficult lighting (low light, dust, shadow transitions)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        prev_gray = clahe.apply(prev_gray)
        curr_gray = clahe.apply(curr_gray)

        if self.method == "dis":
            # Run DIS optical flow
            flow = self._dis_flow.calc(prev_gray, curr_gray, None)
        else:
            # Run Farneback optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None, **self.farneback_params
            )

        flow_raw_x = flow[..., 0]
        flow_raw_y = flow[..., 1]

        # Estimate and compensate global camera motion
        (flow_camera_x, flow_camera_y, inlier_ratio, reprojection_error,
         tracked_features, scene_cut_score, blur_score, camera_motion_unreliable) = self.compensate_camera_motion(
             prev_gray, curr_gray, crowd_mask
         )

        flow_residual_x = flow_raw_x - flow_camera_x
        flow_residual_y = flow_raw_y - flow_camera_y

        # Reject vectors that are numerically invalid, implausibly large, or so
        # small that their angle is dominated by optical-flow noise. A scene cut
        # invalidates the complete field: it is not crowd motion.
        magnitude = np.hypot(flow_residual_x, flow_residual_y)
        valid_mask = (
            np.isfinite(flow_residual_x)
            & np.isfinite(flow_residual_y)
            & (magnitude >= self.min_motion_px)
            & (magnitude <= self.max_motion_px)
        )
        scene_cut_detected = scene_cut_score >= self.scene_cut_threshold
        if scene_cut_detected:
            valid_mask[:] = False

        flow_residual_x = np.where(valid_mask, flow_residual_x, 0.0).astype(np.float32)
        flow_residual_y = np.where(valid_mask, flow_residual_y, 0.0).astype(np.float32)
        valid_flow_ratio = float(np.mean(valid_mask))

        # Quality is deliberately conservative. Camera-registration failure,
        # blur, a scene cut, and sparse usable vectors independently reduce it.
        feature_quality = min(1.0, tracked_features / 80.0)
        registration_quality = float(np.clip(inlier_ratio, 0.0, 1.0))
        if reprojection_error > 0:
            registration_quality *= float(np.exp(-reprojection_error / 2.5))
        texture_quality = float(np.clip(blur_score / max(self.min_blur_score, 1e-6), 0.0, 1.0))
        vector_quality = float(np.clip(valid_flow_ratio / 0.15, 0.0, 1.0))
        flow_quality = (
            0.35 * registration_quality
            + 0.20 * feature_quality
            + 0.20 * texture_quality
            + 0.25 * vector_quality
        )
        if camera_motion_unreliable:
            flow_quality *= 0.55
        if scene_cut_detected:
            flow_quality = 0.0

        t_end = time.perf_counter()
        inference_time_ms = (t_end - t_start) * 1000.0

        return FlowResult(
            flow_x=flow_residual_x,
            flow_y=flow_residual_y,
            inference_time_ms=inference_time_ms,
            inlier_ratio=inlier_ratio,
            reprojection_error=reprojection_error,
            tracked_features=tracked_features,
            scene_cut_score=scene_cut_score,
            blur_score=blur_score,
            camera_motion_unreliable=camera_motion_unreliable,
            valid_mask=valid_mask,
            valid_flow_ratio=valid_flow_ratio,
            flow_quality=float(np.clip(flow_quality, 0.0, 1.0)),
            scene_cut_detected=scene_cut_detected,
        )

    def aggregate_grid_flow(
        self,
        flow_x: np.ndarray,
        flow_y: np.ndarray,
        grids: List[GridBox],
        density_map: Optional[np.ndarray] = None,
    ) -> Dict[str, dict]:
        """Aggregates dense flow fields into grid-wise movement vectors.

        Args:
            flow_x: 2D float32 NumPy array containing horizontal flow.
            flow_y: 2D float32 NumPy array containing vertical flow.
            grids: List of GridBox coordinates.
            density_map: Optional 2D float32 density map for density-weighted mean calculation.

        Returns:
            Dict mapping grid_id to flow metrics dict.
        """
        grid_flows = {}
        h_flow, w_flow = flow_x.shape[:2]

        # Ensure density map matches the flow resolution if provided
        resized_density = None
        if density_map is not None:
            h_den, w_den = density_map.shape[:2]
            if h_den != h_flow or w_den != w_flow:
                resized_density = cv2.resize(density_map, (w_flow, h_flow), interpolation=cv2.INTER_LINEAR)
            else:
                resized_density = density_map

        for g in grids:
            # Slice the grid bounds
            x1, y1, x2, y2 = g.x1, g.y1, g.x2, g.y2

            # Slice flow fields
            local_u = flow_x[y1:y2, x1:x2]
            local_v = flow_y[y1:y2, x1:x2]

            u_bar = 0.0
            v_bar = 0.0

            if resized_density is not None:
                # Density-weighted mean flow calculation
                local_d = resized_density[y1:y2, x1:x2]
                d_sum = np.sum(local_d)

                if d_sum > 1e-6:
                    u_bar = float(np.sum(local_u * local_d) / d_sum)
                    v_bar = float(np.sum(local_v * local_d) / d_sum)
                else:
                    # Fallback to arithmetic mean if grid is empty of crowd density
                    u_bar = float(np.mean(local_u))
                    v_bar = float(np.mean(local_v))
            else:
                # Arithmetic mean
                u_bar = float(np.mean(local_u))
                v_bar = float(np.mean(local_v))

            # Calculate magnitude and direction angle
            magnitude = math.sqrt(u_bar**2 + v_bar**2)

            # Angle in radians mapping (y increases downwards in image coordinates)
            theta = math.atan2(v_bar, u_bar)
            theta_deg = (theta * 180.0 / math.pi + 360.0) % 360.0

            direction_label = get_direction_label(theta_deg)

            grid_flows[g.grid_id] = {
                "flow_x": u_bar,
                "flow_y": v_bar,
                "magnitude": magnitude,
                "direction_deg": theta_deg,
                "direction_label": direction_label,
            }

        return grid_flows
