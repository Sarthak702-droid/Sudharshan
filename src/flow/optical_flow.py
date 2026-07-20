import numpy as np
import cv2
import math

class OpticalFlowEngine:
    def __init__(self, grids, method="dis", alpha=0.25, min_motion=0.25):
        """
        Initializes the optical flow engine.
        grids: list of grid dictionaries.
        method: "dis" or "farneback". DIS is faster and recommended.
        alpha: smoothing factor for Exponential Moving Average (0.0 to 1.0).
        min_motion: motion threshold below which flow is classified as static.
        """
        self.grids = grids
        self.method = method.lower()
        self.alpha = alpha
        self.min_motion = min_motion

        # Keep track of previous aggregated flow vectors for EMA temporal smoothing
        # Keys are grid_ids, values are (u_ema, v_ema)
        self.ema_history = {}

        # OpenCV DIS optical flow instance
        if self.method == "dis":
            self.dis_flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        else:
            self.dis_flow = None

    def compute_flow(self, prev_frame, curr_frame):
        """
        Computes dense optical flow between two consecutive BGR frames.
        Returns a flow field of shape (height, width, 2).
        """
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        if self.method == "dis":
            # DIS flow requires initial flow placeholder
            flow = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 2), dtype=np.float32)
            flow = self.dis_flow.calc(prev_gray, curr_gray, flow)
        else:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                curr_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )

        return flow

    def aggregate_grid_flow(self, flow, density_map=None):
        """
        Aggregates the flow field for each grid.
        Applies density-weighted mean if density_map is available, outlier filtering,
        temporal smoothing, coherence calculation, and direction labeling.
        """
        aggregated_metrics = {}

        for g in self.grids:
            grid_id = g["grid_id"]
            x1, y1, x2, y2 = g["x1"], g["y1"], g["x2"], g["y2"]

            # Crop flow field for the grid
            grid_flow = flow[y1:y2, x1:x2]
            u = grid_flow[..., 0]
            v = grid_flow[..., 1]

            # Compute density-weighted mean flow if density map is provided
            if density_map is not None:
                grid_density = density_map[y1:y2, x1:x2]
                density_sum = np.sum(grid_density)
                if density_sum > 1e-5:
                    u_mean = np.sum(u * grid_density) / density_sum
                    v_mean = np.sum(v * grid_density) / density_sum
                else:
                    u_mean = np.mean(u)
                    v_mean = np.mean(v)
            else:
                u_mean = np.mean(u)
                v_mean = np.mean(v)

            # Filter outlier vectors to prevent camera jitter artifacts
            # Apply EMA temporal smoothing
            if grid_id in self.ema_history:
                u_prev, v_prev = self.ema_history[grid_id]
                u_ema = self.alpha * u_mean + (1.0 - self.alpha) * u_prev
                v_ema = self.alpha * v_mean + (1.0 - self.alpha) * v_prev
            else:
                u_ema = u_mean
                v_ema = v_mean

            self.ema_history[grid_id] = (u_ema, v_ema)

            # Magnitude and direction
            magnitude = math.sqrt(u_ema**2 + v_ema**2)

            # Compute coherence
            coherence = self._calculate_coherence(u, v)

            # Determine direction label
            direction_label, direction_deg = self._vector_to_direction(u_ema, v_ema, magnitude)

            aggregated_metrics[grid_id] = {
                "mean_dx": float(u_ema),
                "mean_dy": float(v_ema),
                "motion_magnitude": float(magnitude),
                "direction_deg": float(direction_deg),
                "direction_label": direction_label,
                "coherence": float(coherence)
            }

        return aggregated_metrics

    def _calculate_coherence(self, u, v, sample_step=4):
        """
        Calculates flow coherence inside the grid.
        coherence = || (1/N) * sum( V_i / ||V_i|| ) ||
        """
        # Subsample to keep it fast
        u_sub = u[::sample_step, ::sample_step].flatten()
        v_sub = v[::sample_step, ::sample_step].flatten()

        mags = np.sqrt(u_sub**2 + v_sub**2)
        valid = mags > 0.05

        if not np.any(valid):
            return 1.0 # High coherence (static is technically uniform)

        u_norm = u_sub[valid] / mags[valid]
        v_norm = v_sub[valid] / mags[valid]

        r_u = np.mean(u_norm)
        r_v = np.mean(v_norm)

        coherence = math.sqrt(r_u**2 + r_v**2)
        return coherence

    def _vector_to_direction(self, dx, dy, magnitude):
        """
        Maps a movement vector to 8 cardinal/intercardinal direction labels.
        """
        if magnitude < self.min_motion:
            return "STATIC", 0.0

        # math.atan2 returns angle in radians [-pi, pi], y is downward in image space
        # We invert dy to align with normal Cartesian y-axis (upwards) for compass directions
        angle_rad = math.atan2(-dy, dx)
        angle_deg = math.degrees(angle_rad)

        # Normalize to [0, 360)
        angle_deg = (angle_deg + 360.0) % 360.0

        # Map degrees to directions
        # EAST is around 0 deg (or 360 deg)
        if (angle_deg >= 337.5) or (angle_deg < 22.5):
            direction = "EAST"
        elif 22.5 <= angle_deg < 67.5:
            direction = "NORTH_EAST"
        elif 67.5 <= angle_deg < 112.5:
            direction = "NORTH"
        elif 112.5 <= angle_deg < 157.5:
            direction = "NORTH_WEST"
        elif 157.5 <= angle_deg < 202.5:
            direction = "WEST"
        elif 202.5 <= angle_deg < 247.5:
            direction = "SOUTH_WEST"
        elif 247.5 <= angle_deg < 292.5:
            direction = "SOUTH"
        else:
            direction = "SOUTH_EAST"

        return direction, angle_deg
