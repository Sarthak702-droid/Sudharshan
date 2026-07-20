import math

class RiskScorer:
    def __init__(self, density_red=4.0, normal_speed=3.0, expected_direction_label="EAST"):
        """
        Initializes the Risk Scorer.
        density_red: crowd density (people/m2 or people/100px) that triggers high-risk state.
        normal_speed: standard motion magnitude representing normal walking flow speed.
        expected_direction_label: dominant direction expected along the corridor (e.g., "EAST").
        """
        self.density_red = density_red
        self.normal_speed = normal_speed
        self.expected_direction_label = expected_direction_label.upper()

        # Mapping direction label to angle (approximate center angle of the ranges)
        self.direction_angles = {
            "EAST": 0.0,
            "NORTH_EAST": 45.0,
            "NORTH": 90.0,
            "NORTH_WEST": 135.0,
            "WEST": 180.0,
            "SOUTH_WEST": 225.0,
            "SOUTH": 270.0,
            "SOUTH_EAST": 315.0
        }

    def compute_risk(self, density_score_val, motion_magnitude, direction_label, direction_deg, coherence, expected_direction=None):
        """
        Computes risk score and level for a single grid cell.
        density_score_val: people density in the grid.
        motion_magnitude: magnitude of flow motion.
        direction_label: direction category (e.g. "EAST").
        direction_deg: exact movement angle in degrees [0, 360).
        coherence: coherence score (0.0 to 1.0).
        expected_direction: override expected direction label for this specific grid if available.
        """
        # 1. Density Score (0.0 to 1.0)
        density_norm = min(density_score_val / self.density_red, 1.0)

        # 2. Speed Drop Score (0.0 to 1.0)
        # If speed is >= normal_speed, drop is 0. If speed is 0, drop is 1.0.
        speed_drop = 1.0 - min(motion_magnitude / self.normal_speed, 1.0)

        # 3. Flow Conflict Score (0.0 to 1.0)
        # We combine local incoherence and reverse flow comparison
        exp_dir = expected_direction.upper() if expected_direction else self.expected_direction_label

        reverse_flow_flag = 0.0
        if direction_label != "STATIC" and exp_dir in self.direction_angles:
            exp_angle = self.direction_angles[exp_dir]
            # Calculate absolute difference between angles
            diff = abs(direction_deg - exp_angle)
            diff = min(diff, 360.0 - diff)

            # If moving opposite (diff > 135 degrees), mark as reverse flow
            if diff > 135.0:
                reverse_flow_flag = 1.0

        # Flow conflict: high when coherence is low, or when moving backwards
        flow_conflict = max(1.0 - coherence, reverse_flow_flag)

        # 4. Final Risk Score (0.0 to 1.0)
        risk_score = (
            density_norm * 0.50 +
            speed_drop * 0.30 +
            flow_conflict * 0.20
        )

        # Scale to 0-100
        risk_percentage = int(round(risk_score * 100))

        # Risk level mapping
        if risk_percentage <= 40:
            risk_level = "GREEN"
        elif risk_percentage <= 60:
            risk_level = "YELLOW"
        elif risk_percentage <= 80:
            risk_level = "ORANGE"
        else:
            risk_level = "RED"

        # Determine speed level label
        if motion_magnitude < 0.25:
            speed_level = "STATIC"
        elif motion_magnitude < 1.0:
            speed_level = "SLOW"
        elif motion_magnitude < 3.0:
            speed_level = "MODERATE"
        else:
            speed_level = "FAST"

        return {
            "density_norm_score": float(density_norm),
            "speed_drop_score": float(speed_drop),
            "flow_conflict_score": float(flow_conflict),
            "risk_score": risk_percentage,
            "risk_level": risk_level,
            "speed_level": speed_level,
            "is_reverse_flow": bool(reverse_flow_flag > 0.5)
        }
