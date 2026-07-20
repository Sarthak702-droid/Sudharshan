from dataclasses import dataclass


@dataclass
class GridMetrics:
    grid_id: str
    count: float
    density: float
    flow_x: float
    flow_y: float
    speed: float
    direction_deg: float
    direction_label: str
    density_score: float
    slow_score: float
    stagnation_score: float
    flow_conflict_score: float
    reverse_score: float
    congestion_score: float
    risk_level: str
    confidence: float
    # New Robust AI features
    turbulence_score: float = 0.0
    speed_surge_warning: bool = False
    stasis_warning: bool = False
    turbulence_warning: bool = False
    crowd_present: bool = False
    # Multimodal direction features
    is_bimodal: bool = False
    primary_direction_deg: float = 0.0
    secondary_direction_deg: float = 0.0

    # New Prediction features
    predicted_count: float = 0.0
    predicted_congestion_score: float = 0.0
    predicted_risk_level: str = "GREEN"
    trend_slope: float = 0.0

    # Confidence-aware classification and calibrated physical telemetry.
    crowd_class: str = "EMPTY"
    crowd_probability: float = 0.0
    flow_quality: float = 1.0
    valid_flow_ratio: float = 1.0
    alert_eligible: bool = True
    physical_calibrated: bool = False
    speed_mps: float = 0.0
    density_people_m2: float = 0.0
    divergence: float = 0.0
    acceleration: float = 0.0
