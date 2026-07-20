from dataclasses import dataclass
import numpy as np


@dataclass
class FlowResult:
    flow_x: np.ndarray  # 2D float32 array of horizontal displacements (u)
    flow_y: np.ndarray  # 2D float32 array of vertical displacements (v)
    inference_time_ms: float
    # Camera motion telemetry
    inlier_ratio: float = 1.0
    reprojection_error: float = 0.0
    tracked_features: int = 0
    scene_cut_score: float = 0.0
    blur_score: float = 999.0
    camera_motion_unreliable: bool = False
    # Per-pixel and frame-level reliability. Invalid vectors are zeroed before
    # fusion, and the mask prevents those zeros from being treated as evidence.
    valid_mask: np.ndarray | None = None
    valid_flow_ratio: float = 1.0
    flow_quality: float = 1.0
    scene_cut_detected: bool = False
