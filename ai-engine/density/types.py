from dataclasses import dataclass
import numpy as np


@dataclass
class DensityInferenceResult:
    density_map: np.ndarray
    estimated_count: float
    crowd_mask: np.ndarray
    inference_time_ms: float
    device: str
    input_width: int
    input_height: int
    model_name: str
    checkpoint_path: str
