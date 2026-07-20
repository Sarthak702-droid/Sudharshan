from dataclasses import dataclass
import numpy as np


@dataclass
class IngestedFrame:
    frame: np.ndarray  # BGR format NumPy array
    frame_index: int   # 0-indexed frame count
    timestamp: float   # Epoch timestamp when the frame was read
    source: str        # String representation of the source (e.g. file path, RTSP URL, or camera index)
    width: int         # Frame width
    height: int        # Frame height
