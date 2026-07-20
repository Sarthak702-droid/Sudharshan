import cv2
import numpy as np


def calculate_count(density_map: np.ndarray) -> float:
    """Calculates the estimated crowd count by summing the density map.
    Negative values are clipped to 0.0.
    """
    clipped = np.clip(density_map, a_min=0.0, a_max=None)
    return float(np.sum(clipped))


def create_crowd_mask(density_map: np.ndarray, threshold: float) -> np.ndarray:
    """Generates a binary crowd presence mask based on normalized density values.

    Args:
        density_map: 2D float NumPy array.
        threshold: Float threshold in range [0, 1].

    Returns:
        mask: 2D binary NumPy array (uint8) where 1 indicates crowd presence.
    """
    den_min = np.min(density_map)
    den_max = np.max(density_map)
    denom = den_max - den_min
    if denom > 1e-10:
        density_norm = (density_map - den_min) / denom
    else:
        density_norm = np.zeros_like(density_map)

    return (density_norm >= threshold).astype(np.uint8)


def normalize_density_for_debug(density_map: np.ndarray) -> np.ndarray:
    """Normalizes density map values to [0, 255] range for debug visualization (uint8)."""
    den_min = np.min(density_map)
    den_max = np.max(density_map)
    normalized = 255.0 * (density_map - den_min + 1e-10) / (1e-10 + den_max - den_min)
    return np.clip(normalized, 0, 255).astype(np.uint8)


def resize_density_preserve_count(
    density_map: np.ndarray, target_height: int, target_width: int
) -> np.ndarray:
    """Resizes a density map while preserving the total crowd count (sum of density values).

    Args:
        density_map: 2D NumPy array.
        target_height: Expected output height.
        target_width: Expected output width.

    Returns:
        resized_density: Resized density map with preserved count.
    """
    orig_sum = np.sum(density_map)
    orig_h, orig_w = density_map.shape[:2]

    if orig_h == target_height and orig_w == target_width:
        return density_map.copy()

    # Resize using bilinear interpolation
    resized = cv2.resize(density_map, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    # Clip negative values that might arise from interpolation
    resized = np.clip(resized, a_min=0.0, a_max=None)

    resized_sum = np.sum(resized)
    if resized_sum > 1e-10:
        resized = resized * (orig_sum / resized_sum)
    elif orig_sum > 1e-10:
        # Fallback to scaling factor from dimensions directly if resized_sum is 0
        scale = (orig_h * orig_w) / (target_height * target_width)
        resized = resized * scale

    return resized
