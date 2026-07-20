import unittest
import numpy as np
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / "ai-engine") not in sys.path:
    sys.path.insert(0, str(project_root / "ai-engine"))

from density.postprocessing import (
    calculate_count,
    create_crowd_mask,
    normalize_density_for_debug,
    resize_density_preserve_count,
)


class TestPostprocessing(unittest.TestCase):
    def test_calculate_count(self):
        # A simple grid
        dm = np.array([[0.1, 0.2], [0.3, -0.5]])

        # Count should clip -0.5 to 0.0, sum should be 0.1 + 0.2 + 0.3 = 0.6
        count = calculate_count(dm)
        self.assertAlmostEqual(count, 0.6)

    def test_create_crowd_mask(self):
        # Create a density map with values 0 to 10
        dm = np.array([[0.0, 5.0], [2.0, 10.0]])

        # Normalized values: [[0.0, 0.5], [0.2, 1.0]]
        # Threshold = 0.5: mask should be [[0, 1], [0, 1]]
        mask = create_crowd_mask(dm, threshold=0.5)
        expected = np.array([[0, 1], [0, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(mask, expected)

        # Threshold = 0.1: mask should be [[0, 1], [1, 1]]
        mask_low = create_crowd_mask(dm, threshold=0.1)
        expected_low = np.array([[0, 1], [1, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(mask_low, expected_low)

    def test_normalize_density_for_debug(self):
        dm = np.array([[1.0, 2.0], [3.0, 4.0]])
        norm = normalize_density_for_debug(dm)

        # The maximum value (4.0) should map to 255 (or near 255)
        # The minimum value (1.0) should map to 0 (or near 0)
        self.assertEqual(norm.dtype, np.uint8)
        self.assertGreaterEqual(norm[1, 1], 254)
        self.assertLessEqual(norm[0, 0], 1)

    def test_count_preserving_resize(self):
        # Create an 8x8 density map with total sum (count) = 16.0
        dm = np.ones((8, 8), dtype=np.float32) * 0.25
        self.assertAlmostEqual(np.sum(dm), 16.0)

        # Resize to 16x16
        resized_16 = resize_density_preserve_count(dm, 16, 16)
        self.assertEqual(resized_16.shape, (16, 16))
        self.assertAlmostEqual(float(np.sum(resized_16)), 16.0, places=4)

        # Resize to 4x4
        resized_4 = resize_density_preserve_count(dm, 4, 4)
        self.assertEqual(resized_4.shape, (4, 4))
        self.assertAlmostEqual(float(np.sum(resized_4)), 16.0, places=4)


if __name__ == "__main__":
    unittest.main()
