import unittest
import numpy as np
import torch
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / "ai-engine") not in sys.path:
    sys.path.insert(0, str(project_root / "ai-engine"))

from density.preprocessing import preprocess_image


class TestPreprocessing(unittest.TestCase):
    def test_invalid_input(self):
        # Must be numpy array
        with self.assertRaises(ValueError):
            preprocess_image("not an array")

        # Must be 3D with 3 channels
        with self.assertRaises(ValueError):
            preprocess_image(np.zeros((100, 100), dtype=np.uint8))
        with self.assertRaises(ValueError):
            preprocess_image(np.zeros((100, 100, 1), dtype=np.uint8))

    def test_preprocessing_shapes_and_types(self):
        # Test default frame: 640x480 (height x width x 3) BGR
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tensor, orig_w, orig_h, nwd, nht = preprocess_image(frame)

        # Output properties
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.dtype, torch.float32)

        # Batch dimension + channels + spatial dims aligned to 32
        self.assertEqual(list(tensor.shape), [1, 3, 480, 640])
        self.assertEqual(orig_w, 640)
        self.assertEqual(orig_h, 480)
        self.assertEqual(nwd, 640)
        self.assertEqual(nht, 480)

    def test_min_size_enforcement(self):
        # A tiny frame (e.g., 100x100) should be upscaled to min_size (320x320)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        tensor, orig_w, orig_h, nwd, nht = preprocess_image(frame)

        self.assertEqual(list(tensor.shape), [1, 3, 320, 320])
        self.assertEqual(nwd, 320)
        self.assertEqual(nht, 320)

    def test_normalization_values(self):
        # Pure white frame (all 255)
        frame = np.ones((320, 320, 3), dtype=np.uint8) * 255
        tensor, _, _, _, _ = preprocess_image(frame)

        # After conversion to ToTensor ([0, 1]) and normalization with mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]:
        # expected channel 0 max: (1.0 - 0.485) / 0.229 = 2.2489
        val_ch0 = tensor[0, 0, 0, 0].item()
        self.assertAlmostEqual(val_ch0, 2.2489, places=3)


if __name__ == "__main__":
    unittest.main()
