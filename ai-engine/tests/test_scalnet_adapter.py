import unittest
import os
import numpy as np
import torch
import sys
from pathlib import Path
import warnings

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / "ai-engine") not in sys.path:
    sys.path.insert(0, str(project_root / "ai-engine"))

from density.scalnet_adapter import (
    SCALNetAdapter,
    SCALNetCheckpointNotFound,
    SCALNetCheckpointIncompatible,
    SCALNetNotLoaded,
    InvalidFrameError,
)
from density.types import DensityInferenceResult


class TestSCALNetAdapter(unittest.TestCase):
    def setUp(self):
        self.scalnet_root = project_root / "SCALNet"
        self.checkpoint_path = Path(
            os.environ.get(
                "SCALNET_CHECKPOINT",
                self.scalnet_root / "checkpoints" / "model.pth",
            )
        )
        self.dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def require_checkpoint(self):
        if not self.checkpoint_path.is_file():
            self.skipTest(
                "SCALNet checkpoint not supplied; set SCALNET_CHECKPOINT to run integration tests"
            )

    def test_infer_before_load_error(self):
        adapter = SCALNetAdapter(
            scalnet_root=self.scalnet_root,
            checkpoint_path=self.checkpoint_path,
            device="cpu",
        )
        self.assertFalse(adapter.is_loaded())
        with self.assertRaises(SCALNetNotLoaded):
            adapter.infer(self.dummy_frame)

    def test_checkpoint_not_found_error(self):
        bad_path = self.scalnet_root / "checkpoints" / "does_not_exist.pth"
        adapter = SCALNetAdapter(
            scalnet_root=self.scalnet_root,
            checkpoint_path=bad_path,
            device="cpu",
        )
        with self.assertRaises(SCALNetCheckpointNotFound):
            adapter.load()

    def test_invalid_frame_inference_error(self):
        self.require_checkpoint()
        adapter = SCALNetAdapter(
            scalnet_root=self.scalnet_root,
            checkpoint_path=self.checkpoint_path,
            device="cpu",
        )
        adapter.load()
        self.assertTrue(adapter.is_loaded())

        # Test invalid inputs
        with self.assertRaises(InvalidFrameError):
            adapter.infer("not a numpy array")
        with self.assertRaises(InvalidFrameError):
            adapter.infer(np.zeros((100, 100), dtype=np.uint8))  # 2D instead of 3D BGR

        adapter.unload()

    def test_load_unload_lifecycle(self):
        self.require_checkpoint()
        adapter = SCALNetAdapter(
            scalnet_root=self.scalnet_root,
            checkpoint_path=self.checkpoint_path,
            device="cpu",
        )
        self.assertFalse(adapter.is_loaded())

        # Load
        adapter.load()
        self.assertTrue(adapter.is_loaded())
        self.assertIsNotNone(adapter._model)

        # Unload
        adapter.unload()
        self.assertFalse(adapter.is_loaded())
        self.assertIsNone(adapter._model)

    def test_cpu_smoke_test(self):
        self.require_checkpoint()
        adapter = SCALNetAdapter(
            scalnet_root=self.scalnet_root,
            checkpoint_path=self.checkpoint_path,
            device="cpu",
        )
        adapter.load()

        # Run inference
        result = adapter.infer(self.dummy_frame)

        # Verify result contract
        self.assertIsInstance(result, DensityInferenceResult)
        self.assertEqual(result.density_map.shape, (480, 640))
        self.assertEqual(result.crowd_mask.shape, (480, 640))
        self.assertGreaterEqual(result.estimated_count, 0.0)
        self.assertGreater(result.inference_time_ms, 0.0)
        self.assertEqual(result.device, "cpu")
        self.assertEqual(result.input_width, 640)
        self.assertEqual(result.input_height, 480)
        self.assertEqual(result.model_name, "SCALNet")

        adapter.unload()

    def test_auto_device_selection_and_fallback(self):
        self.require_checkpoint()
        # We test that creating with device="auto" yields a clean load/inference
        # Even if CUDA is available but failed verification (due to GPU capability sm_61 mismatch),
        # it fallback to CPU and finishes correctly.
        adapter = SCALNetAdapter(
            scalnet_root=self.scalnet_root,
            checkpoint_path=self.checkpoint_path,
            device="auto",
        )

        # This shouldn't raise CUDA kernel mismatch errors; it should handle it and load
        adapter.load()
        self.assertTrue(adapter.is_loaded())

        # Run inference
        result = adapter.infer(self.dummy_frame)
        self.assertIsInstance(result, DensityInferenceResult)

        # Verify device matches the final resolved device (should be cpu if fallback occurred)
        self.assertEqual(result.device, adapter.device)

        adapter.unload()


if __name__ == "__main__":
    unittest.main()
