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

from flow.types import FlowResult
from flow.engine import OpticalFlowEngine, get_direction_label
from grid.types import GridBox


class TestOpticalFlowEngine(unittest.TestCase):
    def test_initialization(self):
        # Default initialization (DIS)
        engine = OpticalFlowEngine(method="dis")
        self.assertEqual(engine.method, "dis")

        # Farneback
        engine_fb = OpticalFlowEngine(method="farneback")
        self.assertEqual(engine_fb.method, "farneback")

        # Unsupported method raises error
        with self.assertRaises(ValueError):
            OpticalFlowEngine(method="invalid_flow")

    def test_direction_label_mapping(self):
        # East (0 deg)
        self.assertEqual(get_direction_label(0.0), "EAST")
        self.assertEqual(get_direction_label(350.0), "EAST")
        self.assertEqual(get_direction_label(10.0), "EAST")

        # South (90 deg)
        self.assertEqual(get_direction_label(90.0), "SOUTH")

        # West (180 deg)
        self.assertEqual(get_direction_label(180.0), "WEST")

        # North (270 deg)
        self.assertEqual(get_direction_label(270.0), "NORTH")

        # Diagonals
        self.assertEqual(get_direction_label(45.0), "SOUTH-EAST")
        self.assertEqual(get_direction_label(135.0), "SOUTH-WEST")
        self.assertEqual(get_direction_label(225.0), "NORTH-WEST")
        self.assertEqual(get_direction_label(315.0), "NORTH-EAST")

    def test_grid_flow_aggregation_arithmetic(self):
        engine = OpticalFlowEngine(method="dis")

        # Create a mock 100x100 flow field with constant displacement
        flow_x = np.ones((100, 100), dtype=np.float32) * 5.0   # 5 pixels right
        flow_y = np.ones((100, 100), dtype=np.float32) * -5.0  # 5 pixels up (NORTH in image coordinates)

        # Grid box covering the whole 100x100 region
        grid = GridBox(
            grid_id="G_00_00",
            row=0,
            col=0,
            x1=0,
            y1=0,
            x2=100,
            y2=100,
            area=10000.0,
            effective_area=10000.0
        )

        metrics = engine.aggregate_grid_flow(flow_x, flow_y, [grid])

        self.assertIn("G_00_00", metrics)
        m = metrics["G_00_00"]
        self.assertAlmostEqual(m["flow_x"], 5.0)
        self.assertAlmostEqual(m["flow_y"], -5.0)

        # Magnitude = sqrt(5^2 + (-5)^2) = sqrt(50) = 7.071
        self.assertAlmostEqual(m["magnitude"], 7.071, places=3)

        # Angle = atan2(-5, 5) = -45 deg -> normalized to 315 deg (NORTH-EAST)
        self.assertAlmostEqual(m["direction_deg"], 315.0)
        self.assertEqual(m["direction_label"], "NORTH-EAST")

    def test_grid_flow_aggregation_density_weighted(self):
        engine = OpticalFlowEngine(method="dis")

        # Create a mock 100x100 flow field
        # Left half has displacement 10.0, Right half has displacement 0.0
        flow_x = np.zeros((100, 100), dtype=np.float32)
        flow_x[:, :50] = 10.0
        flow_y = np.zeros((100, 100), dtype=np.float32)

        # Create density map centered on the left half (x=25, y=50) -> high density in left region
        density_map = np.zeros((100, 100), dtype=np.float32)
        density_map[40:60, 20:30] = 1.0  # high density (sum = 200.0)

        grid = GridBox(
            grid_id="G_00_00",
            row=0,
            col=0,
            x1=0,
            y1=0,
            x2=100,
            y2=100,
            area=10000.0,
            effective_area=10000.0
        )

        # 1. Density-Weighted Aggregation
        metrics_weighted = engine.aggregate_grid_flow(flow_x, flow_y, [grid], density_map=density_map)

        # 2. Arithmetic Aggregation (without density map)
        metrics_arithmetic = engine.aggregate_grid_flow(flow_x, flow_y, [grid])

        # Arithmetic average of flow_x should be: (50*100*10.0 + 50*100*0.0)/10000 = 5.0
        self.assertAlmostEqual(metrics_arithmetic["G_00_00"]["flow_x"], 5.0)

        # Density weighted average should be focused on the left half, so it should be close to 10.0
        self.assertGreater(metrics_weighted["G_00_00"]["flow_x"], 9.9)

    def test_camera_translation_compensation(self):
        import cv2
        # Generate a structured background frame (texture) so features can be tracked
        h, w = 180, 180
        np.random.seed(42)
        frame1 = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
        # Apply blur to make it smooth and realistic
        frame1 = cv2.GaussianBlur(frame1, (3, 3), 0)

        # Shift frame1 by dx=3, dy=2 to make frame2
        M_shift = np.float32([[1, 0, 3], [0, 1, 2]])
        frame2 = cv2.warpAffine(frame1, M_shift, (w, h))

        engine = OpticalFlowEngine(method="dis")
        res = engine.calculate_flow(frame1, frame2)

        # Since the entire frame shifted, flow_x and flow_y should be corrected to ~0.0
        # (excluding boundary edge pixels that were shifted in)
        residual_x = res.flow_x[10:-10, 10:-10]
        residual_y = res.flow_y[10:-10, 10:-10]

        # Check that residual flow is near zero (mean displacement < 0.5 px)
        self.assertLess(np.mean(np.abs(residual_x)), 0.5)
        self.assertLess(np.mean(np.abs(residual_y)), 0.5)

        # Verify reliability metrics
        self.assertGreaterEqual(res.inlier_ratio, 0.55)
        self.assertGreaterEqual(res.tracked_features, 40)
        self.assertFalse(res.camera_motion_unreliable)

    def test_camera_motion_gates(self):
        # Generate featureless flat frames
        h, w = 180, 180
        frame1 = np.zeros((h, w, 3), dtype=np.uint8)
        frame2 = np.zeros((h, w, 3), dtype=np.uint8)

        engine = OpticalFlowEngine(method="dis")
        res = engine.calculate_flow(frame1, frame2)

        # Should mark compensation as unreliable due to lack of trackable background features
        self.assertTrue(res.camera_motion_unreliable)
        self.assertEqual(res.tracked_features, 0)

    def test_scene_cut_invalidates_flow(self):
        frame1 = np.zeros((96, 96, 3), dtype=np.uint8)
        frame2 = np.full((96, 96, 3), 255, dtype=np.uint8)
        engine = OpticalFlowEngine(method="dis", scene_cut_threshold=20.0)

        res = engine.calculate_flow(frame1, frame2)

        self.assertTrue(res.scene_cut_detected)
        self.assertEqual(res.flow_quality, 0.0)
        self.assertEqual(res.valid_flow_ratio, 0.0)
        self.assertFalse(np.any(res.valid_mask))
        self.assertTrue(np.all(res.flow_x == 0.0))


if __name__ == "__main__":
    unittest.main()
