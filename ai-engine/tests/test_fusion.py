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

from fusion.types import GridMetrics
from fusion.aggregator import FusionAggregator
from density.types import DensityInferenceResult
from flow.types import FlowResult
from grid.types import GridBox


class TestFusionAggregator(unittest.TestCase):
    def setUp(self):
        # Create two adjacent grids (grid_size=100)
        self.grids = [
            GridBox(
                grid_id="G_00_00",
                row=0,
                col=0,
                x1=0,
                y1=0,
                x2=100,
                y2=100,
                area=10000.0,
                effective_area=10000.0,
            ),
            GridBox(
                grid_id="G_00_01",
                row=0,
                col=1,
                x1=100,
                y1=0,
                x2=200,
                y2=100,
                area=10000.0,
                effective_area=10000.0,
            ),
        ]

    def test_expected_vector_mapping(self):
        agg = FusionAggregator(expected_direction="EAST")
        vec = agg._get_expected_vector("G_00_00")
        self.assertEqual(vec, (1.0, 0.0))

        agg_west = FusionAggregator(expected_direction="WEST")
        vec_west = agg_west._get_expected_vector("G_00_00")
        self.assertEqual(vec_west, (-1.0, 0.0))

        # Test dict routing
        agg_dict = FusionAggregator(expected_direction={"G_00_00": "NORTH", "G_00_01": "SOUTH"})
        self.assertEqual(agg_dict._get_expected_vector("G_00_00"), (0.0, -1.0))
        self.assertEqual(agg_dict._get_expected_vector("G_00_01"), (0.0, 1.0))

    def test_fusion_normal_flow(self):
        # Normal flow: aligned, low density, high speed
        # density_red=4.0, speed_normal=5.0
        agg = FusionAggregator(density_red=4.0, speed_normal=5.0, expected_direction="EAST")

        # Mock DensityResult: low density (count 0.5 per grid)
        # density map size 100x200
        density_map = np.ones((100, 200), dtype=np.float32) * (0.5 / 10000.0)
        density_res = DensityInferenceResult(
            density_map=density_map,
            estimated_count=1.0,
            inference_time_ms=5.0,
            device="cpu",
            crowd_mask=np.zeros_like(density_map, dtype=np.uint8),
            input_width=200,
            input_height=100,
            model_name="SCALNet",
            checkpoint_path="dummy.pth",
        )

        # Mock FlowResult: high speed EAST (u=5.0, v=0.0)
        flow_x = np.ones((100, 200), dtype=np.float32) * 5.0
        flow_y = np.zeros((100, 200), dtype=np.float32)
        flow_res = FlowResult(flow_x=flow_x, flow_y=flow_y, inference_time_ms=10.0)

        # Default (unscaled)
        metrics = agg.fuse(self.grids, density_res, flow_res)
        self.assertIn("G_00_00", metrics)
        m = metrics["G_00_00"]
        # density = 0.5 / 10000 * 1.0 (unscaled) = 0.00005
        self.assertAlmostEqual(m.density, 0.00005, places=5)
        self.assertAlmostEqual(m.density_score, 0.0, places=2)
        # speed = 5.0 * 1.0 = 5.0
        self.assertAlmostEqual(m.speed, 5.0, places=5)
        self.assertAlmostEqual(m.slow_score, 0.0, places=2)
        self.assertAlmostEqual(m.stagnation_score, 0.0, places=2)
        self.assertAlmostEqual(m.flow_conflict_score, 0.0, places=2)
        self.assertAlmostEqual(m.reverse_score, 0.0, places=2)
        self.assertAlmostEqual(m.congestion_score, 0.0, places=2)
        self.assertEqual(m.risk_level, "GREEN")

        # Legacy Scaled Mode raises ValueError since it's disabled in Sprint 1
        with self.assertRaises(ValueError):
            FusionAggregator(density_red=4.0, speed_normal=5.0, expected_direction="EAST", use_heuristic_perspective_scaling=True)

    def test_fusion_reverse_danger(self):
        # Reverse flow + High Density + Stagnant
        # density_red=2.0, speed_normal=5.0
        agg = FusionAggregator(density_red=2.0, speed_normal=5.0, expected_direction="EAST")

        # Mock DensityResult: high density (count 4.0 in G_00_00 -> density = 4.0/10000 = 0.0004)
        density_map = np.ones((100, 200), dtype=np.float32) * (2.0 / 10000.0)
        density_res = DensityInferenceResult(
            density_map=density_map,
            estimated_count=4.0,
            inference_time_ms=5.0,
            device="cpu",
            crowd_mask=np.zeros_like(density_map, dtype=np.uint8),
            input_width=200,
            input_height=100,
            model_name="SCALNet",
            checkpoint_path="dummy.pth",
        )

        # Mock FlowResult: very slow WEST (u=-0.5, v=0.0)
        flow_x = np.ones((100, 200), dtype=np.float32) * -0.5
        flow_y = np.zeros((100, 200), dtype=np.float32)
        flow_res = FlowResult(flow_x=flow_x, flow_y=flow_y, inference_time_ms=10.0)

        # Unscaled Test (Default)
        agg_pixel = FusionAggregator(density_red=0.0002, speed_normal=5.0, expected_direction="EAST")
        metrics = agg_pixel.fuse(self.grids, density_res, flow_res)
        m = metrics["G_00_00"]
        # density_score = grid_density / density_red = 0.0002 / 0.0002 = 1.0
        self.assertAlmostEqual(m.density_score, 1.0, places=2)
        # speed = 0.5 * 1.0 = 0.5, speed_normal = 5.0 -> slow_score = 1 - 0.5/5.0 = 0.9
        self.assertAlmostEqual(m.slow_score, 0.9, places=2)
        self.assertAlmostEqual(m.stagnation_score, 0.9, places=2)
        self.assertAlmostEqual(m.reverse_score, 1.0, places=2)
        self.assertGreater(m.congestion_score, 60.0)
        self.assertIn(m.risk_level, ("ORANGE", "RED"))

        # Legacy Scaled Test raises ValueError since it's disabled in Sprint 1
        with self.assertRaises(ValueError):
            FusionAggregator(density_red=0.0002, speed_normal=5.0, expected_direction="EAST", use_heuristic_perspective_scaling=True)

    def test_robust_flow_rejects_vector_outliers(self):
        agg = FusionAggregator(density_red=0.001, speed_normal=5.0, expected_direction="EAST")
        density_map = np.ones((100, 200), dtype=np.float32) / 10000.0
        density_res = DensityInferenceResult(
            density_map=density_map, estimated_count=2.0, inference_time_ms=1.0,
            device="cpu", crowd_mask=np.ones_like(density_map, dtype=np.uint8),
            input_width=200, input_height=100, model_name="SCALNet", checkpoint_path="dummy",
        )
        flow_x = np.full((100, 200), 2.0, dtype=np.float32)
        flow_y = np.zeros_like(flow_x)
        # A small set of extreme, opposite vectors must not flip direction.
        flow_x[:10, :10] = -30.0
        flow_res = FlowResult(flow_x=flow_x, flow_y=flow_y, inference_time_ms=1.0)

        metric = agg.fuse(self.grids, density_res, flow_res)["G_00_00"]

        self.assertGreater(metric.flow_x, 1.8)
        self.assertEqual(metric.direction_label, "EAST")
        self.assertLess(metric.reverse_score, 0.01)

    def test_unreliable_flow_suppresses_alerts(self):
        agg = FusionAggregator(
            density_red=0.0001, speed_normal=5.0, expected_direction="EAST", min_confidence=0.9
        )
        density_map = np.ones((100, 200), dtype=np.float32) * (5.0 / 10000.0)
        density_res = DensityInferenceResult(
            density_map=density_map, estimated_count=10.0, inference_time_ms=1.0,
            device="cpu", crowd_mask=np.ones_like(density_map, dtype=np.uint8),
            input_width=200, input_height=100, model_name="SCALNet", checkpoint_path="dummy",
        )
        flow_res = FlowResult(
            flow_x=np.full((100, 200), -0.5, dtype=np.float32),
            flow_y=np.zeros((100, 200), dtype=np.float32), inference_time_ms=1.0,
            valid_mask=np.zeros((100, 200), dtype=bool), flow_quality=0.0,
            valid_flow_ratio=0.0, camera_motion_unreliable=True,
        )

        metric = agg.fuse(self.grids, density_res, flow_res)["G_00_00"]

        self.assertFalse(metric.alert_eligible)
        self.assertEqual(metric.direction_label, "UNKNOWN")

    def test_physical_density_and_speed_when_calibrated(self):
        calibrated_grid = GridBox(
            grid_id="G_00_00", row=0, col=0, x1=0, y1=0, x2=100, y2=100,
            area=10000.0, effective_area=10000.0,
            ground_polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
            image_polygon=[(0, 0), (100, 0), (100, 100), (0, 100)], is_relative=False,
        )
        density_map = np.full((100, 100), 400.0 / 10000.0, dtype=np.float32)
        density_res = DensityInferenceResult(
            density_map=density_map, estimated_count=400.0, inference_time_ms=1.0,
            device="cpu", crowd_mask=np.ones_like(density_map, dtype=np.uint8),
            input_width=100, input_height=100, model_name="SCALNet", checkpoint_path="dummy",
        )
        flow_res = FlowResult(
            flow_x=np.full((100, 100), 5.0, dtype=np.float32),
            flow_y=np.zeros((100, 100), dtype=np.float32), inference_time_ms=1.0,
        )
        agg = FusionAggregator(density_red=4.0, speed_normal=5.0, fps=10.0, meters_per_pixel=0.02)

        metric = agg.fuse([calibrated_grid], density_res, flow_res)["G_00_00"]

        self.assertTrue(metric.physical_calibrated)
        self.assertAlmostEqual(metric.density_people_m2, 4.0, places=2)
        self.assertAlmostEqual(metric.speed_mps, 1.0, places=2)
        self.assertEqual(metric.crowd_class, "DENSE")


if __name__ == "__main__":
    unittest.main()
