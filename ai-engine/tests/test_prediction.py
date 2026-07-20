import unittest
import numpy as np
import sys
from pathlib import Path

# Ensure project root and ai-engine are in sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(project_root / "ai-engine") not in sys.path:
    sys.path.insert(0, str(project_root / "ai-engine"))

from fusion.types import GridMetrics
from prediction.forecaster import CrowdFlowPredictor


class TestCrowdFlowPredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = CrowdFlowPredictor(history_window_size=10, forecast_horizon_frames=5)

    def test_insufficient_history(self):
        # Empty history should default safely to 0
        pred_count, pred_score, pred_risk, slope = self.predictor.predict_next("G_00_00", [])
        self.assertEqual(pred_count, 0.0)
        self.assertEqual(pred_score, 0.0)
        self.assertEqual(pred_risk, "GREEN")
        self.assertEqual(slope, 0.0)

    def test_linear_trend_forecasting(self):
        # Create a series with a clear upward trend in congestion score (increasing by 2 points per frame)
        # congestion_score: 20, 22, 24, 26, 28, 30, 32, 34, 36, 38
        # Last index is 9 (corresponds to score 38.0).
        # We predict 5 frames ahead -> index 14.
        # Predicted score should be: 20 + 2 * 14 = 48.0 (YELLOW range)
        history = []
        for i in range(10):
            m = GridMetrics(
                grid_id="G_00_00",
                count=float(i + 1),  # count increases from 1.0 to 10.0
                density=0.0001,
                flow_x=1.0,
                flow_y=0.0,
                speed=1.0,
                direction_deg=0.0,
                direction_label="EAST",
                density_score=0.1,
                slow_score=0.1,
                stagnation_score=0.01,
                flow_conflict_score=0.0,
                reverse_score=0.0,
                congestion_score=float(20 + 2 * i),
                risk_level="GREEN",
                confidence=0.9,
            )
            history.append(m)

        pred_count, pred_score, pred_risk, slope = self.predictor.predict_next("G_00_00", history)

        # Count: increases by 1.0 per frame. Frame 9 count is 10.0.
        # Predict 5 frames ahead (index 14) -> Count should be 1 + 1 * 14 = 15.0
        self.assertAlmostEqual(pred_count, 15.0, places=5)
        self.assertAlmostEqual(pred_score, 48.0, places=5)
        self.assertEqual(pred_risk, "YELLOW")
        self.assertAlmostEqual(slope, 2.0, places=5)


if __name__ == "__main__":
    unittest.main()
