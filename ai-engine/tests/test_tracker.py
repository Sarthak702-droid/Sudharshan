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
from fusion.tracker import TemporalTracker


class TestTemporalTracker(unittest.TestCase):
    def setUp(self):
        # Create a mock base GridMetrics
        self.mock_metrics = GridMetrics(
            grid_id="G_00_00",
            count=10.0,
            density=0.001,
            flow_x=2.0,
            flow_y=0.0,
            speed=2.0,
            direction_deg=0.0,
            direction_label="EAST",
            density_score=0.2,
            slow_score=0.2,
            stagnation_score=0.04,
            flow_conflict_score=0.0,
            reverse_score=0.0,
            congestion_score=30.0,  # GREEN risk
            risk_level="GREEN",
            confidence=0.9,
        )

    def test_ema_smoothing(self):
        # Alpha=0.5. Frame 1 count = 10, Frame 2 count = 20. EMA = 0.5*20 + 0.5*10 = 15
        tracker = TemporalTracker(alpha=0.5, fps=5.0)

        # Frame 1
        metrics_1 = {"G_00_00": self.mock_metrics}
        res_1 = tracker.track(metrics_1)
        self.assertEqual(res_1["G_00_00"].count, 10.0)

        # Frame 2
        metrics_2 = {"G_00_00": GridMetrics(
            grid_id="G_00_00",
            count=20.0,
            density=0.002,
            flow_x=2.0,
            flow_y=0.0,
            speed=2.0,
            direction_deg=0.0,
            direction_label="EAST",
            density_score=0.2,
            slow_score=0.2,
            stagnation_score=0.04,
            flow_conflict_score=0.0,
            reverse_score=0.0,
            congestion_score=30.0,
            risk_level="GREEN",
            confidence=0.9,
        )}

        res_2 = tracker.track(metrics_2)
        self.assertEqual(res_2["G_00_00"].count, 15.0)

    def test_temporal_persistence(self):
        # Yellow alert persistence: 2.0 seconds at 5 FPS = 10 frames
        tracker = TemporalTracker(alpha=1.0, fps=5.0, persistence_yellow_sec=2.0)

        # Frame 1 to 9: Score is 50.0 (YELLOW range)
        # Should remain GREEN until frame 10 (persistence constraint)
        for i in range(1, 10):
            mock_yellow = GridMetrics(
                grid_id="G_00_00",
                count=10.0,
                density=0.001,
                flow_x=2.0,
                flow_y=0.0,
                speed=2.0,
                direction_deg=0.0,
                direction_label="EAST",
                density_score=0.2,
                slow_score=0.2,
                stagnation_score=0.04,
                flow_conflict_score=0.0,
                reverse_score=0.0,
                congestion_score=50.0,  # YELLOW range
                risk_level="GREEN",  # Input level from base fusion is green
                confidence=0.9,
            )
            res = tracker.track({"G_00_00": mock_yellow})
            self.assertEqual(res["G_00_00"].risk_level, "GREEN", f"Failed at frame {i}")

        # Frame 10: YELLOW category persists for 10 consecutive frames -> risk level should transition to YELLOW
        mock_yellow_10 = GridMetrics(
            grid_id="G_00_00",
            count=10.0,
            density=0.001,
            flow_x=2.0,
            flow_y=0.0,
            speed=2.0,
            direction_deg=0.0,
            direction_label="EAST",
            density_score=0.2,
            slow_score=0.2,
            stagnation_score=0.04,
            flow_conflict_score=0.0,
            reverse_score=0.0,
            congestion_score=50.0,
            risk_level="GREEN",
            confidence=0.9,
        )
        res_10 = tracker.track({"G_00_00": mock_yellow_10})
        self.assertEqual(res_10["G_00_00"].risk_level, "YELLOW")

    def test_hysteresis_rule(self):
        # Bypass persistence (set yellow/orange persistence to 0/1 frame)
        tracker = TemporalTracker(
            alpha=1.0,
            fps=5.0,
            persistence_yellow_sec=0.0,
            persistence_orange_sec=0.0,
            persistence_red_sec=0.0
        )

        # 1. Enter ORANGE range (congestion = 65.0)
        mock_orange = GridMetrics(
            grid_id="G_00_00",
            count=10.0,
            density=0.001,
            flow_x=2.0,
            flow_y=0.0,
            speed=2.0,
            direction_deg=0.0,
            direction_label="EAST",
            density_score=0.2,
            slow_score=0.2,
            stagnation_score=0.04,
            flow_conflict_score=0.0,
            reverse_score=0.0,
            congestion_score=65.0,  # ORANGE range
            risk_level="GREEN",
            confidence=0.9,
        )
        res = tracker.track({"G_00_00": mock_orange})
        self.assertEqual(res["G_00_00"].risk_level, "ORANGE")

        # 2. Score drops slightly to 58.0 (YELLOW range)
        # Hysteresis prevents leaving ORANGE (margin is < 54.0)
        mock_yellow_high = GridMetrics(
            grid_id="G_00_00",
            count=10.0,
            density=0.001,
            flow_x=2.0,
            flow_y=0.0,
            speed=2.0,
            direction_deg=0.0,
            direction_label="EAST",
            density_score=0.2,
            slow_score=0.2,
            stagnation_score=0.04,
            flow_conflict_score=0.0,
            reverse_score=0.0,
            congestion_score=58.0,  # YELLOW range but > 54
            risk_level="ORANGE",
            confidence=0.9,
        )
        res_high = tracker.track({"G_00_00": mock_yellow_high})
        self.assertEqual(res_high["G_00_00"].risk_level, "ORANGE")

        # 3. Score drops to 52.0 (below hysteresis exit limit 54.0) -> transitions to YELLOW
        mock_yellow_low = GridMetrics(
            grid_id="G_00_00",
            count=10.0,
            density=0.001,
            flow_x=2.0,
            flow_y=0.0,
            speed=2.0,
            direction_deg=0.0,
            direction_label="EAST",
            density_score=0.2,
            slow_score=0.2,
            stagnation_score=0.04,
            flow_conflict_score=0.0,
            reverse_score=0.0,
            congestion_score=52.0,  # YELLOW range < 54
            risk_level="ORANGE",
            confidence=0.9,
        )
        res_low = tracker.track({"G_00_00": mock_yellow_low})
        self.assertEqual(res_low["G_00_00"].risk_level, "YELLOW")

    def test_low_confidence_metric_cannot_raise_alert(self):
        tracker = TemporalTracker(
            alpha=1.0, fps=5.0, persistence_red_sec=0.0,
            persistence_orange_sec=0.0, persistence_yellow_sec=0.0,
        )
        metric = GridMetrics(
            grid_id="G_00_00", count=20.0, density=0.002, flow_x=-1.0, flow_y=0.0,
            speed=1.0, direction_deg=180.0, direction_label="WEST", density_score=1.0,
            slow_score=0.8, stagnation_score=0.8, flow_conflict_score=0.8,
            reverse_score=1.0, congestion_score=95.0, risk_level="RED", confidence=0.2,
            alert_eligible=False, crowd_present=True, crowd_class="CRITICAL",
            crowd_probability=0.95, flow_quality=0.0,
        )

        result = tracker.track({"G_00_00": metric})["G_00_00"]

        self.assertEqual(result.risk_level, "GREEN")
        self.assertFalse(result.alert_eligible)
        self.assertFalse(result.turbulence_warning)

    def test_crowd_class_requires_temporal_consistency(self):
        tracker = TemporalTracker(alpha=1.0, fps=5.0)
        tracker.track({"G_00_00": self.mock_metrics})
        dense = GridMetrics(**{
            **self.mock_metrics.__dict__, "crowd_class": "DENSE",
            "crowd_probability": 0.9, "crowd_present": True,
        })

        first = tracker.track({"G_00_00": dense})["G_00_00"]
        second = tracker.track({"G_00_00": dense})["G_00_00"]
        third = tracker.track({"G_00_00": dense})["G_00_00"]

        self.assertNotEqual(first.crowd_class, "DENSE")
        self.assertNotEqual(second.crowd_class, "DENSE")
        self.assertEqual(third.crowd_class, "DENSE")


if __name__ == "__main__":
    unittest.main()
