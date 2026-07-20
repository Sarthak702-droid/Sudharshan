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

from grid.types import GridBox
from grid.generator import GridGenerator


class TestGridGenerator(unittest.TestCase):
    def test_validation_rules(self):
        # Overlap ratio exceeds max
        with self.assertRaises(ValueError):
            GridGenerator(grid_size=100, overlap_ratio=0.30, max_overlap_ratio=0.25)

        # Overlap ratio close to 33%
        with self.assertRaises(ValueError):
            GridGenerator(grid_size=100, overlap_ratio=0.33, max_overlap_ratio=0.40)

        # Overlap ratio close to 50%
        with self.assertRaises(ValueError):
            GridGenerator(grid_size=100, overlap_ratio=0.50, max_overlap_ratio=0.60)

    def test_grid_math_and_dimensions(self):
        # grid_size=100, overlap=20% -> step=80, overlap=20
        gen = GridGenerator(grid_size=100, overlap_ratio=0.20)
        self.assertEqual(gen.step_size, 80)
        self.assertEqual(gen.overlap_size, 20)

        # Monitored section size: Width=200, Height=100
        # Grid Size = 100
        # Grids in X:
        # 1. 0 to 100
        # 2. 80 to 180
        # 3. (uncovered tail at 180-200 -> final grid) 100 to 200
        # Grids in Y:
        # 1. 0 to 100
        # Total grids: 3 columns * 1 row = 3 grids
        grids = gen.generate_grids(width=200, height=100)
        self.assertEqual(len(grids), 3)

        # Check coordinates of the third grid (should be starting at 100, ending at 200)
        self.assertEqual(grids[2].grid_id, "G_00_02")
        self.assertEqual(grids[2].x1, 100)
        self.assertEqual(grids[2].x2, 200)
        self.assertEqual(grids[2].area, 10000.0)
        self.assertEqual(grids[2].effective_area, 10000.0)

    def test_boundary_polygon_clipping(self):
        gen = GridGenerator(grid_size=100, overlap_ratio=0.20)

        # Define a boundary polygon covering the left half of a 400x100 region
        # Monitored area is x in [0, 150], y in [0, 100]
        boundary = [
            (0, 0),
            (150, 0),
            (150, 100),
            (0, 100)
        ]

        grids = gen.generate_grids(width=400, height=100, boundary_polygon=boundary)

        # Col 0: x in [0, 100] -> fully inside boundary -> effective_area = 10000.0
        # Col 1: x in [80, 180] -> partially inside (80 to 150 is inside -> width 70px) -> effective_area = 7000.0
        # Col 2: x in [160, 260] -> outside -> discarded
        # Col 3: x in [240, 340] -> outside -> discarded
        # Col 4: (final edge) x in [300, 400] -> outside -> discarded
        self.assertEqual(len(grids), 2)

        self.assertEqual(grids[0].grid_id, "G_00_00")
        self.assertAlmostEqual(grids[0].effective_area, 10000.0, delta=10)

        self.assertEqual(grids[1].grid_id, "G_00_01")
        # Overlap of [80, 180] and [0, 150] is [80, 150] -> 70px (or 71px inclusive) width * 100px height = ~7000
        self.assertAlmostEqual(grids[1].effective_area, 7000.0, delta=150)

    def test_homography_projection(self):
        # Setup an identity-like homography matrix
        H = np.eye(3, dtype=np.float32)
        H[0, 2] = 10.0  # translate X by +10
        H[1, 2] = 20.0  # translate Y by +20

        gen = GridGenerator()
        pts = np.array([[0.0, 0.0], [5.0, 5.0]], dtype=np.float32)

        proj = gen.project_points(pts, H)
        np.testing.assert_allclose(proj, [[10.0, 20.0], [15.0, 25.0]])

        inv_proj = gen.inverse_project_points(proj, H)
        np.testing.assert_allclose(inv_proj, pts, atol=1e-5)

    def test_overlap_weights_correction(self):
        # We test for different overlap ratios: 0%, 10%, 20%, 25%
        for overlap in [0.0, 0.10, 0.20, 0.25]:
            gen = GridGenerator(grid_size=100, overlap_ratio=overlap, max_overlap_ratio=0.30)
            grids = gen.generate_grids(width=300, height=200)

            weights = gen.compute_overlap_weights(width=300, height=200, grids=grids)

            # Sum of weights in each pixel should equal 1.0 (where covered by grids)
            # Create a combined coverage map:
            coverage = np.zeros((200, 300), dtype=np.float32)
            for g in grids:
                coverage[g.y1:g.y2, g.x1:g.x2] += 1.0

            expected_weights = np.where(coverage >= 1.0, 1.0 / coverage, 1.0)
            np.testing.assert_allclose(weights, expected_weights)

            # Setup a random density map
            np.random.seed(42)
            density_map = np.random.rand(200, 300).astype(np.float32)

            # Sum of count_g using weights should equal total density in covered area
            total_weighted_grid_count = 0.0
            for g in grids:
                local_d = density_map[g.y1:g.y2, g.x1:g.x2]
                local_w = weights[g.y1:g.y2, g.x1:g.x2]
                total_weighted_grid_count += np.sum(local_d * local_w)

            # Active area density (where coverage >= 1)
            active_density = np.sum(density_map[coverage >= 1.0])
            self.assertAlmostEqual(total_weighted_grid_count, active_density, places=2)

    def test_adjacency_graph(self):
        # Setup a simple 2x2 grid in image space
        gen = GridGenerator(grid_size=100, overlap_ratio=0.20)
        grids = gen.generate_grids(width=180, height=180) # should produce 2x2 grids (step=80)

        graph = gen.build_adjacency_graph(grids, expected_direction="EAST")

        # Grid G_00_00 should have neighbors: G_00_01, G_01_00, G_01_01
        self.assertIn("G_00_00", graph)
        neighbors = [e.target_id for e in graph["G_00_00"]]
        self.assertEqual(len(neighbors), 3)
        self.assertCountEqual(neighbors, ["G_00_01", "G_01_00", "G_01_01"])

        # Check one of the edges (from G_00_00 to G_00_01, moving EAST in image coordinates)
        east_edge = [e for e in graph["G_00_00"] if e.target_id == "G_00_01"][0]
        self.assertEqual(east_edge.distance_m, 80.0) # step size = 80px (fallback to pixels)
        self.assertEqual(east_edge.direction_deg, 0.0) # EAST is 0 degrees
        self.assertEqual(east_edge.expected_direction, "EAST")


if __name__ == "__main__":
    unittest.main()
