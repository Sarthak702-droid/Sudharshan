import math
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict

from .types import GridBox, GridAdjacencyEdge


class GridGenerator:
    def __init__(
        self,
        grid_size: int = 100,
        overlap_ratio: float = 0.20,
        max_overlap_ratio: float = 0.25,
    ) -> None:
        """Initializes the Grid Generator.

        Args:
            grid_size: Width and height of each grid in pixels (image-relative fallback).
            overlap_ratio: Overlap ratio between adjacent grids (0.0 to 1.0).
            max_overlap_ratio: Maximum allowed overlap ratio.
        """
        if overlap_ratio > max_overlap_ratio:
            raise ValueError(f"Overlap ratio {overlap_ratio} exceeds the maximum allowed {max_overlap_ratio}")

        # Disallow ~33% or ~50% overlaps as per guidelines
        if math.isclose(overlap_ratio, 0.33, abs_tol=0.01) or math.isclose(overlap_ratio, 0.50, abs_tol=0.01):
            raise ValueError(f"Overlap ratio of {overlap_ratio} is not allowed (avoid 33% and 50% overlaps).")

        self.grid_size = grid_size
        self.overlap_ratio = overlap_ratio

        # Calculate step size
        self.step_size = int(grid_size * (1.0 - overlap_ratio))
        if self.step_size <= 0:
            raise ValueError("Grid size and overlap ratio result in a step size of 0. Increase grid size or decrease overlap.")

        self.overlap_size = grid_size - self.step_size

    def project_points(self, points: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Projects points from ground coordinates to image coordinates using homography H.

        Args:
            points: np.ndarray of shape (N, 2)
            H: np.ndarray of shape (3, 3)

        Returns:
            np.ndarray of shape (N, 2)
        """
        N = points.shape[0]
        pts_hom = np.hstack([points, np.ones((N, 1))])
        proj_hom = np.dot(H, pts_hom.T)
        w = proj_hom[2, :]
        w = np.where(np.abs(w) < 1e-9, 1e-9, w)
        proj_pts = proj_hom[:2, :] / w
        return proj_pts.T

    def inverse_project_points(self, points: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Projects points from image coordinates to ground coordinates using inverse homography H^-1.

        Args:
            points: np.ndarray of shape (N, 2)
            H: np.ndarray of shape (3, 3)

        Returns:
            np.ndarray of shape (N, 2)
        """
        H_inv = np.linalg.inv(H)
        return self.project_points(points, H_inv)

    def generate_grids(
        self,
        width: int,
        height: int,
        boundary_polygon: Optional[List[Tuple[float, float]]] = None,
        homography: Optional[np.ndarray] = None,
        ground_roi: Optional[List[Tuple[float, float]]] = None,
        camera_roi: Optional[List[Tuple[float, float]]] = None,
        grid_size_m: float = 10.0,
        overlap_ratio_m: float = 0.20,
    ) -> List[GridBox]:
        """Divides the event region into overlapping grids.
        Supports calibrated ground-plane projection if homography and ground_roi are provided.

        Args:
            width: Image width.
            height: Image height.
            boundary_polygon: List of (x, y) pixel coordinates defining monitored boundary (fallback).
            homography: 3x3 homography mapping ground coordinates to image space.
            ground_roi: Monitored ground corridor polygon in meters.
            camera_roi: Camera viewport polygon in pixels.
            grid_size_m: Target ground grid cell size in meters (default: 10.0m).
            overlap_ratio_m: Target overlap ratio on ground (default: 0.20).

        Returns:
            List of GridBox objects.
        """
        # --- GROUND-PLANE CALIBRATED MODE ---
        if homography is not None and ground_roi is not None:
            # 1. Project ground ROI to image space to create active boundary mask
            ground_roi_px = self.project_points(np.array(ground_roi, dtype=np.float32), homography)
            monitored_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(monitored_mask, [np.array(ground_roi_px, dtype=np.int32)], 1)

            # 2. Setup camera ROI mask
            camera_mask = np.zeros((height, width), dtype=np.uint8)
            if camera_roi is not None:
                cv2.fillPoly(camera_mask, [np.array(camera_roi, dtype=np.int32)], 1)
            else:
                camera_mask.fill(1)

            active_mask = cv2.bitwise_and(monitored_mask, camera_mask)

            # 3. Determine loop ranges on ground plane
            ground_pts = np.array(ground_roi, dtype=np.float32)
            x_min, y_min = np.min(ground_pts, axis=0)
            x_max, y_max = np.max(ground_pts, axis=0)

            step_m = grid_size_m * (1.0 - overlap_ratio_m)

            y_coords = []
            y = y_min
            while y + grid_size_m <= y_max:
                y_coords.append(y)
                y += step_m
            if y < y_max and (not y_coords or y_coords[-1] + grid_size_m < y_max):
                y_coords.append(y_max - grid_size_m)

            x_coords = []
            x = x_min
            while x + grid_size_m <= x_max:
                x_coords.append(x)
                x += step_m
            if x < x_max and (not x_coords or x_coords[-1] + grid_size_m < x_max):
                x_coords.append(x_max - grid_size_m)

            grids: List[GridBox] = []

            # 4. Generate cells in ground plane and project
            for row, y_start in enumerate(y_coords):
                for col, x_start in enumerate(x_coords):
                    grid_id = f"G_{row:02d}_{col:02d}"

                    ground_poly = [
                        (x_start, y_start),
                        (x_start + grid_size_m, y_start),
                        (x_start + grid_size_m, y_start + grid_size_m),
                        (x_start, y_start + grid_size_m),
                    ]

                    # Project ground corners to pixels
                    image_poly = self.project_points(np.array(ground_poly, dtype=np.float32), homography)
                    image_poly_pts = [(float(p[0]), float(p[1])) for p in image_poly]

                    # Generate mask of projected polygon to calculate intersection
                    image_poly_mask = np.zeros((height, width), dtype=np.uint8)
                    cv2.fillPoly(image_poly_mask, [np.array(image_poly, dtype=np.int32)], 1)

                    intersection_mask = cv2.bitwise_and(image_poly_mask, active_mask)
                    clipped_area = float(np.sum(intersection_mask))
                    full_area = float(np.sum(image_poly_mask))

                    if full_area <= 0:
                        continue

                    valid_ratio = clipped_area / full_area
                    if valid_ratio >= 0.40:
                        # Extract bounding box clipped to image frame limits
                        xs = [p[0] for p in image_poly_pts]
                        ys = [p[1] for p in image_poly_pts]
                        x1 = int(max(0, min(xs)))
                        y1 = int(max(0, min(ys)))
                        x2 = int(min(width, max(xs)))
                        y2 = int(min(height, max(ys)))

                        grids.append(
                            GridBox(
                                grid_id=grid_id,
                                row=row,
                                col=col,
                                x1=x1,
                                y1=y1,
                                x2=x2,
                                y2=y2,
                                area=full_area,
                                effective_area=clipped_area,
                                boundary_polygon=ground_roi_px.tolist(),
                                ground_polygon=ground_poly,
                                image_polygon=image_poly_pts,
                                is_relative=False,
                            )
                        )
            return grids

        # --- IMAGE-RELATIVE FALLBACK MODE ---
        mask = None
        if boundary_polygon is not None:
            mask = np.zeros((height, width), dtype=np.uint8)
            poly_pts = np.array(boundary_polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [poly_pts], 1)

        grids = []
        y_coords = []
        y = 0
        while y + self.grid_size <= height:
            y_coords.append(y)
            y += self.step_size
        if y < height and (not y_coords or y_coords[-1] + self.grid_size < height):
            y_coords.append(height - self.grid_size)

        x_coords = []
        x = 0
        while x + self.grid_size <= width:
            x_coords.append(x)
            x += self.step_size
        if x < width and (not x_coords or x_coords[-1] + self.grid_size < width):
            x_coords.append(width - self.grid_size)

        for row, y_start in enumerate(y_coords):
            for col, x_start in enumerate(x_coords):
                grid_id = f"G_{row:02d}_{col:02d}"
                x1, y1 = x_start, y_start
                x2, y2 = x_start + self.grid_size, y_start + self.grid_size

                full_area = float(self.grid_size * self.grid_size)
                effective_area = full_area

                if mask is not None:
                    active_pixels = int(np.sum(mask[y1:y2, x1:x2]))
                    effective_area = float(active_pixels)
                    if effective_area <= 0.01 * full_area:
                        continue

                grids.append(
                    GridBox(
                        grid_id=grid_id,
                        row=row,
                        col=col,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        area=full_area,
                        effective_area=effective_area,
                        boundary_polygon=boundary_polygon,
                        is_relative=True,
                    )
                )

        return grids

    def build_adjacency_graph(
        self,
        grids: List[GridBox],
        expected_direction: str = "EAST",
    ) -> Dict[str, List[GridAdjacencyEdge]]:
        """Builds neighbor adjacency graph for the active grids.

        Args:
            grids: List of generated GridBox objects.
            expected_direction: Expected flow direction label (default: "EAST").

        Returns:
            Dict mapping grid_id to List of GridAdjacencyEdge objects.
        """
        graph: Dict[str, List[GridAdjacencyEdge]] = {}
        grid_map = {(g.row, g.col): g for g in grids}

        for g in grids:
            graph[g.grid_id] = []
            row, col = g.row, g.col

            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    neighbor = grid_map.get((row + dr, col + dc))
                    if neighbor is not None:
                        if not g.is_relative and g.ground_polygon is not None and neighbor.ground_polygon is not None:
                            # Use ground coordinates for center calculations
                            g_pts = np.array(g.ground_polygon)
                            n_pts = np.array(neighbor.ground_polygon)
                            c1 = np.mean(g_pts, axis=0)
                            c2 = np.mean(n_pts, axis=0)

                            dx = float(c2[0] - c1[0])
                            dy = float(c2[1] - c1[1])
                            distance = float(math.sqrt(dx**2 + dy**2))
                        else:
                            # Fallback to image-relative center distance in pixels
                            c1_x = (g.x1 + g.x2) / 2.0
                            c1_y = (g.y1 + g.y2) / 2.0
                            c2_x = (neighbor.x1 + neighbor.x2) / 2.0
                            c2_y = (neighbor.y1 + neighbor.y2) / 2.0

                            dx = float(c2_x - c1_x)
                            dy = float(c2_y - c1_y)
                            distance = float(math.sqrt(dx**2 + dy**2))

                        direction_rad = math.atan2(dy, dx)
                        direction_deg = (direction_rad * 180.0 / math.pi + 360.0) % 360.0

                        edge = GridAdjacencyEdge(
                            source_id=g.grid_id,
                            target_id=neighbor.grid_id,
                            direction_deg=direction_deg,
                            distance_m=distance,
                            expected_direction=expected_direction,
                        )
                        graph[g.grid_id].append(edge)
        return graph

    def compute_overlap_weights(
        self,
        width: int,
        height: int,
        grids: List[GridBox],
    ) -> np.ndarray:
        """Precomputes a global frame-level overlap weight matrix W(x, y).

        Args:
            width: Image width.
            height: Image height.
            grids: List of active GridBox structures.

        Returns:
            np.ndarray of shape (height, width) containing weights.
        """
        coverage = np.zeros((height, width), dtype=np.float32)

        for g in grids:
            mask = np.zeros((height, width), dtype=np.uint8)
            if not g.is_relative and g.image_polygon is not None:
                cv2.fillPoly(mask, [np.array(g.image_polygon, dtype=np.int32)], 1)
            else:
                mask[g.y1:g.y2, g.x1:g.x2] = 1
            coverage += mask

        weights = np.where(coverage >= 1.0, 1.0 / coverage, 1.0)
        return weights

    def _is_point_in_polygon(self, x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
        """Legacy helper to check if a point is inside a polygon."""
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
