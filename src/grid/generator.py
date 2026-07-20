import math

class GridGenerator:
    def __init__(self, grid_size=100, overlap_ratio=0.20, max_overlap_ratio=0.25):
        """
        Initializes the Grid Generator.
        grid_size: width and height of each grid in pixels (or meters in real-world mode).
        overlap_ratio: ratio of overlap between adjacent grids (0.0 to 1.0).
        """
        if overlap_ratio > max_overlap_ratio:
            raise ValueError(f"Overlap ratio {overlap_ratio} exceeds the maximum allowed {max_overlap_ratio}")

        # Avoid 33% or 50% overlap as per requirements
        if math.isclose(overlap_ratio, 0.33, abs_tol=0.01) or math.isclose(overlap_ratio, 0.50, abs_tol=0.01):
            raise ValueError(f"Overlap ratio of {overlap_ratio} is not allowed (avoid 33% and 50%)")

        self.grid_size = grid_size
        self.overlap_ratio = overlap_ratio
        self.step_size = int(grid_size * (1 - overlap_ratio))
        self.overlap_size = grid_size - self.step_size

    def generate_grids(self, width, height, boundary_polygon=None):
        """
        Generates grid boxes for a given width and height.
        Optionally filters/clips them based on a boundary polygon.
        Returns a list of grid dictionaries.
        """
        grids = []
        row = 0
        y = 0

        while y + self.grid_size <= height:
            col = 0
            x = 0
            while x + self.grid_size <= width:
                grid_id = f"G_{row:02d}_{col:02d}"

                # Default coordinates
                x1, y1 = x, y
                x2, y2 = x + self.grid_size, y + self.grid_size

                # Check boundary clipping if polygon provided (dummy/simple bounding box check for now, can expand)
                is_valid = True
                effective_area = self.grid_size * self.grid_size

                if boundary_polygon:
                    # Simple check: at least center of grid must be inside boundary
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    is_valid = self._is_point_in_polygon(cx, cy, boundary_polygon)

                if is_valid:
                    grids.append({
                        "grid_id": grid_id,
                        "row": row,
                        "col": col,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "area": effective_area
                    })

                x += self.step_size
                col += 1
            y += self.step_size
            row += 1

        return grids

    def _is_point_in_polygon(self, x, y, polygon):
        """
        Ray-casting algorithm to determine if a point is inside a polygon.
        polygon: list of (x, y) tuples.
        """
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
