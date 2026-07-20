from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class GridBox:
    grid_id: str
    row: int
    col: int
    x1: int
    y1: int
    x2: int
    y2: int
    area: float
    effective_area: float
    boundary_polygon: Optional[List[Tuple[float, float]]] = None
    # Ground-plane extensions
    ground_polygon: Optional[List[Tuple[float, float]]] = None
    image_polygon: Optional[List[Tuple[float, float]]] = None
    is_relative: bool = True


@dataclass
class GridAdjacencyEdge:
    source_id: str
    target_id: str
    direction_deg: float      # angle from source center to target center in ground plane
    distance_m: float         # distance in meters between centers
    expected_direction: str   # expected flow direction at source
