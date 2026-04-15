import numpy as np

from .config import SimulationConfig
from .models import CellType


def build_grid_network(config: SimulationConfig) -> np.ndarray:
    """Build a simple orthogonal road grid with intersections at road crossings."""

    grid = np.full((config.grid_height, config.grid_width), CellType.EMPTY, dtype=np.int8)

    for row in range(0, config.grid_height, config.road_spacing):
        grid[row, :] = CellType.ROAD_HORIZONTAL

    for col in range(0, config.grid_width, config.road_spacing):
        vertical_mask = grid[:, col] == CellType.ROAD_HORIZONTAL
        grid[:, col] = CellType.ROAD_VERTICAL
        grid[vertical_mask, col] = CellType.INTERSECTION

    return grid


def intersection_mask(grid: np.ndarray) -> np.ndarray:
    """Return a boolean mask identifying which cells are intersections."""

    return grid == CellType.INTERSECTION
