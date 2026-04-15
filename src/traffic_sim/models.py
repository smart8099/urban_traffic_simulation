from dataclasses import dataclass
from enum import IntEnum


class CellType(IntEnum):
    """Enumerates the supported grid cell types in the road network."""

    EMPTY = 0
    ROAD_HORIZONTAL = 1
    ROAD_VERTICAL = 2
    INTERSECTION = 3


class LightPhase(IntEnum):
    """Represents the active traffic-light phase at intersections."""

    HORIZONTAL_GREEN = 0
    VERTICAL_GREEN = 1


class Direction(IntEnum):
    """Represents the movement direction assigned to a vehicle."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


@dataclass(slots=True)
class SimulationMetrics:
    """Collects summary statistics produced after each simulation step."""

    steps: int = 0
    active_vehicles: int = 0
    completed_vehicles: int = 0
    average_queue_length: float = 0.0
