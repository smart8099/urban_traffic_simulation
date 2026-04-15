from __future__ import annotations

import numpy as np

from .config import SimulationConfig
from .models import Direction, LightPhase, SimulationMetrics
from .network import build_grid_network, intersection_mask


class CPUTrafficSimulation:
    """Run the reference traffic simulation using CPU-backed NumPy arrays."""

    def __init__(self, config: SimulationConfig) -> None:
        """Initialize simulation state, vehicle storage, and occupancy buffers."""

        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self.grid = build_grid_network(config)
        self.intersections = intersection_mask(self.grid)
        self.occupancy = np.full(self.grid.shape, -1, dtype=np.int32)
        self.entry_points = self._build_entry_points()

        self.vehicle_active = np.zeros(config.max_vehicles, dtype=bool)
        self.vehicle_x = np.zeros(config.max_vehicles, dtype=np.int32)
        self.vehicle_y = np.zeros(config.max_vehicles, dtype=np.int32)
        self.vehicle_direction = np.zeros(config.max_vehicles, dtype=np.int8)

        self.step_count = 0
        self.completed_vehicles = 0

    def current_light_phase(self) -> LightPhase:
        """Return the current global traffic-light phase for this timestep."""

        cycle_position = (self.step_count // self.config.light_cycle_steps) % 2
        return LightPhase(cycle_position)

    def _build_entry_points(self) -> np.ndarray:
        """Enumerate valid boundary spawn cells and their initial travel directions."""

        entry_points: list[tuple[int, int, int]] = []

        for y in range(self.config.grid_height):
            if self.grid[y, 0] in (1, 3):
                entry_points.append((0, y, int(Direction.RIGHT)))
            if self.grid[y, self.config.grid_width - 1] in (1, 3):
                entry_points.append((self.config.grid_width - 1, y, int(Direction.LEFT)))

        for x in range(self.config.grid_width):
            if self.grid[0, x] in (2, 3):
                entry_points.append((x, 0, int(Direction.DOWN)))
            if self.grid[self.config.grid_height - 1, x] in (2, 3):
                entry_points.append((x, self.config.grid_height - 1, int(Direction.UP)))

        return np.asarray(entry_points, dtype=np.int32)

    def _direction_has_green(self, direction: int, phase: LightPhase) -> bool:
        """Return whether a vehicle direction is permitted by the active light phase."""

        if direction in (Direction.LEFT, Direction.RIGHT):
            return phase == LightPhase.HORIZONTAL_GREEN
        return phase == LightPhase.VERTICAL_GREEN

    def _turn_options(self, direction: int) -> tuple[int, int, int]:
        """Return the forward, left-turn, and right-turn directions for a vehicle."""

        if direction == Direction.RIGHT:
            return int(Direction.RIGHT), int(Direction.UP), int(Direction.DOWN)
        if direction == Direction.LEFT:
            return int(Direction.LEFT), int(Direction.DOWN), int(Direction.UP)
        if direction == Direction.UP:
            return int(Direction.UP), int(Direction.LEFT), int(Direction.RIGHT)
        return int(Direction.DOWN), int(Direction.RIGHT), int(Direction.LEFT)

    def _choose_intersection_direction(self, direction: int) -> int:
        """Select a new direction for a vehicle leaving an intersection."""

        if self.rng.random() >= self.config.turn_probability:
            return direction

        forward, left_turn, right_turn = self._turn_options(direction)
        return int(self.rng.choice(np.asarray((forward, left_turn, right_turn), dtype=np.int8)))

    def spawn_vehicle(self) -> None:
        """Attempt to create new vehicles at free boundary entry points."""

        inactive_ids = np.flatnonzero(~self.vehicle_active)
        if inactive_ids.size == 0 or self.entry_points.size == 0:
            return

        shuffled_entries = self.entry_points[self.rng.permutation(len(self.entry_points))]
        spawned = 0

        for x, y, direction in shuffled_entries:
            if spawned >= self.config.spawn_attempts_per_step:
                break
            if self.rng.random() > self.config.spawn_rate:
                continue
            if self.occupancy[y, x] != -1:
                continue

            available = np.flatnonzero(~self.vehicle_active)
            if available.size == 0:
                break

            vehicle_id = int(available[0])
            self.vehicle_active[vehicle_id] = True
            self.vehicle_x[vehicle_id] = int(x)
            self.vehicle_y[vehicle_id] = int(y)
            self.vehicle_direction[vehicle_id] = int(direction)
            self.occupancy[y, x] = vehicle_id
            spawned += 1

    def _next_position(self, x: int, y: int, direction: int) -> tuple[int, int]:
        """Compute the next grid cell for a vehicle moving one step forward."""

        if direction == Direction.RIGHT:
            return x + 1, y
        if direction == Direction.LEFT:
            return x - 1, y
        if direction == Direction.UP:
            return x, y - 1
        return x, y + 1

    def _can_enter_next_cell(self, x: int, y: int, direction: int, next_x: int, next_y: int) -> bool:
        """Return whether the vehicle may move into its next target cell this step."""

        if not (0 <= next_x < self.config.grid_width and 0 <= next_y < self.config.grid_height):
            return True

        next_is_intersection = bool(self.intersections[next_y, next_x])
        current_is_intersection = bool(self.intersections[y, x])

        if next_is_intersection and not current_is_intersection:
            return self._direction_has_green(direction, self.current_light_phase())

        return True

    def step(self) -> SimulationMetrics:
        """Advance the simulation by one timestep and return updated metrics."""

        self.step_count += 1
        self.spawn_vehicle()

        active_ids = np.flatnonzero(self.vehicle_active)
        next_occupancy = np.full_like(self.occupancy, -1)
        queue_count = 0

        for vehicle_id in active_ids:
            x = int(self.vehicle_x[vehicle_id])
            y = int(self.vehicle_y[vehicle_id])
            direction = int(self.vehicle_direction[vehicle_id])
            if self.intersections[y, x]:
                direction = self._choose_intersection_direction(direction)
                self.vehicle_direction[vehicle_id] = direction

            next_x, next_y = self._next_position(x, y, direction)

            if not (0 <= next_x < self.config.grid_width and 0 <= next_y < self.config.grid_height):
                self.vehicle_active[vehicle_id] = False
                self.completed_vehicles += 1
                continue

            if not self._can_enter_next_cell(x, y, direction, next_x, next_y):
                next_occupancy[y, x] = vehicle_id
                queue_count += 1
                continue

            if self.occupancy[next_y, next_x] != -1 or next_occupancy[next_y, next_x] != -1:
                next_occupancy[y, x] = vehicle_id
                queue_count += 1
                continue

            self.vehicle_x[vehicle_id] = next_x
            self.vehicle_y[vehicle_id] = next_y
            next_occupancy[next_y, next_x] = vehicle_id

        self.occupancy = next_occupancy

        active_vehicles = int(self.vehicle_active.sum())
        average_queue = queue_count / active_vehicles if active_vehicles else 0.0

        return SimulationMetrics(
            steps=self.step_count,
            active_vehicles=active_vehicles,
            completed_vehicles=self.completed_vehicles,
            average_queue_length=average_queue,
        )
