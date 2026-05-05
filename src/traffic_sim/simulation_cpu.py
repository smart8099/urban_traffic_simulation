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
        """Advance the simulation by one timestep using an explicit 3-phase pipeline."""

        self.step_count += 1
        self.spawn_vehicle()

        active_ids = np.flatnonzero(self.vehicle_active)
        if active_ids.size == 0:
            self.occupancy = np.full_like(self.occupancy, -1)
            return SimulationMetrics(
                steps=self.step_count,
                active_vehicles=0,
                completed_vehicles=self.completed_vehicles,
                average_queue_length=0.0,
            )

        width = self.config.grid_width
        height = self.config.grid_height
        phase = self.current_light_phase()
        n = active_ids.size

        # Phase 1: per-vehicle intents (side-effect free).
        intent_target = np.full(n, -1, dtype=np.int64)
        wants_move = np.zeros(n, dtype=bool)
        exits = np.zeros(n, dtype=bool)
        new_direction = np.zeros(n, dtype=np.int8)

        for i in range(n):
            vid = int(active_ids[i])
            x = int(self.vehicle_x[vid])
            y = int(self.vehicle_y[vid])
            d = int(self.vehicle_direction[vid])

            if self.intersections[y, x]:
                d = self._choose_intersection_direction(d)
            new_direction[i] = d

            nx, ny = self._next_position(x, y, d)

            if not (0 <= nx < width and 0 <= ny < height):
                exits[i] = True
                continue

            if not self._can_enter_next_cell(x, y, d, nx, ny):
                continue  # blocked by light, stays put

            intent_target[i] = ny * width + nx
            wants_move[i] = True

        self.vehicle_direction[active_ids] = new_direction

        # Phase 2: conflict resolution — lowest active_ids index wins its target cell.
        # active_ids is sorted ascending, so iterating in order gives "lowest vehicle id first".
        cell_winner: dict[int, int] = {}
        for i in range(n):
            if wants_move[i]:
                t = int(intent_target[i])
                if t not in cell_winner:
                    cell_winner[t] = i

        wins_conflict = np.zeros(n, dtype=bool)
        for i in range(n):
            if wants_move[i] and cell_winner[int(intent_target[i])] == i:
                wins_conflict[i] = True

        # Phase 3: chain-move fixed-point resolution.
        # A vehicle will_move iff it wins conflict AND (target is empty OR occupant will_move).
        id_to_local: dict[int, int] = {int(vid): i for i, vid in enumerate(active_ids)}
        will_move = np.zeros(n, dtype=bool)

        # Seed: vehicles whose target is currently empty.
        for i in range(n):
            if wins_conflict[i]:
                t = int(intent_target[i])
                ty, tx = divmod(t, width)
                if self.occupancy[ty, tx] == -1:
                    will_move[i] = True

        # Propagate along chains until no changes.
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if will_move[i] or not wins_conflict[i]:
                    continue
                t = int(intent_target[i])
                ty, tx = divmod(t, width)
                occ = int(self.occupancy[ty, tx])
                if occ == -1:
                    continue
                occ_idx = id_to_local.get(occ, -1)
                if occ_idx >= 0 and will_move[occ_idx]:
                    will_move[i] = True
                    changed = True

        # Apply exits.
        exit_ids = active_ids[exits]
        if exit_ids.size > 0:
            self.vehicle_active[exit_ids] = False
            self.completed_vehicles += int(exit_ids.size)

        # Apply moves and rebuild occupancy.
        new_occupancy = np.full_like(self.occupancy, -1)
        queue_count = 0
        for i in range(n):
            if exits[i]:
                continue
            vid = int(active_ids[i])
            if will_move[i]:
                t = int(intent_target[i])
                ty, tx = divmod(t, width)
                self.vehicle_x[vid] = tx
                self.vehicle_y[vid] = ty
                new_occupancy[ty, tx] = vid
            else:
                x = int(self.vehicle_x[vid])
                y = int(self.vehicle_y[vid])
                new_occupancy[y, x] = vid
                queue_count += 1

        self.occupancy = new_occupancy

        active_vehicles = int(self.vehicle_active.sum())
        average_queue = queue_count / active_vehicles if active_vehicles else 0.0

        return SimulationMetrics(
            steps=self.step_count,
            active_vehicles=active_vehicles,
            completed_vehicles=self.completed_vehicles,
            average_queue_length=average_queue,
        )
