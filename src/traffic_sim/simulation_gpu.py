from __future__ import annotations

import numpy as np

from .config import SimulationConfig
from .models import Direction, LightPhase, SimulationMetrics
from .network import build_grid_network, intersection_mask

try:
    import cupy as cp
    import cupyx
except ImportError:  # pragma: no cover - depends on local CUDA setup
    cp = None
    cupyx = None


_DX = np.array([0, 0, -1, 1], dtype=np.int32)  # UP, DOWN, LEFT, RIGHT
_DY = np.array([-1, 1, 0, 0], dtype=np.int32)

# turn_table[current_direction] = [forward, left_turn, right_turn]
_TURN_TABLE = np.array(
    [
        [int(Direction.UP), int(Direction.LEFT), int(Direction.RIGHT)],
        [int(Direction.DOWN), int(Direction.RIGHT), int(Direction.LEFT)],
        [int(Direction.LEFT), int(Direction.DOWN), int(Direction.UP)],
        [int(Direction.RIGHT), int(Direction.UP), int(Direction.DOWN)],
    ],
    dtype=np.int32,
)


class GPUTrafficSimulation:
    """Vectorized CuPy-backed traffic simulation."""

    def __init__(self, config: SimulationConfig) -> None:
        if cp is None:
            raise RuntimeError("CuPy is not installed or not available in this environment.")

        self.config = config
        self.host_rng = np.random.default_rng(config.random_seed)
        self.device_rng = cp.random.default_rng(config.random_seed)

        grid_np = build_grid_network(config)
        intersections_np = intersection_mask(grid_np)

        self.grid = cp.asarray(grid_np)
        self.intersections = cp.asarray(intersections_np)
        self.occupancy = cp.full(self.grid.shape, -1, dtype=cp.int32)

        self.entry_points = cp.asarray(
            self._build_entry_points(grid_np), dtype=cp.int32
        )

        self.vehicle_active = cp.zeros(config.max_vehicles, dtype=cp.bool_)
        self.vehicle_x = cp.zeros(config.max_vehicles, dtype=cp.int32)
        self.vehicle_y = cp.zeros(config.max_vehicles, dtype=cp.int32)
        self.vehicle_direction = cp.zeros(config.max_vehicles, dtype=cp.int8)

        self._dx = cp.asarray(_DX)
        self._dy = cp.asarray(_DY)
        self._turn_table = cp.asarray(_TURN_TABLE)

        self.step_count = 0
        self.completed_vehicles = 0

    def _build_entry_points(self, grid_np: np.ndarray) -> np.ndarray:
        entry_points: list[tuple[int, int, int]] = []
        height, width = grid_np.shape

        for y in range(height):
            if grid_np[y, 0] in (1, 3):
                entry_points.append((0, y, int(Direction.RIGHT)))
            if grid_np[y, width - 1] in (1, 3):
                entry_points.append((width - 1, y, int(Direction.LEFT)))

        for x in range(width):
            if grid_np[0, x] in (2, 3):
                entry_points.append((x, 0, int(Direction.DOWN)))
            if grid_np[height - 1, x] in (2, 3):
                entry_points.append((x, height - 1, int(Direction.UP)))

        return np.asarray(entry_points, dtype=np.int32)

    def current_light_phase(self) -> LightPhase:
        cycle_position = (self.step_count // self.config.light_cycle_steps) % 2
        return LightPhase(cycle_position)

    def spawn_vehicle(self) -> None:
        if int(self.entry_points.shape[0]) == 0:
            return

        available = cp.flatnonzero(~self.vehicle_active)
        if available.size == 0:
            return

        n_entries = int(self.entry_points.shape[0])
        perm = self.host_rng.permutation(n_entries)
        shuffled = cp.asnumpy(self.entry_points)[perm]

        attempts = min(self.config.spawn_attempts_per_step, n_entries)
        draws = self.host_rng.random(attempts)

        available_host = cp.asnumpy(available)
        slot_idx = 0

        for i in range(attempts):
            if slot_idx >= available_host.size:
                break
            if draws[i] > self.config.spawn_rate:
                continue
            x, y, direction = int(shuffled[i, 0]), int(shuffled[i, 1]), int(shuffled[i, 2])
            if int(self.occupancy[y, x].get()) != -1:
                continue

            vehicle_id = int(available_host[slot_idx])
            slot_idx += 1
            self.vehicle_active[vehicle_id] = True
            self.vehicle_x[vehicle_id] = x
            self.vehicle_y[vehicle_id] = y
            self.vehicle_direction[vehicle_id] = direction
            self.occupancy[y, x] = vehicle_id

    def step(self) -> SimulationMetrics:
        self.step_count += 1
        self.spawn_vehicle()

        active_ids = cp.flatnonzero(self.vehicle_active)
        if active_ids.size == 0:
            return SimulationMetrics(
                steps=self.step_count,
                active_vehicles=0,
                completed_vehicles=self.completed_vehicles,
                average_queue_length=0.0,
            )

        vx = self.vehicle_x[active_ids]
        vy = self.vehicle_y[active_ids]
        vd = self.vehicle_direction[active_ids].astype(cp.int32)

        at_intersection = self.intersections[vy, vx].astype(cp.bool_)
        turn_draws = self.device_rng.random(int(active_ids.size))
        will_turn = at_intersection & (turn_draws < self.config.turn_probability)
        choice_draws = self.device_rng.integers(0, 3, size=int(active_ids.size))
        new_direction = self._turn_table[vd, choice_draws]
        vd = cp.where(will_turn, new_direction, vd)
        self.vehicle_direction[active_ids] = vd.astype(cp.int8)

        next_x = vx + self._dx[vd]
        next_y = vy + self._dy[vd]

        width = self.config.grid_width
        height = self.config.grid_height
        out_of_bounds = (
            (next_x < 0) | (next_x >= width) | (next_y < 0) | (next_y >= height)
        )

        safe_nx = cp.where(out_of_bounds, 0, next_x)
        safe_ny = cp.where(out_of_bounds, 0, next_y)

        next_is_intersection = self.intersections[safe_ny, safe_nx].astype(cp.bool_)
        entering_intersection = next_is_intersection & ~at_intersection

        phase = self.current_light_phase()
        is_horizontal_dir = (vd == int(Direction.LEFT)) | (vd == int(Direction.RIGHT))
        if phase == LightPhase.HORIZONTAL_GREEN:
            light_allows = is_horizontal_dir
        else:
            light_allows = ~is_horizontal_dir

        blocked_by_light = entering_intersection & ~light_allows

        target_occupancy = self.occupancy[safe_ny, safe_nx]
        target_empty = (~out_of_bounds) & (target_occupancy == -1)

        can_attempt = (~out_of_bounds) & (~blocked_by_light) & target_empty

        target_1d = safe_ny * width + safe_nx
        sentinel = cp.int32(self.config.max_vehicles + 1)
        intent = cp.full(height * width, sentinel, dtype=cp.int32)

        cand_idx = cp.flatnonzero(can_attempt)
        moved = cp.zeros(int(active_ids.size), dtype=cp.bool_)
        if cand_idx.size > 0:
            cand_targets = target_1d[cand_idx]
            cand_vids = active_ids[cand_idx].astype(cp.int32)
            cupyx.scatter_min(intent, cand_targets, cand_vids)
            winners = intent[cand_targets] == cand_vids
            moved[cand_idx] = winners

        queued = (~out_of_bounds) & (~moved)

        exited_idx = cp.flatnonzero(out_of_bounds)
        if exited_idx.size > 0:
            exiting_vids = active_ids[exited_idx]
            self.vehicle_active[exiting_vids] = False
            self.completed_vehicles += int(exited_idx.size)

        moved_idx = cp.flatnonzero(moved)
        if moved_idx.size > 0:
            moved_vids = active_ids[moved_idx]
            self.vehicle_x[moved_vids] = next_x[moved_idx]
            self.vehicle_y[moved_vids] = next_y[moved_idx]

        new_occupancy = cp.full(self.occupancy.shape, -1, dtype=cp.int32)
        active_after = cp.flatnonzero(self.vehicle_active)
        if active_after.size > 0:
            ax = self.vehicle_x[active_after]
            ay = self.vehicle_y[active_after]
            new_occupancy[ay, ax] = active_after.astype(cp.int32)
        self.occupancy = new_occupancy

        active_vehicles = int(self.vehicle_active.sum())
        queue_count = int(queued.sum())
        average_queue = queue_count / active_vehicles if active_vehicles else 0.0

        return SimulationMetrics(
            steps=self.step_count,
            active_vehicles=active_vehicles,
            completed_vehicles=self.completed_vehicles,
            average_queue_length=average_queue,
        )

    def occupancy_cpu(self) -> np.ndarray:
        return cp.asnumpy(self.occupancy)

    def grid_cpu(self) -> np.ndarray:
        return cp.asnumpy(self.grid)
