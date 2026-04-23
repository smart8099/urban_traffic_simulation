from dataclasses import dataclass


@dataclass(slots=True)
class SimulationConfig:
    """Stores the configurable parameters used by the simulation and UI."""

    grid_width: int = 64
    grid_height: int = 64
    road_spacing: int = 8
    max_vehicles: int = 500
    spawn_rate: float = 0.6
    spawn_attempts_per_step: int = 2
    light_cycle_steps: int = 20
    turn_probability: float = 0.2
    cell_size: int = 10
    fps: int = 30
    random_seed: int = 7
