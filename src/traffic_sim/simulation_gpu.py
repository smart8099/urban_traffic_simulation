from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .config import SimulationConfig
from .models import SimulationMetrics
from .simulation_cpu import CPUTrafficSimulation

try:
    import cupy as cp
except ImportError:  # pragma: no cover - depends on local CUDA setup
    cp = None


class GPUTrafficSimulation:
    """Expose a CuPy-backed interface for the traffic simulation state."""

    def __init__(self, config: SimulationConfig) -> None:
        """Initialize GPU arrays and seed them from the CPU reference state."""

        if cp is None:
            raise RuntimeError("CuPy is not installed or not available in this environment.")

        self.config = config
        self.cpu_reference = CPUTrafficSimulation(config)
        self.grid = cp.asarray(self.cpu_reference.grid)
        self.occupancy = cp.asarray(self.cpu_reference.occupancy)
        self.vehicle_active = cp.asarray(self.cpu_reference.vehicle_active)
        self.vehicle_x = cp.asarray(self.cpu_reference.vehicle_x)
        self.vehicle_y = cp.asarray(self.cpu_reference.vehicle_y)
        self.vehicle_direction = cp.asarray(self.cpu_reference.vehicle_direction)

    def _sync_from_cpu(self) -> None:
        """Refresh GPU arrays from the CPU reference implementation state."""

        self.occupancy = cp.asarray(self.cpu_reference.occupancy)
        self.vehicle_active = cp.asarray(self.cpu_reference.vehicle_active)
        self.vehicle_x = cp.asarray(self.cpu_reference.vehicle_x)
        self.vehicle_y = cp.asarray(self.cpu_reference.vehicle_y)
        self.vehicle_direction = cp.asarray(self.cpu_reference.vehicle_direction)

    def step(self) -> SimulationMetrics:
        """Advance one step and mirror the resulting state onto the GPU arrays."""

        metrics = self.cpu_reference.step()
        self._sync_from_cpu()
        return SimulationMetrics(**asdict(metrics))

    @property
    def step_count(self) -> int:
        """Return the number of simulated steps completed so far."""

        return self.cpu_reference.step_count

    def current_light_phase(self):
        """Return the traffic-light phase from the reference simulation."""

        return self.cpu_reference.current_light_phase()

    def occupancy_cpu(self) -> np.ndarray:
        """Copy the current GPU occupancy grid back to a NumPy array."""

        return cp.asnumpy(self.occupancy)

    def grid_cpu(self) -> np.ndarray:
        """Copy the static GPU grid layout back to a NumPy array."""

        return cp.asnumpy(self.grid)
