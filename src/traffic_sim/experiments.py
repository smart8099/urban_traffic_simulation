from __future__ import annotations

from dataclasses import asdict
from time import perf_counter

from .config import SimulationConfig
from .simulation_cpu import CPUTrafficSimulation
from .simulation_gpu import GPUTrafficSimulation


def run_benchmark(backend: str, steps: int, config: SimulationConfig) -> dict:
    """Run a fixed-length benchmark and return timing plus final metrics."""

    if backend == "cpu":
        simulation = CPUTrafficSimulation(config)
    elif backend == "gpu":
        simulation = GPUTrafficSimulation(config)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    start = perf_counter()
    final_metrics = None
    for _ in range(steps):
        final_metrics = simulation.step()
    elapsed = perf_counter() - start

    result = asdict(final_metrics) if final_metrics is not None else {}
    result["backend"] = backend
    result["elapsed_seconds"] = elapsed
    result["steps_per_second"] = steps / elapsed if elapsed > 0 else 0.0
    return result
