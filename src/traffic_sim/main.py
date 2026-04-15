from __future__ import annotations

import argparse

from .config import SimulationConfig
from .experiments import run_benchmark
from .simulation_cpu import CPUTrafficSimulation
from .simulation_gpu import GPUTrafficSimulation


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for simulation, visualization, and benchmarking."""

    parser = argparse.ArgumentParser(description="GPU-accelerated traffic simulation")
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--max-vehicles", type=int, default=500)
    parser.add_argument("--road-spacing", type=int, default=8)
    parser.add_argument("--spawn-rate", type=float, default=0.6)
    parser.add_argument("--spawn-attempts", type=int, default=2)
    parser.add_argument("--turn-probability", type=float, default=0.2)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> SimulationConfig:
    """Construct a simulation configuration object from parsed CLI arguments."""

    return SimulationConfig(
        grid_width=args.width,
        grid_height=args.height,
        max_vehicles=args.max_vehicles,
        road_spacing=args.road_spacing,
        spawn_rate=args.spawn_rate,
        spawn_attempts_per_step=args.spawn_attempts,
        turn_probability=args.turn_probability,
    )


def create_simulation(backend: str, config: SimulationConfig):
    """Create either the CPU or GPU simulation backend."""

    if backend == "cpu":
        return CPUTrafficSimulation(config)
    return GPUTrafficSimulation(config)


def run_visual_demo(args: argparse.Namespace, config: SimulationConfig) -> None:
    """Run the pygame visualization loop for the selected simulation backend."""

    from .visualization import TrafficVisualizer

    simulation = create_simulation(args.backend, config)
    visualizer = TrafficVisualizer(config)

    try:
        for _ in range(args.steps):
            metrics = simulation.step()
            if not visualizer.draw(simulation, metrics):
                break
    finally:
        visualizer.close()


def main() -> None:
    """Dispatch the application into benchmark, headless, or visualization mode."""

    args = parse_args()
    config = build_config(args)

    if args.benchmark:
        result = run_benchmark(args.backend, args.steps, config)
        for key, value in result.items():
            print(f"{key}: {value}")
        return

    if args.headless:
        simulation = create_simulation(args.backend, config)
        metrics = None
        for _ in range(args.steps):
            metrics = simulation.step()
        if metrics is not None:
            print(metrics)
        return

    run_visual_demo(args, config)


if __name__ == "__main__":
    main()
