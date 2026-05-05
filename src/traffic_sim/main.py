from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

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
    parser.add_argument(
        "--save-gif",
        action="store_true",
        help="Record the visualization to outputs/gifs/<run_name>/run.gif.",
    )
    parser.add_argument(
        "--gif-tag",
        type=str,
        default=None,
        help="Optional suffix appended to the auto-generated run folder name.",
    )
    parser.add_argument(
        "--gif-fps",
        type=int,
        default=10,
        help="Playback frame rate for the saved GIF.",
    )
    return parser.parse_args()


def resolve_gif_path(args: argparse.Namespace) -> Path:
    """Build a unique outputs/gifs/<run_name>/run.gif path for this run."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [args.backend, timestamp]
    if args.gif_tag:
        parts.append(args.gif_tag)
    run_name = "_".join(parts)
    return Path("outputs") / "gifs" / run_name / "run.gif"


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

    gif_path = str(resolve_gif_path(args)) if args.save_gif else None
    simulation = create_simulation(args.backend, config)
    visualizer = TrafficVisualizer(
        config,
        save_gif_path=gif_path,
        gif_fps=args.gif_fps,
        headless=args.headless,
    )

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

    if args.headless and not args.save_gif:
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
