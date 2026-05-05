"""Benchmark sweep — runs the simulation across a cross-product of grid sizes
and spawn-attempt values, appending one row per config to a CSV.

Usage:
    PYTHONPATH=src python scripts/sweep.py \
        --backend gpu \
        --output logs/sweeps/sweep_gpu.csv \
        --grid-list 64,128,256,512,1024 \
        --spawn-attempts-list 30 \
        --steps 1000

Pass a single value in either list to fix that axis. Pass multiple values in
both to run the full cross-product.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from time import perf_counter

# Make src/ importable when run from the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from traffic_sim.config import SimulationConfig  # noqa: E402
from traffic_sim.simulation_cpu import CPUTrafficSimulation  # noqa: E402
from traffic_sim.simulation_gpu import GPUTrafficSimulation  # noqa: E402


def make_simulation(backend: str, config: SimulationConfig):
    if backend == "cpu":
        return CPUTrafficSimulation(config)
    if backend == "gpu":
        return GPUTrafficSimulation(config)
    raise ValueError(f"Unsupported backend: {backend}")


def run_one(backend: str, steps: int, config: SimulationConfig):
    sim = make_simulation(backend, config)
    start = perf_counter()
    final = None
    for _ in range(steps):
        final = sim.step()
    elapsed = perf_counter() - start
    return final, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cpu", "gpu"], required=True)
    parser.add_argument("--output", required=True, help="CSV output path (appended to)")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--max-vehicles", type=int, default=100000)
    parser.add_argument("--spawn-rate", type=float, default=1.0)
    parser.add_argument(
        "--grid-list",
        default="512",
        help="Comma-separated list of square grid sizes (width = height)",
    )
    parser.add_argument(
        "--spawn-attempts-list",
        default="30",
        help="Comma-separated list of spawn_attempts_per_step values",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=50,
        help="Steps to run once before timing (absorbs GPU JIT cost). 0 to skip.",
    )
    args = parser.parse_args()

    grid_sizes = [int(v) for v in args.grid_list.split(",")]
    spawn_values = [int(v) for v in args.spawn_attempts_list.split(",")]

    if args.warmup_steps > 0:
        warmup_config = SimulationConfig(
            grid_width=grid_sizes[0],
            grid_height=grid_sizes[0],
            max_vehicles=args.max_vehicles,
            spawn_rate=args.spawn_rate,
            spawn_attempts_per_step=spawn_values[0],
        )
        print(f"Warming up {args.backend} for {args.warmup_steps} steps...", flush=True)
        run_one(args.backend, args.warmup_steps, warmup_config)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not output_path.exists()

    fieldnames = [
        "backend",
        "spawn_attempts",
        "steps",
        "width",
        "height",
        "max_vehicles",
        "spawn_rate",
        "elapsed_seconds",
        "steps_per_second",
        "active_vehicles",
        "completed_vehicles",
        "average_queue_length",
        "vehicle_steps_per_second",
    ]

    with output_path.open("a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if new_file:
            writer.writeheader()

        for size in grid_sizes:
            for spawn_attempts in spawn_values:
                config = SimulationConfig(
                    grid_width=size,
                    grid_height=size,
                    max_vehicles=args.max_vehicles,
                    spawn_rate=args.spawn_rate,
                    spawn_attempts_per_step=spawn_attempts,
                )
                print(
                    f"  running {args.backend} grid={size}x{size} "
                    f"spawn_attempts={spawn_attempts}...",
                    flush=True,
                )
                metrics, elapsed = run_one(args.backend, args.steps, config)
                steps_per_second = args.steps / elapsed if elapsed > 0 else 0.0
                row = {
                    "backend": args.backend,
                    "spawn_attempts": spawn_attempts,
                    "steps": args.steps,
                    "width": size,
                    "height": size,
                    "max_vehicles": args.max_vehicles,
                    "spawn_rate": args.spawn_rate,
                    "elapsed_seconds": elapsed,
                    "steps_per_second": steps_per_second,
                    "active_vehicles": metrics.active_vehicles,
                    "completed_vehicles": metrics.completed_vehicles,
                    "average_queue_length": metrics.average_queue_length,
                    "vehicle_steps_per_second": steps_per_second * metrics.active_vehicles,
                }
                writer.writerow(row)
                csv_file.flush()
                print(
                    f"    -> {steps_per_second:.2f} steps/s, "
                    f"active={metrics.active_vehicles}, "
                    f"completed={metrics.completed_vehicles}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
