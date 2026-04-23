# GPU-Accelerated Urban Traffic Simulation

This project implements a simplified urban traffic simulator for CSCI 6356. It is inspired by the 2024 paper "Large scale multi-GPU based parallel traffic simulation for accelerated traffic assignment and propagation" by Xuan Jiang, Raja Sengupta, James Demmel, and Samuel Williams, but intentionally reduces the scope to a manageable class-project implementation.

The simulator models vehicle movement on a 2D grid road network with intersections and traffic lights. The project is organized around three goals:

- build a CPU baseline using `NumPy`
- build a matching GPU implementation using `CuPy`
- visualize traffic behavior with `pygame`

## Project Status

This repository contains a working CPU + GPU simulation:

- package layout
- scenario configuration
- grid generation
- CPU simulation with boundary spawning, lane-aligned movement, traffic lights, and basic queueing
- **Fully vectorized CuPy GPU implementation** — parallel vehicle spawning, turn resolution via GPU turn tables, traffic-light gating, and conflict-free movement using `cupyx.scatter_min`
- `pygame` visualization entry point
- experiment runner for timing and metric collection
- SLURM job scripts for running CPU and GPU benchmarks on the cluster (`run_slurm_cpu.sh`, `run_slurm_gpu.sh`)

## Tech Stack

- Python 3.11+
- `numpy` for CPU-side data structures and updates
- `cupy` for GPU-side arrays and acceleration when running on a supported CUDA environment
- `pygame` for visualization
- `matplotlib` for plots

## Repository Layout

```text
final_project/
├── README.md
├── requirements.txt
└── src/
    └── traffic_sim/
        ├── __init__.py
        ├── config.py
        ├── experiments.py
        ├── main.py
        ├── models.py
        ├── network.py
        ├── simulation_cpu.py
        ├── simulation_gpu.py
        └── visualization.py
```

## Simulation Model

The design uses a discrete time-stepped simulation.

- The city is represented as a 2D grid.
- Roads run horizontally and vertically at fixed spacing.
- Intersections contain traffic lights with alternating phases.
- Vehicles occupy grid cells and attempt to move one cell per step.
- Vehicles spawn from valid road entry points on the map boundary.
- Vehicles stop for red lights and occupied cells.
- Vehicles may turn at intersections with a configurable turn probability.
- Metrics will include throughput, travel time, and queue length.

This keeps the model simple enough for a CPU/GPU comparison while still showing congestion behavior.

## How To Read The Simulation

The current visualizer uses a simple color map:

- light background: empty space, not part of the road network
- dark gray cells: road segments
- darker gray cells: intersections
- red blocks: vehicles
- green circle on an intersection: that direction currently has the green light
- red circle on an intersection: that direction is currently stopped

Important detail: the light phase is global in the current prototype.

- when the phase is `horizontal green`, left-right traffic can enter intersections
- when the phase is `vertical green`, up-down traffic can enter intersections

The bottom bar shows:

- `step`: current timestep
- `active`: vehicles currently on the map
- `completed`: vehicles that reached the boundary and exited
- `avg_queue`: fraction of active vehicles blocked during the latest step

## What You'll See

When you run the visual demo, the window shows:

- a light background for non-road space
- dark gray horizontal and vertical road lines
- darker gray intersection cells
- red rectangles moving along the road network as vehicles
- colored traffic-light markers at intersections
- a bottom status panel with the current step count and traffic metrics

As the simulation runs, vehicles enter from the map boundary, move through the grid, stop when blocked, and form visible queues near intersections when demand increases.

## Installation

This project uses a **conda environment**. Make sure `conda` (or `mamba`) is available before proceeding.

Create and activate the environment:

```bash
conda create -n traffic_sim python=3.11 -y
conda activate traffic_sim
pip install -r requirements.txt
```

`requirements.txt` installs the CPU and visualization stack. `CuPy` must be installed separately because the right package depends on the CUDA version on your system. Check the CUDA version first:

```bash
nvcc --version   # or: nvidia-smi
```

Then install the matching CuPy build, for example:

```bash
pip install cupy-cuda12x   # CUDA 12.x
# pip install cupy-cuda11x # CUDA 11.x
```

## Run the Visual Demo

From the repository root:

```bash
conda activate traffic_sim
PYTHONPATH=src python -m traffic_sim.main --backend cpu --steps 500
```

Try the GPU backend once `CuPy` is installed:

```bash
conda activate traffic_sim
PYTHONPATH=src python -m traffic_sim.main --backend gpu --steps 500
```

## Useful Commands

Run a headless simulation and print final metrics:

```bash
PYTHONPATH=src python -m traffic_sim.main --backend cpu --steps 200 --headless
```

Run a denser traffic scenario:

```bash
PYTHONPATH=src python -m traffic_sim.main --backend cpu --steps 500 --spawn-rate 0.8 --spawn-attempts 3
```

Run a smaller grid for easier visualization:

```bash
PYTHONPATH=src python -m traffic_sim.main --backend cpu --width 48 --height 48 --steps 500
```

Run the benchmark mode:

```bash
PYTHONPATH=src python -m traffic_sim.main --backend cpu --steps 500 --benchmark
```

## Current CLI Options

- `--backend {cpu,gpu}` selects the simulation backend
- `--steps` sets the number of simulation timesteps
- `--width` and `--height` set the grid dimensions
- `--max-vehicles` sets the vehicle pool size
- `--road-spacing` controls the spacing between road lines
- `--spawn-rate` controls the probability of each spawn attempt succeeding
- `--spawn-attempts` controls how many spawn attempts happen each step
- `--turn-probability` controls how often vehicles turn at intersections
- `--headless` runs without opening the `pygame` window
- `--benchmark` prints timing and summary metrics

## Running on the Cluster (SLURM)

Two SLURM job scripts are provided at the project root.

Submit the CPU benchmark:

```bash
sbatch run_slurm_cpu.sh
```

Submit the GPU benchmark (requests one GPU):

```bash
sbatch run_slurm_gpu.sh
```

Output and error files land in `logs/slurm_success/` and `logs/slurm_errors/` respectively.  Both scripts load CUDA 12.3, activate the `traffic_sim` conda environment, and run a 512 × 512 grid with 20 000 vehicles for 500 steps.

## Immediate Development Plan

1. Track richer metrics such as travel time, throughput by window, and intersection queue length.
2. Add benchmark scenarios for different grid sizes and traffic densities.
3. Generate CPU vs. GPU speedup plots for the final report.

## Mapping to the Paper

The paper motivates the project in three ways:

- traffic simulation is naturally time-stepped
- per-vehicle updates are highly parallel
- large-scale speedup comes from parallel hardware

This class project does not attempt the full paper scope such as:

- regional-scale road graphs
- route assignment engines
- multi-GPU partitioning
- cloud deployment

Instead, it focuses on a smaller but defensible demonstration of GPU acceleration on repeated vehicle updates.

## Notes for the Report

When writing the final report, emphasize:

- the simplified grid-based traffic model
- the identical logic used for CPU and GPU backends
- runtime scaling with traffic density and grid size
- why the update step is suitable for parallel execution

## Reference

- Paper: https://www.sciencedirect.com/science/article/pii/S0968090X24003942
- Reference repository: https://github.com/Xuan-1998/LPSim
