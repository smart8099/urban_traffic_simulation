"""Plot CPU vs GPU sweep results.

Reads the two backend CSVs produced by scripts/sweep.py and emits a 2x2 figure:
- throughput (vehicle-steps/sec) vs active vehicles
- simulation rate (steps/sec) vs active vehicles
- GPU/CPU speedup vs active vehicles
- elapsed wall time vs active vehicles

Usage:
    PYTHONPATH=src python scripts/plot_sweeps.py \
        --cpu-csv logs/sweeps/sweep_cpu.csv \
        --gpu-csv logs/sweeps/sweep_gpu.csv \
        --output logs/sweeps/sweep_plot.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CONFIG_KEYS = ["width", "height", "spawn_attempts"]


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Average duplicate (backend, config) rows so each config plots as one point."""
    return (
        df.groupby(["backend", *CONFIG_KEYS], as_index=False)
        .agg(
            elapsed_seconds=("elapsed_seconds", "mean"),
            steps_per_second=("steps_per_second", "mean"),
            active_vehicles=("active_vehicles", "mean"),
            vehicle_steps_per_second=("vehicle_steps_per_second", "mean"),
        )
        .sort_values("active_vehicles")
    )


def compute_speedup(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(
        index=CONFIG_KEYS,
        columns="backend",
        values=["steps_per_second", "active_vehicles"],
    )
    paired = pivot.dropna()
    speedup = paired[("steps_per_second", "gpu")] / paired[("steps_per_second", "cpu")]
    active = (
        paired[("active_vehicles", "gpu")] + paired[("active_vehicles", "cpu")]
    ) / 2.0
    out = pd.DataFrame({"active_vehicles": active.values, "speedup": speedup.values})
    return out.sort_values("active_vehicles")


def line_plot(ax, df: pd.DataFrame, y_col: str, ylabel: str, title: str, log_y: bool):
    colors = {"cpu": "tab:blue", "gpu": "tab:orange"}
    for backend, group in df.groupby("backend"):
        ax.plot(
            group["active_vehicles"],
            group[y_col],
            marker="o",
            label=backend.upper(),
            color=colors.get(backend, None),
        )
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Active vehicles")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-csv", default="logs/sweeps/sweep_cpu.csv")
    parser.add_argument("--gpu-csv", default="logs/sweeps/sweep_gpu.csv")
    parser.add_argument("--output", default="logs/sweeps/sweep_plot.png")
    args = parser.parse_args()

    df_cpu = pd.read_csv(args.cpu_csv)
    df_gpu = pd.read_csv(args.gpu_csv)
    df = pd.concat([df_cpu, df_gpu], ignore_index=True)
    df = aggregate(df)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    line_plot(
        axes[0, 0],
        df,
        "vehicle_steps_per_second",
        "Throughput (vehicle-steps / sec)",
        "Throughput vs workload size",
        log_y=True,
    )
    line_plot(
        axes[0, 1],
        df,
        "steps_per_second",
        "Simulation rate (steps / sec)",
        "Simulation rate vs workload size",
        log_y=True,
    )

    speedup = compute_speedup(df)
    ax = axes[1, 0]
    ax.plot(
        speedup["active_vehicles"],
        speedup["speedup"],
        marker="o",
        color="tab:purple",
    )
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="break-even")
    ax.set_xscale("log")
    ax.set_xlabel("Active vehicles")
    ax.set_ylabel("Speedup (GPU steps/sec ÷ CPU steps/sec)")
    ax.set_title("GPU speedup over CPU")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    line_plot(
        axes[1, 1],
        df,
        "elapsed_seconds",
        "Elapsed time (sec, 1000 steps)",
        "Wall time per 1000 steps",
        log_y=True,
    )

    fig.suptitle("Urban Traffic Simulation — CPU vs GPU sweep results", fontsize=14)
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
