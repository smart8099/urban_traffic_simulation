"""Plot CPU vs GPU sweep results, one clean figure per sweep dimension.

Each sweep holds two axes fixed and varies one:

- grid           : varies width (= height); spawn_attempts=30, spawn_rate=1.0
- spawn_attempts : varies spawn_attempts; width=512, spawn_rate=1.0
- density        : varies spawn_rate; width=256, spawn_attempts=30

For each sweep the script emits four figures:
- simulation_rate_<sweep>_<tag>.png  (steps / sec)
- throughput_<sweep>_<tag>.png       (vehicle-steps / sec)
- wall_time_<sweep>_<tag>.png        (seconds per 1000 steps)
- speedup_<sweep>_<tag>.png          (GPU steps/sec ÷ CPU steps/sec)

Usage:
    python scripts/plot_sweeps.py
    python scripts/plot_sweeps.py --tag final
    python scripts/plot_sweeps.py --sweeps grid,density
"""

from __future__ import annotations

import argparse
import glob
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CONFIG_KEYS = ["width", "height", "spawn_attempts", "spawn_rate"]


SWEEP_DEFS = {
    "grid": {
        "x_col": "width",
        "x_label": "Grid size (cells per side)",
        "title_suffix": "vs grid size",
        "x_log": True,
        "filter": {"spawn_attempts": 30, "spawn_rate": 1.0},
    },
    "spawn_attempts": {
        "x_col": "spawn_attempts",
        "x_label": "Spawn attempts per step",
        "title_suffix": "vs spawn attempts (512x512)",
        "x_log": True,
        "filter": {"width": 512, "spawn_rate": 1.0},
    },
    "density": {
        "x_col": "spawn_rate",
        "x_label": "Spawn rate (density knob)",
        "title_suffix": "vs spawn rate (256x256)",
        "x_log": False,
        "filter": {"width": 256, "spawn_attempts": 30},
    },
}


METRIC_DEFS = {
    "simulation_rate": {
        "y_col": "steps_per_second",
        "y_label": "Simulation rate (steps / sec)",
        "y_log": True,
    },
    "throughput": {
        "y_col": "vehicle_steps_per_second",
        "y_label": "Throughput (vehicle-steps / sec)",
        "y_log": True,
    },
    "wall_time": {
        "y_col": "elapsed_seconds",
        "y_label": "Elapsed time (sec, 1000 steps)",
        "y_log": True,
    },
}


def load_csvs(pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No CSVs matched pattern: {pattern}")
    print(f"Loaded {len(paths)} file(s) for {pattern}:")
    for p in paths:
        print(f"  - {p}")
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


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
    )


def filter_sweep(df: pd.DataFrame, fixed: dict) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    for col, val in fixed.items():
        mask &= df[col] == val
    return df.loc[mask].copy()


def save_figure(fig, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_metric(
    df: pd.DataFrame,
    sweep_name: str,
    metric_name: str,
    output_dir: Path,
    tag: str,
) -> None:
    sweep = SWEEP_DEFS[sweep_name]
    metric = METRIC_DEFS[metric_name]
    sub = filter_sweep(df, sweep["filter"]).sort_values(sweep["x_col"])
    if sub.empty:
        print(f"  skip {metric_name} {sweep_name}: no rows match {sweep['filter']}")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"cpu": "tab:blue", "gpu": "tab:orange"}
    for backend, group in sub.groupby("backend"):
        ax.plot(
            group[sweep["x_col"]],
            group[metric["y_col"]],
            marker="o",
            label=backend.upper(),
            color=colors.get(backend),
        )
    if sweep["x_log"]:
        ax.set_xscale("log")
    if metric["y_log"]:
        ax.set_yscale("log")
    ax.set_xlabel(sweep["x_label"])
    ax.set_ylabel(metric["y_label"])
    ax.set_title(f"{metric['y_label']} {sweep['title_suffix']}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    save_figure(fig, output_dir, f"{metric_name}_{sweep_name}_{tag}")


def plot_speedup(
    df: pd.DataFrame,
    sweep_name: str,
    output_dir: Path,
    tag: str,
) -> None:
    sweep = SWEEP_DEFS[sweep_name]
    sub = filter_sweep(df, sweep["filter"])
    if sub.empty:
        print(f"  skip speedup {sweep_name}: no rows match {sweep['filter']}")
        return

    pivot = sub.pivot_table(
        index=CONFIG_KEYS,
        columns="backend",
        values="steps_per_second",
    ).dropna()
    if pivot.empty:
        print(f"  skip speedup {sweep_name}: no paired CPU/GPU rows")
        return

    ratio = (pivot["gpu"] / pivot["cpu"]).rename("speedup").reset_index()
    ratio = ratio.sort_values(sweep["x_col"])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        ratio[sweep["x_col"]],
        ratio["speedup"],
        marker="o",
        color="tab:purple",
    )
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="break-even")
    if sweep["x_log"]:
        ax.set_xscale("log")
    ax.set_xlabel(sweep["x_label"])
    ax.set_ylabel("Speedup (GPU steps/sec ÷ CPU steps/sec)")
    ax.set_title(f"GPU speedup {sweep['title_suffix']}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    save_figure(fig, output_dir, f"speedup_{sweep_name}_{tag}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu-csv",
        default="logs/sweeps/sweep_cpu_*.csv",
        help="Path or glob for CPU sweep CSV(s); multiple files are merged.",
    )
    parser.add_argument(
        "--gpu-csv",
        default="logs/sweeps/sweep_gpu_*.csv",
        help="Path or glob for GPU sweep CSV(s); multiple files are merged.",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/sweeps",
        help="Directory to write PNGs to.",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Suffix appended to each filename. Defaults to a UTC-style timestamp.",
    )
    parser.add_argument(
        "--sweeps",
        default=",".join(SWEEP_DEFS.keys()),
        help=f"Comma-separated subset of {list(SWEEP_DEFS.keys())}.",
    )
    args = parser.parse_args()

    df_cpu = load_csvs(args.cpu_csv)
    df_gpu = load_csvs(args.gpu_csv)
    df = aggregate(pd.concat([df_cpu, df_gpu], ignore_index=True))

    output_dir = Path(args.output_dir)
    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    requested_sweeps = [s.strip() for s in args.sweeps.split(",") if s.strip()]

    for sweep_name in requested_sweeps:
        if sweep_name not in SWEEP_DEFS:
            print(f"Unknown sweep '{sweep_name}', skipping. "
                  f"Known: {list(SWEEP_DEFS.keys())}")
            continue
        print(f"\n=== sweep: {sweep_name} ===")
        for metric_name in METRIC_DEFS:
            plot_metric(df, sweep_name, metric_name, output_dir, tag)
        plot_speedup(df, sweep_name, output_dir, tag)


if __name__ == "__main__":
    main()
