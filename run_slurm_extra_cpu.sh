#!/bin/bash
#SBATCH --job-name=traffic_sim_extra_cpu
#SBATCH --output=/home/dmasamba01/parallel_computing/urban_traffic_simulation/logs/slurm_success/extra_cpu.out.%j
#SBATCH --error=/home/dmasamba01/parallel_computing/urban_traffic_simulation/logs/slurm_errors/extra_cpu.err.%j
#SBATCH -N 1
#SBATCH -p sxmq
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

# Fills the gaps needed to update the presentation:
#   [1] Variance trials over the existing recipe (3 repeats).
#   [2] Tiny-grid runs (16,32) to make the GPU-overhead-at-small-scale claim defensible.
#   [3] Density (spawn_rate) sweep at 256x256 to populate slide 11.

echo "Start time: $(date)"

source /home/dmasamba01/anaconda3/etc/profile.d/conda.sh
conda activate /home/dmasamba01/anaconda3/envs/traffic_sim

cd /home/dmasamba01/parallel_computing/urban_traffic_simulation

RUN_ID=${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}
CSV=logs/sweeps/sweep_cpu_extra_${RUN_ID}.csv
echo "Writing results to $CSV"

echo "[1/3] CPU variance trials — repeat the 512x512 spawn-attempts sweep 3x"
PYTHONPATH=src python scripts/sweep.py \
  --backend cpu --output "$CSV" \
  --grid-list 512 \
  --spawn-attempts-list 1,3,10,30,100 \
  --steps 1000 --max-vehicles 100000 --spawn-rate 1.0 \
  --repeats 3

echo "[2/3] CPU tiny-grid sweep at spawn_attempts=30"
PYTHONPATH=src python scripts/sweep.py \
  --backend cpu --output "$CSV" \
  --grid-list 16,32 \
  --spawn-attempts-list 30 \
  --steps 1000 --max-vehicles 100000 --spawn-rate 1.0 \
  --repeats 3

echo "[3/3] CPU density sweep at 256x256, spawn_attempts=30"
PYTHONPATH=src python scripts/sweep.py \
  --backend cpu --output "$CSV" \
  --grid-list 256 \
  --spawn-attempts-list 30 \
  --spawn-rate-list 0.3,0.6,0.9,1.0 \
  --steps 1000 --max-vehicles 100000 \
  --repeats 3

echo "Done."
echo "End time  : $(date)"
