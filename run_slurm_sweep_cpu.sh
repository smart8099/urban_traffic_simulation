#!/bin/bash
#SBATCH --job-name=traffic_sim_sweep_cpu
#SBATCH --output=/home/dmasamba01/parallel_computing/urban_traffic_simulation/logs/slurm_success/sweep_cpu.out.%j
#SBATCH --error=/home/dmasamba01/parallel_computing/urban_traffic_simulation/logs/slurm_errors/sweep_cpu.err.%j
#SBATCH -N 1
#SBATCH -p sxmq
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

echo "Start time: $(date)"

source /home/dmasamba01/anaconda3/etc/profile.d/conda.sh
conda activate /home/dmasamba01/anaconda3/envs/traffic_sim

cd /home/dmasamba01/parallel_computing/urban_traffic_simulation

RUN_ID=${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}
CSV=logs/sweeps/sweep_cpu_${RUN_ID}.csv
echo "Writing results to $CSV"

echo "[1/2] CPU spawn-attempts sweep at fixed 512x512 grid"
PYTHONPATH=src python scripts/sweep.py \
  --backend cpu \
  --output "$CSV" \
  --grid-list 512 \
  --spawn-attempts-list 1,3,10,30,100 \
  --steps 1000 \
  --max-vehicles 100000 \
  --spawn-rate 1.0

echo "[2/2] CPU grid-size sweep at fixed spawn-attempts=30"
PYTHONPATH=src python scripts/sweep.py \
  --backend cpu \
  --output "$CSV" \
  --grid-list 64,128,256,512,1024 \
  --spawn-attempts-list 30 \
  --steps 1000 \
  --max-vehicles 100000 \
  --spawn-rate 1.0

echo "Done."
echo "End time  : $(date)"
