#!/bin/bash
#SBATCH --job-name=traffic_sim_sweep_gpu
#SBATCH --output=/home/dmasamba01/parallel_computing/urban_traffic_simulation/logs/slurm_success/sweep_gpu.out.%j
#SBATCH --error=/home/dmasamba01/parallel_computing/urban_traffic_simulation/logs/slurm_errors/sweep_gpu.err.%j
#SBATCH -N 1
#SBATCH -p sxmq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

module load cuda/12.3
source /home/dmasamba01/anaconda3/etc/profile.d/conda.sh
conda activate /home/dmasamba01/anaconda3/envs/traffic_sim

cd /home/dmasamba01/parallel_computing/urban_traffic_simulation

CSV=logs/sweeps/sweep_gpu.csv

echo "[1/2] GPU spawn-attempts sweep at fixed 512x512 grid"
PYTHONPATH=src python scripts/sweep.py \
  --backend gpu \
  --output "$CSV" \
  --grid-list 512 \
  --spawn-attempts-list 1,3,10,30,100 \
  --steps 1000 \
  --max-vehicles 100000 \
  --spawn-rate 1.0

echo "[2/2] GPU grid-size sweep at fixed spawn-attempts=30"
PYTHONPATH=src python scripts/sweep.py \
  --backend gpu \
  --output "$CSV" \
  --grid-list 64,128,256,512,1024 \
  --spawn-attempts-list 30 \
  --steps 1000 \
  --max-vehicles 100000 \
  --spawn-rate 1.0

echo "Done."
echo "End time  : $(date)"
