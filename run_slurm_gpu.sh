#!/bin/bash
### Sets the job's name.
#SBATCH --job-name=traffic_sim_gpu

### Sets the job's output file and path.
#SBATCH --output=/home/dmasamba01/parallel_computing/urban_traffic_simulation/logs/slurm_success/output_gpu.out.%j

### Sets the job's error output file and path.
#SBATCH --error=/home/dmasamba01/parallel_computing/urban_traffic_simulation/logs/slurm_errors/errors_gpu.err.%j

### Requested number of nodes for this job.
#SBATCH -N 1

### Requested partition (GPU partition).
#SBATCH -p sxmq

### Requested number of GPUs (single-GPU CuPy implementation for now).
#SBATCH --gres=gpu:1

### Requested number of CPUs.
#SBATCH --cpus-per-task=4

### Limit on the total run time of the job allocation.
#SBATCH --time=01:00:00

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

### Load CUDA module.
module load cuda/12.3

### Activate conda environment.
echo "Activating traffic_sim environment"
source /home/dmasamba01/anaconda3/etc/profile.d/conda.sh
conda activate /home/dmasamba01/anaconda3/envs/traffic_sim

### Move to project root.
cd /home/dmasamba01/parallel_computing/urban_traffic_simulation

### Launch benchmark.
echo "Running GPU benchmark"
PYTHONPATH=src python -m traffic_sim.main \
  --backend gpu \
  --width 512 \
  --height 512 \
   --max-vehicles 20000 \
  --steps 2000 \
  --spawn-rate 0.8 \
  --spawn-attempts 10 \
  --headless \
  --benchmark

### Record a small visualization GIF for this run (headless, dummy SDL driver).
# echo "Recording GPU visualization GIF"
# PYTHONPATH=src python -m traffic_sim.main \
#   --backend gpu \
#   --width 64 \
#   --height 64 \
#   --max-vehicles 500 \
#   --steps 300 \
#   --spawn-rate 0.8 \
#   --spawn-attempts 3 \
#   --headless \
#   --save-gif \
#   --gif-tag slurm \
#   --gif-fps 15

echo "Deactivating environment"
conda deactivate

echo "Done."
echo "End time  : $(date)"
