#!/bin/bash
#SBATCH --job-name=traffic_sim_gifs
#SBATCH --output=/home/dmasamba01/parallel_computing/urban_traffic_simulation/logs/slurm_success/gifs.out.%j
#SBATCH --error=/home/dmasamba01/parallel_computing/urban_traffic_simulation/logs/slurm_errors/gifs.err.%j
#SBATCH -N 1
#SBATCH -p sxmq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00

# Records three GIFs to outputs/gifs/ for slide 13:
#   light    — sparse traffic, easy to follow individual vehicles
#   dense    — visible queues at intersections
#   smallgrid — clearer view of intersections and turning behavior

echo "Start time: $(date)"

module load cuda/12.3
source /home/dmasamba01/anaconda3/etc/profile.d/conda.sh
conda activate /home/dmasamba01/anaconda3/envs/traffic_sim

cd /home/dmasamba01/parallel_computing/urban_traffic_simulation

export SDL_VIDEODRIVER=dummy   # pygame surface without X server

echo "[1/3] light traffic"
PYTHONPATH=src python -m traffic_sim.main \
  --backend gpu --width 64 --height 64 --steps 400 \
  --spawn-rate 0.4 --spawn-attempts 1 \
  --headless --save-gif --gif-tag light --gif-fps 12

echo "[2/3] dense traffic"
PYTHONPATH=src python -m traffic_sim.main \
  --backend gpu --width 64 --height 64 --steps 400 \
  --spawn-rate 0.9 --spawn-attempts 3 \
  --headless --save-gif --gif-tag dense --gif-fps 12

echo "[3/3] small grid for clarity"
PYTHONPATH=src python -m traffic_sim.main \
  --backend gpu --width 32 --height 32 --steps 300 \
  --spawn-rate 0.7 --spawn-attempts 2 \
  --headless --save-gif --gif-tag smallgrid --gif-fps 12

echo "Done."
echo "End time  : $(date)"
