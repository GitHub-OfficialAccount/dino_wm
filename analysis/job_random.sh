#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=30
#SBATCH --job-name=iwmRand
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -x
source /tc1apps/anaconda3/etc/profile.d/conda.sh
conda activate iwm
cd ~/iwm/dino_wm
export DATASET_DIR=$HOME/iwm/data/database
export WANDB_MODE=offline PYTHONPATH=. PYTHONUNBUFFERED=1
MODEL=$HOME/iwm/data/checkpoints/outputs/outputs/wall_single
python analysis/run_random.py --model_path $MODEL --n_evals 50 --R 80 --record_path $HOME/iwm/records/random_R80.npz --device cpu
echo "RANDOM_DONE $?"
