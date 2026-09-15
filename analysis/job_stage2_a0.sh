#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=120
#SBATCH --job-name=iwmS2R
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Stage 2 replay with the record's own alpha (alpha=0 arms) and optional --all_nodes (ALL=1, random rollouts).
set -x
source /tc1apps/anaconda3/etc/profile.d/conda.sh
conda activate iwm
cd ~/iwm/dino_wm
export DATASET_DIR=$HOME/iwm/data/database
export WANDB_MODE=offline PYTHONPATH=. PYTHONUNBUFFERED=1
MODEL=$HOME/iwm/data/checkpoints/outputs/outputs/wall_single
ALPHA=$(python -c "import numpy as np; print(float(np.load('$REC')['alpha']))")
X=""; [ -n "$ALL" ] && X="--all_nodes"
python analysis/stage2_replay.py --model_path $MODEL --record_path $REC \
  --out $HOME/iwm/records/${OUT:-stage2}.npz --device cuda --n_q ${NQ:-30} --n_episodes 20 --alpha $ALPHA $X
echo "S2R_DONE $?"
