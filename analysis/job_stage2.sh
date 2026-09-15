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
# Stage 2 replay of an MCTS record:  REC=~/iwm/records/mcts_wm_c0.2_K2000.npz OUT=stage2_wm sbatch analysis/job_stage2.sh
set -x
source /tc1apps/anaconda3/etc/profile.d/conda.sh
conda activate iwm
cd ~/iwm/dino_wm
export DATASET_DIR=$HOME/iwm/data/database
export WANDB_MODE=offline PYTHONPATH=. PYTHONUNBUFFERED=1
MODEL=$HOME/iwm/data/checkpoints/outputs/outputs/wall_single
python analysis/stage2_replay.py --model_path $MODEL --record_path $REC \
  --out $HOME/iwm/records/${OUT:-stage2}.npz --device cuda --n_q ${NQ:-30} --n_episodes 20
echo "S2R_DONE $?"
