#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=330
#SBATCH --job-name=iwmConf
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -x
source /tc1apps/anaconda3/etc/profile.d/conda.sh
conda activate iwm
cd ~/iwm/dino_wm
export DATASET_DIR=$HOME/iwm/data/database WANDB_MODE=offline PYTHONPATH=. PYTHONUNBUFFERED=1
python analysis/confound.py \
  --model_path $HOME/iwm/data/checkpoints/outputs/outputs/wall_single \
  --record_path ${REC} --data_path $HOME/iwm/data/database/wall_single \
  --device cuda --n_evals ${NEVALS:-50} --n_episodes 20 \
  --out ${REC%.npz}_confound.npz
echo "CONFOUND_DONE $?"
