#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=180
#SBATCH --job-name=iwmCache
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# L0 (coverage-sweep outline §4): cache DINO latents for all episodes (fp16), verify against the
# in-loop encoder (B3), then the door-vs-strip one-step error on the RELEASED model (§3.1 difficulty check).
set -x
source /tc1apps/anaconda3/etc/profile.d/conda.sh
conda activate iwm
cd ~/iwm/dino_wm
export DATASET_DIR=$HOME/iwm/data/database
export WANDB_MODE=offline PYTHONPATH=. PYTHONUNBUFFERED=1
MODEL=$HOME/iwm/data/checkpoints/outputs/outputs/wall_single
DATA=$DATASET_DIR/wall_single
python datasets/latent_cache.py --model_path $MODEL --data_path $DATA --out_dir $DATA/latents --device cuda --verify 100
echo "CACHE_DONE $?"
ls $DATA/latents | wc -l
python analysis/door_error.py --model_path $MODEL --data_path $DATA --latent_dir $DATA/latents \
  --masks $HOME/iwm/records/coverage_masks.npz --level r100 --device cuda --n_max 4000 --out $HOME/iwm/records/door_error_release.npz
echo "DOOR_ERROR_DONE $?"
