#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=330
#SBATCH --job-name=iwmA1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -x
source /tc1apps/anaconda3/etc/profile.d/conda.sh
conda activate iwm
cd ~/iwm/dino_wm
export DATASET_DIR=$HOME/iwm/data/database
export WANDB_MODE=offline
export PYTHONPATH=.
mkdir -p ~/iwm/records
MODEL=$HOME/iwm/data/checkpoints/outputs/outputs/wall_single
REC=$HOME/iwm/records/${TAG:-a1}_ns${NSAMP:-300}.npz

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# stage 1: planning with candidate recording
python analysis/run_record.py --model_path $MODEL \
  --n_evals ${NEVALS:-50} --goal_H 5 --num_samples ${NSAMP:-300} --topk ${TOPK:-30} \
  --opt_steps ${OPTSTEPS:-10} --n_rollout 20 --device cuda \
  --record_all_evals 0 1 --record_path $REC
echo "STAGE1_DONE $?"

# stage 2: replay + metrics. Separate stage so a failure here does not lose the record.
python analysis/aggregate.py --model_path $MODEL --record_path $REC \
  --device cuda --n_evals ${NEVALS:-50}
echo "STAGE2_DONE $?"
