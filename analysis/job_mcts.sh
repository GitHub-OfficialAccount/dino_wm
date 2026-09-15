#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=330
#SBATCH --job-name=iwmMCTS
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Stage 1 sweep. Submit with --gres=gpu:1 for the wm / latent arms; without it for 1.1-S:
#   SIM=env VALUE=state  sbatch analysis/job_mcts.sh                      # 1.1-S, CPU only
#   SIM=env VALUE=latent GREEDY=1 sbatch --gres=gpu:1 analysis/job_mcts.sh # 1.1-L + 1.1-G
#   SIM=wm  VALUE=latent GREEDY=1 sbatch --gres=gpu:1 analysis/job_mcts.sh # 1.2   + 1.2-G
#   SIM=env VALUE=state CUCTS="0.2 0.35 0.5 0.7 1.0 1.4142" K=5000 sbatch ... # c_uct on 1.1-S
#   SIM=wm  VALUE=latent CUCT=0.25 sbatch --gres=gpu:1 ...                 # c_uct sensitivity
set -x
source /tc1apps/anaconda3/etc/profile.d/conda.sh
conda activate iwm
cd ~/iwm/dino_wm
export DATASET_DIR=$HOME/iwm/data/database
export WANDB_MODE=offline PYTHONPATH=. PYTHONUNBUFFERED=1
mkdir -p ~/iwm/records
MODEL=$HOME/iwm/data/checkpoints/outputs/outputs/wall_single
SIM=${SIM:-wm}; VALUE=${VALUE:-latent}; K=${K:-2000}; NEVALS=${NEVALS:-50}
DEV=cpu; if [ -n "$CUDA_VISIBLE_DEVICES" ]; then DEV=cuda; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; fi
G=""; [ -n "$GREEDY" ] && G="--greedy_too"
SNAPS=${SNAPS:-"10 30 100 300 1000 2000"}

# CUCTS="0.2 0.5 1.0" sweeps c_uct (one record per value); default is a single run.
for C in ${CUCTS:-${CUCT:-0.5}}; do
  REC=$HOME/iwm/records/mcts_${TAG:-${SIM}_${VALUE}}_c${C}_K${K}.npz
  python analysis/run_mcts.py --model_path $MODEL --n_evals $NEVALS --goal_H 5 \
    --sim $SIM --value $VALUE --K $K --c_uct $C --snapshots $SNAPS \
    --cache_dtype ${CACHE:-fp16} --normalize ${NORM:-minmax} --alpha ${ALPHA:-1.0} --seed ${SEED:-0} --n_rollout 20 --device $DEV \
    --record_path $REC --log_every 100 $G
  echo "MCTS_DONE c=$C $?"
done
