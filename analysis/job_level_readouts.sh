#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=120
#SBATCH --job-name=iwmLvl
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Per-level readouts (coverage-sweep outline §3.3): Stage 1 arms (1.2+1.2-G, 1.1-L+1.1-G) on the level's
# model, Stage 2 replays of both trees, and the door/strip one-step error at the level. LEVEL required.
set -x
source /tc1apps/anaconda3/etc/profile.d/conda.sh
conda activate iwm
cd ~/iwm/dino_wm
export DATASET_DIR=$HOME/iwm/data/database
export WANDB_MODE=offline PYTHONPATH=. PYTHONUNBUFFERED=1
LEVEL=${LEVEL:?set LEVEL}
MODEL=$HOME/iwm/retrain/$LEVEL/plan
DATA=$DATASET_DIR/wall_single
R=$HOME/iwm/records/cov_$LEVEL; mkdir -p $R
for ARM in "wm latent" "env latent"; do set -- $ARM
  python analysis/run_mcts.py --model_path $MODEL --n_evals 50 --goal_H 5 --sim $1 --value $2 --K 2000 --c_uct 0.2 \
    --snapshots 10 30 100 300 1000 2000 --cache_dtype fp16 --seed 0 --n_rollout 20 --device cuda \
    --record_path $R/mcts_${1}_c0.2_K2000.npz --log_every 500 --greedy_too
  echo "MCTS_DONE $1 $?"
done
for SIM in wm env; do
  python analysis/stage2_replay.py --model_path $MODEL --record_path $R/mcts_${SIM}_c0.2_K2000.npz \
    --out $R/stage2_${SIM}.npz --device cuda --n_q 30 --n_episodes 20
  echo "S2R_DONE $SIM $?"
done
python analysis/door_error.py --model_path $MODEL --data_path $DATA --latent_dir $DATA/latents \
  --masks $HOME/iwm/records/coverage_masks.npz --level $LEVEL --device cuda --n_max 4000 --out $R/door_error.npz
echo "DOOR_ERROR_DONE $?"
echo "LEVEL_DONE $LEVEL"
