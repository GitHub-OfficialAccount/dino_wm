#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=330
#SBATCH --job-name=iwmRetrain
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# Coverage-sweep retrain (outline §3.2). LEVEL in {r100, r20, r10, r5, r10ctrl, x5}. Resumes from the
# level's model_latest.pth if present and trains up to TARGET epochs total; resubmit the same job to
# continue past the 6-hour cap. After training, writes plan/hydra.yaml (DINO encoder, no decoder) and a
# symlink so the Stage 1/2 readouts load the checkpoint unchanged.
set -x
source /tc1apps/anaconda3/etc/profile.d/conda.sh
conda activate iwm
cd ~/iwm/dino_wm
export DATASET_DIR=$HOME/iwm/data/database
export WANDB_MODE=offline PYTHONPATH=. PYTHONUNBUFFERED=1
LEVEL=${LEVEL:?set LEVEL}; TARGET=${TARGET:-65}
DIR=$HOME/iwm/retrain/$LEVEL; mkdir -p $DIR
DATA=$DATASET_DIR/wall_single
EPOCH=0
if [ -f $DIR/run/checkpoints/model_latest.pth ]; then
  EPOCH=$(python -c "import torch; print(int(torch.load('$DIR/run/checkpoints/model_latest.pth', map_location='cpu', weights_only=False)['epoch']))")
fi
REMAIN=$((TARGET - EPOCH)); echo "level $LEVEL: at epoch $EPOCH, $REMAIN to go"
if [ $REMAIN -gt 0 ]; then
  python train.py --config-name train_coverage training.epochs=$REMAIN \
    env.dataset._target_=datasets.cached_wall_dset.load_cached_wall_slice_train_val \
    +env.dataset.latent_dir=$DATA/latents +env.dataset.mask_path=$HOME/iwm/records/coverage_masks.npz +env.dataset.level=$LEVEL \
    env.num_workers=8 ckpt_base_path=$DIR hydra.run.dir=$DIR/run
  echo "TRAIN_EXIT $?"
fi
EPOCH=$(python -c "import torch; print(int(torch.load('$DIR/run/checkpoints/model_latest.pth', map_location='cpu', weights_only=False)['epoch']))")
echo "level $LEVEL now at epoch $EPOCH"
if [ $EPOCH -ge $TARGET ]; then
  mkdir -p $DIR/plan/checkpoints
  sed 's/^has_decoder: true/has_decoder: false/' $HOME/iwm/data/checkpoints/outputs/outputs/wall_single/hydra.yaml > $DIR/plan/hydra.yaml
  ln -sf $DIR/run/checkpoints/model_latest.pth $DIR/plan/checkpoints/model_latest.pth
  echo "RETRAIN_DONE $LEVEL"
else
  echo "RETRAIN_PARTIAL $LEVEL $EPOCH"
fi
