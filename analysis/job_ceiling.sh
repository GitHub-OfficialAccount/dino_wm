#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=120
#SBATCH --job-name=iwmCeil
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -x
source /tc1apps/anaconda3/etc/profile.d/conda.sh
conda activate iwm
cd ~/iwm/dino_wm
export PYTHONPATH=. PYTHONUNBUFFERED=1
python analysis/wall_ceiling.py --record ${REC:-$HOME/iwm/records/mcts_s_csweep_c0.2_K5000.npz} \
  --out $HOME/iwm/records/ceiling.npz --workers 4
echo "CEIL_DONE $?"
