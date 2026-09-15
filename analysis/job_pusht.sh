#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=330
#SBATCH --job-name=iwmPT
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# E0-extended on PushT (experiment-0-extended-outline.md). STAGE=gates | a1 | a2 | conf
#   gates: A0.1-A0.7 at n_evals=10 (A0.7 on 200 transitions)
#   a1/a2: planning with candidate recording (300x10 / 1000x10) + single-pass replay analysis
#   conf:  the mean-reversion / collider / variance confound tests on REC
set -x
source /tc1apps/anaconda3/etc/profile.d/conda.sh
conda activate iwm
cd ~/iwm/dino_wm
export DATASET_DIR=$HOME/iwm/data/database
export WANDB_MODE=offline PYTHONPATH=. PYTHONUNBUFFERED=1 SDL_VIDEODRIVER=dummy
mkdir -p ~/iwm/records/pusht
MODEL=$HOME/iwm/data/checkpoints/outputs/outputs/pusht
STAGE=${STAGE:-gates}
case $STAGE in
  gates)
    python analysis/verify_gates_ext.py --model_path $MODEL --n_evals ${NEVALS:-10} --n_trans 200 \
      --num_samples 64 --opt_steps 2 --record_path ~/iwm/records/pusht/gates_rec.npz --device cuda
    echo "PT_GATES_DONE $?" ;;
  a1|a2)
    NS=300; [ $STAGE = a2 ] && NS=1000
    REC=$HOME/iwm/records/pusht/${STAGE}_ns${NS}.npz
    python analysis/run_record.py --model_path $MODEL --n_evals ${NEVALS:-50} --goal_H 5 --num_samples $NS \
      --topk 30 --opt_steps 10 --n_rollout 20 --device cuda --record_all_evals 0 1 --record_path $REC
    echo "STAGE1_DONE $?"
    python analysis/aggregate.py --model_path $MODEL --record_path $REC --device cuda --n_evals ${NEVALS:-50} \
      $( [ $STAGE = a1 ] && echo "--max_per_group 15" )
    echo "STAGE2_DONE $?" ;;
  a2agg)
    # capped contrasts on the full record (the uncapped replay exceeded the 5.5 h limit on PushT)
    python analysis/aggregate.py --model_path $MODEL --record_path $HOME/iwm/records/pusht/a2_ns1000.npz --device cuda \
      --n_evals ${NEVALS:-50} --max_per_group ${MAXPG:-30}
    echo "STAGE2_DONE $?" ;;
  a2regret)
    # selection regret needs every candidate of instances 0-1: replay only those, uncapped
    python - <<'PY'
import numpy as np
R = dict(np.load("$HOME/iwm/records/pusht/a2_ns1000.npz".replace("$HOME", __import__("os").path.expanduser("~"))))
keep = np.isin(R["eval_idx"], [0, 1])
row_keys = [k for k, v in R.items() if getattr(v, "ndim", 0) >= 1 and v.shape[0] == len(R["eval_idx"]) and not k.startswith("spread_")]
for k in row_keys: R[k] = R[k][keep]
np.savez_compressed(__import__("os").path.expanduser("~/iwm/records/pusht/a2_ns1000_evals01.npz"), **R)
print("filtered record:", int(keep.sum()), "rows")
PY
    cp $HOME/iwm/records/pusht/a2_ns1000_targets.pkl $HOME/iwm/records/pusht/a2_ns1000_evals01_targets.pkl
    python analysis/aggregate.py --model_path $MODEL --record_path $HOME/iwm/records/pusht/a2_ns1000_evals01.npz --device cuda \
      --n_evals ${NEVALS:-50}
    echo "STAGE2_DONE $?" ;;
  # ---- 30 refinement rounds (outline §10): planning and analysis are separate jobs so each fits 5.5 h
  plan30)
    NS=${NS:-300}; REC=$HOME/iwm/records/pusht/os30_ns${NS}.npz
    python analysis/run_record.py --model_path $MODEL --n_evals ${NEVALS:-50} --goal_H 5 --num_samples $NS \
      --topk 30 --opt_steps 30 --n_rollout 20 --device cuda --record_all_evals 0 1 --record_path $REC
    echo "STAGE1_DONE $?" ;;
  plan30h)
    # instance-range half of a 30-round planning run (1000 samples does not fit one job): H0=0 H1=25 etc.
    NS=${NS:-1000}; REC=$HOME/iwm/records/pusht/os30_ns${NS}_h${HALF:?}.npz
    python analysis/run_record.py --model_path $MODEL --n_evals ${NEVALS:-50} --goal_H 5 --num_samples $NS \
      --topk 30 --opt_steps 30 --n_rollout 20 --device cuda --record_all_evals 0 1 --record_path $REC \
      --eval_range ${H0:?} ${H1:?}
    echo "STAGE1_DONE $?" ;;
  merge30)
    NS=${NS:-1000}
    python analysis/merge_records.py --out $HOME/iwm/records/pusht/os30_ns${NS}.npz \
      --parts $HOME/iwm/records/pusht/os30_ns${NS}_h0.npz $HOME/iwm/records/pusht/os30_ns${NS}_h1.npz
    echo "MERGE_DONE $?" ;;
  agg30)
    NS=${NS:-300}; REC=$HOME/iwm/records/pusht/os30_ns${NS}.npz
    python analysis/aggregate.py --model_path $MODEL --record_path $REC --device cuda --n_evals ${NEVALS:-50} \
      --max_per_group ${MAXPG:-10} ${STEPS:+--steps $STEPS}
    echo "STAGE2_DONE $?" ;;
  regret30)
    # all candidates of instances 0-1, every third opt_step (11 of 30) to fit the job
    NS=${NS:-1000}; REC=$HOME/iwm/records/pusht/os30_ns${NS}.npz
    NS=$NS python - <<'PY'
import numpy as np, os
src = os.path.expanduser("~/iwm/records/pusht/os30_ns" + os.environ["NS"] + ".npz")
R = dict(np.load(src)); keep = np.isin(R["eval_idx"], [0, 1]); n = len(R["eval_idx"])
for k in [k for k, v in R.items() if getattr(v, "ndim", 0) >= 1 and v.shape[0] == n and not k.startswith("spread_")]: R[k] = R[k][keep]
np.savez_compressed(src.replace(".npz", "_evals01.npz"), **R); print("filtered record:", int(keep.sum()), "rows")
PY
    cp ${REC%.npz}_targets.pkl ${REC%.npz}_evals01_targets.pkl
    python analysis/aggregate.py --model_path $MODEL --record_path ${REC%.npz}_evals01.npz --device cuda --n_evals ${NEVALS:-50} \
      --steps 0 3 6 9 12 15 18 21 24 27 29
    echo "STAGE2_DONE $?" ;;
  conf30)
    NS=${NS:-300}; REC=$HOME/iwm/records/pusht/os30_ns${NS}.npz
    python analysis/confound.py --model_path $MODEL --record_path $REC --device cuda --n_evals ${NEVALS:-50} \
      --data_path $DATASET_DIR/pusht_noise/val --n_episodes 20 --max_per_group ${MAXPG:-8} --out ${REC%.npz}_confound.npz
    echo "CONF_DONE $?" ;;
  conf)
    python analysis/confound.py --model_path $MODEL --record_path ${REC:?} --device cuda --n_evals ${NEVALS:-50} \
      --data_path $DATASET_DIR/pusht_noise/val --n_episodes 20 --max_per_group ${MAXPG:-30} --out ${REC%.npz}_confound.npz
    echo "CONF_DONE $?" ;;
esac
