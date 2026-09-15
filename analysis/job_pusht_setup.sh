#!/bin/bash
#SBATCH --partition=UGGPU-TC1
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=40
#SBATCH --job-name=iwmPTsetup
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
# PushT deps into the iwm env (pymunk 6 API: add_collision_handler was removed in 7), then an
# import + headless-render check. CPU only; no GPU needed.
set -x
source /tc1apps/anaconda3/etc/profile.d/conda.sh
conda activate iwm
cd ~/iwm/dino_wm
pip install --no-deps "pymunk==6.2.1" "pygame" "shapely" "scikit-image" "opencv-python-headless" "decord" "lazy-loader" "tifffile" "cffi" "pycparser" 2>&1 | tail -3
pip install matplotlib 2>&1 | tail -1
python -c "import numpy, torch; print('numpy', numpy.__version__, 'torch', torch.__version__)"
export SDL_VIDEODRIVER=dummy PYTHONPATH=. DATASET_DIR=$HOME/iwm/data/database
python - <<'PY'
import warnings; warnings.filterwarnings("ignore")
import numpy as np, gym, env
e = gym.make("pusht", with_velocity=True, with_target=True)
init, goal = e.sample_random_init_goal_states(1)
obs, s = e.prepare(1, init)
print("pusht env ok: obs visual", obs["visual"].shape, obs["visual"].dtype, "proprio", obs["proprio"].shape, "state", np.asarray(s).shape)
PY
echo "ENV_CHECK_EXIT $?"
python -c "
from decord import VideoReader; import os
r = VideoReader(os.path.expanduser('~/iwm/data/database/pusht_noise/val/obses/episode_000.mp4'), num_threads=1); print('decord ok, frames', len(r), r[0].shape)"
echo "PT_SETUP_DONE $?"
