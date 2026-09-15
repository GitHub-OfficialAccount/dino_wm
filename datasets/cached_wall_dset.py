"""Wall dataset served from cached DINO latents, with an optional training-slice list
(coverage-sweep outline §3.1–3.2).

`CachedWallDataset.get_frames` returns obs["visual"] as the cached (T, 196, 384) latents in
place of images; everything else (actions, proprio, normalisation, episode split) is the
parent class. `load_cached_wall_slice_train_val` mirrors `load_wall_slice_train_val` and,
when `mask_path` and `level` are given, replaces the training slices with the level's list
from coverage_masks.npz (global episode index -> subset index), permuted with the trainer's
seed as TrajSlicerDataset does. The validation split is untouched.
"""
from pathlib import Path
from typing import Callable, Optional
import numpy as np, torch
from .wall_dset import WallDataset
from .traj_dset import get_train_val_sliced


class CachedWallDataset(WallDataset):
    def __init__(self, latent_dir: str, **kw):
        super().__init__(**kw)
        self.latent_dir = Path(latent_dir)

    def get_frames(self, idx, frames):
        z = torch.load(self.latent_dir / f"episode_{idx:03d}.pt", weights_only=True)
        act = self.actions[idx, frames]
        state = self.states[idx, frames]
        proprio = self.proprios[idx, frames]
        door_location = self.door_locations[idx, frames]
        wall_location = self.wall_locations[idx, frames]
        obs = {"visual": z[frames], "proprio": proprio}
        return obs, act, state, {"fix_door_location": door_location[0], "fix_wall_location": wall_location[0]}


def load_cached_wall_slice_train_val(transform=None, n_rollout=None, data_path="data/wall_single",
                                     latent_dir=None, normalize_action=False, split_ratio=0.9,
                                     split_mode="random", num_hist=0, num_pred=0, frameskip=0,
                                     mask_path=None, level=None):
    assert split_mode == "random"
    dset = CachedWallDataset(latent_dir=latent_dir or str(Path(data_path) / "latents"), n_rollout=n_rollout,
                             transform=None, data_path=data_path, normalize_action=normalize_action)
    dset_train, dset_val, train_slices, val_slices = get_train_val_sliced(
        traj_dataset=dset, train_fraction=split_ratio, num_frames=num_hist + num_pred, frameskip=frameskip)
    if mask_path is not None and level is not None:
        M = np.load(mask_path)
        assert list(M["train_episodes"]) == list(dset_train.indices), "mask split != trainer split"
        sub = {int(g): k for k, g in enumerate(dset_train.indices)}
        L = M[f"{level}_slices"]; n = (num_hist + num_pred) * frameskip
        slices = [(sub[int(e)], int(s), int(s) + n) for e, s in L]
        assert len(slices) == int(M["n_train_slices"]), (len(slices), int(M["n_train_slices"]))
        train_slices.slices = np.random.permutation(np.array(slices))      # as TrajSlicerDataset.__init__
        print(f"[cached_wall_dset] level {level}: {len(train_slices.slices)} training slices from {mask_path}")
    datasets = {"train": train_slices, "valid": val_slices}
    return datasets, {"train": dset_train, "valid": dset_val}
