"""Coverage-sweep holdout masks (stage-2-coverage-sweep-outline.md §3.1). Local, seconds.

Replicates the trainer's 90/10 episode split exactly (torch randperm, seed 42, as
`datasets.traj_dset.split_traj_datasets`), then over the training split's slices
(episode, start) with frames (start, start+frameskip):

  H      door neighbourhood: either frame has |x-32| <= 6 and |y-30| <= 10
  STRIP  count-matched control off the search's path: either frame has y >= 56.75, not in H,
         subsampled to exactly |H|

Levels (each exactly the full training slice count):
  r100, r20, r10, r5   keep p% of H (fixed-seed subsample), refill the removed count by
                       resampling non-H slices with replacement
  r10ctrl              keep 10% of STRIP, refill from non-STRIP
  x5                   H repeated 5x, non-H thinned (without replacement) to fit

Writes coverage_masks.npz: per level `<level>_slices` (n, 2) int32 of (global episode
index, start); plus `train_episodes`, `H_slices`, `STRIP_slices`, geometry and seeds.

    python analysis/coverage_masks.py --data_path ../data/database/wall_single --out coverage_masks.npz
"""
import argparse
import numpy as np, torch

WALL_X, DOOR_Y, H_DX, H_DY, STRIP_Y = 32.0, 30.0, 6.0, 10.0, 56.75
LEVELS = {"r100": 1.0, "r20": 0.2, "r10": 0.1, "r5": 0.05}


def train_split(n, train_fraction=0.9, random_seed=42):
    """Identical to datasets.traj_dset.split_traj_datasets."""
    lengths = [int(train_fraction * n), n - int(train_fraction * n)]
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(random_seed)).tolist()
    return np.array(idx[: lengths[0]]), np.array(idx[lengths[0]:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--out", default="coverage_masks.npz")
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--num_frames", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    st = torch.load(f"{a.data_path}/states.pth", weights_only=True).numpy()   # (N, T, 2)
    N, T, _ = st.shape
    tr, va = train_split(N)
    starts = np.arange(T - a.num_frames * a.frameskip + 1)
    ep = np.repeat(tr, len(starts)); s0 = np.tile(starts, len(tr))          # all training slices
    xa, ya = st[ep, s0, 0], st[ep, s0, 1]; xb, yb = st[ep, s0 + a.frameskip, 0], st[ep, s0 + a.frameskip, 1]
    in_door = lambda x, y: (np.abs(x - WALL_X) <= H_DX) & (np.abs(y - DOOR_Y) <= H_DY)
    H = in_door(xa, ya) | in_door(xb, yb)
    strip_raw = ((ya >= STRIP_Y) | (yb >= STRIP_Y)) & ~H
    rng = np.random.default_rng(a.seed)
    strip = np.zeros_like(H); strip[rng.choice(np.flatnonzero(strip_raw), H.sum(), replace=False)] = True
    allsl = np.stack([ep, s0], 1).astype(np.int32)
    n_total = len(allsl)
    out = {"train_episodes": tr, "val_episodes": va, "H_slices": allsl[H], "STRIP_slices": allsl[strip],
           "geometry": np.array([WALL_X, DOOR_Y, H_DX, H_DY, STRIP_Y]), "seed": np.asarray(a.seed),
           "n_train_slices": np.asarray(n_total), "frameskip": np.asarray(a.frameskip)}
    print(f"episodes: train {len(tr)}, val {len(va)}; training slices {n_total}")
    print(f"H {H.sum()} ({H.mean():.3%}); STRIP raw {strip_raw.sum()} -> matched {strip.sum()}; "
          f"crossing slices {(np.sign(xa - WALL_X) != np.sign(xb - WALL_X)).sum()}")

    def level(keep_mask, p, name, rng):
        """Keep p of keep_mask's slices, refill from the complement with replacement."""
        held = np.flatnonzero(keep_mask); rest = np.flatnonzero(~keep_mask)
        n_keep = int(round(len(held) * p))
        kept = rng.choice(held, n_keep, replace=False) if n_keep < len(held) else held
        refill = rng.choice(rest, len(held) - n_keep, replace=True)
        sl = allsl[np.concatenate([kept, rest, refill])]
        assert len(sl) == n_total
        out[f"{name}_slices"] = sl
        out[f"{name}_kept_held"] = allsl[kept]
        print(f"  {name:>8}: kept {n_keep:>5} of {len(held)} held-out slices, refill {len(refill):>5}, total {len(sl)}")

    for k, (name, p) in enumerate(LEVELS.items()):
        level(H, p, name, np.random.default_rng([a.seed, 100 + k]))   # fixed per-level streams
    level(strip, 0.1, "r10ctrl", np.random.default_rng([a.seed, 1010]))
    # x5: H repeated 5x, non-H thinned without replacement to hold the total
    held = np.flatnonzero(H); rest = np.flatnonzero(~H)
    n_rest = n_total - 5 * len(held)
    r5 = np.random.default_rng([a.seed, 5555]).choice(rest, n_rest, replace=False)
    sl = allsl[np.concatenate([np.tile(held, 5), r5])]
    out["x5_slices"] = sl; out["x5_kept_held"] = allsl[held]
    print(f"  {'x5':>8}: H x5 = {5 * len(held)}, non-H thinned to {n_rest}, total {len(sl)}")
    np.savez_compressed(a.out, **out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
