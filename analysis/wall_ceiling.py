"""Exact feasibility ceiling of the Stage 1 task: for each (start, goal) instance, the
minimum final distance over ALL 16^5 open-loop plans under the true Wall dynamics.

Dynamic programming over positions: sequences that reach the same position (to 1e-4)
merge, so the frontier stays ~1e4 wide instead of 1e6. The result is the denominator G1
needs — 1.1-S can only be judged against what the action set + horizon allow.

    python analysis/wall_ceiling.py --record mcts_env_state_c0.2_K5000.npz --out ceiling.npz
"""
import argparse, time, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, gym
from multiprocessing import Pool
import env  # registers "wall"

R_SUCCESS = 4.5


def ceiling_one(args):
    j, seed, s0, sg, raw, H, fs = args
    e = gym.make("wall").unwrapped
    e.prepare(seed, s0)
    assert int(e.wall_x) == 32 and int(e.hole_y) == 30, (e.wall_x, e.hole_y)
    goal = np.asarray(sg[:2], float)
    level = {(round(float(s0[0]), 4), round(float(s0[1]), 4)): e.dot_position.clone()}
    mins, sizes = [], []
    for d in range(H):
        nxt = {}
        for pos in level.values():
            for a in range(len(raw)):
                p = pos
                for f in range(fs):
                    e.dot_position = p
                    p = e._calculate_next_position(raw[a, f])
                key = (round(float(p[0]), 4), round(float(p[1]), 4))
                if key not in nxt:
                    nxt[key] = p
        level = nxt
        P = np.array(list(level.keys()))
        mins.append(float(np.linalg.norm(P - goal, axis=-1).min()))
        sizes.append(len(level))
    return j, mins, sizes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--out", default="ceiling.npz")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--H", type=int, default=None, help="override depth (smoke tests)")
    a = ap.parse_args()
    R = np.load(a.record)
    raw, H, fs = R["action_set_raw"], a.H or int(R["H"]), int(R["frameskip"])
    jobs = [(j, int(R["eval_seed"][j]), R["state_0"][j].astype(np.float64),
             R["state_g"][j].astype(np.float64), raw, H, fs) for j in range(len(R["eval_seed"]))]
    t0 = time.time()
    with Pool(a.workers) as pool:
        res = pool.map(ceiling_one, jobs)
    res.sort()
    mins = np.array([r[1] for r in res]); sizes = np.array([r[2] for r in res])
    feas = mins[:, -1] < R_SUCCESS
    print(f"{len(res)} instances in {time.time() - t0:.0f}s; frontier sizes (mean per depth): "
          f"{sizes.mean(0).round(0)}")
    print(f"feasible at depth H={H}: {feas.sum()}/{len(feas)} = {feas.mean():.2f}; "
          f"min final dist: median {np.median(mins[:, -1]):.2f}, max {mins[:, -1].max():.2f}")
    print("min dist by depth (mean over instances):", mins.mean(0).round(2))
    print("infeasible instances:", np.flatnonzero(~feas).tolist(),
          "min dist:", mins[~feas, -1].round(2).tolist())
    np.savez_compressed(a.out, min_dist=mins, frontier=sizes, feasible=feas,
                        state_0=R["state_0"], state_g=R["state_g"], eval_seed=R["eval_seed"])


if __name__ == "__main__":
    main()
