"""Test 2 as outline.md specifies it: random rollouts instead of a search trajectory.

From each instance's root, R uniformly random depth-H action sequences over the same
16-action set, written in the MCTS record schema (a "tree" that is R disjoint chains of
depth H, all marked as siblings of nothing) so `stage2_replay.py` replays them unchanged.
No planning happens; the record exists to be replayed.

    python analysis/run_random.py --model_path M --n_evals 50 --R 80 --record_path random.npz
"""
import argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
from analysis.run_mcts import build
from planning.mcts import make_action_set


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--n_evals", type=int, default=50)
    ap.add_argument("--goal_H", type=int, default=5)
    ap.add_argument("--R", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--record_path", default="random_records.npz")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    ws, model, env, mcfg = build(a.model_path, a.n_evals, a.goal_H, device=a.device,
                                 planner={"n_sims": 1, "simulator": "env", "value": "state"}, alpha=a.alpha)
    pre, fs, H = ws.data_preprocessor, int(mcfg.frameskip), a.goal_H
    raw, flat, exec_ = make_action_set(pre, frameskip=fs)
    A = len(flat)
    rng = np.random.default_rng(a.seed)
    seqs = rng.integers(0, A, size=(a.n_evals, a.R, H))
    # record schema: per instance, root (node 0) + R chains of H nodes
    cols = {k: [] for k in ("inst", "node_id", "parent_id", "depth", "action_idx", "N", "Q_raw",
                            "V_leaf", "V_leaf_vis", "V_leaf_pro", "rho", "sim_idx_expanded", "on_plan")}
    for j in range(a.n_evals):
        nid = 0
        def add(parent, depth, act, sim):
            nonlocal nid
            cols["inst"].append(j); cols["node_id"].append(nid); cols["parent_id"].append(parent)
            cols["depth"].append(depth); cols["action_idx"].append(act); cols["N"].append(1)
            cols["Q_raw"].append(np.nan); cols["V_leaf"].append(np.nan); cols["V_leaf_vis"].append(np.nan)
            cols["V_leaf_pro"].append(np.nan); cols["rho"].append(1.0); cols["sim_idx_expanded"].append(sim)
            cols["on_plan"].append(False)
            nid += 1
            return nid - 1
        add(-1, 0, -1, -1)
        for r in range(a.R):
            parent = 0
            for h in range(H):
                parent = add(parent, h + 1, int(seqs[j, r, h]), r)
    R = {k: np.array(v) for k, v in cols.items()}
    R["is_random_rollout"] = np.asarray(True)
    R.update({"state_0": np.asarray(ws.state_0, np.float32), "state_g": np.asarray(ws.state_g, np.float32),
              "eval_seed": np.asarray(ws.eval_seed, np.int64), "K": np.asarray(0), "c_uct": np.asarray(np.nan),
              "norm_floor": np.asarray(np.nan), "alpha": np.asarray(a.alpha), "H": np.asarray(H),
              "frameskip": np.asarray(fs), "goal_H": np.asarray(H), "seed": np.asarray(a.seed),
              "simulator": np.asarray("random"), "value": np.asarray("latent"), "mode": np.asarray("random"),
              "cache_dtype": np.asarray("n/a"), "normalize": np.asarray("n/a"),
              "action_set_raw": raw, "action_set_flat": flat.cpu().numpy(),
              "snap_K": np.array([0]), "snap_action_idx": seqs[:, 0][None],
              "snap_actions": flat[torch.as_tensor(seqs[:, 0])].cpu().numpy()[None],
              "Q_min": np.full(a.n_evals, np.nan), "Q_max": np.full(a.n_evals, np.nan),
              "nodes_created": np.full(a.n_evals, a.R * H), "n_calls": np.asarray(a.n_evals * a.R * H)})
    np.savez_compressed(a.record_path, **R)
    print(f"[random] wrote {a.record_path}: {a.n_evals} instances x {a.R} rollouts x depth {H}")


if __name__ == "__main__":
    main()
