"""Stage 1 driver: MCTS (or open-loop greedy) on Wall, evaluate every K-snapshot, record.

Mirrors `run_record.py`: builds the E0 PlanWorkspace (same seeds, same instances, same
evaluator) with `MCTSPlanner` in place of CEM. One run yields the whole K-sweep via the
planner's anytime snapshots; each snapshot's open-loop plan is executed in the true env
by the unchanged `PlanEvaluator.eval_actions` (gate G6) and run through `wm.rollout` for
its imagined objective and the G3 round-trip check.

    python analysis/run_mcts.py --model_path M --sim wm  --value latent --K 2000 --record_path r.npz
    python analysis/run_mcts.py --model_path M --sim env --value state  --K 2000 ...
    python analysis/run_mcts.py ... --mode greedy          # arms 1.1-G / 1.2-G
"""
import os, sys, warnings, argparse, time
warnings.filterwarnings("ignore")
import numpy as np, gym, torch
from omegaconf import OmegaConf
import hydra
from pathlib import Path

import plan as P
from plan import PlanWorkspace, DummyWandbRun
from utils import move_to_device


def build(model_path, n_evals=2, goal_H=5, n_rollout=20, seed=99, device="cpu",
          planner=None, alpha=1.0):
    model_path = Path(model_path)
    model_cfg = OmegaConf.load(model_path / "hydra.yaml")
    model_cfg.env.dataset.n_rollout = n_rollout
    _, dset = hydra.utils.call(model_cfg.env.dataset, num_hist=model_cfg.num_hist,
                               num_pred=model_cfg.num_pred, frameskip=model_cfg.frameskip)
    dset = dset["valid"]
    model = P.load_model(model_path / "checkpoints" / "model_latest.pth", model_cfg,
                         model_cfg.num_action_repeat, device=torch.device(device))
    model.eval()
    model.decoder = None
    from env.serial_vector_env import SerialVectorEnv
    env = SerialVectorEnv([gym.make(model_cfg.env.name, *model_cfg.env.args,
                                    **model_cfg.env.kwargs) for _ in range(n_evals)])
    planner = dict(planner or {})
    planner.setdefault("_target_", "planning.mcts.MCTSPlanner")
    planner.setdefault("horizon", goal_H)
    planner.setdefault("frameskip", int(model_cfg.frameskip))
    planner.setdefault("alpha", alpha)
    cfg_dict = {
        "seed": seed, "n_evals": n_evals, "goal_source": "random_state",
        "goal_H": goal_H, "n_plot_samples": 0, "debug_dset_init": False,
        "objective": {"_target_": "planning.objectives.create_objective_fn",
                      "alpha": alpha, "base": 2, "mode": "last"},
        "planner": planner,
    }
    ws = PlanWorkspace(cfg_dict=cfg_dict, wm=model, dset=dset, env=env,
                       env_name=model_cfg.env.name, frameskip=model_cfg.frameskip,
                       wandb_run=DummyWandbRun())
    return ws, model, env, model_cfg


def imagined(ws, actions):
    """Imagined final latent + objective of an open-loop plan through `wm.rollout`."""
    pre, dev = ws.data_preprocessor, ws.device
    with torch.no_grad():
        z_g = ws.wm.encode_obs(move_to_device(pre.transform_obs(ws.obs_g), dev))
        i_z, z_full = ws.wm.rollout(obs_0=move_to_device(pre.transform_obs(ws.obs_0), dev),
                                    act=actions)
    loss = ws.planner.objective_fn(i_z, z_g)
    return loss.cpu().numpy(), z_full[:, -1]


def evaluate_snapshots(ws, planner, tag=""):
    """Execute every snapshot's plan in the env; attach success, final state, imagined
    objective, and (when the plan came wholly from the tree) the G3 round-trip error."""
    rows = []
    for s in planner.snapshots:
        acts = torch.as_tensor(s["actions"]).to(ws.device)
        logs, succ, _, e_states = ws.evaluator.eval_actions(acts, filename=f"{tag}K{s['K']}")
        final = e_states[:, -1, :2]
        s["success"] = np.asarray(succ, bool)
        s["final_state"] = final
        s["state_dist"] = np.linalg.norm(final - np.asarray(ws.state_g)[:, :2], axis=-1)
        s["imagined_loss"], z_last = imagined(ws, acts)
        rt = np.full(len(succ), np.nan)
        if planner.mode == "mcts" and planner.sim_kind == "wm" and s is planner.snapshots[-1]:
            for i in range(len(succ)):
                if s["plan_depth"][i] == planner.horizon:
                    zc = planner.pv_state(i).float().to(ws.device)
                    rt[i] = (zc - z_last[i]).abs().max().item()
        s["pv_roundtrip"] = rt
        rows.append((s["K"], s["success"].mean(), s["state_dist"].mean(),
                     float(np.mean(s["plan_depth"])), float(np.mean(s["imagined_loss"])),
                     s.get("n_calls", 0), s.get("elapsed", 0.0)))
    print(f"\n=== {tag} K-sweep (n={len(planner.snapshots[0]['success'])}) ===")
    print(f"{'K':>6} {'success':>8} {'dist':>7} {'depth':>6} {'imag.loss':>10} {'calls':>8} {'sec':>7}")
    for r in rows:
        print(f"{r[0]:>6} {r[1]:>8.3f} {r[2]:>7.2f} {r[3]:>6.2f} {r[4]:>10.4f} {r[5]:>8} {r[6]:>7.1f}")
    return rows


def snapshot_arrays(planner, prefix="snap_"):
    keys = ("success", "final_state", "state_dist", "imagined_loss", "pv_roundtrip")
    return {f"{prefix}{k}": np.stack([s[k] for s in planner.snapshots]) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--n_evals", type=int, default=2)
    ap.add_argument("--goal_H", type=int, default=5)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--sim", choices=["wm", "env"], default="wm")
    ap.add_argument("--value", choices=["latent", "state"], default="latent")
    ap.add_argument("--mode", choices=["mcts", "greedy"], default="mcts")
    ap.add_argument("--c_uct", type=float, default=float(np.sqrt(2)))
    ap.add_argument("--cache_dtype", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--normalize", choices=["minmax", "none"], default="minmax")
    ap.add_argument("--alpha", type=float, default=1.0, help="objective weight on proprio (0 = visual-only)")
    ap.add_argument("--snapshots", type=int, nargs="*", default=[10, 30, 100, 300, 1000, 2000])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_rollout", type=int, default=20)
    ap.add_argument("--record_path", default="mcts_records.npz")
    ap.add_argument("--greedy_too", action="store_true",
                    help="also run the open-loop greedy arm on the same simulator/value "
                         "and store it in the same record (greedy_* keys)")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    planner_cfg = {"n_sims": a.K, "simulator": a.sim, "value": a.value, "mode": a.mode,
                   "c_uct": a.c_uct, "cache_dtype": a.cache_dtype, "snapshot_Ks": a.snapshots,
                   "normalize": a.normalize,
                   "seed": a.seed, "log_every": a.log_every}
    ws, model, env, mcfg = build(a.model_path, a.n_evals, a.goal_H, a.n_rollout,
                                 device=a.device, planner=planner_cfg, alpha=a.alpha)
    pl = ws.planner
    t0 = time.time()
    pl.plan(ws.obs_0, ws.obs_g)
    print(f"[run_mcts] planning done in {time.time() - t0:.1f}s; calls {pl.sim.n_calls}")
    rows = evaluate_snapshots(ws, pl, tag=f"{a.sim}-{a.value}-{a.mode}")
    R = pl.record()
    R.update(snapshot_arrays(pl))

    if a.greedy_too and a.mode == "mcts":
        from planning.mcts import MCTSPlanner
        g = MCTSPlanner(horizon=a.goal_H, n_sims=0, wm=model, action_dim=pl.action_dim,
                        objective_fn=pl.objective_fn, preprocessor=pl.preprocessor,
                        evaluator=pl.evaluator, wandb_run=ws.wandb_run, env=env,
                        simulator=a.sim, value=a.value, mode="greedy", alpha=pl.alpha,
                        frameskip=pl.frameskip, cache_dtype=a.cache_dtype, seed=a.seed)
        g.plan(ws.obs_0, ws.obs_g)
        evaluate_snapshots(ws, g, tag=f"{a.sim}-{a.value}-greedy")
        G = g.record(); G.update(snapshot_arrays(g))
        for k in ("snap_actions", "snap_action_idx", "snap_pv_value", "greedy_values",
                  "snap_success", "snap_final_state", "snap_state_dist", "snap_imagined_loss",
                  "snap_n_calls", "snap_elapsed"):
            R["greedy_" + k.replace("snap_", "")] = G[k][0] if k.startswith("snap_") else G[k]

    np.savez_compressed(a.record_path, **R)
    print(f"[run_mcts] wrote {a.record_path}")
    if a.mode == "mcts":
        nodes = R["nodes_created"]
        print(f"[run_mcts] G4: calls {int(R['n_calls'])} == nodes {int(nodes.sum())}: "
              f"{int(R['n_calls']) == int(nodes.sum())}; shortfall K-nodes per instance "
              f"mean {np.mean(a.K - nodes):.1f} max {int(np.max(a.K - nodes))}")
        for d in range(1, a.goal_H + 1):
            n = R[f"N_depth{d}"]
            if len(n):
                print(f"[run_mcts] N at depth {d}: n={len(n)} median {np.median(n):.0f} "
                      f"p90 {np.percentile(n, 90):.0f} max {n.max()} frac(N=1) {(n == 1).mean():.2f}")
        rt = R["snap_pv_roundtrip"][-1]
        if np.isfinite(rt).any():
            print(f"[run_mcts] G3 round-trip max|z_cache - z_rollout| over {np.isfinite(rt).sum()} "
                  f"full-depth plans: {np.nanmax(rt):.3e}")
    print("PLANNING_DONE")


if __name__ == "__main__":
    main()
