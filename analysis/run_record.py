"""Minimal end-to-end driver: run CEM with candidate recording, no hydra/wandb/submitit.

Mirrors plan.py's planning_main for the wall env only, so A0 can be iterated on CPU.
"""
import os, sys, warnings, argparse, shutil
warnings.filterwarnings("ignore")
import numpy as np, gym, torch
from omegaconf import OmegaConf
import hydra
from pathlib import Path

import plan as P
from plan import PlanWorkspace, DummyWandbRun


# Per-environment planning settings, matching each checkpoint's released plan config
# (experiment-0-extended-outline.md §3.2): goal source and objective weight.
ENV_SETTINGS = {"wall": {"goal_source": "random_state", "alpha": 1},
                "pusht": {"goal_source": "dset", "alpha": 1}}


def targets_path_for(record_path):
    """Planning targets (obs_0/obs_g/states/gt_actions) saved beside a record, so analysis
    stages rebuild exactly the instances the record was planned on. Needed for `dset` goals,
    which PlanWorkspace samples with an unseeded RNG; harmless for `random_state` goals."""
    return str(record_path).replace(".npz", "") + "_targets.pkl"


def build(model_path, n_evals=2, goal_H=3, num_samples=8, topk=3, opt_steps=2,
          n_rollout=20, record_path="cem_records.npz", record_all_evals=(), seed=99,
          device="cpu", goal_source=None, alpha=None, targets_path=None):
    model_path = Path(model_path)
    model_cfg = OmegaConf.load(model_path / "hydra.yaml")
    model_cfg.env.dataset.n_rollout = n_rollout      # keep dataset reads small
    env_name = str(model_cfg.env.name)
    es = ENV_SETTINGS.get(env_name, ENV_SETTINGS["wall"])
    goal_source = goal_source or es["goal_source"]; alpha = es["alpha"] if alpha is None else alpha
    import random
    random.seed(seed); np.random.seed(seed)          # dset-goal segment sampling uses `random`
    if targets_path is not None and os.path.exists(targets_path):
        goal_source = "file"
        print(f"[run_record] planning targets from {targets_path}")

    _, dset = hydra.utils.call(model_cfg.env.dataset, num_hist=model_cfg.num_hist,
                               num_pred=model_cfg.num_pred, frameskip=model_cfg.frameskip)
    dset = dset["valid"]

    model = P.load_model(model_path / "checkpoints" / "model_latest.pth", model_cfg,
                         model_cfg.num_action_repeat, device=torch.device(device))
    model.eval()   # defensive; predictor already saved in eval mode
    # We never decode: the objective is entirely in latent space. Dropping the decoder
    # skips VQVAE decoding and the plotting path, and saves compute.
    model.decoder = None

    from env.serial_vector_env import SerialVectorEnv
    env = SerialVectorEnv([gym.make(model_cfg.env.name, *model_cfg.env.args,
                                    **model_cfg.env.kwargs) for _ in range(n_evals)])

    cfg_dict = {
        "seed": seed, "n_evals": n_evals, "goal_source": goal_source, "goal_file_path": targets_path,
        "goal_H": goal_H, "n_plot_samples": 0, "debug_dset_init": False,
        "objective": {"_target_": "planning.objectives.create_objective_fn",
                      "alpha": alpha, "base": 2, "mode": "last"},
        "planner": {
            "_target_": "planning.cem_record.RecordingCEMPlanner",
            "horizon": goal_H, "topk": topk, "num_samples": num_samples,
            "var_scale": 1, "opt_steps": opt_steps, "eval_every": opt_steps,
            "record_path": record_path, "record_all_evals": list(record_all_evals),
            # num_hist > 1 multiplies the predictor's attention by num_hist^2; chunk the rollout
            "rollout_chunk": 250 if int(model_cfg.num_hist) > 1 else None,
        },
    }
    ws = PlanWorkspace(cfg_dict=cfg_dict, wm=model, dset=dset, env=env,
                       env_name=model_cfg.env.name, frameskip=model_cfg.frameskip,
                       wandb_run=DummyWandbRun())
    return ws, model, env, model_cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--n_evals", type=int, default=2)
    ap.add_argument("--goal_H", type=int, default=3)
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--opt_steps", type=int, default=2)
    ap.add_argument("--n_rollout", type=int, default=20)
    ap.add_argument("--record_path", default="cem_records.npz")
    ap.add_argument("--record_all_evals", type=int, nargs="*", default=[])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eval_range", type=int, nargs=2, default=None,
                    help="plan only instances [a, b) of the n_evals built (same instances, global ids); "
                         "merge halves with analysis/merge_records.py")
    a = ap.parse_args()
    ws, model, env, mcfg = build(a.model_path, a.n_evals, a.goal_H, a.num_samples,
                                 a.topk, a.opt_steps, a.n_rollout, a.record_path,
                                 record_all_evals=a.record_all_evals, device=a.device)
    if a.eval_range is not None:
        lo, hi = a.eval_range
        # slice every per-instance structure the planner and evaluator touch; ids stay global
        for k in ws.obs_0: ws.obs_0[k] = ws.obs_0[k][lo:hi]
        for k in ws.obs_g: ws.obs_g[k] = ws.obs_g[k][lo:hi]
        ws.state_0, ws.state_g, ws.eval_seed = ws.state_0[lo:hi], ws.state_g[lo:hi], ws.eval_seed[lo:hi]
        if ws.gt_actions is not None: ws.gt_actions = ws.gt_actions[lo:hi]
        ev = ws.evaluator; ev.obs_0, ev.obs_g, ev.state_0, ev.state_g, ev.seed = ws.obs_0, ws.obs_g, ws.state_0, ws.state_g, ws.eval_seed
        ws.env.envs = ws.env.envs[lo:hi]; ws.env.num_envs = hi - lo
        ws.planner.eval_offset = lo
        ws.planner.record_all_evals = {i - lo for i in ws.planner.record_all_evals if lo <= i < hi}
        print(f"[run_record] planning instances [{lo}, {hi}) of {a.n_evals}")
    logs = ws.perform_planning()
    shutil.copy("plan_targets.pkl", targets_path_for(a.record_path))   # self-contained record (all n_evals)

    # Make the record self-contained: the env is deterministic, so storing the initial
    # and goal states plus seeds lets the analysis pass regenerate obs_0/obs_g exactly
    # rather than carrying ~60 MB of images around.
    R = dict(np.load(a.record_path))
    R["state_0"] = np.asarray(ws.state_0, dtype=np.float32)
    R["state_g"] = np.asarray(ws.state_g, dtype=np.float32)
    R["eval_seed"] = np.asarray(ws.eval_seed, dtype=np.int64)
    R["frameskip"] = np.asarray(mcfg.frameskip)
    R["goal_H"] = np.asarray(a.goal_H)
    np.savez_compressed(a.record_path, **R)
    print("PLANNING_DONE", logs)
