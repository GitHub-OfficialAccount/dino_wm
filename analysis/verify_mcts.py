"""Stage 1 gates G0–G5 at tiny scale (`stage-1-outline.md` §7). CPU, n_evals=2, K<=30.

G0  EnvSimulator exactness: no-render step == env.rollout positions; rendered+encoded
    latent == encode of the rollout's frame; _LatentValue == objective_fn.
G1  search works: 1.1-S (env, V_state) reaches the goal at small K on 2 instances.
    (The real G1 — >=0.9 at K=1000 on 50 — is the cluster sweep.)
G2  determinism: same seed twice -> identical node tables (env and wm simulators).
G3  cache round-trip: WMSimulator composed 5 deep (fp32 cache) == wm.rollout bit-exact;
    fp16 deviation measured against it.
G4  budget accounting: calls == nodes_created; shortfall reported.
G5  lockstep equivalence: instance planned alone == the same instance in a batch.

    PYTHONPATH=. python analysis/verify_mcts.py --model_path ../data/checkpoints/outputs/outputs/wall_single
"""
import warnings, argparse, copy, time
warnings.filterwarnings("ignore")
import numpy as np, torch
from einops import rearrange
from utils import move_to_device
from analysis.run_mcts import build
from planning.mcts import MCTSPlanner, EnvSimulator, WMSimulator, _LatentValue, make_action_set
from env.serial_vector_env import SerialVectorEnv


def same_tables(A, B, tol=0.0, keys=("parent_id", "depth", "action_idx", "N", "sim_idx_expanded")):
    """Structure must be identical; values within `tol` (0 for the env simulator, float
    rounding for the world model: batch-1 vs batch-n GEMMs are not bit-identical)."""
    if any(A[k].shape != B[k].shape for k in keys):
        return False, "shape"
    bad = [k for k in keys if not np.array_equal(A[k], B[k])]
    dq = np.nanmax(np.abs(A["Q_raw"] - B["Q_raw"]))
    dv = np.nanmax(np.abs(A["V_leaf"] - B["V_leaf"]))
    return (not bad) and dq <= tol and dv <= tol, f"mismatch {bad} dQ {dq:.2e} dV {dv:.2e}"


def new_planner(ws, **kw):
    base = dict(horizon=ws.goal_H, n_sims=20, wm=ws.wm, action_dim=ws.action_dim,
                objective_fn=ws.planner.objective_fn, preprocessor=ws.data_preprocessor,
                evaluator=ws.evaluator, wandb_run=ws.wandb_run, env=ws.env,
                frameskip=ws.frameskip, snapshot_Ks=(), seed=0)
    base.update(kw)
    return MCTSPlanner(**base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--K_env", type=int, default=2000)
    ap.add_argument("--K_wm", type=int, default=20)
    ap.add_argument("--c_uct", type=float, default=0.5)
    ap.add_argument("--wm_tol", type=float, default=1e-5)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    ok = {}

    ws, model, env, mcfg = build(a.model_path, n_evals=2, goal_H=5, device=a.device,
                                 planner={"n_sims": 1, "simulator": "env", "value": "state"})
    pre, dev, fs = ws.data_preprocessor, ws.device, int(mcfg.frameskip)
    raw, flat, exec_ = make_action_set(pre, frameskip=fs)
    print(f"action set: {len(flat)} actions; per-planner-step displacement (env units, 2x action) "
          f"{sorted(set(np.round(2 * np.linalg.norm(raw.sum(1), axis=-1), 3)))}")

    # ---------------- G0: env simulator exactness
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(flat), size=(2, 5))
    acts_t = flat[torch.as_tensor(idx)]                                   # (2, 5, 10)
    exec_a = pre.denormalize_actions(rearrange(acts_t, "b t (f d) -> b (t f) d", f=fs)).numpy()
    e_obs, e_states = env.rollout(ws.eval_seed, ws.state_0, exec_a)       # (2, 26, 2)
    with torch.no_grad():
        z_g = model.encode_obs(move_to_device(pre.transform_obs(ws.obs_g), dev))
    lv = _LatentValue(z_g, 1.0)
    sim = EnvSimulator(env.envs, exec_, ws.eval_seed, ws.state_0, ws.state_g, value="latent",
                       wm=model, preprocessor=pre, latent_value=lv, frameskip=fs, device=dev)
    st = sim.root([0, 1])
    dpos = 0.0
    for h in range(5):
        st = sim.step([0, 1], st, list(idx[:, h]))
        p = np.stack([s.numpy() for s in st])
        dpos = max(dpos, np.abs(p - e_states[:, (h + 1) * fs]).max())
    print(f"G0 no-render step vs env.rollout: max|dpos| = {dpos:.3e}")
    # encoded latent of the rendered final state vs the rollout's own frame
    obs_last = {k: v[:, -1:] for k, v in e_obs.items()}
    with torch.no_grad():
        z_roll = model.encode_obs(move_to_device(pre.transform_obs(obs_last), dev))
    rend = [sim._render(i, st[i]) for i in range(2)]
    obs_r = {"visual": np.stack([r[0] for r in rend])[:, None],
             "proprio": np.stack([r[1] for r in rend])[:, None]}
    with torch.no_grad():
        z_rend = model.encode_obs(move_to_device(pre.transform_obs(obs_r), dev))
    dz = max((z_rend[k] - z_roll[k]).abs().max().item() for k in z_roll)
    v_lv = lv(z_roll, [0, 1])[0]
    v_obj = -ws.planner.objective_fn(z_roll, z_g).cpu().numpy()
    print(f"G0 rendered-encode vs rollout-encode: max|dz| = {dz:.3e}; "
          f"_LatentValue vs objective_fn: {np.abs(v_lv - v_obj).max():.3e}")
    ok["G0"] = dpos == 0 and dz == 0 and np.abs(v_lv - v_obj).max() < 1e-5

    # ---------------- G1 (small): 1.1-S solves 2 instances at K_env
    t0 = time.time()
    p = new_planner(ws, n_sims=a.K_env, simulator="env", value="state", c_uct=a.c_uct,
                    snapshot_Ks=(10, 30, 100, 300, 1000, 2000))
    acts, _ = p.plan(ws.obs_0, ws.obs_g)
    print(f"G1 1.1-S K={a.K_env} on 2 instances: {time.time() - t0:.1f}s, calls {p.sim.n_calls}")
    for s in p.snapshots:
        exec_a = pre.denormalize_actions(rearrange(torch.as_tensor(s["actions"]),
                                                   "b t (f d) -> b (t f) d", f=fs)).numpy()
        _, es = env.rollout(ws.eval_seed, ws.state_0, exec_a)
        d = np.linalg.norm(es[:, -1, :2] - np.asarray(ws.state_g)[:, :2], axis=-1)
        print(f"   K={s['K']:>5} plan_depth {s['plan_depth']} final dist {np.round(d, 2)} "
              f"success {d < 4.5}  pv_value {np.round(s['pv_value'], 2)}")
    ok["G1 (2 inst.)"] = bool((d < 4.5).all())
    R_env = p.record()
    print(f"G4 (env): calls {p.sim.n_calls} == nodes {int(R_env['nodes_created'].sum())}; "
          f"shortfall per instance {a.K_env - R_env['nodes_created']}")
    ok["G4"] = p.sim.n_calls == int(R_env["nodes_created"].sum())

    # ---------------- G2: determinism (env and wm)
    p2 = new_planner(ws, n_sims=a.K_env, simulator="env", value="state", c_uct=a.c_uct)
    p2.plan(ws.obs_0, ws.obs_g)
    same, msg = same_tables(R_env, p2.record())
    print(f"G2 env determinism: {same} ({msg})")
    t0 = time.time()
    w1 = new_planner(ws, n_sims=a.K_wm, simulator="wm", value="latent", cache_dtype="fp32")
    w1.plan(ws.obs_0, ws.obs_g)
    print(f"   wm K={a.K_wm} on 2 instances: {time.time() - t0:.1f}s")
    w2 = new_planner(ws, n_sims=a.K_wm, simulator="wm", value="latent", cache_dtype="fp32")
    w2.plan(ws.obs_0, ws.obs_g)
    same_w, msg_w = same_tables(w1.record(), w2.record())   # same batch both times: exact
    print(f"G2 wm determinism:  {same_w} ({msg_w})")
    ok["G2"] = same and same_w
    ok["G4"] &= w1.sim.n_calls == int(w1.record()["nodes_created"].sum())

    # ---------------- G3: cache round-trip vs wm.rollout, fp32 then fp16
    trans0 = move_to_device(pre.transform_obs(ws.obs_0), dev)
    with torch.no_grad():
        _, z_full = model.rollout(obs_0=trans0, act=acts_t.to(dev))
    ref = z_full[:, -1]
    out = {}
    for dt in (torch.float32, torch.float16):
        s = WMSimulator(model, flat, lv, dev, cache_dtype=dt)
        st = s.root(trans0)
        for h in range(5):
            st = [s.cache(z) for z in s.step([0, 1], st, list(idx[:, h]))]
        z5 = torch.stack(st).float()
        out[dt] = (z5 - ref).abs().max().item()
        zo, _ = model.separate_emb(z5.unsqueeze(1))
        dv = np.abs(lv(zo, [0, 1])[0] - lv(model.separate_emb(ref.unsqueeze(1))[0], [0, 1])[0]).max()
        out[dt] = dv
        print(f"G3 {str(dt):>14} cache, 5 composed steps vs wm.rollout: max|dz| {(z5 - ref).abs().max().item():.3e}, "
              f"|dV| {dv:.3e}")
    # fp32 is not bit-exact against rollout (cat/slice changes the GEMM path), so the
    # criterion is float rounding; fp16's deviation is the number the outline asked for.
    ok["G3"] = out[torch.float32] < a.wm_tol

    # ---------------- G5: lockstep equivalence (instance 1 alone vs in the batch of 2)
    def alone(ws, j, **kw):
        ev = copy.copy(ws.evaluator)
        ev.obs_0 = {k: v[j:j + 1] for k, v in ws.obs_0.items()}
        ev.obs_g = {k: v[j:j + 1] for k, v in ws.obs_g.items()}
        ev.state_0, ev.state_g = ws.state_0[j:j + 1], ws.state_g[j:j + 1]
        ev.seed = [ws.eval_seed[j]]
        ev.env = SerialVectorEnv([ws.env.envs[j]])
        p = new_planner(ws, evaluator=ev, env=ev.env, **kw)
        p.plan(ev.obs_0, ev.obs_g)
        return p.record()

    def sub(R, j):
        sel = R["inst"] == j
        return {k: v[sel] for k, v in R.items() if k in
                ("parent_id", "depth", "action_idx", "N", "sim_idx_expanded", "Q_raw", "V_leaf")}

    g5 = True
    for kind, kw, tol in (("env", dict(n_sims=a.K_env, simulator="env", value="state", c_uct=a.c_uct), 0.0),
                          ("wm", dict(n_sims=a.K_wm, simulator="wm", value="latent", cache_dtype="fp32"), a.wm_tol)):
        Rb = R_env if kind == "env" else w1.record()
        for j in (0, 1):
            Ra = alone(ws, j, **kw)
            same, msg = same_tables(sub(Ra, 0), sub(Rb, j), tol=tol)
            print(f"G5 {kind} instance {j} alone vs batched: {same} ({msg})")
            g5 &= same
    ok["G5"] = g5

    print("\n==== gates ====")
    for k, v in ok.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print("G6: PlanWorkspace + eval_actions on MCTS output is exercised by run_mcts.py")


if __name__ == "__main__":
    main()
