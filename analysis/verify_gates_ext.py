"""E0-extended gates A0.1–A0.7 for a new environment (experiment-0-extended-outline.md §5).

A0.1 env determinism: same seed/state/actions twice -> identical states and frames.
A0.2 model determinism (two rollouts of the same actions are identical; dropout off).
A0.3 replay reproducibility: recomputed imagined loss == logged loss (via verify_gates' gate 3).
A0.4 frameskip alignment: drift[h=0] masked (nan) and tiny before masking.
A0.5 goal reachability: the dataset's own 25-step action sequence, executed open-loop from
     the instance's start, reaches success on >= 80 % of instances (dset goals only).
A0.6 baseline success: reported (the rule and fallback live in the outline).
A0.7 truncated-history cost: one-step visual |e| from a 1-frame root vs a 3-frame root on
     N dataset transitions; pass if ratio <= 1.5.

    PYTHONPATH=. python analysis/verify_gates_ext.py --model_path M --n_evals 10 --n_trans 200
"""
import warnings, argparse; warnings.filterwarnings("ignore")
import numpy as np, torch
from einops import rearrange
from utils import move_to_device
from analysis.run_record import build
from analysis.optimism import metrics, align_imagined_to_real


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--n_evals", type=int, default=10)
    ap.add_argument("--n_trans", type=int, default=200)
    ap.add_argument("--num_samples", type=int, default=64)
    ap.add_argument("--opt_steps", type=int, default=2)
    ap.add_argument("--record_path", default="rec_ext.npz")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    ws, model, env, mcfg = build(a.model_path, n_evals=a.n_evals, goal_H=5, num_samples=a.num_samples,
                                 topk=8, opt_steps=a.opt_steps, record_path=a.record_path, device=a.device)
    fs, H = int(mcfg.frameskip), 5
    pre, dev = ws.data_preprocessor, ws.device
    ok = {}
    print(f"env {ws.env_name}; num_hist {int(mcfg.num_hist)}; goal_source {ws.goal_source}; n_evals {a.n_evals}")

    # ---- A0.1 env determinism
    j = 0; env_j = ws.env.envs[j]
    rng = np.random.default_rng(0)
    acts = rng.normal(size=(H * fs, ws.dset.action_dim)).astype(np.float32) * 0.5
    exec_a = pre.denormalize_actions(torch.tensor(acts)).numpy()
    o1, s1 = env_j.rollout(ws.eval_seed[j], ws.state_0[j], exec_a)
    o2, s2 = env_j.rollout(ws.eval_seed[j], ws.state_0[j], exec_a)
    ds, dv = np.abs(s1 - s2).max(), np.abs(o1["visual"].astype(float) - o2["visual"].astype(float)).max()
    print(f"A0.1 env determinism: max|Δstate| {ds:.2e}, max|Δframe| {dv:.2e}")
    ok["A0.1"] = ds == 0 and dv == 0

    # ---- A0.2 model determinism
    obs0 = {k: v[j:j + 1] for k, v in ws.obs_0.items()}
    t0 = move_to_device(pre.transform_obs(obs0), dev)
    at = torch.tensor(rearrange(acts, "(t f) d -> 1 t (f d)", f=fs)).to(dev)
    with torch.no_grad():
        z1, _ = model.rollout(obs_0=t0, act=at); z2, _ = model.rollout(obs_0=t0, act=at)
    dz = max((z1[k] - z2[k]).abs().max().item() for k in z1)
    print(f"A0.2 model determinism: max|Δz| {dz:.2e}; training mode {model.training}")
    ok["A0.2"] = dz == 0 and not model.training

    # ---- A0.5 goal reachability (dset goals carry the ground-truth actions)
    if ws.gt_actions is not None:
        gt = ws.gt_actions.to(dev)                                  # (B, H, f*d) normalised
        logs, succ, _, e_states = ws.evaluator.eval_actions(gt, filename="gt_check")
        fin = e_states[:, -1]
        print(f"A0.5 goal reachability: ground-truth plan success {np.mean(succ):.2f} on {len(succ)} instances; "
              f"mean final state dist {np.mean(np.linalg.norm(fin - np.asarray(ws.state_g), axis=-1)):.2f}")
        ok["A0.5"] = np.mean(succ) >= 0.8
    else:
        print("A0.5 skipped (random_state goals)")

    # ---- A0.3 / A0.4 / A0.6 via a short planning run
    ws.perform_planning()
    R = np.load(a.record_path)
    sel = (R["eval_idx"] == j) & (R["opt_step"] == 0)
    acts_t = torch.tensor(R["action_seq"][sel]).to(dev)
    K = acts_t.shape[0]
    obs0 = {k: np.repeat(v[j:j + 1], K, axis=0) for k, v in ws.obs_0.items()}
    obsg = {k: np.repeat(v[j:j + 1], K, axis=0) for k, v in ws.obs_g.items()}
    with torch.no_grad():
        i_z, _ = model.rollout(obs_0=move_to_device(pre.transform_obs(obs0), dev), act=acts_t)
        z_g = model.encode_obs(move_to_device(pre.transform_obs(obsg), dev))
    obj = __import__("planning.objectives", fromlist=["x"]).create_objective_fn(ws.planner.objective_fn.__closure__ and 1 or 1, 2, "last")
    loss_re = ws.planner.objective_fn(i_z, {k: v[:, -1:] for k, v in z_g.items()}).detach().cpu().numpy()
    d3 = np.abs(loss_re - R["imagined_loss"][sel]).max()
    print(f"A0.3 replay reproducibility: max|recomputed − logged| = {d3:.2e}")
    ok["A0.3"] = d3 < 1e-5
    exec_a = pre.denormalize_actions(rearrange(acts_t.cpu(), "b t (f d) -> b (t f) d", f=fs)).numpy()
    e = [env_j.rollout(ws.eval_seed[j], ws.state_0[j], exec_a[k]) for k in range(K)]
    raw = {"visual": np.stack([o[0]["visual"] for o in e]), "proprio": np.stack([o[0]["proprio"] for o in e])}
    with torch.no_grad():
        z_r = model.encode_obs(move_to_device(pre.transform_obs(raw), dev))
    idx = align_imagined_to_real(H, fs)
    zr_al = {k: v[:, idx] for k, v in z_r.items()}
    zg_b = {k: v[:, -1:].repeat(1, H + 1, *([1] * (v.ndim - 2))) for k, v in z_g.items()}
    m = metrics({k: v.cpu().numpy() for k, v in i_z.items()}, {k: v.cpu().numpy() for k, v in zr_al.items()},
                {k: v.cpu().numpy() for k, v in zg_b.items()}, alpha=1.0, part="visual")
    print(f"A0.4 alignment: drift[h=0] max {np.nanmax(m['drift'][:, 0]):.2e} (masked to nan in A); visual drift by depth "
          f"{np.round(np.nanmean(m['drift'], 0), 4)}; A by depth {np.round(np.nanmean(m['alignment'], 0), 3)}")
    ok["A0.4"] = np.nanmax(m["drift"][:, 0]) < 1e-4 and np.isnan(m["alignment"][:, 0]).all()
    import json
    last = json.loads(open(ws.log_filename).read().strip().split("\n")[-1])
    sr = [v for k, v in last.items() if k.endswith("success_rate")]
    print(f"A0.6 baseline success (this short run, {a.num_samples}x{a.opt_steps}): {sr[0] if sr else 'n/a'} — the A1 run decides")

    # ---- A0.7 truncated-history cost (only meaningful when num_hist > 1)
    nh = int(mcfg.num_hist)
    if nh > 1:
        ds_ = getattr(ws.dset, "dataset", ws.dset)   # TrajSubset (random split) or the dataset itself (folder split)
        e1, e3 = [], []
        rng = np.random.default_rng(0); n = 0
        for i in range(len(ds_)):
            T = ds_.get_seq_length(i)
            if T < (nh + 1) * fs + 1:
                continue
            obs, act, state, _ = ds_.get_frames(i, range(T))
            for t in rng.choice(np.arange((nh - 1) * fs, T - fs), size=min(4, T - nh * fs), replace=False):
                # 1-frame root at t; 3-frame root at t-2fs, t-fs, t; predict t+fs; target = encoded frame t+fs
                for k, hist in ((1, [t]), (nh, [t - (nh - 1 - h) * fs for h in range(nh)])):
                    vis = obs["visual"][hist].float()[None]; pro = obs["proprio"][hist].float()[None]
                    a_hist = torch.stack([act[h:h + fs].reshape(-1) for h in hist]).float()[None]   # (1, k, f*d)
                    with torch.no_grad():
                        z_pred, _ = model.rollout(obs_0={"visual": vis.to(dev), "proprio": pro.to(dev)}, act=a_hist.to(dev))
                        z_tgt = model.encode_obs({"visual": obs["visual"][[t + fs]].float()[None].to(dev),
                                                  "proprio": obs["proprio"][[t + fs]].float()[None].to(dev)})
                    err = ((z_pred["visual"][0, -1] - z_tgt["visual"][0, 0]) ** 2).mean().sqrt().item()
                    (e1 if k == 1 else e3).append(err)
                n += 1
            if n >= a.n_trans:
                break
        r = np.mean(e1) / np.mean(e3)
        print(f"A0.7 truncated history: one-step visual |e| 1-frame root {np.mean(e1):.4f} vs {nh}-frame root {np.mean(e3):.4f} "
              f"on {len(e1)} transitions; ratio {r:.3f} (pass <= 1.5)")
        ok["A0.7"] = r <= 1.5
    else:
        print("A0.7 skipped (num_hist = 1)")

    print("\n==== gates ====")
    for k, v in ok.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")


if __name__ == "__main__":
    main()
