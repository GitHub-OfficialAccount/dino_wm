"""Confound tests for the Phase A headline (findings §4.1, §9 items 1–3).

Runs on an existing record — no re-planning. Replays every logged candidate once and
saves per-candidate quantities, then reports:

  1. A against the dataset-mean direction (z̄ − z) beside A against the goal (z_g − z).
     If the elite/control contrast is comparably positive on the mean direction, the
     "goal-directed" reading is not supported: regression to the mean is state-dependent
     and the pairing does not cancel it.
  2. Within the CONTROL group alone, A regressed on |u|; and the |u|-stratified contrast.
     If A varies with |u| in an unselected group, the collider produces the contrast
     mechanically and the stratified version must be primary.
  3. Var(ε) and Var(d̂) within control only, and over all candidates where all were
     logged. The logged elite+control set is bimodal, so the headline 17× is an upper
     bound until this runs.
"""
import argparse, collections, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
from utils import move_to_device
from analysis.optimism import wdot, _sub
from analysis.aggregate import replay_batch, goal_latents, boot, ELITE, RANDOM
from analysis.run_record import build

PARTS = ("both", "visual", "proprio")


def dataset_mean_latent(ws, model, data_path, n_episodes, device):
    """Mean latent over dataset frames, encoded through the SAME pipeline as replayed obs.

    Wall: stored episodes are (T, 3, 224, 224) float32 in [0, 255]; the env path hands
    transform_obs uint8 HWC. Cast to exactly that so the encoding is byte-identical (this
    is the path every E0 and Stage 2 number used; unchanged).
    Other environments (PushT: mp4 via decord): frames come through the workspace's own
    dataset class, whose `get_frames` applies the training transform; they are handed to
    `encode_obs` directly, proprio normalised as the dataset does it.
    """
    pre = ws.data_preprocessor
    vs, ps = [], []
    if (Path(data_path) / "obses" / "episode_000.pth").exists():                  # Wall
        states = torch.load(Path(data_path) / "states.pth", weights_only=True)  # (N, T, 2)
        for i in range(n_episodes):
            img = torch.load(Path(data_path) / "obses" / f"episode_{i:03d}.pth", weights_only=True)
            T = min(img.shape[0], states.shape[1])
            vis = img[:T].round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()[None]
            pro = states[i, :T].numpy()[None]
            with torch.no_grad():
                z = model.encode_obs(move_to_device(pre.transform_obs(
                    {"visual": vis, "proprio": pro}), device))
            vs.append(z["visual"][0].cpu().numpy()); ps.append(z["proprio"][0].cpu().numpy())
    else:
        ds = getattr(ws.dset, "dataset", ws.dset)   # TrajSubset or the dataset itself (folder split)
        for i in range(min(n_episodes, len(ds))):
            obs, _, _, _ = ds.get_frames(i, range(ds.get_seq_length(i)))
            vis = obs["visual"].float()[None]; pro = obs["proprio"].float()[None]
            with torch.no_grad():
                z = model.encode_obs({"visual": vis.to(device), "proprio": pro.to(device)})
            vs.append(z["visual"][0].cpu().numpy()); ps.append(z["proprio"][0].cpu().numpy())
    zv = np.concatenate(vs, 0); zp = np.concatenate(ps, 0)
    return {"visual": zv.mean(0), "proprio": zp.mean(0)}, zv.shape[0]


def per_candidate(ws, model, R, zbar, frameskip, device, max_per_group=None, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    zr_means = []
    for j in np.unique(R["eval_idx"]):
        for i in np.unique(R["opt_step"]):
            sel = (R["eval_idx"] == j) & (R["opt_step"] == i)
            if sel.sum() == 0:
                continue
            if max_per_group is not None:
                idx = np.where(sel)[0]; keep = []
                for t in (ELITE, RANDOM):
                    g = idx[R["tag"][idx] == t]
                    keep.append(g if len(g) <= max_per_group else rng.choice(g, max_per_group, replace=False))
                sel = np.zeros_like(sel); sel[np.concatenate(keep)] = True
            acts = torch.tensor(R["action_seq"][sel])
            i_z, z_r = replay_batch(ws, model, int(j), acts, frameskip)
            z_g = goal_latents(ws, model, int(j), acts.shape[0], i_z["visual"].shape[1])
            zb = {k: np.broadcast_to(zbar[k][None, None], z_r[k].shape) for k in z_r}
            zr_means.append({k: z_r[k][:, -1].mean(0) for k in z_r})
            e = _sub(i_z, z_r); u = _sub(z_r, z_g); mvec = _sub(zb, z_r)  # mvec = z̄ − z
            rec = {"eval": np.full(sel.sum(), j), "step": np.full(sel.sum(), i),
                   "tag": R["tag"][sel], "rank": R["imagined_rank"][sel],
                   "dhat": R["imagined_loss"][sel]}
            for part in PARTS:
                ee = wdot(e, e, 1.0, part); uu = wdot(u, u, 1.0, part); mm = wdot(mvec, mvec, 1.0, part)
                ue = wdot(u, e, 1.0, part); me = wdot(mvec, e, 1.0, part)
                ne, nu, nm = np.sqrt(np.maximum(ee, 0)), np.sqrt(np.maximum(uu, 0)), np.sqrt(np.maximum(mm, 0))
                A_goal = np.where(ne > 1e-4, -ue / np.maximum(ne * nu, 1e-12), np.nan)
                A_mean = np.where(ne > 1e-4, me / np.maximum(ne * nm, 1e-12), np.nan)
                rec[f"{part}_A_goal"] = A_goal; rec[f"{part}_A_mean"] = A_mean
                rec[f"{part}_ne"] = ne; rec[f"{part}_nu"] = nu; rec[f"{part}_nm"] = nm
                rec[f"{part}_G"] = -2.0 * ue - ee
            rows.append(rec)
    out = {k: np.concatenate([r[k] for r in rows], 0) for k in rows[0]}
    zr_mean = {k: np.mean([m[k] for m in zr_means], 0) for k in zr_means[0]}
    return out, zr_mean


def ols(x, y):
    m = ~(np.isnan(x) | np.isnan(y)); x, y = x[m], y[m]
    if len(x) < 3: return np.nan, np.nan, len(x)
    b = np.polyfit(x, y, 1)[0]; r = np.corrcoef(x, y)[0, 1]
    return b, r, len(x)


def report(P, part):
    Ag = P[f"{part}_A_goal"][:, -1]; Am = P[f"{part}_A_mean"][:, -1]
    nu = P[f"{part}_nu"][:, -1]; ne = P[f"{part}_ne"][:, -1]; G = P[f"{part}_G"][:, -1]
    tag, ev, st = P["tag"], P["eval"], P["step"]
    el, ra = tag == ELITE, tag == RANDOM
    keys = sorted(set(zip(ev, st)))
    def contrast(v):
        c = []
        for (j, i) in keys:
            m = (ev == j) & (st == i)
            a, b = v[m & el], v[m & ra]
            if len(a) and len(b): c.append(np.nanmean(a) - np.nanmean(b))
        return np.array(c)
    print(f"\n############ part = {part} ############")
    print("=== 1. goal direction vs mean direction (elite − control, paired, bootstrap over instances) ===")
    for name, v in (("A_goal", Ag), ("A_mean", Am)):
        mu, lo, hi = boot(contrast(v))
        print(f"  {name:7s} contrast = {mu:+.4f}  [{lo:+.4f}, {hi:+.4f}]   levels: elite {np.nanmean(v[el]):+.4f}  control {np.nanmean(v[ra]):+.4f}")
    print(f"  corr(A_goal, A_mean) over all candidates = {np.corrcoef(Ag[~np.isnan(Ag)&~np.isnan(Am)], Am[~np.isnan(Ag)&~np.isnan(Am)])[0,1]:+.3f}")
    print("  depth profile, mean over all candidates:")
    print("    A_goal:", np.round(np.nanmean(P[f"{part}_A_goal"], 0), 3))
    print("    A_mean:", np.round(np.nanmean(P[f"{part}_A_mean"], 0), 3))

    print("=== 2. within CONTROL only: does A depend on |u|? ===")
    b, r, n = ols(nu[ra], Ag[ra])
    print(f"  pooled OLS  A_goal ~ |u|:  slope {b:+.4f}   r {r:+.3f}   n {n}")
    slopes = [ols(nu[(ev == j) & ra], Ag[(ev == j) & ra])[0] for j in np.unique(ev)]
    mu, lo, hi = boot(slopes)
    print(f"  per-instance slopes: mean {mu:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
    q = np.nanquantile(nu[ra], np.linspace(0, 1, 6))
    print("  control A_goal by |u| quintile:", [f"{np.nanmean(Ag[ra & (nu >= q[k]) & (nu < q[k+1] + (k==4)*1e9)]):+.3f}" for k in range(5)])
    print("  |u| levels: elite %.3f  control %.3f" % (np.nanmean(nu[el]), np.nanmean(nu[ra])))
    print("=== 2b. |u|-stratified contrast (pooled deciles of |u|) ===")
    d = np.nanquantile(nu, np.linspace(0, 1, 11)); strat = []
    for k in range(10):
        m = (nu >= d[k]) & (nu < d[k+1] + (k == 9) * 1e9)
        a, b_ = Ag[m & el], Ag[m & ra]
        if len(a) >= 5 and len(b_) >= 5:
            strat.append((min(len(a), len(b_)), np.nanmean(a) - np.nanmean(b_)))
            print(f"  decile {k}: |u|∈[{d[k]:.2f},{d[k+1]:.2f}]  n_e={len(a):5d} n_c={len(b_):5d}  contrast {np.nanmean(a)-np.nanmean(b_):+.4f}")
    if strat:
        w = np.array([s[0] for s in strat]); c = np.array([s[1] for s in strat])
        print(f"  weighted stratified contrast = {np.sum(w*c)/np.sum(w):+.4f}   (unstratified {np.nanmean(Ag[el]) - np.nanmean(Ag[ra]):+.4f})")

    print("=== 3. Var(ε), Var(d̂) on clean sets ===")
    byv = collections.defaultdict(lambda: ([], [], [], []))
    for (j, i) in keys:
        m = (ev == j) & (st == i)
        c = m & ra
        byv[i][0].append(np.nanvar(G[c])); byv[i][1].append(np.nanvar(P["dhat"][c]))
        if m.sum() >= 200:  # all candidates logged for this instance
            byv[i][2].append(np.nanvar(G[m])); byv[i][3].append(np.nanvar(P["dhat"][m]))
    print("  step   control-only Var(ε)  Var(d̂)   |  all-candidates Var(ε)  Var(d̂)")
    for i in sorted(byv):
        ce, cd, ae, ad = byv[i]
        s = f"  {i:4d}   {np.nanmean(ce):.5f}   {np.nanmean(cd):.5f}"
        s += f"   |  {np.nanmean(ae):.5f}   {np.nanmean(ad):.5f}" if ae else "   |  (n/a)"
        print(s)
    ce0, ce9 = np.nanmean(byv[min(byv)][0]), np.nanmean(byv[max(byv)][0])
    print(f"  control-only Var(ε) ratio last/first = {ce9/ce0:.2f}×")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True); ap.add_argument("--record_path", required=True)
    ap.add_argument("--data_path", required=True); ap.add_argument("--device", default="cpu")
    ap.add_argument("--n_evals", type=int, default=None); ap.add_argument("--n_episodes", type=int, default=20)
    ap.add_argument("--max_per_group", type=int, default=None); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = dict(np.load(a.record_path)); n_evals = a.n_evals or int(R["eval_idx"].max()) + 1
    ws, model, env, mcfg = build(a.model_path, n_evals=n_evals, goal_H=int(R["goal_H"]),
                                 device=a.device, record_path="/dev/null",
                                 targets_path=__import__('analysis.run_record', fromlist=['x']).targets_path_for(a.record_path))
    assert np.abs(np.asarray(ws.state_0) - R["state_0"]).max() < 1e-4
    zbar, nframes = dataset_mean_latent(ws, model, a.data_path, a.n_episodes, a.device)
    print(f"[confound] z̄ from {nframes} dataset frames")
    P, zr_mean = per_candidate(ws, model, R, zbar, int(R["frameskip"]), a.device, a.max_per_group)
    for k in ("visual", "proprio"):  # sanity: z̄ must sit on the replayed-latent manifold
        rel = np.linalg.norm(zbar[k] - zr_mean[k]) / np.linalg.norm(zbar[k])
        print(f"[confound] |z̄ − mean(z_real)| / |z̄|  ({k}) = {rel:.3f}   (expect ≪ 1)")
    np.savez_compressed(a.out, **P, zbar_visual=zbar["visual"], zbar_proprio=zbar["proprio"])
    print(f"[confound] wrote per-candidate quantities to {a.out}")
    for part in PARTS:
        report(P, part)


if __name__ == "__main__":
    main()
