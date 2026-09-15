"""Stage 2 replay: per-node optimism quantities on an MCTS record (stage-2-outline.md §3–§6).

For every instance: the principal-variation node at each depth and ALL its siblings (the
children of the PV node one level up), plus two samples for B_search — nodes drawn
proportional to N (the tree's query distribution) and uniformly. Each (instance, depth)
is one batched replay: imagined latent from `wm.rollout` (fp32, from the encoded root),
true latent by executing the path in the env and encoding (E0's `replay_batch`).

Writes one .npz with a per-node table and a per-(instance, depth) sibling-set table:
  nodes:   inst, depth, node_id, N, on_plan, is_sib, is_qN, is_qU, action_idx (H, padded -1),
           and per part in {both, visual, proprio}: A_goal, A_mean, G, drift (|e|), dist (|u|),
           V_imag, V_true, dzbar_model, dzbar_true
  sibsets: inst, depth, n_sib, S_model, S_true (per part), null_sd (per part)
  gates:   S1 |V_imag(PV) - V_leaf|, S2 for env records |V_true(PV) - V_leaf|, PV actions equal
Plus a training-distribution reference (one-step error on dataset transitions).

    python analysis/stage2_replay.py --model_path M --record_path mcts_wm_c0.2_K2000.npz \
        --out stage2_wm.npz --device cuda
"""
import argparse, time, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
from einops import rearrange
from utils import move_to_device
from analysis.optimism import wdot, _sub, metrics
from analysis.aggregate import replay_batch, goal_latents, tree_paths
from analysis.confound import dataset_mean_latent
from analysis.run_mcts import build

PARTS = ("both", "visual", "proprio")


def part_norm(x, part, alpha=1.0):
    return np.sqrt(np.maximum(wdot(x, x, alpha, part), 0.0))


def pairwise_spread(z, part, alpha=1.0):
    """Mean pairwise distance (planner metric) among the rows of z (dict of (n, ...))."""
    n = z["visual"].shape[0]
    if n < 2:
        return np.nan
    d = []
    for i in range(n):
        for j in range(i + 1, n):
            zi = {k: v[i:i + 1] for k, v in z.items()}; zj = {k: v[j:j + 1] for k, v in z.items()}
            d.append(float(part_norm(_sub(zi, zj), part, alpha)[0]))
    return float(np.mean(d))


def shuffled_null_sd(i_z, z_r, z_g, part, alpha=1.0, n=40, seed=0):
    """sd of A under pairing each node's e with another node's u (E0's null), within a set."""
    rng = np.random.default_rng(seed)
    e = _sub(i_z, z_r); u = _sub(z_r, z_g)
    B = e["visual"].shape[0]
    if B < 3:
        return np.nan
    out = []
    for _ in range(n):
        p = rng.permutation(B)
        up = {k: u[k][p] for k in u}
        ee = wdot(e, e, alpha, part); uu = wdot(up, up, alpha, part); ue = wdot(up, e, alpha, part)
        out.append(-ue / np.maximum(np.sqrt(np.maximum(ee * uu, 0)), 1e-12))
    return float(np.nanstd(np.concatenate(out)))


def select_nodes(R, j, rng, n_q=30):
    """Per depth: PV node + all its siblings; plus B_search samples (N-proportional, uniform)."""
    sel, paths = tree_paths(R, j)
    depth, N, on_plan, par, nid = (R[k][sel] for k in ("depth", "N", "on_plan", "parent_id", "node_id"))
    pv_ids = set(int(x) for x in nid[on_plan])
    is_sib = np.zeros(len(sel), bool); is_pv = on_plan.copy()
    for d in range(1, int(depth.max()) + 1):
        at = np.flatnonzero(depth == d)
        is_sib[at[(~on_plan[at]) & np.isin(par[at], list(pv_ids))]] = True
    nonroot = np.flatnonzero(depth > 0)
    w = N[nonroot].astype(float); w /= w.sum()
    qN = np.unique(rng.choice(nonroot, size=min(n_q, len(nonroot)), replace=True, p=w))
    qU = rng.choice(nonroot, size=min(n_q, len(nonroot)), replace=False)
    is_qN = np.zeros(len(sel), bool); is_qN[qN] = True
    is_qU = np.zeros(len(sel), bool); is_qU[qU] = True
    keep = np.flatnonzero((is_pv | is_sib | is_qN | is_qU) & (depth > 0))   # root has no path
    return sel, paths, keep, is_pv, is_sib, is_qN, is_qU


def training_reference(ws, model, zbar, n_episodes, frameskip, alpha, device):
    """One-step model error on dataset transitions: |e| and A_mean at depth 1, plus the
    sibling-style spread of predicted vs encoded outcomes is not defined here (one action
    per state), so only |e|, A_mean and V-independent quantities are reported."""
    pre = ws.data_preprocessor
    ds = ws.dset.dataset
    out = {p: {"drift": [], "A_mean": []} for p in PARTS}
    n_trans = 0
    for i in range(n_episodes):
        img = torch.load(Path(ds.data_path) / "obses" / f"episode_{i:03d}.pth", weights_only=True)
        T = min(img.shape[0], ds.states.shape[1])
        ts = np.arange(0, T - frameskip, frameskip)
        vis = img.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
        raw_states = torch.load(Path(ds.data_path) / "states.pth", weights_only=True)[i, :T].numpy()
        obs0 = {"visual": vis[ts][:, None], "proprio": raw_states[ts][:, None]}
        obs1 = {"visual": vis[ts + frameskip][:, None], "proprio": raw_states[ts + frameskip][:, None]}
        act = ds.actions[i][:T]                                   # normalised (T, 2)
        a = torch.stack([act[t:t + frameskip].reshape(-1) for t in ts]).float()[:, None]  # (n, 1, 10)
        with torch.no_grad():
            t0 = move_to_device(pre.transform_obs(obs0), device)
            i_z, _ = model.rollout(obs_0=t0, act=a.to(device))    # (n, 2, ...): t=0 and prediction
            z1 = model.encode_obs(move_to_device(pre.transform_obs(obs1), device))
        zh = {k: v[:, -1].cpu().numpy() for k, v in i_z.items()}
        zr = {k: v[:, 0].cpu().numpy() for k, v in z1.items()}
        zb = {k: np.broadcast_to(zbar[k], zr[k].shape) for k in zr}
        for p in PARTS:
            m = metrics({k: v[:, None] for k, v in zh.items()}, {k: v[:, None] for k, v in zr.items()},
                        {k: v[:, None] for k, v in zb.items()}, alpha=alpha, part=p)
            out[p]["drift"].append(m["drift"][:, 0]); out[p]["A_mean"].append(m["alignment"][:, 0])
        n_trans += len(ts)
    return {f"train_{p}_{q}": np.concatenate(v) for p, d in out.items() for q, v in d.items()}, n_trans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--record_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--n_evals", type=int, default=None)
    ap.add_argument("--n_q", type=int, default=30, help="B_search samples per instance, each kind")
    ap.add_argument("--n_episodes", type=int, default=20)
    ap.add_argument("--skip_train_ref", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all_nodes", action="store_true",
                    help="replay every non-root node (Test 2 random-rollout records: no PV, no siblings)")
    a = ap.parse_args()

    R = dict(np.load(a.record_path))
    n_evals = a.n_evals or int(R["inst"].max()) + 1
    H, fs = int(R["H"]), int(R["frameskip"])
    is_env = str(R["simulator"]) == "env"
    ws, model, env, mcfg = build(a.model_path, n_evals=n_evals, goal_H=H, device=a.device,
                                 planner={"n_sims": 1, "simulator": "env", "value": "state"})
    d0 = np.abs(np.asarray(ws.state_0) - R["state_0"]).max(); dg = np.abs(np.asarray(ws.state_g) - R["state_g"]).max()
    assert list(ws.eval_seed) == list(R["eval_seed"]) and d0 < 1e-4 and dg < 1e-4, "conditions differ from record"
    print(f"[stage2] conditions reproduced (state_0 {d0:.1e}, state_g {dg:.1e}); record simulator={R['simulator']}")
    data_path = Path(mcfg.env.dataset.data_path)
    zbar, n_frames = dataset_mean_latent(ws, model, data_path, a.n_episodes, ws.device)
    print(f"[stage2] dataset-mean latent over {n_frames} frames")
    flat = torch.as_tensor(R["action_set_flat"])
    rng = np.random.default_rng(a.seed)

    nodes = {k: [] for k in ("inst", "depth", "node_id", "N", "on_plan", "is_sib", "is_qN", "is_qU", "V_leaf")}
    nodes["action_idx"] = []
    for p in PARTS:
        for q in ("A_goal", "A_mean", "G", "drift", "dist", "V_imag", "V_true", "dzbar_model", "dzbar_true"):
            nodes[f"{p}_{q}"] = []
    sibsets = {k: [] for k in ("inst", "depth", "n_sib")}
    for p in PARTS:
        for q in ("S_model", "S_true", "null_sd"):
            sibsets[f"{p}_{q}"] = []
    t0 = time.time(); n_done = 0
    for j in range(n_evals):
        sel, paths, keep, is_pv, is_sib, is_qN, is_qU = select_nodes(R, j, rng, a.n_q)
        if a.all_nodes:
            keep = np.flatnonzero(R["depth"][sel] > 0)
        depth = R["depth"][sel]
        for d in sorted(set(int(depth[k]) for k in keep)):
            ks = [k for k in keep if depth[k] == d]
            acts = flat[torch.as_tensor(np.array([paths[k] for k in ks]))]        # (n, d, 10)
            i_z, z_r = replay_batch(ws, model, j, acts, fs)
            z_g = goal_latents(ws, model, j, acts.shape[0], i_z["visual"].shape[1])
            zh = {k: v[:, -1] for k, v in i_z.items()}; zr = {k: v[:, -1] for k, v in z_r.items()}
            zg = {k: v[:, -1] for k, v in z_g.items()}
            zb = {k: np.broadcast_to(zbar[k], zr[k].shape) for k in zr}
            for k in ks:
                nodes["inst"].append(j); nodes["depth"].append(d); nodes["node_id"].append(int(R["node_id"][sel][k]))
                nodes["N"].append(int(R["N"][sel][k])); nodes["on_plan"].append(bool(is_pv[k]))
                nodes["is_sib"].append(bool(is_sib[k])); nodes["is_qN"].append(bool(is_qN[k])); nodes["is_qU"].append(bool(is_qU[k]))
                nodes["V_leaf"].append(float(R["V_leaf"][sel][k]))
                nodes["action_idx"].append(np.array(paths[k] + [-1] * (H - len(paths[k])), np.int32))
            for p in PARTS:
                mg = metrics({k: v[:, None] for k, v in zh.items()}, {k: v[:, None] for k, v in zr.items()},
                             {k: v[:, None] for k, v in zg.items()}, alpha=a.alpha, part=p)
                mm = metrics({k: v[:, None] for k, v in zh.items()}, {k: v[:, None] for k, v in zr.items()},
                             {k: v[:, None] for k, v in zb.items()}, alpha=a.alpha, part=p)
                v_imag = -wdot(_sub(zh, zg), _sub(zh, zg), a.alpha, p)
                v_true = -wdot(_sub(zr, zg), _sub(zr, zg), a.alpha, p)
                nodes[f"{p}_A_goal"] += list(mg["alignment"][:, 0]); nodes[f"{p}_A_mean"] += list(mm["alignment"][:, 0])
                nodes[f"{p}_G"] += list(mg["gap"][:, 0]); nodes[f"{p}_drift"] += list(mg["drift"][:, 0])
                nodes[f"{p}_dist"] += list(mg["dist_to_goal"][:, 0])
                nodes[f"{p}_V_imag"] += list(v_imag); nodes[f"{p}_V_true"] += list(v_true)
                nodes[f"{p}_dzbar_model"] += list(part_norm(_sub(zh, zb), p, a.alpha))
                nodes[f"{p}_dzbar_true"] += list(part_norm(_sub(zr, zb), p, a.alpha))
            # sibling set at this depth = PV node + its siblings (both are children of the PV parent)
            ss = [i for i, k in enumerate(ks) if is_pv[k] or is_sib[k]]
            if len(ss) >= 2:
                sibsets["inst"].append(j); sibsets["depth"].append(d); sibsets["n_sib"].append(len(ss))
                zh_s = {k: v[ss] for k, v in zh.items()}; zr_s = {k: v[ss] for k, v in zr.items()}; zg_s = {k: v[ss] for k, v in zg.items()}
                for p in PARTS:
                    sibsets[f"{p}_S_model"].append(pairwise_spread(zh_s, p, a.alpha))
                    sibsets[f"{p}_S_true"].append(pairwise_spread(zr_s, p, a.alpha))
                    sibsets[f"{p}_null_sd"].append(shuffled_null_sd(zh_s, zr_s, zg_s, p, a.alpha, seed=a.seed + j))
            n_done += len(ks)
        if j % 5 == 4:
            print(f"[stage2] instance {j + 1}/{n_evals}: {n_done} nodes, {time.time() - t0:.0f}s", flush=True)

    out = {"nodes_" + k: np.array(v) for k, v in nodes.items()}
    out.update({"sib_" + k: np.array(v) for k, v in sibsets.items()})
    for k in ("state_0", "state_g", "eval_seed", "K", "c_uct", "H", "frameskip", "simulator", "value", "alpha",
              "snap_actions", "snap_K", "snap_success", "snap_state_dist"):
        if k in R:
            out[k] = R[k]
    # gates S1/S2: replayed V on the PV vs recorded V_leaf; PV actions vs the extracted plan
    pv = out["nodes_on_plan"]
    v_re = out["nodes_both_V_true"] if is_env else out["nodes_both_V_imag"]
    out["gate_dV_pv"] = np.abs(v_re[pv] - out["nodes_V_leaf"][pv]) if pv.any() else np.array([np.nan])
    out["gate_dV_all"] = np.abs(v_re - out["nodes_V_leaf"]) if np.isfinite(out["nodes_V_leaf"]).any() else np.array([np.nan])
    ok = True
    for j in range(n_evals):
        m = pv & (out["nodes_inst"] == j) & (out["nodes_depth"] == H)
        if m.any():
            ok &= bool((out["nodes_action_idx"][m][0] == R["snap_action_idx"][-1][j]).all())
    out["gate_pv_actions_match"] = np.asarray(ok)
    print(f"[stage2] {n_done} nodes replayed in {time.time() - t0:.0f}s; "
          f"gate {'S2' if is_env else 'S1'} max|dV| on PV {np.nanmax(out['gate_dV_pv']):.2e}, all nodes {np.nanmax(out['gate_dV_all']):.2e}; "
          f"PV actions match plan: {ok}")
    if not a.skip_train_ref:
        ref, n_tr = training_reference(ws, model, zbar, a.n_episodes, fs, a.alpha, ws.device)
        out.update(ref); out["train_n"] = np.asarray(n_tr)
        print(f"[stage2] training reference: {n_tr} transitions; visual |e| {np.nanmean(ref['train_visual_drift']):.4f}, "
              f"A_mean {np.nanmean(ref['train_visual_A_mean']):.4f}")
    np.savez_compressed(a.out, **out)
    print(f"[stage2] wrote {a.out}")
    print("STAGE2_REPLAY_DONE")


if __name__ == "__main__":
    main()
