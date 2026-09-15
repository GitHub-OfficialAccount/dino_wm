"""Turn a CEM candidate record into the Experiment 1 readouts (§6, §7).

Primary estimand: A(elite) - A(random), paired within (eval instance, opt_step).
Bootstraps over eval instances, never candidates: candidates within an instance share a
root and a proposal distribution, so treating them as independent inflates significance.
"""
import argparse, collections, warnings; warnings.filterwarnings("ignore")
import numpy as np, torch
from einops import rearrange, repeat
from utils import move_to_device
from analysis.optimism import metrics, align_imagined_to_real, shuffled_null

ELITE, RANDOM = 1, 2
PARTS = ("both", "visual", "proprio")


def replay_batch(ws, model, j, acts_t, frameskip, chunk=64):
    """Imagined and replayed latents for candidates of eval instance j.

    Also leaves the replayed true state trajectories on `ws.last_states` (K, T+1, d) for
    environment-specific readouts (PushT: agent/T decomposition, contact)."""
    pre, dev = ws.data_preprocessor, ws.device
    env_j = ws.env.envs[j]
    iz, zr, st = [], [], []
    for s in range(0, acts_t.shape[0], chunk):
        a = acts_t[s:s + chunk].to(dev)
        K = a.shape[0]
        obs0 = {k: np.repeat(v[j:j + 1], K, axis=0) for k, v in ws.obs_0.items()}
        with torch.no_grad():
            i_z, _ = model.rollout(obs_0=move_to_device(pre.transform_obs(obs0), dev), act=a)
        exec_a = pre.denormalize_actions(
            rearrange(a.cpu(), "b t (f d) -> b (t f) d", f=frameskip)).numpy()
        e = [env_j.rollout(ws.eval_seed[j], ws.state_0[j], exec_a[k]) for k in range(K)]
        raw = {"visual": np.stack([o[0]["visual"] for o in e]),
               "proprio": np.stack([o[0]["proprio"] for o in e])}
        st.append(np.stack([np.asarray(o[1]) for o in e]))
        with torch.no_grad():
            z_r = model.encode_obs(move_to_device(pre.transform_obs(raw), dev))
        idx = align_imagined_to_real(a.shape[1], frameskip)
        iz.append({k: v.cpu().numpy() for k, v in i_z.items()})
        zr.append({k: v[:, idx].cpu().numpy() for k, v in z_r.items()})
    j2 = lambda L: {k: np.concatenate([d[k] for d in L], 0) for k in L[0]}
    ws.last_states = np.concatenate(st, 0)
    return j2(iz), j2(zr)


def pusht_columns(ws, j, rec, tag):
    """PushT-specific readouts (experiment-0-extended-outline.md §3.3), from the replayed
    true states of the last replay_batch: final agent / T progress toward the goal state,
    contact with the T during the 25 env steps, and arena-bound proximity.
    State layout: [agent_x, agent_y, T_x, T_y, angle, agent_vx, agent_vy]."""
    S = ws.last_states                                  # (K, T+1, 7)
    g = np.asarray(ws.state_g[j])
    final = S[:, -1]
    d_agent = np.linalg.norm(final[:, :2] - g[:2], axis=-1)
    d_T = np.linalg.norm(final[:, 2:4] - g[2:4], axis=-1)
    d_ang = np.abs(final[:, 4] - g[4]); d_ang = np.minimum(d_ang, 2 * np.pi - d_ang)
    # contact: agent-T centre distance below the agent radius (15) + half the T's extent (~30)
    contact = (np.linalg.norm(S[:, :, :2] - S[:, :, 2:4], axis=-1) < 45).any(1)
    bound = np.minimum(np.minimum(final[:, :2].min(1), final[:, 2:4].min(1)),
                       np.minimum(512 - final[:, :2].max(1), 512 - final[:, 2:4].max(1)))
    for grp, m in (("elite", tag == ELITE), ("random", tag == RANDOM)):
        if m.any():
            rec[f"d_agent_{grp}"] = float(d_agent[m].mean()); rec[f"d_T_{grp}"] = float(d_T[m].mean())
            rec[f"d_ang_{grp}"] = float(d_ang[m].mean()); rec[f"contact_{grp}"] = float(contact[m].mean())
            rec[f"bound_{grp}"] = float(bound[m].mean())
    rec["contact_mask"] = contact


def goal_latents(ws, model, j, B, T):
    pre, dev = ws.data_preprocessor, ws.device
    obsg = {k: np.repeat(v[j:j + 1], B, axis=0) for k, v in ws.obs_g.items()}
    with torch.no_grad():
        z_g = model.encode_obs(move_to_device(pre.transform_obs(obsg), dev))
    return {k: repeat(v[:, -1:], "b 1 ... -> b t ...", t=T).cpu().numpy()
            for k, v in z_g.items()}


def boot(x, n=10000, seed=0):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def run(ws, model, R, frameskip, alpha=1.0, parts=PARTS, max_per_group=None, seed=0, steps=None):
    """Replay once per candidate set; all parts share that replay.

    max_per_group caps how many elite / control candidates are replayed per
    (eval, opt_step). The env replay dominates stage-2 cost, and precision is
    limited by the number of eval instances (the bootstrap unit), not by
    candidates within one, so capping trades little power for a lot of time.
    """
    rng = np.random.default_rng(seed)
    out = {p: [] for p in parts}
    for j in np.unique(R["eval_idx"]):
        for i in np.unique(R["opt_step"]):
            if steps is not None and int(i) not in steps:
                continue
            sel = (R["eval_idx"] == j) & (R["opt_step"] == i)
            if sel.sum() == 0:
                continue
            if max_per_group is not None:
                idx = np.where(sel)[0]
                keep = []
                for t in (ELITE, RANDOM):
                    g = idx[R["tag"][idx] == t]
                    keep.append(g if len(g) <= max_per_group
                                else rng.choice(g, max_per_group, replace=False))
                sel = np.zeros_like(sel)
                sel[np.concatenate(keep)] = True
            acts = torch.tensor(R["action_seq"][sel])
            # Replay once: the parts differ only in how the metric weights visual vs
            # proprio, so replaying per part would be 3x wasted work.
            i_z, z_r = replay_batch(ws, model, int(j), acts, frameskip)
            z_g = goal_latents(ws, model, int(j), acts.shape[0], i_z["visual"].shape[1])
            tag = R["tag"][sel]
            env_name = getattr(ws, "env_name", "wall")
            for part in parts:
                m = metrics(i_z, z_r, z_g, alpha=alpha, part=part)
                rec = {"eval": int(j), "opt_step": int(i)}
                if env_name == "pusht":
                    pusht_columns(ws, int(j), rec, tag)
                    c = rec.pop("contact_mask")
                    for grp, cm in (("contact", c), ("nocontact", ~c)):
                        e_, r_ = (tag == ELITE) & cm, (tag == RANDOM) & cm
                        rec[f"A_contrast_{grp}"] = (float(np.nanmean(m["alignment"][e_, -1]) - np.nanmean(m["alignment"][r_, -1]))
                                                    if e_.any() and r_.any() else np.nan)
                for name in ("alignment", "gap", "drift", "dist_to_goal"):
                    rec[f"{name}_elite"] = (np.nanmean(m[name][tag == ELITE, -1])
                                            if (tag == ELITE).any() else np.nan)
                    rec[f"{name}_random"] = (np.nanmean(m[name][tag == RANDOM, -1])
                                             if (tag == RANDOM).any() else np.nan)
                rec["A_contrast"] = rec["alignment_elite"] - rec["alignment_random"]
                rec["G_contrast"] = rec["gap_elite"] - rec["gap_random"]
                rec["depth_drift"] = np.nanmean(m["drift"], 0)
                rec["depth_align"] = np.nanmean(m["alignment"], 0)
                # Variance share, reported as numerator and denominator separately:
                # both collapse as sigma anneals, so the ratio degenerates to 0/0.
                rec["var_eps"] = float(np.nanvar(m["gap"][:, -1]))
                rec["var_dhat"] = float(np.nanvar(R["imagined_loss"][sel]))
                # Selection regret needs ground truth for *every* candidate, so it is
                # only defined where all of them were logged.
                if (tag == ELITE).any() and sel.sum() >= 200:
                    dstar = m["dist_to_goal"][:, -1] ** 2
                    chosen = dstar[np.argmin(R["imagined_rank"][sel])]
                    rec["regret_model"] = float(chosen - np.nanmin(dstar))
                    rec["regret_random"] = float(np.nanmean(dstar) - np.nanmin(dstar))
                rec["null_sd"] = float(np.nanstd(shuffled_null(
                    i_z, z_r, z_g, alpha=alpha, n=20)[..., -1]))
                out[part].append(rec)
    return out


def report(rows):
    print("\n=== primary: A(elite) - A(random), paired within (eval, opt_step) ===")
    allc = [r["A_contrast"] for r in rows]
    mu, lo, hi = boot(allc)
    nsd = np.nanmean([r["null_sd"] for r in rows])
    print(f"pooled  A_contrast = {mu:+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]  n={len(allc)}")
    print(f"        in empirical-null units: {mu / nsd:+.3f}  (null sd = {nsd:.5f})")
    a = np.asarray(allc, dtype=float); a = a[~np.isnan(a)]
    print(f"sign test: {int((a > 0).sum())}/{len(a)} positive")

    by = collections.defaultdict(list)
    for r in rows:
        by[r["opt_step"]].append(r["A_contrast"])
    print("\nby opt_step (hump expected: rises, then collapses as sigma anneals):")
    for k in sorted(by):
        mu, lo, hi = boot(by[k])
        print(f"  step {k:2d}: {mu:+.5f}  [{lo:+.5f}, {hi:+.5f}]  n={len(by[k])}")

    print("\n=== depth profile ===")
    print("drift    :", np.round(np.nanmean([r["depth_drift"] for r in rows], 0), 5))
    print("alignment:", np.round(np.nanmean([r["depth_align"] for r in rows], 0), 5))

    print("\n=== mechanism diagnostics ===")
    print(f"  Var(eps)  = {np.nanmean([r['var_eps'] for r in rows]):.6f}  (model-error spread)")
    print(f"  Var(dhat) = {np.nanmean([r['var_dhat'] for r in rows]):.6f}  (spread CEM ranks on)")
    rm = [r["regret_model"] for r in rows if "regret_model" in r]
    rr = [r["regret_random"] for r in rows if "regret_random" in r]
    if rm:
        print(f"  selection regret: model={np.nanmean(rm):+.5f}  random-pick={np.nanmean(rr):+.5f}"
              f"  (model < random => still useful for ranking despite bias)")
    byv = collections.defaultdict(lambda: ([], []))
    for r in rows:
        byv[r["opt_step"]][0].append(r["var_eps"])
        byv[r["opt_step"]][1].append(r["var_dhat"])
    print("  by opt_step:")
    for k in sorted(byv):
        e, d = byv[k]
        print(f"    step {k:2d}: Var(eps)={np.nanmean(e):.6f}  Var(dhat)={np.nanmean(d):.6f}")

    if any("d_T_elite" in r for r in rows):
        print("\n=== PushT: object vs agent, contact, bounds (elite / random) ===")
        for q in ("d_agent", "d_T", "d_ang", "contact", "bound"):
            e = np.nanmean([r.get(f"{q}_elite", np.nan) for r in rows]); rr_ = np.nanmean([r.get(f"{q}_random", np.nan) for r in rows])
            print(f"  {q:>8}: elite={e:.3f}  random={rr_:.3f}  diff={e - rr_:+.3f}")
        cf = np.nanmean([r.get("contact_elite", np.nan) for r in rows] + [r.get("contact_random", np.nan) for r in rows])
        print(f"  contact fraction (pooled) = {cf:.2f}  -> stratified contrast {'READABLE' if 0.2 <= cf <= 0.8 else 'UNDERPOWERED (outside 20-80%)'}")
        for grp in ("contact", "nocontact"):
            v = [r.get(f"A_contrast_{grp}", np.nan) for r in rows]
            mu, lo, hi = boot(v)
            print(f"  A_contrast | {grp:>9}: {mu:+.5f}  [{lo:+.5f}, {hi:+.5f}]  n={int(np.isfinite(np.asarray(v, float)).sum())}")
    print("\n=== levels (final depth) ===")
    for k in ("alignment", "gap", "drift", "dist_to_goal"):
        e = np.nanmean([r[f"{k}_elite"] for r in rows])
        rr_ = np.nanmean([r[f"{k}_random"] for r in rows])
        print(f"  {k:14s} elite={e:+.5f}  random={rr_:+.5f}  diff={e - rr_:+.5f}")


# ----------------------------------------------------------------------------- tree mode
def tree_paths(R, j):
    """Node table of instance j and, per node, its action-index path from the root."""
    sel = np.flatnonzero(R["inst"] == j)
    nid, par, act = R["node_id"][sel], R["parent_id"][sel], R["action_idx"][sel]
    lookup = {int(n): k for k, n in enumerate(nid)}
    paths = []
    for k in range(len(sel)):
        p, cur = [], k
        while par[cur] >= 0:
            p.append(int(act[cur])); cur = lookup[int(par[cur])]
        paths.append(p[::-1])
    return sel, paths


def sample_tree_nodes(R, j, rng, max_per_group=None):
    """Which nodes of instance j to replay, and their group label.

    Test 1.3 wants an N-contrast at fixed depth. Where N varies within a depth
    (typically depth 1-2) nodes are binned by N; where it does not (depth 3+, all
    N = 1 except along the principal variation) the contrast is PV node vs its
    off-PV siblings. Both are emitted; the report pairs them within instance.
    """
    sel, paths = tree_paths(R, j)
    depth, N, on_plan, par = (R[k][sel] for k in ("depth", "N", "on_plan", "parent_id"))
    pv_parents = set(int(x) for x in R["node_id"][sel][on_plan])
    rows = []
    for d in range(1, int(depth.max()) + 1):
        at = np.flatnonzero(depth == d)
        if len(at) == 0:
            continue
        pv = at[on_plan[at]]
        sib = at[(~on_plan[at]) & np.isin(par[at], list(pv_parents))]
        if max_per_group is not None and len(sib) > max_per_group:
            sib = rng.choice(sib, max_per_group, replace=False)
        rows += [(int(k), d, "pv") for k in pv] + [(int(k), d, "sib") for k in sib]
        # N-bins: only where N takes more than one value at this depth
        if len(np.unique(N[at])) > 1:
            hi = at[N[at] >= np.percentile(N[at], 75)]
            lo = at[N[at] <= np.percentile(N[at], 25)]
            for grp, g in (("N_hi", hi), ("N_lo", lo)):
                if max_per_group is not None and len(g) > max_per_group:
                    g = rng.choice(g, max_per_group, replace=False)
                rows += [(int(k), d, grp) for k in g]
    return sel, paths, rows


def run_tree(ws, model, R, frameskip, alpha=1.0, parts=PARTS, max_per_group=None, seed=0):
    """Replay sampled tree nodes (imagined vs real at the node's depth); one row per
    (node, group, part). Nodes at the same depth share one batched replay."""
    rng = np.random.default_rng(seed)
    flat = torch.as_tensor(R["action_set_flat"])
    out = {p: [] for p in parts}
    for j in np.unique(R["inst"]):
        sel, paths, rows = sample_tree_nodes(R, int(j), rng, max_per_group)
        for d in sorted(set(r[1] for r in rows)):
            rd = [r for r in rows if r[1] == d]
            ks = sorted(set(r[0] for r in rd))
            acts = flat[torch.as_tensor(np.array([paths[k] for k in ks]))]       # (n, d, 10)
            i_z, z_r = replay_batch(ws, model, int(j), acts, frameskip)
            z_g = goal_latents(ws, model, int(j), acts.shape[0], i_z["visual"].shape[1])
            pos = {k: i for i, k in enumerate(ks)}
            for part in parts:
                m = metrics(i_z, z_r, z_g, alpha=alpha, part=part)
                for k, _, grp in rd:
                    i = pos[k]
                    out[part].append({
                        "inst": int(j), "depth": int(d), "node": int(R["node_id"][sel][k]),
                        "N": int(R["N"][sel][k]), "group": grp,
                        "V_leaf": float(R["V_leaf"][sel][k]),
                        "alignment": float(m["alignment"][i, -1]), "gap": float(m["gap"][i, -1]),
                        "drift": float(m["drift"][i, -1]), "dist": float(m["dist_to_goal"][i, -1]),
                    })
    return out


def report_tree(rows):
    """Paired contrasts at fixed depth: PV vs off-PV siblings, and high-N vs low-N."""
    def paired(g1, g2):
        by = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rows:
            by[(r["inst"], r["depth"])][r["group"]].append(r)
        out = collections.defaultdict(list)
        for (inst, d), g in by.items():
            if g[g1] and g[g2]:
                for q in ("alignment", "gap", "drift", "dist"):
                    a = np.nanmean([r[q] for r in g[g1]]); b = np.nanmean([r[q] for r in g[g2]])
                    out[(d, q)].append(a - b)
        return out
    for g1, g2 in (("pv", "sib"), ("N_hi", "N_lo")):
        c = paired(g1, g2)
        if not c:
            continue
        print(f"\n=== {g1} - {g2}, paired within (instance, depth); bootstrap over instances ===")
        for d in sorted(set(k[0] for k in c)):
            line = f"  depth {d}:"
            for q in ("alignment", "gap", "drift", "dist"):
                v = c.get((d, q), [])
                if v:
                    mu, lo, hi = boot(v)
                    line += f"  {q} {mu:+.4f} [{lo:+.4f},{hi:+.4f}] n={len(v)}"
            print(line)
    print("\n=== levels by depth (all sampled nodes) ===")
    by = collections.defaultdict(list)
    for r in rows:
        by[r["depth"]].append(r)
    for d in sorted(by):
        g = by[d]
        print(f"  depth {d}: n={len(g)}  A={np.nanmean([r['alignment'] for r in g]):+.4f}  "
              f"G={np.nanmean([r['gap'] for r in g]):+.4f}  |e|={np.nanmean([r['drift'] for r in g]):.4f}  "
              f"|u|={np.nanmean([r['dist'] for r in g]):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--record_path", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--n_evals", type=int, default=None)
    ap.add_argument("--max_per_group", type=int, default=None)
    ap.add_argument("--steps", type=int, nargs="*", default=None,
                    help="replay only these opt_steps (e.g. every third of 30) to fit a job")
    ap.add_argument("--tree", action="store_true",
                    help="record is an MCTS node table (analysis/run_mcts.py); replay "
                         "nodes sampled by (depth, N-bin) / PV-vs-sibling instead of CEM candidates")
    a = ap.parse_args()

    R = dict(np.load(a.record_path))
    key = "inst" if a.tree else "eval_idx"
    n_evals = a.n_evals or int(R[key].max()) + 1
    goal_H, fs = int(R["goal_H"]), int(R["frameskip"])

    from analysis.run_record import build
    from analysis.run_record import targets_path_for
    ws, model, env, mcfg = build(a.model_path, n_evals=n_evals, goal_H=goal_H,
                                 device=a.device, record_path="/dev/null",
                                 targets_path=targets_path_for(a.record_path))
    # The env and goal sampling are deterministic given the seed, so rebuilding the
    # workspace reproduces the recorded conditions. Verify rather than overwrite --
    # overwriting state_0 alone would desync it from obs_0, which prepare_targets()
    # built and which the rollouts actually start from.
    d0 = np.abs(np.asarray(ws.state_0) - R["state_0"]).max()
    dg = np.abs(np.asarray(ws.state_g) - R["state_g"]).max()
    assert list(ws.eval_seed) == list(R["eval_seed"]), "eval_seed mismatch vs record"
    assert d0 < 1e-4 and dg < 1e-4, (
        f"rebuilt conditions differ from record (state_0 {d0}, state_g {dg})")
    print(f"[aggregate] conditions reproduced (state_0 {d0:.2e}, state_g {dg:.2e})")

    runner, reporter = (run_tree, report_tree) if a.tree else (run, report)
    kw = {"max_per_group": a.max_per_group}
    if not a.tree:
        kw["steps"] = set(a.steps) if a.steps else None
    for part, rows in runner(ws, model, R, fs, alpha=a.alpha, **kw).items():
        print(f"\n############ part = {part} ############")
        reporter(rows)


if __name__ == "__main__":
    main()
