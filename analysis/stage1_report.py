"""Stage 1 readouts from MCTS records (stage-1-outline.md §5, §8). Runs locally on the
records alone — no model, no env.

    python analysis/stage1_report.py --records ../data/experiment-outputs/stage1/*.npz --c 0.5

Prints, for each arm and c: the K-sweep (success, plan depth, state distance, imagined
objective, shortfall); the c_uct sensitivity of 1.1-S; and, at the chosen c, the paired
comparisons of §8: 1.2-G − 1.1-G, 1.1-L − 1.1-G, 1.2 − 1.2-G, and the double difference
d_i = (s_1.2 − s_1.2-G) − (s_1.1-L − s_1.1-G) bootstrapped over instances.
"""
import argparse, collections, glob, re
import numpy as np


def load(paths):
    recs = {}
    for p in paths:
        R = dict(np.load(p, allow_pickle=False))
        sim, val = str(R["simulator"]), str(R["value"])
        arm = {("env", "state"): "1.1-S", ("env", "latent"): "1.1-L", ("wm", "latent"): "1.2"}[(sim, val)]
        recs[(arm, round(float(R["c_uct"]), 4))] = R
    return recs


def boot(x, n=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def mcnemar(a, b):
    """Paired success vectors -> (rate diff, #discordant, two-sided exact p)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
    n = n01 + n10
    from math import comb
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(0, min(n01, n10) + 1)) / 2 ** n)
    return a.mean() - b.mean(), n, p


def sweep_table(R, label):
    K, s = R["snap_K"], R["snap_success"]
    print(f"\n--- {label}: K-sweep, n={s.shape[1]} ---")
    print(f"{'K':>6} {'success':>8} {'depth':>6} {'dist':>7} {'imag.loss':>10} {'pv_value':>9} {'nodes':>7} {'sec':>7}")
    for i, k in enumerate(K):
        nodes = (R["sim_idx_expanded"] < k).sum() / s.shape[1] if "sim_idx_expanded" in R else np.nan
        print(f"{k:>6} {s[i].mean():>8.3f} {R['snap_plan_depth'][i].mean():>6.2f} "
              f"{R['snap_state_dist'][i].mean():>7.2f} {R['snap_imagined_loss'][i].mean():>10.4f} "
              f"{R['snap_pv_value'][i].mean():>9.3f} {nodes:>7.0f} {R['snap_elapsed'][i]:>7.1f}")
    if "greedy_success" in R:
        print(f"{'greedy':>6} {R['greedy_success'].mean():>8.3f} {5:>6.2f} "
              f"{R['greedy_state_dist'].mean():>7.2f} {R['greedy_imagined_loss'].mean():>10.4f} "
              f"{R['greedy_pv_value'].mean():>9.3f}")
    if "N_depth1" in R:
        print("  N per depth (median / p90 / max / frac N=1): " + "; ".join(
            f"d{d}: {np.median(R[f'N_depth{d}']):.0f}/{np.percentile(R[f'N_depth{d}'], 90):.0f}/"
            f"{R[f'N_depth{d}'].max()}/{(R[f'N_depth{d}'] == 1).mean():.2f}"
            for d in range(1, int(R["H"]) + 1) if f"N_depth{d}" in R and len(R[f"N_depth{d}"])))
        print(f"  G4: calls {int(R['n_calls'])}, nodes {int(R['nodes_created'].sum())}, "
              f"shortfall K-nodes mean {np.mean(int(R['K']) - R['nodes_created']):.0f}")
        rt = R["snap_pv_roundtrip"][-1]
        if np.isfinite(rt).any():
            print(f"  G3 PV round-trip max|dz| over {np.isfinite(rt).sum()} full-depth plans: {np.nanmax(rt):.2e}")


def at_K(R, K):
    i = int(np.flatnonzero(R["snap_K"] == K)[0])
    return R["snap_success"][i].astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--c", type=float, default=None, help="c_uct at which the arms are read")
    ap.add_argument("--K", type=int, default=1000)
    ap.add_argument("--fresh_from", type=int, default=None,
                    help="also report the paired comparisons on instances >= this index alone "
                         "(Stage 2 §5.2: instances 0-49 are Stage 1's; 50+ are the fresh test)")
    a = ap.parse_args()
    recs = load([p for g in a.records for p in sorted(glob.glob(g))])
    arms = sorted(set(k[0] for k in recs)); cs = sorted(set(k[1] for k in recs))
    print("records:", ", ".join(f"{k[0]}@c={k[1]}" for k in sorted(recs)))

    # ---- c_uct sensitivity, every arm that has more than one c
    for arm in arms:
        rows = [(c, recs[(arm, c)]) for c in cs if (arm, c) in recs]
        if len(rows) < 2:
            continue
        Ks = rows[0][1]["snap_K"]
        print(f"\n=== {arm}: success by c_uct (columns K) ===")
        print(f"{'c':>7} " + " ".join(f"{k:>7}" for k in Ks) + "   depth@Kmax")
        for c, R in rows:
            print(f"{c:>7} " + " ".join(f"{R['snap_success'][i].mean():>7.3f}" for i in range(len(Ks)))
                  + f"   {R['snap_plan_depth'][-1].mean():.2f}")
    for (arm, c), R in sorted(recs.items()):
        sweep_table(R, f"{arm} c={c}")

    # ---- paired comparisons at the chosen c
    c = a.c if a.c is not None else cs[0]
    g = {arm: recs.get((arm, c)) for arm in ("1.1-S", "1.1-L", "1.2")}
    if g["1.2"] is None:
        return
    K = a.K
    print(f"\n=== paired comparisons at c={c}, K={K} (McNemar exact, two-sided) ===")
    S = {}
    for arm in ("1.1-S", "1.1-L", "1.2"):
        if g[arm] is not None:
            S[arm] = at_K(g[arm], K)
            if "greedy_success" in g[arm]:
                S[arm + "-G"] = g[arm]["greedy_success"].astype(bool)
    for x, y in (("1.2-G", "1.1-G"), ("1.1-L", "1.1-G"), ("1.2", "1.2-G"), ("1.2", "1.1-L"),
                 ("1.1-L", "1.1-S"), ("1.1-S", "1.1-G")):
        x, y = x.replace("1.1-G", "1.1-L-G"), y.replace("1.1-G", "1.1-L-G")
        if x in S and y in S:
            d, n, p = mcnemar(S[x], S[y])
            print(f"  {x.replace('1.1-L-G', '1.1-G'):>7} − {y.replace('1.1-L-G', '1.1-G'):<7}: "
                  f"{S[x].mean():.2f} − {S[y].mean():.2f} = {d:+.3f}   discordant {n:>2}   p = {p:.3f}")
    if all(k in S for k in ("1.2", "1.2-G", "1.1-L", "1.1-L-G")):
        di = (S["1.2"].astype(int) - S["1.2-G"].astype(int)) - (S["1.1-L"].astype(int) - S["1.1-L-G"].astype(int))
        D = {arm: g[a2]["snap_state_dist"][int(np.flatnonzero(g[a2]["snap_K"] == K)[0])] if not arm.endswith("-G")
             else g[a2]["greedy_state_dist"] for arm, a2 in (("1.2", "1.2"), ("1.2-G", "1.2"), ("1.1-L", "1.1-L"), ("1.1-G", "1.1-L"))}
        dd = (D["1.2"] - D["1.2-G"]) - (D["1.1-L"] - D["1.1-G"])
        s0, sg = g["1.2"]["state_0"][:, :2].astype(float), g["1.2"]["state_g"][:, :2].astype(float)
        det = np.linalg.norm(s0 - [32, 30.], axis=1) + np.linalg.norm([32, 30.] - sg, axis=1) - np.linalg.norm(s0 - sg, axis=1)
        subsets = [("all", np.ones(len(di), bool)), ("straight", det <= 3), ("detour", det > 3)]
        if a.fresh_from is not None:
            f = np.arange(len(di)) >= a.fresh_from
            subsets += [("fresh", f), ("fresh straight", f & (det <= 3)), ("stage-1 50", ~f)]
        for lab, m in subsets:
            mu, lo, hi = boot(di[m]); md, ld, hd = boot(dd[m])
            print(f"  double difference [{lab:>14}, n={m.sum():>3}]: success {mu:+.3f} [{lo:+.3f}, {hi:+.3f}]   "
                  f"distance {md:+.2f} [{ld:+.2f}, {hd:+.2f}]   "
                  f"rates 1.1-G {S['1.1-L-G'][m].mean():.2f} 1.1-L {S['1.1-L'][m].mean():.2f} 1.2-G {S['1.2-G'][m].mean():.2f} 1.2 {S['1.2'][m].mean():.2f}")
    # K-trend, paired: per-instance success at each K for 1.2 vs 1.1-L
    if g["1.1-L"] is not None:
        print("\n  paired K-trend (1.2 − 1.1-L) per K:")
        for i, k in enumerate(g["1.2"]["snap_K"]):
            d, n, p = mcnemar(g["1.2"]["snap_success"][i], g["1.1-L"]["snap_success"][i])
            print(f"    K={k:>5}: {d:+.3f}  discordant {n:>2}  p={p:.3f}")


if __name__ == "__main__":
    main()
