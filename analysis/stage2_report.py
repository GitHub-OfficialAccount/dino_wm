"""Stage 2 readouts from stage2_replay.py outputs (stage-2-outline.md §4, §6, §7).
Runs locally on the replay files alone.

    python analysis/stage2_report.py --wm stage2_wm.npz --env stage2_envL.npz [--ceiling ceiling.npz]

C1  PV vs siblings on A_goal (+ marginals |e|, |u|; residual-adjusted form)     pooled / straight
C1m the same on A_mean, and the residual of A_goal on A_mean
C2  Spearman(A_goal, log N) among siblings; high-N minus low-N quartile
C3  sibling latent spread S_model / S_true by depth (env record gives S_true independently)
C4  ordering fidelity: Spearman(V_imag, V_true) among siblings; top-1 / top-3 agreement
C5  |e| and A_goal by depth, PV vs siblings
C6  B_search: N-weighted G vs uniform G vs training-distribution |e|
C7  dzbar model vs true; corr(spread, A_mean)
Gates S1–S5.
"""
import argparse, collections
import numpy as np


def boot(x, n=10000, seed=0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), len(x)


def fmt(t):
    m, lo, hi, n = t
    return f"{m:+.4f} [{lo:+.4f},{hi:+.4f}] n={n}" if n else "   -   "


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    ra = np.argsort(np.argsort(a[ok])); rb = np.argsort(np.argsort(b[ok]))
    return float(np.corrcoef(ra, rb)[0, 1])


def detour_mask(Z, thresh=3.0):
    s0, sg = Z["state_0"][:, :2].astype(float), Z["state_g"][:, :2].astype(float)
    det = np.linalg.norm(s0 - [32, 30.], axis=1) + np.linalg.norm([32, 30.] - sg, axis=1) - np.linalg.norm(s0 - sg, axis=1)
    return det <= thresh


def per_set(Z, part, inst_mask):
    """Group per (instance, depth): PV row index and sibling row indices."""
    inst, depth, pv, sib = Z["nodes_inst"], Z["nodes_depth"], Z["nodes_on_plan"], Z["nodes_is_sib"]
    out = {}
    for j in np.unique(inst):
        if not inst_mask[j]:
            continue
        for d in np.unique(depth[inst == j]):
            m = (inst == j) & (depth == d)
            p = np.flatnonzero(m & pv); s = np.flatnonzero(m & sib)
            if len(p) == 1 and len(s) >= 1:
                out[(int(j), int(d))] = (int(p[0]), s)
    return out


def contrast(Z, part, q, sets, resid=None):
    """Per (inst, depth): value at PV minus mean over siblings; bootstrap over instances by depth."""
    x = Z[f"nodes_{part}_{q}"] if resid is None else resid
    by = collections.defaultdict(list)
    for (j, d), (p, s) in sets.items():
        by[d].append(x[p] - np.nanmean(x[s]))
    return {d: boot(v) for d, v in sorted(by.items())}


def residualise(Z, part, sets, covariates):
    """A_goal regressed on covariates with (instance, depth) fixed effects over PV + siblings;
    returns residuals aligned with the node table (NaN where not in a set)."""
    y = Z[f"nodes_{part}_A_goal"]
    X = np.stack([Z[f"nodes_{part}_{c}"] for c in covariates], 1)
    rows, groups = [], []
    for g, (p, s) in enumerate(sets.values()):
        idx = np.concatenate([[p], s]); rows.append(idx); groups.append(np.full(len(idx), g))
    rows = np.concatenate(rows); groups = np.concatenate(groups)
    yy, XX = y[rows], X[rows]
    ok = np.isfinite(yy) & np.isfinite(XX).all(1)
    # within-group demeaning = fixed effects
    def demean(v):
        out = v.copy().astype(float)
        for g in np.unique(groups[ok]):
            m = ok & (groups == g); out[m] = v[m] - v[m].mean(0)
        return out
    yd, Xd = demean(yy), demean(XX)
    beta = np.linalg.lstsq(Xd[ok], yd[ok], rcond=None)[0]
    res = np.full(len(y), np.nan); res[rows[ok]] = (yd - Xd @ beta)[ok]
    return res, beta


def report(Z, E, straight, part, label):
    print(f"\n################ part = {part}   [{label}] ################")
    sets = per_set(Z, part, straight)
    print(f"sets: {len(sets)} (instance, depth) pairs; siblings per set by depth: " +
          ", ".join(f"d{d}: {np.median([len(s) for (j, dd), (p, s) in sets.items() if dd == d]):.0f}"
                    for d in sorted(set(d for _, d in sets))))
    print("\n=== C1: A_goal(PV) − mean A_goal(siblings), by depth ===")
    for d, t in contrast(Z, part, "A_goal", sets).items():
        print(f"  depth {d}: {fmt(t)}")
    print("  marginals (PV vs siblings), by depth:")
    for q in ("drift", "dist", "V_imag", "V_true"):
        by = collections.defaultdict(lambda: ([], []))
        for (j, d), (p, s) in sets.items():
            by[d][0].append(Z[f"nodes_{part}_{q}"][p]); by[d][1].append(np.nanmean(Z[f"nodes_{part}_{q}"][s]))
        print(f"    {q:>7}: " + "  ".join(f"d{d} {np.nanmean(a):.3f}/{np.nanmean(b):.3f}" for d, (a, b) in sorted(by.items())))
    res, beta = residualise(Z, part, sets, ("dist", "drift"))
    print(f"  residual-adjusted (A_goal ~ |u| + |e| with (inst,depth) FE; beta={np.round(beta, 3)}):")
    for d, t in contrast(Z, part, "A_goal", sets, resid=res).items():
        print(f"    depth {d}: {fmt(t)}")
    print("\n=== C1m: A_mean contrast, and A_goal residual on A_mean ===")
    for d, t in contrast(Z, part, "A_mean", sets).items():
        print(f"  A_mean depth {d}: {fmt(t)}")
    res_m, beta_m = residualise(Z, part, sets, ("A_mean",))
    for d, t in contrast(Z, part, "A_goal", sets, resid=res_m).items():
        print(f"  residual depth {d}: {fmt(t)}   (beta on A_mean {beta_m[0]:.3f})")
    # null sd per depth from the sibling sets
    sd = collections.defaultdict(list)
    for j, d, v in zip(Z["sib_inst"], Z["sib_depth"], Z[f"sib_{part}_null_sd"]):
        if straight[j]: sd[int(d)].append(v)
    print("  shuffled-null sd of A by depth: " + ", ".join(f"d{d} {np.nanmean(v):.3f}" for d, v in sorted(sd.items())))

    print("\n=== C2: Spearman(A_goal, log N) among siblings incl. PV; high-N − low-N quartile ===")
    by_r, by_q = collections.defaultdict(list), collections.defaultdict(list)
    for (j, d), (p, s) in sets.items():
        idx = np.concatenate([[p], s]); A = Z[f"nodes_{part}_A_goal"][idx]; N = Z["nodes_N"][idx]
        if len(np.unique(N)) > 1:
            by_r[d].append(spearman(A, np.log(N)))
            hi, lo = A[N >= np.percentile(N, 75)], A[N <= np.percentile(N, 25)]
            by_q[d].append(np.nanmean(hi) - np.nanmean(lo))
    for d in sorted(by_r):
        print(f"  depth {d}: rho {fmt(boot(by_r[d]))}   hi−lo {fmt(boot(by_q[d]))}")

    print("\n=== C3: sibling latent spread, S_model / S_true by depth (this record) ===")
    for d in sorted(np.unique(Z["sib_depth"])):
        m = (Z["sib_depth"] == d) & straight[Z["sib_inst"]]
        r = Z[f"sib_{part}_S_model"][m] / Z[f"sib_{part}_S_true"][m]
        print(f"  depth {d}: ratio {fmt(boot(r))}   S_model {np.nanmean(Z[f'sib_{part}_S_model'][m]):.4f}  S_true {np.nanmean(Z[f'sib_{part}_S_true'][m]):.4f}")
    print("  value-spread ratio on the same siblings, sd(V_imag)/sd(V_true) by depth: " + ", ".join(
        f"d{d} {fmt(boot([Z[f'nodes_{part}_V_imag'][idx].std() / Z[f'nodes_{part}_V_true'][idx].std() for (j, dd), (p, s) in sets.items() if dd == d for idx in [np.concatenate([[p], s])] if len(idx) >= 3]))}"
        for d in sorted(set(d for _, d in sets))))
    if E is not None:
        print("  S_true from the env record's own sibling sets, by depth: " + ", ".join(
            f"d{d} {np.nanmean(E[f'sib_{part}_S_true'][(E['sib_depth'] == d) & straight[E['sib_inst']]]):.4f}"
            for d in sorted(np.unique(E["sib_depth"]))))

    print("\n=== C4: ordering fidelity among siblings incl. PV: Spearman(V_imag, V_true); top-1 / top-3 ===")
    by_s, by_1, by_3 = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    for (j, d), (p, s) in sets.items():
        idx = np.concatenate([[p], s]); vi = Z[f"nodes_{part}_V_imag"][idx]; vt = Z[f"nodes_{part}_V_true"][idx]
        by_s[d].append(spearman(vi, vt))
        order_t = np.argsort(-vt); best_i = int(np.argmax(vi))
        by_1[d].append(float(best_i == order_t[0])); by_3[d].append(float(best_i in order_t[:3]))
    for d in sorted(by_s):
        print(f"  depth {d}: rho {fmt(boot(by_s[d]))}   top1 {np.mean(by_1[d]):.2f}  top3 {np.mean(by_3[d]):.2f}")

    print("\n=== C5: |e| and A_goal by depth, PV / siblings ===")
    for d in sorted(set(d for _, d in sets)):
        pe = [Z[f"nodes_{part}_drift"][p] for (j, dd), (p, s) in sets.items() if dd == d]
        se = [np.nanmean(Z[f"nodes_{part}_drift"][s]) for (j, dd), (p, s) in sets.items() if dd == d]
        pa = [Z[f"nodes_{part}_A_goal"][p] for (j, dd), (p, s) in sets.items() if dd == d]
        sa = [np.nanmean(Z[f"nodes_{part}_A_goal"][s]) for (j, dd), (p, s) in sets.items() if dd == d]
        print(f"  depth {d}: |e| PV {np.nanmean(pe):.4f} sib {np.nanmean(se):.4f}   A PV {np.nanmean(pa):+.4f} sib {np.nanmean(sa):+.4f}")

    print("\n=== C6: B_search — mean G (and |e|) over N-weighted sample vs uniform sample; training reference ===")
    inst = Z["nodes_inst"]; sm = straight[inst]
    for lab, m in (("N-weighted", Z["nodes_is_qN"] & sm), ("uniform", Z["nodes_is_qU"] & sm)):
        per_inst = [np.nanmean(Z[f"nodes_{part}_G"][m & (inst == j)]) for j in np.unique(inst[m])]
        per_e = [np.nanmean(Z[f"nodes_{part}_drift"][m & (inst == j)]) for j in np.unique(inst[m])]
        print(f"  {lab:>10}: G {fmt(boot(per_inst))}   |e| {np.nanmean(per_e):.4f}   mean depth {Z['nodes_depth'][m].mean():.2f}")
    if f"train_{part}_drift" in Z:
        print(f"  training transitions (one step): |e| {np.nanmean(Z[f'train_{part}_drift']):.4f}  "
              f"A_mean {np.nanmean(Z[f'train_{part}_A_mean']):.4f}  n={int(Z['train_n'])}")
        d1 = (Z["nodes_depth"] == 1) & sm
        print(f"  tree depth-1 nodes:            |e| {np.nanmean(Z[f'nodes_{part}_drift'][d1]):.4f}  "
              f"A_mean {np.nanmean(Z[f'nodes_{part}_A_mean'][d1]):.4f}  n={int(d1.sum())}")

    print("\n=== C7: distance to z̄, model vs true, by depth; corr(spread ratio, A_mean) ===")
    for d in sorted(set(d for _, d in sets)):
        idx = np.concatenate([np.concatenate([[p], s]) for (j, dd), (p, s) in sets.items() if dd == d])
        dm, dt = Z[f"nodes_{part}_dzbar_model"][idx], Z[f"nodes_{part}_dzbar_true"][idx]
        print(f"  depth {d}: d(ẑ, z̄) {np.nanmean(dm):.4f}   d(z, z̄) {np.nanmean(dt):.4f}   ratio {np.nanmean(dm / dt):.3f}")
    ratio = Z[f"sib_{part}_S_model"] / Z[f"sib_{part}_S_true"]
    am = np.array([np.nanmean(Z[f"nodes_{part}_A_mean"][(Z["nodes_inst"] == j) & (Z["nodes_depth"] == d) & (Z["nodes_is_sib"] | Z["nodes_on_plan"])])
                   for j, d in zip(Z["sib_inst"], Z["sib_depth"])])
    ok = np.isfinite(ratio) & np.isfinite(am) & straight[Z["sib_inst"]]
    print(f"  Spearman(S_model/S_true, mean A_mean of the set) over sets: {spearman(ratio[ok], am[ok]):+.3f} (n={ok.sum()})")


def gates(Z, E):
    print("\n==== gates ====")
    print(f"S1 (wm) max|V_imag − V_leaf| on PV nodes: {Z['gate_dV_pv'].max():.2e}; all replayed nodes: {Z['gate_dV_all'].max():.2e}; PV actions == plan: {bool(Z['gate_pv_actions_match'])}")
    if E is not None:
        print(f"S2 (env) max|V_true − V_leaf| on PV nodes: {E['gate_dV_pv'].max():.2e}; all: {E['gate_dV_all'].max():.2e}; PV actions == plan: {bool(E['gate_pv_actions_match'])}")
    print("S3 siblings per PV node by depth (median / min / frac ≥ 8): " + ", ".join(
        f"d{d} {np.median(Z['sib_n_sib'][Z['sib_depth'] == d]) - 1:.0f}/{Z['sib_n_sib'][Z['sib_depth'] == d].min() - 1}/"
        f"{np.mean(Z['sib_n_sib'][Z['sib_depth'] == d] - 1 >= 8):.2f}" for d in sorted(np.unique(Z["sib_depth"]))))
    if E is not None:
        a = {(int(j), int(d)): v for j, d, v in zip(Z["sib_inst"], Z["sib_depth"], Z["sib_both_S_true"]) if d == 1}
        b = {(int(j), int(d)): v for j, d, v in zip(E["sib_inst"], E["sib_depth"], E["sib_both_S_true"]) if d == 1}
        common = [k for k in a if k in b and Z["sib_n_sib"][list(zip(Z["sib_inst"], Z["sib_depth"])).index(k)] == 16
                  and E["sib_n_sib"][list(zip(E["sib_inst"], E["sib_depth"])).index(k)] == 16]
        if common:
            dd = max(abs(a[k] - b[k]) for k in common)
            print(f"S4 depth-1 S_true, wm record vs env record on {len(common)} complete sets: max|Δ| = {dd:.2e}")
        else:
            print("S4: no complete depth-1 sibling sets in both records")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wm", required=True)
    ap.add_argument("--env", default=None)
    ap.add_argument("--parts", nargs="*", default=["visual", "both", "proprio"])
    a = ap.parse_args()
    Z = dict(np.load(a.wm)); E = dict(np.load(a.env)) if a.env else None
    n = int(Z["nodes_inst"].max()) + 1
    straight = detour_mask(Z)
    print(f"record: {a.wm}  simulator={Z['simulator']} c={float(Z['c_uct'])} K={int(Z['K'])} n={n}; "
          f"replayed nodes {len(Z['nodes_inst'])}; straight-line instances {straight.sum()}/{n}")
    gates(Z, E)
    for part in a.parts:
        report(Z, E, np.ones(n, bool), part, "pooled")
        report(Z, E, straight, part, "straight-line")


if __name__ == "__main__":
    main()
