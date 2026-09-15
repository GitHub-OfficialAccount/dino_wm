"""Coverage-sweep readouts (stage-2-coverage-sweep-outline.md §6–§7). Runs locally on the
per-level record directories.

    python analysis/coverage_report.py --root ../data/experiment-outputs/coverage \
        --levels r100 r20 r10 r5 --extra r10ctrl x5 --release ../data/experiment-outputs/stage2

Per level: door/strip one-step error (T2), Stage 1 success/distance for the four arms (T1,
P4, P4b, P7), visual/proprio/both C1 at every depth on the model tree and the 1.1-L tree
(P1, P-baseline, P2), B_search (P3), spread ratio and ordering (P5). Dose-response: paired
per-instance slope on log p over the thinned levels, bootstrapped over instances, with the
1.1-L visual C1 slope beside it. P10: R10 vs R10-ctrl, paired.
"""
import argparse, collections
from pathlib import Path
import numpy as np
from analysis.stage2_report import per_set, contrast, residualise, spearman


def boot(x, n=10000, seed=0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5), len(x)


def fmt(t, d=3):
    m, lo, hi = t[:3]
    return f"{m:+.{d}f} [{lo:+.{d}f},{hi:+.{d}f}]" if np.isfinite(m) else "   -   "


def at_K(R, K, key="snap_success"):
    return R[key][int(np.flatnonzero(R["snap_K"] == K)[0])]


def c1_by_depth(Z, part, mask=None):
    sets = per_set(Z, part, np.ones(int(Z["nodes_inst"].max()) + 1, bool) if mask is None else mask)
    per_inst = collections.defaultdict(dict)                  # depth -> inst -> value
    for (j, d), (p, s) in sets.items():
        per_inst[d][j] = Z[f"nodes_{part}_A_goal"][p] - np.nanmean(Z[f"nodes_{part}_A_goal"][s])
    return per_inst


def bsearch(Z, part):
    inst = Z["nodes_inst"]
    qn = np.array([np.nanmean(Z[f"nodes_{part}_G"][Z["nodes_is_qN"] & (inst == j)]) for j in np.unique(inst)])
    qu = np.array([np.nanmean(Z[f"nodes_{part}_G"][Z["nodes_is_qU"] & (inst == j)]) for j in np.unique(inst)])
    return qn, qu


def load_level(root, lvl):
    d = Path(root) / lvl
    out = {"W": np.load(d / "mcts_wm_c0.2_K2000.npz"), "L": np.load(d / "mcts_env_c0.2_K2000.npz"),
           "Zw": dict(np.load(d / "stage2_wm.npz")), "Zl": dict(np.load(d / "stage2_env.npz"))}
    if (d / "door_error.npz").exists():
        out["E"] = dict(np.load(d / "door_error.npz"))
    return out


def slope_on_logp(values_by_level, ps):
    """values_by_level: {p: array over instances}; paired per-instance OLS slope on log p."""
    ps = sorted(ps); x = np.log(ps); x = x - x.mean()
    Y = np.stack([values_by_level[p] for p in ps], 1)         # (inst, levels)
    ok = np.isfinite(Y).all(1)
    return (Y[ok] @ x) / (x @ x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--levels", nargs="+", default=["r100", "r20", "r10", "r5"])
    ap.add_argument("--extra", nargs="*", default=["r10ctrl", "x5"])
    ap.add_argument("--release", default=None, help="Stage 2 dir with stage2_wm.npz / stage2_envL.npz for T1")
    ap.add_argument("--parts", nargs="*", default=["visual", "proprio", "both"])
    a = ap.parse_args()
    pmap = {"r100": 1.0, "r20": 0.2, "r10": 0.1, "r5": 0.05, "r10ctrl": 0.1, "x5": 5.0}
    lv = {l: load_level(a.root, l) for l in a.levels + a.extra if (Path(a.root) / l / "stage2_wm.npz").exists()}
    print("levels loaded:", ", ".join(lv))

    print("\n=== T2 manipulation: one-step visual |e| by region (door_all / door_removed / strip / non_door) and ratios ===")
    for l, D in lv.items():
        if "E" not in D:
            continue
        E = D["E"]; nd = E["non_door_visual"]
        print(f"  {l:>7}: door_all {E['door_all_visual']:.4f}  removed {E['door_removed_visual']:.4f}  strip {E['strip_visual']:.4f}  "
              f"non_door {nd:.4f}   ratios door/non {E['door_all_visual'] / nd:.3f}  strip/non {E['strip_visual'] / nd:.3f}  "
              f"door/strip {E['door_all_visual'] / E['strip_visual']:.3f}   proprio door {E['door_all_proprio']:.4f} non {E['non_door_proprio']:.4f}")

    print("\n=== Stage 1 arms at K=2000 (T1, P4, P4b, P7): success / mean final distance ===")
    print(f"  {'level':>7} {'1.1-G':>7} {'1.1-L':>7} {'1.2-G':>7} {'1.2':>7} | {'dd succ':>22} | {'1.2 K=300':>9} {'1.2 dist':>9} {'1.1-L dist':>10}")
    for l, D in lv.items():
        W, L = D["W"], D["L"]
        s12, s12g, s11, s11g = at_K(W, 2000).astype(int), W["greedy_success"].astype(int), at_K(L, 2000).astype(int), L["greedy_success"].astype(int)
        dd = (s12 - s12g) - (s11 - s11g)
        print(f"  {l:>7} {s11g.mean():>7.2f} {s11.mean():>7.2f} {s12g.mean():>7.2f} {s12.mean():>7.2f} | {fmt(boot(dd), 2):>22} | "
              f"{at_K(W, 300).mean():>9.2f} {at_K(W, 2000, 'snap_state_dist').mean():>9.1f} {at_K(L, 2000, 'snap_state_dist').mean():>10.1f}")

    for part in a.parts:
        print(f"\n=== C1 by depth, part = {part}: model tree (1.2) | truth tree (1.1-L)  [P1, P-baseline, P2] ===")
        c1w = {l: c1_by_depth(D["Zw"], part) for l, D in lv.items()}
        c1l = {l: c1_by_depth(D["Zl"], part) for l, D in lv.items()}
        for l in lv:
            row = "  ".join(f"d{d} {fmt(boot(list(c1w[l][d].values())), 2)}" for d in range(1, 6) if d in c1w[l])
            rowl = "  ".join(f"d{d} {boot(list(c1l[l][d].values()))[0]:+.2f}" for d in range(1, 6) if d in c1l[l])
            print(f"  {l:>7} 1.2:   {row}\n  {'':>7} 1.1-L: {rowl}")
        # dose-response over thinned levels
        thin = [l for l in a.levels if l in lv]
        if len(thin) >= 3:
            for d in (4, 5):
                vals = {pmap[l]: np.array([c1w[l][d].get(j, np.nan) for j in range(50)]) for l in thin}
                valsl = {pmap[l]: np.array([c1l[l][d].get(j, np.nan) for j in range(50)]) for l in thin}
                sw, sl = slope_on_logp(vals, list(vals)), slope_on_logp(valsl, list(valsl))
                print(f"  slope of C1 on log p at depth {d}: model tree {fmt(boot(sw))} (n={len(sw)})   truth tree {fmt(boot(sl))}   difference {fmt(boot(sw - sl))}")
        print(f"  B_search ({part}): " + "; ".join(
            f"{l} N-wt {fmt(boot(bsearch(D['Zw'], part)[0]))} uni {boot(bsearch(D['Zw'], part)[1])[0]:+.3f}" for l, D in lv.items()))
        print(f"  spread ratio d1/d5 ({part}): " + "; ".join(
            f"{l} {np.nanmean(D['Zw'][f'sib_{part}_S_model'][D['Zw']['sib_depth'] == 1] / D['Zw'][f'sib_{part}_S_true'][D['Zw']['sib_depth'] == 1]):.2f}/"
            f"{np.nanmean(D['Zw'][f'sib_{part}_S_model'][D['Zw']['sib_depth'] == 5] / D['Zw'][f'sib_{part}_S_true'][D['Zw']['sib_depth'] == 5]):.2f}" for l, D in lv.items()))
        ords = []
        for l, D in lv.items():
            sets = per_set(D["Zw"], part, np.ones(50, bool)); r1 = [spearman(D["Zw"][f"nodes_{part}_V_imag"][np.concatenate([[p], s])], D["Zw"][f"nodes_{part}_V_true"][np.concatenate([[p], s])]) for (j, d), (p, s) in sets.items() if d == 1]
            r5 = [spearman(D["Zw"][f"nodes_{part}_V_imag"][np.concatenate([[p], s])], D["Zw"][f"nodes_{part}_V_true"][np.concatenate([[p], s])]) for (j, d), (p, s) in sets.items() if d == 5]
            ords.append(f"{l} {np.nanmean(r1):.2f}/{np.nanmean(r5):.2f}")
        print(f"  ordering Spearman d1/d5 ({part}): " + "; ".join(ords))

    if "r10" in lv and "r10ctrl" in lv:
        print("\n=== P10: R10 vs R10-ctrl, visual C1 at depths 4-5, paired per instance ===")
        for d in (4, 5):
            a10, ac = c1_by_depth(lv["r10"]["Zw"], "visual")[d], c1_by_depth(lv["r10ctrl"]["Zw"], "visual")[d]
            common = sorted(set(a10) & set(ac)); diff = np.array([a10[j] - ac[j] for j in common])
            print(f"  depth {d}: R10 − R10-ctrl {fmt(boot(diff))} (n={len(diff)});  R10-ctrl − R100 "
                  f"{fmt(boot([ac[j] - c1_by_depth(lv['r100']['Zw'], 'visual')[d].get(j, np.nan) for j in common]))}" if "r100" in lv else "")
    if a.release and "r100" in lv:
        print("\n=== T1 reproduction: R100 vs the released model ===")
        Zr = dict(np.load(Path(a.release) / "stage2_wm.npz"))
        for d in (4, 5):
            print(f"  visual C1 depth {d}: release {boot(list(c1_by_depth(Zr, 'visual')[d].values()))[0]:+.3f}   R100 {boot(list(c1_by_depth(lv['r100']['Zw'], 'visual')[d].values()))[0]:+.3f}")
        print(f"  training |e| (visual): release {np.nanmean(Zr['train_visual_drift']):.4f}   R100 {np.nanmean(lv['r100']['Zw']['train_visual_drift']):.4f}")


if __name__ == "__main__":
    main()
