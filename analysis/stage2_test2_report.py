"""Test 2 (outline.md): random rollouts vs the search tree, per depth.

    python analysis/stage2_test2_report.py --random stage2_random.npz --tree stage2_wm.npz

Per depth and part: |e|, |u|, G, A_goal, A_mean on the random-rollout nodes beside the
tree's PV nodes and its siblings; the shuffled null; and the |u|-matched form — random
rollouts whose |u| lies inside the tree nodes' interquartile range at that depth
(stage-2-outline.md §11.2: if raw and matched disagree, or < 25 % match at depths 4-5,
Test 2's literal form is weaker evidence than §3b).
"""
import argparse, collections
import numpy as np


def boot(x, n=10000, seed=0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return "   -   "
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(1)
    return f"{x.mean():+.4f} [{np.percentile(m, 2.5):+.4f},{np.percentile(m, 97.5):+.4f}]"


def per_inst(Z, mask, key):
    inst = Z["nodes_inst"]
    return [np.nanmean(Z[key][mask & (inst == j)]) for j in np.unique(inst[mask])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--parts", nargs="*", default=["visual", "proprio", "both"])
    a = ap.parse_args()
    Rr, T = dict(np.load(a.random)), dict(np.load(a.tree))
    print(f"random: {len(Rr['nodes_inst'])} nodes; tree: {len(T['nodes_inst'])} nodes (PV {T['nodes_on_plan'].sum()}, siblings {T['nodes_is_sib'].sum()})")
    for part in a.parts:
        print(f"\n################ part = {part} ################")
        print(f"{'depth':>5} {'set':>14} {'n':>6} {'|e|':>7} {'|u|':>7} {'G':>28} {'A_goal':>28} {'A_mean':>28}")
        for d in range(1, int(T["nodes_depth"].max()) + 1):
            tm_pv = (T["nodes_depth"] == d) & T["nodes_on_plan"]; tm_sib = (T["nodes_depth"] == d) & T["nodes_is_sib"]
            rm = Rr["nodes_depth"] == d
            u_tree = T[f"nodes_{part}_dist"][tm_pv | tm_sib]
            lo, hi = np.nanpercentile(u_tree, 25), np.nanpercentile(u_tree, 75)
            rm_match = rm & (Rr[f"nodes_{part}_dist"] >= lo) & (Rr[f"nodes_{part}_dist"] <= hi)
            for lab, Z, m in (("tree PV", T, tm_pv), ("tree siblings", T, tm_sib), ("random", Rr, rm), ("random |u|-match", Rr, rm_match)):
                if m.sum() == 0:
                    print(f"{d:>5} {lab:>14} {0:>6}"); continue
                e = np.nanmean(Z[f"nodes_{part}_drift"][m]); u = np.nanmean(Z[f"nodes_{part}_dist"][m])
                print(f"{d:>5} {lab:>14} {m.sum():>6} {e:>7.3f} {u:>7.3f} {boot(per_inst(Z, m, f'nodes_{part}_G')):>28} "
                      f"{boot(per_inst(Z, m, f'nodes_{part}_A_goal')):>28} {boot(per_inst(Z, m, f'nodes_{part}_A_mean')):>28}")
            print(f"      matched fraction of random rollouts at depth {d}: {rm_match.sum() / max(rm.sum(), 1):.2f}  (tree |u| IQR [{lo:.3f}, {hi:.3f}])")
        # depth-1 sibling-style spread on random rollouts is not defined (chains, not siblings);
        # the proprio compression signature is read from |e| and from dzbar instead
        for d in (1, 5):
            rm = Rr["nodes_depth"] == d
            print(f"  depth {d} random: d(z_hat, zbar)/d(z, zbar) = {np.nanmean(Rr[f'nodes_{part}_dzbar_model'][rm] / Rr[f'nodes_{part}_dzbar_true'][rm]):.3f}")


if __name__ == "__main__":
    main()
