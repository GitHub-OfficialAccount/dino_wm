"""Gate for the alpha=0 arm (stage-2-outline.md §11.1): the visual objective is the visual column.

The alpha=0 tree and the alpha=1 tree share their first level (same root, same 16 actions).
The alpha=0 record's depth-1 V_leaf must equal the alpha=1 replay's nodes_visual_V_imag for
the same (instance, action) within 1e-3. Also prints the visual per-tree Q range against the
1e-3 normalisation floor.

    python analysis/check_alpha0_gate.py --a0 mcts_wm_a0_c0.2_K2000.npz --replay stage2_wm.npz
"""
import argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--tol", type=float, default=1e-3)
    a = ap.parse_args()
    R = np.load(a.a0); Z = np.load(a.replay)
    assert float(R["alpha"]) == 0.0, f"record alpha is {float(R['alpha'])}, not 0"
    ref = {}
    for i, d, act, v in zip(Z["nodes_inst"], Z["nodes_depth"], Z["nodes_action_idx"], Z["nodes_visual_V_imag"]):
        if d == 1:
            ref[(int(i), int(act[0]))] = float(v)
    diffs, missing = [], 0
    d1 = np.flatnonzero(R["depth"] == 1)
    for k in d1:
        key = (int(R["inst"][k]), int(R["action_idx"][k]))
        if key in ref:
            diffs.append(abs(float(R["V_leaf"][k]) - ref[key]))
        else:
            missing += 1
    diffs = np.array(diffs)
    rng = R["Q_max"] - R["Q_min"]
    print(f"depth-1 nodes compared: {len(diffs)} (missing from replay: {missing})")
    print(f"max |V_leaf(alpha=0) - V_imag_visual(alpha=1 replay)| = {diffs.max():.2e}  (tolerance {a.tol:.0e})  "
          f"-> {'PASS' if diffs.max() <= a.tol else 'FAIL'}")
    print(f"visual per-tree Q range: median {np.median(rng):.3f}, min {rng.min():.3f}, floor 1e-3 -> "
          f"{'clear' if rng.min() > 1e-3 else 'FLOOR ENGAGED'}")


if __name__ == "__main__":
    main()
