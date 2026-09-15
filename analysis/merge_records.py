"""Merge instance-range records (run_record.py --eval_range) into one record.

    python analysis/merge_records.py --out os30_ns1000.npz --parts os30_ns1000_h0.npz os30_ns1000_h1.npz
Row arrays are concatenated (eval_idx is already global); state_0/state_g/eval_seed come from
the planning targets (all instances), so the analysis stages' condition check passes.
"""
import argparse, pickle, shutil
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", nargs="+", required=True)
    a = ap.parse_args()
    P = [dict(np.load(p)) for p in a.parts]
    n0 = len(P[0]["eval_idx"])
    row_keys = [k for k, v in P[0].items() if getattr(v, "ndim", 0) >= 1 and v.shape[0] == n0 and not k.startswith("spread_")]
    sp_keys = [k for k in P[0] if k.startswith("spread_")]
    out = {k: np.concatenate([p[k] for p in P], 0) for k in row_keys + sp_keys}
    for k, v in P[0].items():
        if k not in out:
            out[k] = v
    tp = a.parts[0].replace(".npz", "") + "_targets.pkl"
    T = pickle.load(open(tp, "rb"))
    out["state_0"] = np.asarray(T["state_0"], np.float32); out["state_g"] = np.asarray(T["state_g"], np.float32)
    n = len(T["state_0"]); seed = int(P[0]["eval_seed"][0]) if len(P[0]["eval_seed"]) else 1
    out["eval_seed"] = np.array([99 * i + 1 for i in range(n)], np.int64)
    np.savez_compressed(a.out, **out); shutil.copy(tp, a.out.replace(".npz", "") + "_targets.pkl")
    ids = np.unique(out["eval_idx"]); print(f"merged {len(a.parts)} parts: {len(out['eval_idx'])} rows, instances {ids.min()}..{ids.max()} ({len(ids)}), eval_seed for {n}")


if __name__ == "__main__":
    main()
