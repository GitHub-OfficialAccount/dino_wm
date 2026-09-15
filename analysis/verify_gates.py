"""Gates 3 and 4: replay reproducibility and frameskip alignment (Experiment 1, §8)."""
import warnings, argparse; warnings.filterwarnings("ignore")
import numpy as np, torch
from einops import rearrange, repeat
from utils import move_to_device
from analysis.run_record import build
from analysis.optimism import metrics, align_imagined_to_real


def replay(ws, model, j, acts_t, frameskip):
    """Imagined + real rollouts for candidate actions of eval instance j."""
    K = acts_t.shape[0]
    pre = ws.data_preprocessor
    obs0 = {k: np.repeat(v[j:j+1], K, axis=0) for k, v in ws.obs_0.items()}
    trans0 = move_to_device(pre.transform_obs(obs0), ws.device)
    obsg = {k: np.repeat(v[j:j+1], K, axis=0) for k, v in ws.obs_g.items()}
    z_g = model.encode_obs(move_to_device(pre.transform_obs(obsg), ws.device))

    with torch.no_grad():
        i_z, _ = model.rollout(obs_0=trans0, act=acts_t)

    # real rollout: expand each planner action into `frameskip` env actions
    exec_a = rearrange(acts_t.cpu(), "b t (f d) -> b (t f) d", f=frameskip)
    exec_a = pre.denormalize_actions(exec_a).numpy()
    env_j = ws.env.envs[j]
    e_obs = [env_j.rollout(ws.eval_seed[j], ws.state_0[j], exec_a[k]) for k in range(K)]
    e_vis = np.stack([o[0]["visual"] for o in e_obs])
    e_pro = np.stack([o[0]["proprio"] for o in e_obs])
    with torch.no_grad():
        z_r = model.encode_obs(move_to_device(
            pre.transform_obs({"visual": e_vis, "proprio": e_pro}), ws.device))
    return i_z, z_r, z_g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--record_path", default="rec.npz")
    a = ap.parse_args()

    ws, model, env, mcfg = build(a.model_path, record_path=a.record_path)
    fs = mcfg.frameskip
    ws.perform_planning()
    R = np.load(a.record_path)

    j = 0
    sel = (R["eval_idx"] == j) & (R["opt_step"] == 0)
    acts = torch.tensor(R["action_seq"][sel])
    print(f"\ncandidates for eval {j}, opt_step 0: {acts.shape}")

    i_z, z_r, z_g = replay(ws, model, j, acts, fs)
    print("imagined z:", i_z["visual"].shape, " replayed z:", z_r["visual"].shape)

    # --- GATE 3: recomputed imagined objective must match the logged loss ---
    obj = __import__("planning.objectives", fromlist=["x"]).create_objective_fn(1, 2, "last")
    zg1 = {k: v[:, -1:] for k, v in z_g.items()}
    loss_re = obj(i_z, zg1).detach().cpu().numpy()
    print("GATE3 max|recomputed loss - logged loss| =",
          np.abs(loss_re - R["imagined_loss"][sel]).max())

    # --- GATE 4: alignment map; drift at h=0 must be exactly 0 ---
    idx = align_imagined_to_real(acts.shape[1], fs)
    print("GATE4 alignment map:", idx, " real T =", z_r["visual"].shape[1])
    zr_al = {k: v[:, idx] for k, v in z_r.items()}
    zg_b = {k: repeat(v[:, -1:], "b 1 ... -> b t ...", t=zr_al["visual"].shape[1])
            for k, v in z_g.items()}
    m = metrics({k: v.detach().cpu().numpy() for k, v in i_z.items()},
                {k: v.detach().cpu().numpy() for k, v in zr_al.items()},
                {k: v.detach().cpu().numpy() for k, v in zg_b.items()}, alpha=1.0)
    print("GATE4 drift[h=0] (must be 0):", m["drift"][:, 0])
    print("drift by depth      :", np.round(m["drift"].mean(0), 5))
    print("alignment by depth  :", np.round(m["alignment"].mean(0), 5))
    print("gap by depth        :", np.round(m["gap"].mean(0), 5))


if __name__ == "__main__":
    main()
