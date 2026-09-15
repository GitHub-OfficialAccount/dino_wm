"""One-step model error on held-out door slices, control-strip slices and non-door slices
(coverage-sweep outline §3.3 item 1, gate T2, and the door-vs-strip difficulty check).

Uses the cached DINO latents: for each slice (episode, start) the model predicts z_{t+5}
from the cached z_t, the proprio state and the 5 actions; the target is the cached z_{t+5}.
Reports mean visual / proprio |e| per set and the ratios the gates read.

    python analysis/door_error.py --model_path M --data_path D --latent_dir D/latents \
        --masks coverage_masks.npz --level r10 --device cuda
`--level` selects which held-out slices are "door" (that level's removed H slices; for
r100 the full H); `--n_max` caps each set for speed (default 2000, fixed seed).
"""
import argparse, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
from omegaconf import OmegaConf
import plan as P


def load_set(latent_dir, states, actions, slices, fs, device, n_max, rng):
    if len(slices) > n_max:
        slices = slices[rng.choice(len(slices), n_max, replace=False)]
    z0, z1, p0, p1, a = [], [], [], [], []
    cache = {}
    for e, s in slices:
        e, s = int(e), int(s)
        if e not in cache:
            cache[e] = torch.load(Path(latent_dir) / f"episode_{e:03d}.pt", weights_only=True).float()
        z0.append(cache[e][s]); z1.append(cache[e][s + fs])
        p0.append(states[e, s]); p1.append(states[e, s + fs]); a.append(actions[e, s:s + fs].reshape(-1))
    return (torch.stack(z0).to(device), torch.stack(z1).to(device), torch.stack(p0).to(device),
            torch.stack(p1).to(device), torch.stack(a).to(device))


def one_step_error(model, pre, z0, z1, p0, p1, a, batch=256):
    ev, ep = [], []
    with torch.no_grad():
        for s in range(0, len(z0), batch):
            sl = slice(s, s + batch)
            pn0 = pre.normalize_proprios(p0[sl]); pn1 = pre.normalize_proprios(p1[sl])
            an = pre.normalize_actions(a[sl].view(-1, 5, 2)).reshape(-1, 1, 10)
            zp0 = model.encode_proprio(pn0[:, None]); zp1 = model.encode_proprio(pn1[:, None])
            act_emb = model.encode_act(an)
            zv = z0[sl][:, None]
            P_ = zp0.unsqueeze(2).repeat(1, 1, zv.shape[2], 1); A_ = act_emb.unsqueeze(2).repeat(1, 1, zv.shape[2], 1)
            z = torch.cat([zv, P_, A_], dim=3)
            zpred = model.predict(z)
            zo, _ = model.separate_emb(zpred)
            ev.append(((zo["visual"][:, 0] - z1[sl]) ** 2).mean(dim=(1, 2)).sqrt().cpu())
            ep.append(((zo["proprio"][:, 0] - zp1[:, 0]) ** 2).mean(dim=1).sqrt().cpu())
    return torch.cat(ev).numpy(), torch.cat(ep).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--latent_dir", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--level", default="r100")
    ap.add_argument("--n_max", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = OmegaConf.load(Path(a.model_path) / "hydra.yaml")
    model = P.load_model(Path(a.model_path) / "checkpoints" / "model_latest.pth", cfg, cfg.num_action_repeat,
                         device=torch.device(a.device)); model.eval()
    from datasets.wall_dset import WallDataset
    from preprocessor import Preprocessor
    ds = WallDataset(data_path=a.data_path, normalize_action=True)
    dev = torch.device(a.device)
    pre = Preprocessor(action_mean=ds.action_mean.to(dev), action_std=ds.action_std.to(dev), state_mean=ds.state_mean.to(dev),
                       state_std=ds.state_std.to(dev), proprio_mean=ds.proprio_mean.to(dev), proprio_std=ds.proprio_std.to(dev),
                       transform=None)   # stats on the model's device
    raw_states = torch.load(Path(a.data_path) / "states.pth", weights_only=True).float()
    raw_actions = torch.load(Path(a.data_path) / "actions.pth", weights_only=True).float()
    M = np.load(a.masks); fs = int(M["frameskip"]); rng = np.random.default_rng(0)
    H, S = M["H_slices"], M["STRIP_slices"]
    kept = M[f"{a.level}_kept_held"] if a.level != "r10ctrl" else M["r100_kept_held"]
    kept_set = set(map(tuple, kept.tolist()))
    removed = np.array([x for x in H.tolist() if tuple(x) not in kept_set], dtype=np.int32) if len(kept) < len(H) else H
    Hs, Ss = set(map(tuple, H.tolist())), set(map(tuple, S.tolist()))
    tr = M["train_episodes"]; starts = np.arange(50 - 2 * fs + 1)
    allsl = np.stack([np.repeat(tr, len(starts)), np.tile(starts, len(tr))], 1)
    non = np.array([x for x in allsl.tolist() if tuple(x) not in Hs and tuple(x) not in Ss], dtype=np.int32)
    sets = {"door_all": H, "door_removed": removed, "strip": S, "non_door": non}
    res = {}
    for name, sl in sets.items():
        ev, ep = one_step_error(model, pre, *load_set(a.latent_dir, raw_states, raw_actions, sl, fs, a.device, a.n_max, rng))
        res[name] = (ev.mean(), ep.mean(), len(ev))
        print(f"{name:>13}: n={len(ev):>5}  visual |e| {ev.mean():.4f}  proprio |e| {ep.mean():.4f}")
    print(f"ratios (visual): door_all/non_door {res['door_all'][0] / res['non_door'][0]:.3f}   "
          f"door_removed/non_door {res['door_removed'][0] / res['non_door'][0]:.3f}   "
          f"strip/non_door {res['strip'][0] / res['non_door'][0]:.3f}   door_all/strip {res['door_all'][0] / res['strip'][0]:.3f}")
    if a.out:
        np.savez(a.out, **{f"{k}_{q}": v[i] for k, v in res.items() for i, q in enumerate(("visual", "proprio", "n"))},
                 level=a.level, model_path=str(a.model_path))


if __name__ == "__main__":
    main()
