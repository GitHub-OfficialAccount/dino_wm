"""Cache the frozen DINO encoder's latents for every frame of the wall dataset, exactly as
the training path computes them (coverage-sweep outline §3.2, B3).

Training path (WallDataset.get_frames -> VWorldModel.encode_obs):
    image = episode[frames] / 255 ; transform (Resize 224, CenterCrop 224, Normalize .5/.5)
    encoder_transform (Resize 224, a no-op at 224) ; DinoV2Encoder.forward -> (T, 196, 384)
This script does the same per episode and writes latents/episode_XXX.pt as fp16 (or fp32
with --fp32). --verify N re-encodes N frames through the model's own encode_obs and reports
the max deviation of the cached latents and of the planner objective computed from them.

    python datasets/latent_cache.py --model_path M --data_path D --out_dir D/latents --device cuda --verify 100
"""
import argparse, time, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
from omegaconf import OmegaConf
import hydra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--episodes", type=int, nargs=2, default=None, help="range [a, b) of episodes; default all")
    ap.add_argument("--verify", type=int, default=100)
    ap.add_argument("--batch", type=int, default=50)
    a = ap.parse_args()
    cfg = OmegaConf.load(Path(a.model_path) / "hydra.yaml")
    transform = hydra.utils.instantiate(cfg.env.dataset.transform)
    encoder = hydra.utils.instantiate(cfg.encoder).to(a.device).eval()
    for p in encoder.parameters():
        p.requires_grad = False
    from torchvision import transforms
    enc_tf = transforms.Resize((cfg.img_size // 16) * encoder.patch_size)   # VWorldModel.encoder_transform
    obs_dir = Path(a.data_path) / "obses"; out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    n = len(list(obs_dir.glob("episode_*.pth")))
    lo, hi = (0, n) if a.episodes is None else a.episodes
    dtype = torch.float32 if a.fp32 else torch.float16
    t0 = time.time(); done = 0
    for i in range(lo, hi):
        f = out / f"episode_{i:03d}.pt"
        if f.exists():
            continue
        img = torch.load(obs_dir / f"episode_{i:03d}.pth", weights_only=True)       # (T, 3, 224, 224) in [0, 255]
        zs = []
        with torch.no_grad():
            for s in range(0, img.shape[0], a.batch):
                x = transform(img[s:s + a.batch] / 255).to(a.device)
                zs.append(encoder(enc_tf(x)).to(dtype).cpu())
        torch.save(torch.cat(zs), f); done += 1
        if done % 100 == 0:
            print(f"[cache] {done} episodes, {time.time() - t0:.0f}s", flush=True)
    print(f"[cache] wrote {done} episodes to {out} ({dtype}) in {time.time() - t0:.0f}s")

    if a.verify:
        # the model's own path, on the first episode: encode_obs on uint8-HWC obs as the env/replay
        # hands them, vs. the cached latents. Also the planner objective from each.
        import plan as P
        from preprocessor import Preprocessor
        model = P.load_model(Path(a.model_path) / "checkpoints" / "model_latest.pth", cfg, cfg.num_action_repeat,
                             device=torch.device(a.device))
        model.eval()   # returns None (nn.Module.eval is overridden without a return)
        i = lo; img = torch.load(obs_dir / f"episode_{i:03d}.pth", weights_only=True)[: a.verify]
        obs = {"visual": (img / 255)[None], "proprio": torch.zeros(1, img.shape[0], 2)}
        obs["visual"] = transform(obs["visual"][0])[None]                          # what get_frames yields
        with torch.no_grad():
            z_model = model.encode_obs({"visual": obs["visual"].to(a.device), "proprio": obs["proprio"].to(a.device)})["visual"][0]
        z_cache = torch.load(out / f"episode_{i:03d}.pt", weights_only=True)[: a.verify].float().to(a.device)
        dz = (z_model - z_cache).abs().max().item()
        # objective deviation: mean squared distance to a fixed reference latent (the last frame)
        ref = z_model[-1:]
        v_model = ((z_model - ref) ** 2).mean(dim=(1, 2)); v_cache = ((z_cache - ref) ** 2).mean(dim=(1, 2))
        print(f"[verify] {a.verify} frames: max|z_cache - z_model| = {dz:.2e}; max|dV| = {(v_model - v_cache).abs().max().item():.2e} "
              f"(Stage 1 G3 fp16 figure 8.9e-5)")


if __name__ == "__main__":
    main()
