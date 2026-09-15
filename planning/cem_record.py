"""CEM planner that logs candidate action sequences for the optimism-gap measurement.

Logs *actions*, never latents: rollouts are deterministic given actions (verified), so the
analysis pass recomputes imagined latents for whatever subset it replays. A full run's log
is a few MB rather than the ~250 GB the latents would take.

Per (eval instance, opt_step) it records the elite set and an equal-sized set drawn
uniformly from the same candidates. The elite-vs-random contrast is what identifies
selection-induced optimism, since the model's baseline bias cancels in the pairing.
"""

import numpy as np
import torch

from .cem import CEMPlanner


class _ChunkedWM:
    """Thin proxy over the world model that runs `rollout` in batches of `chunk` candidates.

    Needed where num_hist > 1: the predictor attends over num_hist x 196 tokens, so 1000
    candidates at once exceed the V100 (PushT, num_hist = 3). Everything else delegates.
    """

    def __init__(self, wm, chunk):
        self._wm, self._chunk = wm, chunk

    def __getattr__(self, name):
        return getattr(self._wm, name)

    def rollout(self, obs_0, act):
        n = act.shape[0]
        if n <= self._chunk:
            return self._wm.rollout(obs_0=obs_0, act=act)
        zo, zs = [], []
        for s in range(0, n, self._chunk):
            o = {k: v[s:s + self._chunk] for k, v in obs_0.items()}
            z_obs, z = self._wm.rollout(obs_0=o, act=act[s:s + self._chunk])
            zo.append(z_obs); zs.append(z)
        return {k: torch.cat([d[k] for d in zo], 0) for k in zo[0]}, torch.cat(zs, 0)


class RecordingCEMPlanner(CEMPlanner):
    def __init__(
        self,
        *args,
        record_path="cem_records.npz",
        n_random=None,
        record_all_evals=(),
        record_seed=0,
        rollout_chunk=None,
        eval_offset=0,
        **kwargs,
    ):
        """
        Args:
            record_path: where the .npz log is written at the end of plan()
            n_random: size of the uniform control set (default: same as topk)
            record_all_evals: eval indices for which *all* candidates are logged.
                Needed for selection regret, which requires ground truth for every
                candidate: min_j d*_j over a 60-sample subset is not the same quantity.
            record_seed: seed for the control-set draw, so runs are reproducible
        """
        super().__init__(*args, **kwargs)
        if rollout_chunk:
            self.wm = _ChunkedWM(self.wm, rollout_chunk)
        self.eval_offset = eval_offset          # global instance id = traj + offset (instance-range runs)
        self.record_path = record_path
        self.n_random = n_random
        self.record_all_evals = set(record_all_evals)
        self._rng = np.random.default_rng(record_seed)
        self._records = []
        self._spread = []

    def _record_candidates(self, traj, opt_step, action, loss, topk_idx):
        n_samples = action.shape[0]
        traj = traj + self.eval_offset
        topk_idx = topk_idx.detach().cpu().numpy()
        loss_np = loss.detach().cpu().numpy()

        if traj in self.record_all_evals:
            # Everything not in the elite set is the control group; keeping all of it
            # (rather than a subsample) is what makes selection regret computable.
            keep = np.arange(n_samples)
            tag = np.full(n_samples, 2, dtype=np.int8)  # 2 = control (non-elite)
            tag[topk_idx] = 1                           # 1 = elite
        else:
            n_rand = self.n_random if self.n_random is not None else len(topk_idx)
            pool = np.setdiff1d(np.arange(n_samples), topk_idx, assume_unique=False)
            n_rand = min(n_rand, len(pool))
            rand_idx = self._rng.choice(pool, size=n_rand, replace=False)
            keep = np.concatenate([topk_idx, rand_idx])
            tag = np.concatenate([
                np.ones(len(topk_idx), dtype=np.int8),
                np.full(n_rand, 2, dtype=np.int8),      # 2 = random control
            ])

        # rank of each kept candidate under the imagined objective (0 = best)
        order = np.argsort(loss_np)
        rank_of = np.empty(n_samples, dtype=np.int32)
        rank_of[order] = np.arange(n_samples)

        acts = action[keep].detach().cpu().numpy().astype(np.float32)
        self._records.append({
            "eval_idx": np.full(len(keep), traj, dtype=np.int32),
            "opt_step": np.full(len(keep), opt_step, dtype=np.int32),
            "candidate_idx": keep.astype(np.int32),
            "tag": tag,
            "imagined_rank": rank_of[keep],
            "imagined_loss": loss_np[keep].astype(np.float32),
            "action_seq": acts,
        })

        # candidate-set spread: distinguishes a null elite-vs-random contrast from one
        # that collapsed because sigma annealed the candidates together (§6).
        a_flat = action.reshape(n_samples, -1)
        sub = a_flat[: min(64, n_samples)]
        d = torch.cdist(sub, sub)
        n = sub.shape[0]
        self._spread.append((traj, opt_step,
                             float(d.sum() / max(n * (n - 1), 1)),
                             float(action.std().item())))

    def _dump(self):
        if not self._records:
            return
        out = {k: np.concatenate([r[k] for r in self._records], axis=0)
               for k in self._records[0]}
        sp = np.array(self._spread, dtype=np.float32)
        out["spread_eval_idx"] = sp[:, 0]
        out["spread_opt_step"] = sp[:, 1]
        out["spread_mean_pairwise"] = sp[:, 2]
        out["spread_sigma"] = sp[:, 3]
        np.savez_compressed(self.record_path, **out)
        print(f"[cem_record] wrote {len(out['eval_idx'])} candidate records "
              f"to {self.record_path}")

    def plan(self, obs_0, obs_g, actions=None):
        result = super().plan(obs_0, obs_g, actions)
        self._dump()
        return result
