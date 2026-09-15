"""MCTS planner for DINO-WM (Stage 1, `stage-1-outline.md` §3–§6).

Drop-in replacement for `CEMPlanner`: same constructor kwargs from `PlanWorkspace`, same
`plan(obs_0, obs_g) -> (actions (B, H, f*d), action_len)` output, so the evaluator, replay
and analysis code from Experiment 0 run unchanged.

The search is plain UCT with an unlearned heuristic at the leaf and no rollouts past it:
one simulator call per simulation (the expanded node), min-max normalised Q, open-loop
plan by argmax visit count. Two simulators share one interface so the true env (1.1) and
the world model (1.2) differ by a constructor argument:

    EnvSimulator  state = dot position (float64, as the evaluator's env holds it)
    WMSimulator   state = latent z (196 x 404), cached per node in fp16 or fp32

All instances' trees are grown in lockstep: every simulation round gathers the leaves
that need expanding across instances into one batched simulator call. Selection and
backup stay in Python. Trees are deterministic given `seed`, so the tree after
simulation k of a K-run is the K=k tree; `snapshot_Ks` records the extracted plan at
those points and the whole K-sweep comes from one run.

Records store raw values only (`V_leaf`, `Q = W/N`) plus the final (Q_min, Q_max);
anything normalised is derived at read time. Actions only, never latents.
"""
import math
import time

import numpy as np
import torch
from einops import rearrange

from .base_planner import BasePlanner
from utils import move_to_device

ENV_ACTION_DIM = 2


# --------------------------------------------------------------------------- actions
def make_action_set(preprocessor, n_dirs=8, mags=(0.6, 1.2), frameskip=5):
    """Discrete planner actions: (direction, magnitude) held for `frameskip` env steps.

    Index a = m_idx * n_dirs + d_idx. Returns
        raw   (A, frameskip, 2) float32, env units (the env moves 2x this per step)
        flat  (A, frameskip*2) float32, normalised and flattened "(f d)" as the
              evaluator expects
        exec  (A, frameskip, 2) float32, what the env will actually execute: the
              normalise/denormalise round trip of `raw`, so EnvSimulator moves the dot
              by exactly what `eval_actions` will
    """
    angles = np.arange(n_dirs) * 2 * np.pi / n_dirs
    raw = np.array([np.tile([m * np.cos(t), m * np.sin(t)], (frameskip, 1))
                    for m in mags for t in angles], dtype=np.float32)
    norm = preprocessor.normalize_actions(torch.tensor(raw)).float()
    flat = rearrange(norm, "a f d -> a (f d)").contiguous()
    exec_ = preprocessor.denormalize_actions(norm).numpy().astype(np.float32)
    return raw, flat, exec_


# --------------------------------------------------------------------------- tree
class Tree:
    """One instance's search tree as flat arrays. Node 0 is the root."""

    def __init__(self, n_actions, capacity=64):
        self.A = n_actions
        self.n = 0
        self._alloc(capacity)
        self.qmin, self.qmax = math.inf, -math.inf

    def _alloc(self, cap):
        self.parent = np.full(cap, -1, np.int32)
        self.depth = np.zeros(cap, np.int32)
        self.action = np.full(cap, -1, np.int32)
        self.N = np.zeros(cap, np.int64)
        self.W = np.zeros(cap, np.float64)
        self.V = np.zeros(cap, np.float64)       # heuristic value at expansion, raw
        self.V_vis = np.zeros(cap, np.float64)   # visual / proprio parts of -V
        self.V_pro = np.zeros(cap, np.float64)
        self.sim_idx = np.full(cap, -1, np.int32)
        self.children = np.full((cap, self.A), -1, np.int32)

    def _grow(self):
        old = {k: getattr(self, k) for k in
               ("parent", "depth", "action", "N", "W", "V", "V_vis", "V_pro",
                "sim_idx", "children")}
        self._alloc(2 * len(self.parent))
        for k, v in old.items():
            getattr(self, k)[: len(v)] = v

    def add(self, parent, action, value, v_vis, v_pro, sim_idx):
        if self.n == len(self.parent):
            self._grow()
        i = self.n
        self.n += 1
        self.parent[i] = parent
        self.depth[i] = 0 if parent < 0 else self.depth[parent] + 1
        self.action[i] = action
        self.V[i], self.V_vis[i], self.V_pro[i] = value, v_vis, v_pro
        self.sim_idx[i] = sim_idx
        if parent >= 0:
            self.children[parent, action] = i
        return i

    def backup(self, path, value):
        for i in path:
            self.N[i] += 1
            self.W[i] += value
            if i != 0:
                q = self.W[i] / self.N[i]
                self.qmin, self.qmax = min(self.qmin, q), max(self.qmax, q)

    def q(self, ids):
        return self.W[ids] / self.N[ids]

    def path_to_root(self, i):
        out = []
        while i > 0:
            out.append(int(self.action[i]))
            i = self.parent[i]
        return out[::-1]

    def table(self):
        n = self.n
        return {
            "node_id": np.arange(n, dtype=np.int32), "parent_id": self.parent[:n].copy(),
            "depth": self.depth[:n].copy(), "action_idx": self.action[:n].copy(),
            "N": self.N[:n].copy(), "Q_raw": np.where(self.N[:n] > 0,
                                                       self.W[:n] / np.maximum(self.N[:n], 1),
                                                       np.nan),
            "V_leaf": self.V[:n].copy(), "V_leaf_vis": self.V_vis[:n].copy(),
            "V_leaf_pro": self.V_pro[:n].copy(),
            "rho": np.ones(n, np.float32), "sim_idx_expanded": self.sim_idx[:n].copy(),
        }


# --------------------------------------------------------------------------- simulators
class _LatentValue:
    """V_latent(z_obs) = -(mse_visual + alpha * mse_proprio) against z_g, per instance.

    Same quantity as `planning.objectives.create_objective_fn(alpha, ., "last")`, computed
    here so the visual and proprio parts can be recorded separately (E0: report both).
    """

    def __init__(self, z_g, alpha):
        self.z_g, self.alpha = z_g, alpha

    def __call__(self, z_obs, inst):
        inst = torch.as_tensor(inst, device=z_obs["visual"].device)
        gv = self.z_g["visual"][inst]          # (n, 1, P, D)
        gp = self.z_g["proprio"][inst]         # (n, 1, d)
        vis = ((z_obs["visual"][:, -1:] - gv) ** 2).mean(dim=(1, 2, 3))
        pro = ((z_obs["proprio"][:, -1:] - gp) ** 2).mean(dim=(1, 2))
        v = -(vis + self.alpha * pro)
        f = lambda x: x.detach().double().cpu().numpy()
        return f(v), f(vis), f(pro)


class EnvSimulator:
    """True Wall dynamics as the transition model (arm 1.1).

    `step` calls each env's own `_calculate_next_position`, which is everything
    `DotWall.step` does except drawing the frame, so it is exact and ~free. Values are
    either the true state distance (`value="state"`) or the latent heuristic on the
    rendered-and-encoded observation (`value="latent"`), the latter batched across
    instances through one DINO forward.
    """

    def __init__(self, envs, exec_actions, seeds, state_0, state_g, value="state",
                 wm=None, preprocessor=None, latent_value=None, frameskip=5,
                 device="cpu", max_batch=512):
        # gym.make wraps the env; the wrapper hides private attributes and would take
        # the `dot_position` writes itself, so work on the underlying WallEnvWrapper.
        self.envs = [getattr(e, "unwrapped", e) for e in envs]
        self.exec = exec_actions
        self.frameskip = frameskip
        self.state_g = np.asarray(state_g, dtype=np.float64)
        self.value_kind = value
        self.wm, self.pre, self.lv = wm, preprocessor, latent_value
        self.device, self.max_batch = device, max_batch
        self.n_calls = 0
        for e, s, s0 in zip(self.envs, seeds, state_0):
            e.prepare(s, s0)     # seeds the env and draws wall_img; deterministic

    def root(self, inst):
        return [self.envs[i].dot_position.clone() for i in inst]

    def step(self, inst, states, action_idx):
        out = []
        for i, pos, a in zip(inst, states, action_idx):
            env = self.envs[i]
            for f in range(self.frameskip):
                env.dot_position = pos
                pos = env._calculate_next_position(self.exec[a, f])
            out.append(pos)
        self.n_calls += len(out)
        return out

    def _render(self, i, pos):
        env = self.envs[i]
        vis = env.channels_to_img(env.wall_img, env._render_dot(pos)).float()
        vis = env.transform(vis).permute(1, 2, 0)          # (224, 224, 3), as `prepare`
        return vis.numpy(), pos.float().numpy()

    def value(self, inst, states):
        n = len(states)
        if self.value_kind == "state":
            p = np.stack([s.detach().cpu().numpy() for s in states])
            d = np.linalg.norm(p - self.state_g[np.asarray(inst)], axis=-1)
            return -d, np.full(n, np.nan), np.full(n, np.nan)
        v, vv, vp = [], [], []
        for s in range(0, n, self.max_batch):
            sl = slice(s, s + self.max_batch)
            rend = [self._render(i, pos) for i, pos in zip(inst[sl], states[sl])]
            obs = {"visual": np.stack([r[0] for r in rend])[:, None],
                   "proprio": np.stack([r[1] for r in rend])[:, None]}
            with torch.no_grad():
                z = self.wm.encode_obs(move_to_device(self.pre.transform_obs(obs), self.device))
            a, b, c = self.lv(z, inst[sl])
            v.append(a); vv.append(b); vp.append(c)
        return np.concatenate(v), np.concatenate(vv), np.concatenate(vp)


class WMSimulator:
    """World-model transitions (arm 1.2). State = latent z (P, D) incl. proprio/action
    channels. Edge (z, a): replace the action channels, one predictor call."""

    def __init__(self, wm, flat_actions, latent_value, device, cache_dtype=torch.float16,
                 max_batch=512):
        self.wm, self.act = wm, flat_actions.to(device)
        self.lv, self.device = latent_value, device
        self.cache_dtype, self.max_batch = cache_dtype, max_batch
        self.n_calls = 0

    def root(self, trans_obs_0):
        b = trans_obs_0["visual"].shape[0]
        with torch.no_grad():
            z = self.wm.encode(trans_obs_0, torch.zeros(b, 1, self.act.shape[1], device=self.device))
        return [z[i, 0].to(self.cache_dtype) for i in range(b)]

    def step(self, inst, states, action_idx):
        out = []
        for s in range(0, len(states), self.max_batch):
            z = torch.stack(states[s:s + self.max_batch]).to(self.device).float().unsqueeze(1)
            a = self.act[torch.as_tensor(action_idx[s:s + self.max_batch], device=self.device)]
            with torch.no_grad():
                z = self.wm.replace_actions_from_z(z, a.unsqueeze(1))
                z = self.wm.predict(z)[:, -1]
            out.extend(z[i] for i in range(z.shape[0]))
        self.n_calls += len(out)
        return out

    def value(self, inst, states):
        v, vv, vp = [], [], []
        for s in range(0, len(states), self.max_batch):
            z = torch.stack(states[s:s + self.max_batch]).to(self.device).float().unsqueeze(1)
            z_obs, _ = self.wm.separate_emb(z)
            a, b, c = self.lv(z_obs, inst[s:s + self.max_batch])
            v.append(a); vv.append(b); vp.append(c)
        return np.concatenate(v), np.concatenate(vv), np.concatenate(vp)

    def cache(self, z):
        return z.to(self.cache_dtype)


# --------------------------------------------------------------------------- planner
class MCTSPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        n_sims,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        env=None,
        simulator="wm",          # "wm" (1.2) | "env" (1.1)
        value="latent",          # "latent" | "state" (env only)
        mode="mcts",             # "mcts" | "greedy" (open-loop greedy, arms 1.1-G / 1.2-G)
        c_uct=0.5,               # sqrt(2) over-explores 16 untried children; see G1 sweep
        norm_floor=1e-3,
        normalize="minmax",      # "minmax": Q scaled by the tree's (min, max); "none": raw Q, c in objective units
        alpha=1.0,
        n_dirs=8,
        mags=(0.6, 1.2),
        frameskip=5,
        cache_dtype="fp16",
        snapshot_Ks=(10, 30, 100, 300, 1000, 2000),
        seed=0,
        record_path=None,
        log_every=0,
        log_filename=None,
        logging_prefix="plan_0",
        **kwargs,
    ):
        super().__init__(wm, action_dim, objective_fn, preprocessor, evaluator,
                         wandb_run, log_filename)
        assert action_dim == ENV_ACTION_DIM * frameskip, (action_dim, frameskip)
        self.horizon, self.K = horizon, n_sims
        self.env, self.sim_kind, self.value_kind, self.mode = env, simulator, value, mode
        self.c_uct, self.norm_floor, self.alpha = c_uct, norm_floor, alpha
        assert normalize in ("minmax", "none"), normalize
        self.normalize = normalize
        self.frameskip, self.seed = frameskip, seed
        self.cache_dtype = {"fp16": torch.float16, "fp32": torch.float32}[cache_dtype]
        self.snapshot_Ks = sorted({k for k in snapshot_Ks if k <= n_sims} | {n_sims})
        self.record_path, self.log_every = record_path, log_every
        self.raw_actions, self.flat_actions, self.exec_actions = make_action_set(
            preprocessor, n_dirs, mags, frameskip)
        self.A = len(self.flat_actions)
        self.snapshots = []
        self.trees = []
        self.states = []

    # ---- setup
    def _goal_latents(self, obs_g):
        with torch.no_grad():
            return self.wm.encode_obs(move_to_device(self.preprocessor.transform_obs(obs_g),
                                                     self.device))

    def _make_simulator(self, obs_0, obs_g):
        z_g = self._goal_latents(obs_g)
        self.z_g = z_g
        lv = _LatentValue(z_g, self.alpha)
        if self.sim_kind == "wm":
            sim = WMSimulator(self.wm, self.flat_actions, lv, self.device, self.cache_dtype)
            trans0 = move_to_device(self.preprocessor.transform_obs(obs_0), self.device)
            roots = sim.root(trans0)
        else:
            ev = self.evaluator
            envs = self.env.envs if hasattr(self.env, "envs") else self.env
            sim = EnvSimulator(envs, self.exec_actions, ev.seed, ev.state_0, ev.state_g,
                               value=self.value_kind, wm=self.wm, preprocessor=self.preprocessor,
                               latent_value=lv, frameskip=self.frameskip, device=self.device)
            roots = sim.root(list(range(len(envs))))
        return sim, roots

    # ---- search primitives
    def _select(self, t, rng):
        """Descend by UCT until a node with an untried action (expand) or depth H (re-eval)."""
        node, path = 0, [0]
        while t.depth[node] < self.horizon:
            ch = t.children[node]
            untried = np.flatnonzero(ch < 0)
            if len(untried):
                a = untried[0] if len(untried) == 1 else rng.choice(untried)
                return path, node, int(a)
            q = t.q(ch)
            if self.normalize == "minmax":
                # MuZero-style min-max normalisation with a floor: below the floor the
                # values are indistinguishable and the exploration term decides (§10 B2).
                qn = (q - t.qmin) / max(t.qmax - t.qmin, self.norm_floor)
            else:
                qn = q                       # raw Q; c_uct is then in objective units (Stage 2 §5.3)
            u = qn + self.c_uct * np.sqrt(np.log(t.N[node]) / t.N[ch])
            best = np.flatnonzero(u == u.max())
            a = best[0] if len(best) == 1 else rng.choice(best)
            node = int(ch[a])
            path.append(node)
        return path, None, None

    def _extract(self, t):
        """Open-loop plan: argmax N from the root (ties -> higher Q). Returns the action
        list (len = depth reached), the node ids on it, and the raw V_leaf at its end."""
        node, acts, nodes = 0, [], [0]
        for _ in range(self.horizon):
            ch = t.children[node]
            if (ch < 0).all():
                break
            n = np.where(ch >= 0, t.N[np.maximum(ch, 0)], -1)
            q = np.where(ch >= 0, t.W[np.maximum(ch, 0)] / np.maximum(t.N[np.maximum(ch, 0)], 1), -np.inf)
            a = int(np.lexsort((q, n))[-1])
            acts.append(a)
            node = int(ch[a])
            nodes.append(node)
        return acts, nodes, float(t.V[node])

    def _plans(self):
        """Extracted plans for all instances, padded past the tree by repeating the last
        action (only happens at tiny K, where the PV does not reach depth H; recorded
        as `plan_depth`)."""
        B, H = len(self.trees), self.horizon
        idx = np.zeros((B, H), np.int64)
        depth, pv_val, pv_nodes = np.zeros(B, np.int32), np.zeros(B), []
        for i, t in enumerate(self.trees):
            acts, nodes, v = self._extract(t)
            depth[i], pv_val[i] = len(acts), v
            pv_nodes.append(nodes)
            fill = acts[-1] if acts else 0
            idx[i] = acts + [fill] * (H - len(acts))
        actions = self.flat_actions[torch.as_tensor(idx)].to(self.device)
        return actions, idx, depth, pv_val, pv_nodes

    def _snapshot(self, k):
        actions, idx, depth, pv_val, pv_nodes = self._plans()
        maxn = max(t.n for t in self.trees)
        N = np.zeros((len(self.trees), maxn), np.int64)
        for i, t in enumerate(self.trees):
            N[i, : t.n] = t.N[: t.n]
        self.snapshots.append({"K": k, "actions": actions.cpu().numpy(), "action_idx": idx,
                               "plan_depth": depth, "pv_value": pv_val, "pv_nodes": pv_nodes,
                               "N": N, "n_calls": self.sim.n_calls,
                               "Qrange": np.array([[t.qmin, t.qmax] for t in self.trees]),
                               "nodes": np.array([t.n for t in self.trees]),
                               "elapsed": time.time() - self._t0})

    # ---- main loops
    def _search(self, sim, roots):
        B = len(roots)
        self.sim = sim
        self.trees = [Tree(self.A) for _ in range(B)]
        self.states = [[r] for r in roots]     # per instance, indexed by node id
        # Seeded by the instance's own eval_seed, not its batch position, so a tree is
        # the same whether the instance is planned alone or in lockstep (G5).
        rngs = [np.random.default_rng([self.seed, int(self.evaluator.seed[i])]) for i in range(B)]
        v0, vv0, vp0 = sim.value(list(range(B)), roots)
        for i, t in enumerate(self.trees):
            t.add(-1, -1, v0[i], vv0[i], vp0[i], -1)
        self._t0 = time.time()
        next_snap = 0
        for k in range(1, self.K + 1):
            paths, req = [], []          # req: (inst, parent, action, path)
            for i, t in enumerate(self.trees):
                path, node, a = self._select(t, rngs[i])
                if node is None:          # depth-H leaf: re-evaluate, no simulator call
                    t.backup(path, t.V[path[-1]])
                else:
                    req.append((i, node, a, path))
            if req:
                inst = [r[0] for r in req]
                st = [self.states[i][n] for i, n, _, _ in req]
                acts = [r[2] for r in req]
                new = sim.step(inst, st, acts)
                v, vv, vp = sim.value(inst, new)
                for j, (i, node, a, path) in enumerate(req):
                    t = self.trees[i]
                    c = t.add(node, a, v[j], vv[j], vp[j], k - 1)
                    self.states[i].append(sim.cache(new[j]) if hasattr(sim, "cache") else new[j])
                    assert c == len(self.states[i]) - 1
                    t.backup(path + [c], v[j])
            if k == self.snapshot_Ks[next_snap]:
                self._snapshot(k)
                next_snap += 1
            if self.log_every and k % self.log_every == 0:
                print(f"[mcts] sim {k}/{self.K}  calls {sim.n_calls}  "
                      f"nodes/inst {np.mean([t.n for t in self.trees]):.0f}  "
                      f"{time.time() - self._t0:.1f}s", flush=True)

    def _greedy(self, sim, roots):
        """Open-loop greedy (arms 1.1-G / 1.2-G): argmax_a V(step(s, a)) over the action
        set, advance the *simulated* state, repeat to depth H."""
        B, H, A = len(roots), self.horizon, self.A
        self.sim = sim
        states = list(roots)
        idx = np.zeros((B, H), np.int64)
        vals = np.zeros((B, H))
        self._t0 = time.time()
        for h in range(H):
            inst = [i for i in range(B) for _ in range(A)]
            st = [states[i] for i in range(B) for _ in range(A)]
            acts = [a for _ in range(B) for a in range(A)]
            new = sim.step(inst, st, acts)
            v, _, _ = sim.value(inst, new)
            v = v.reshape(B, A)
            best = v.argmax(1)
            idx[:, h], vals[:, h] = best, v[np.arange(B), best]
            # advance through the same cache precision as the tree (§3.7: hold it constant)
            states = [sim.cache(new[i * A + best[i]]) if hasattr(sim, "cache") else new[i * A + best[i]]
                      for i in range(B)]
        actions = self.flat_actions[torch.as_tensor(idx)].to(self.device)
        self.snapshots = [{"K": 0, "actions": actions.cpu().numpy(), "action_idx": idx,
                           "plan_depth": np.full(B, H, np.int32), "pv_value": vals[:, -1],
                           "greedy_values": vals, "n_calls": sim.n_calls,
                           "elapsed": time.time() - self._t0}]
        return actions

    def plan(self, obs_0, obs_g, actions=None):
        sim, roots = self._make_simulator(obs_0, obs_g)
        if self.mode == "greedy":
            out = self._greedy(sim, roots)
        else:
            self._search(sim, roots)
            out = torch.as_tensor(self.snapshots[-1]["actions"]).to(self.device)
        if self.record_path:
            self.dump(self.record_path)
        return out, np.full(out.shape[0], np.inf)

    # ---- records
    def pv_state(self, i):
        """Cached simulator state at the end of instance i's final extracted plan."""
        return self.states[i][self.snapshots[-1]["pv_nodes"][i][-1]]

    def record(self):
        """Self-contained record: node tables (all instances, `inst` column), snapshots,
        conditions and settings. Everything Stage 2 needs, nothing derived."""
        ev = self.evaluator
        R = {"state_0": np.asarray(ev.state_0, np.float32), "state_g": np.asarray(ev.state_g, np.float32),
             "eval_seed": np.asarray(ev.seed, np.int64), "K": np.asarray(self.K),
             "c_uct": np.asarray(self.c_uct), "norm_floor": np.asarray(self.norm_floor),
             "alpha": np.asarray(self.alpha), "H": np.asarray(self.horizon),
             "frameskip": np.asarray(self.frameskip), "goal_H": np.asarray(self.horizon),
             "seed": np.asarray(self.seed), "simulator": np.asarray(self.sim_kind),
             "value": np.asarray(self.value_kind), "mode": np.asarray(self.mode),
             "cache_dtype": np.asarray(str(self.cache_dtype)), "normalize": np.asarray(self.normalize),
             "action_set_raw": self.raw_actions, "action_set_flat": self.flat_actions.cpu().numpy(),
             "snap_K": np.array([s["K"] for s in self.snapshots]),
             "snap_actions": np.stack([s["actions"] for s in self.snapshots]),
             "snap_action_idx": np.stack([s["action_idx"] for s in self.snapshots]),
             "snap_plan_depth": np.stack([s["plan_depth"] for s in self.snapshots]),
             "snap_pv_value": np.stack([s["pv_value"] for s in self.snapshots]),
             "snap_n_calls": np.array([s["n_calls"] for s in self.snapshots]),
             "snap_elapsed": np.array([s["elapsed"] for s in self.snapshots])}
        if self.mode == "greedy":
            R["greedy_values"] = self.snapshots[0]["greedy_values"]
            return R
        tabs = [t.table() for t in self.trees]
        for key in tabs[0]:
            R[key] = np.concatenate([tb[key] for tb in tabs])
        R["inst"] = np.concatenate([np.full(t.n, i, np.int32) for i, t in enumerate(self.trees)])
        on_plan = np.zeros(len(R["inst"]), bool)
        off = np.cumsum([0] + [t.n for t in self.trees[:-1]])
        for i, nodes in enumerate(self.snapshots[-1]["pv_nodes"]):
            on_plan[off[i] + np.asarray(nodes)] = True
        R["on_plan"] = on_plan
        R["Q_min"] = np.array([t.qmin for t in self.trees])
        R["Q_max"] = np.array([t.qmax for t in self.trees])
        R["nodes_created"] = np.array([t.n - 1 for t in self.trees])
        R["n_calls"] = np.asarray(self.sim.n_calls)
        maxn = max(t.n for t in self.trees)
        R["snap_N"] = np.stack([np.pad(s["N"], ((0, 0), (0, maxn - s["N"].shape[1])))
                                for s in self.snapshots])
        R["snap_Qrange"] = np.stack([s["Qrange"] for s in self.snapshots])   # (S, B, 2): range trajectory
        # N-per-depth diagnostic (§6): does N vary within a depth at all?
        for d in range(1, self.horizon + 1):
            sel = R["depth"] == d
            R[f"N_depth{d}"] = R["N"][sel]
        return R

    def dump(self, path):
        R = self.record()
        np.savez_compressed(path, **R)
        n = int(R["nodes_created"].sum()) if "nodes_created" in R else 0
        print(f"[mcts] wrote record to {path}  ({n} nodes, {int(R['snap_n_calls'][-1])} calls)")
