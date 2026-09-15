"""Optimism-gap quantities for the CEM measurement (Experiment 1, §6).

Everything is defined against the planner's own objective so the numbers are
commensurable with what CEM actually minimises:

    d(a, b) = mean_v (a_v - b_v)^2 + alpha * mean_p (a_p - b_p)^2

a squared norm under the weighted inner product implemented by `wdot`.
With e = z_hat - z_real (model error) and u = z_real - z_goal:

    G = d(z_real, z_g) - d(z_hat, z_g) = -2<u,e> - <e,e>     (exact)
    A = <e, -u> / (|e| |u|)  in [-1, 1]                       (alignment)
    G = 2|e||u|A - |e|^2                                      (exact identity)

The null is *negative*, not zero: under symmetric error (E[A]=0) the expected gap is
-E|e|^2, because drift alone makes the imagined state look farther from the goal.
Optimism means A > 0; G only turns positive once A > |e| / (2|u|).
"""

import numpy as np


def wdot(x, y, alpha=1.0, part="both"):
    """Weighted inner product matching the planner objective.

    `part` selects the visual-only or proprio-only component. Worth reporting
    separately: proprio averages over 2 dims against visual's 196*384, yet alpha=1
    weights them equally, so proprio carries nearly all the sampling variance of G
    and A while visual self-averages to near-zero noise.
    """
    v = (x["visual"] * y["visual"]).reshape(*x["visual"].shape[:-2], -1).mean(-1)
    p = (x["proprio"] * y["proprio"]).mean(-1)
    if part == "visual":
        return v
    if part == "proprio":
        return alpha * p
    return v + alpha * p


def _sub(a, b):
    return {k: a[k] - b[k] for k in ("visual", "proprio")}


def metrics(z_hat, z_real, z_goal, alpha=1.0, eps=1e-12, part="both", drift_tol=1e-4):
    """Per-depth optimism quantities. Arrays are (B, H, ...); returns (B, H).

    Pass part="visual" or "proprio" to decompose; see `wdot` for why that matters.
    """
    e = _sub(z_hat, z_real)
    u = _sub(z_real, z_goal)
    ee = wdot(e, e, alpha, part)
    uu = wdot(u, u, alpha, part)
    ue = wdot(u, e, alpha, part)
    drift = np.sqrt(np.maximum(ee, 0.0))
    dist = np.sqrt(np.maximum(uu, 0.0))
    gap = -2.0 * ue - ee
    # Alignment is a direction, so it is undefined when there is no error to point
    # anywhere: at h=0, z_hat == z_real by construction. On GPU that is ~1e-7 rather
    # than exactly 0, and the 0/0 ratio would otherwise report a spurious value.
    align = -ue / np.maximum(drift * dist, eps)
    align = np.where((drift <= drift_tol) | (dist <= drift_tol), np.nan, align)
    return {
        "drift": drift,
        "dist_to_goal": dist,
        "alignment": align,
        "gap": gap,
        "gap_from_identity": 2.0 * drift * dist * align - ee,
    }


def shuffled_null(z_hat, z_real, z_goal, alpha=1.0, n=200, seed=0):
    """Empirical null for A: pair each candidate's e with another candidate's u.

    The nominal latent dimension (196x384) badly overstates the effective one, since
    DINO patch tokens are heavily correlated, so 1/sqrt(d) is the wrong null.
    """
    rng = np.random.default_rng(seed)
    e = _sub(z_hat, z_real)
    u = _sub(z_real, z_goal)
    B = e["visual"].shape[0]
    out = []
    for _ in range(n):
        perm = rng.permutation(B)
        up = {k: u[k][perm] for k in u}
        ee = wdot(e, e, alpha); uu = wdot(up, up, alpha); ue = wdot(up, e, alpha)
        out.append(-ue / np.maximum(np.sqrt(ee * uu), 1e-12))
    return np.concatenate(out, axis=0)


def align_imagined_to_real(n_planner_steps, frameskip):
    """Index map from imagined step h to replayed env step (gate 4).

    The planner stores actions flattened as (f d); the evaluator expands them with
    `rearrange(a, "b t (f d) -> b (t f) d")`, so the env advances `frameskip` steps per
    planner step and returns t*frameskip + 1 observations.
    """
    return np.arange(n_planner_steps + 1) * frameskip
