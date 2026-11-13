"""
Reference solutions used to judge the agent:
  * exact_optimum:        true optimum by enumeration (4 workstations only);
  * simulated_annealing:  good solution for any size, same moves as the agent;
  * steepest_descent:     greedy local search (best single action until no action improves).
"""
from itertools import permutations, product

import numpy as np

from flp_env import N_CFG_PER_SITE

_ENUM_CACHE = {}


def _encode(env, cfgs):
    base = env.n * N_CFG_PER_SITE
    key = np.zeros(len(cfgs), dtype=np.int64)
    for m in range(env.n):
        key = key * base + env.rank[cfgs[:, m]]
    return key


def canonical_valid_layouts(env):
    """
    All valid canonical layouts (one per symmetry class). 4 workstations: 480,000 layouts when
    overlapping P/D points are allowed, 313,200 when they are forbidden.
    """
    assert env.n == 4, 'enumeration is only practical for 4 workstations'
    cache_key = (env.n, env.allow_overlap)
    if cache_key not in _ENUM_CACHE:
        per_site = np.array(list(product(range(N_CFG_PER_SITE), repeat=4)), dtype=np.int64)
        cfgs = np.concatenate([per_site + np.array(p, dtype=np.int64) * N_CFG_PER_SITE
                               for p in permutations(range(4))])
        if not env.allow_overlap:
            pts = np.sort(np.concatenate((env.p_pt_np[cfgs], env.d_pt_np[cfgs]), axis=1), axis=1)
            cfgs = cfgs[~(pts[:, 1:] == pts[:, :-1]).any(axis=1)]
        keys = np.stack([_encode(env, env.transform[g][cfgs]) for g in range(env.transform.shape[0])])
        _ENUM_CACHE[cache_key] = cfgs[keys[0] == keys.min(axis=0)]
    return _ENUM_CACHE[cache_key]


def exact_optimum(env, demand):
    """
    True minimum cost for a demand (4 workstations). The loaded cost is a lower bound of the total
    cost, so layouts are examined by increasing loaded cost until the bound exceeds the best total.
    """
    cfgs = canonical_valid_layouts(env)
    env.set_demand(demand, ('exact', np.asarray(demand).tobytes()))
    p, d = env.p_pt_np[cfgs], env.d_pt_np[cfgs]
    loaded = (env.dist[p[:, :, None], d[:, None, :]] * env.demand[None]).sum(axis=(1, 2))
    best, best_cfg = np.inf, None
    for i in np.argsort(loaded, kind='stable'):
        if loaded[i] >= best - 1e-9:
            break
        total = env.cost_of(cfgs[i])[0]
        if total < best - 1e-9:
            best, best_cfg = total, cfgs[i].copy()
    return float(best), best_cfg


def simulated_annealing(env, demand, iterations=20000, t_start=0.05, t_end=0.0005, seed=0, start_cfg=None):
    """Metropolis search on the same action set; temperatures are relative to the initial cost."""
    rng = np.random.default_rng(seed)
    env.set_demand(demand, ('sa', np.asarray(demand).tobytes()))
    cfg = env.default_cfg.copy() if start_cfg is None else np.array(start_cfg)
    cost = env.cost_of(cfg)[0]
    scale = max(env.cost_of(env.default_cfg)[0], 1e-9)
    best, best_cfg = cost, cfg.copy()
    for it in range(iterations):
        temperature = t_start * (t_end / t_start) ** (it / max(iterations - 1, 1))
        new = env._apply(cfg, int(rng.integers(env.n_actions)))
        if env.is_invalid(new):
            continue
        new_cost = env.cost_of(new)[0]
        if new_cost <= cost or rng.random() < np.exp((cost - new_cost) / (scale * temperature)):
            cfg, cost = new, new_cost
            if cost < best - 1e-9:
                best, best_cfg = cost, cfg.copy()
    return float(best), best_cfg


def steepest_descent(env, demand, start_cfg=None, max_steps=1000):
    """Apply the single best action until none improves. Returns (cost, cfg, number of steps)."""
    env.set_demand(demand, ('sd', np.asarray(demand).tobytes()))
    cfg = env.default_cfg.copy() if start_cfg is None else np.array(start_cfg)
    cost = env.cost_of(cfg)[0]
    for step in range(max_steps):
        best_new, best_cost = None, cost
        for a in range(env.n_actions):
            new = env._apply(cfg, a)
            if env.is_invalid(new):
                continue
            c = env.cost_of(new)[0]
            if c < best_cost - 1e-9:
                best_new, best_cost = new, c
        if best_new is None:
            return float(cost), cfg, step
        cfg, cost = best_new, best_cost
    return float(cost), cfg, max_steps
