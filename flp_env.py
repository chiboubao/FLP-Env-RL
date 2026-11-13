"""
Fast multi-demand FLP environment for policy-gradient training (PPO).

Same problem, same actions (same numbering) and same cost as `Environment.Env`, but:
  * a layout is a vector of integers (one "configuration" id per workstation) and every geometric
    quantity is looked up in tables built once at start-up, so a step costs ~0.1-0.3 ms instead
    of 60-900 ms;
  * the transportation demand is part of the observation and changes from one episode to the
    next (fixed list of demand matrices and/or a random generator), so one policy learns to
    solve many FLPs;
  * the reward is normalised per demand (see `step`);
  * `action_mask()` gives the actions that do not create overlapping P/D points.

Cost = loaded travel (sum of demand x shortest path P_i -> D_j) + empty travel (transportation
problem between delivery and pickup points), exactly as in `Environment_FLP.Configuration_cost`.
"""
import json
import os
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import shortest_path

from Environment_FLP.canonical import SHAPES, LayoutGeometry, _key
from Environment_FLP.transport import transport_cost

ROOT = os.path.dirname(os.path.abspath(__file__))
N_SHAPES = len(SHAPES)          # 5
N_CFG_PER_SITE = 4 * N_SHAPES   # 4 reference corners x 5 shapes
REWARDS = ('improvement', 'best', 'legacy')
LOOKAHEAD_SCALE = 20.0          # a 5% loaded-cost gain adds 1 to the score (logit) of the action


def load_demand(path):
    """Demand matrix from a CSV (rows P1..PN = from, columns D1..DN = to)."""
    return pd.read_csv(path, index_col=0).to_numpy(dtype=float)


def random_demand(n, rng, n_products=None, route_length=(2, 4), quantity=(1, 40)):
    """
    Random demand built like the use case of the paper: a few products, each with a random route
    through distinct workstations and a random quantity; flows are accumulated along the routes.
    Default number of products: 2 to 4 for 4 workstations, N/2 to N for larger workshops.
    """
    if n_products is None:
        n_products = (2, 4) if n <= 4 else (n // 2, n)
    demand = np.zeros((n, n))
    for _ in range(int(rng.integers(n_products[0], n_products[1] + 1))):
        length = int(rng.integers(route_length[0], min(route_length[1], n) + 1))
        route = rng.choice(n, size=length, replace=False)
        qty = int(rng.integers(quantity[0], quantity[1] + 1))
        for a, b in zip(route[:-1], route[1:]):
            demand[a, b] += qty
    return demand


class FLPEnv:
    def __init__(self, number_of_machines, demands=None, demand_sampler=None, sampler_probability=1.0,
                 episode_length=40, reward='improvement', canonical=True, random_start=0.0,
                 legacy_bounds=None, invalid_penalty=0.05, allow_overlap=True, lookahead=True,
                 seed=None):
        """
        demands:          list of N x N arrays (or CSV paths); one is drawn at each reset.
        demand_sampler:   optional function rng -> N x N array, used with `sampler_probability`
                          (always, if `demands` is empty).
        reward:           'improvement' | 'best' | 'legacy' (see `step`).
        canonical:        map every layout to its representative under the grid symmetries.
        random_start:     probability that an episode starts from a random valid layout instead
                          of the default initial layout.
        allow_overlap:    True (as in the paper/thesis): a pickup or delivery point may share its
                          spot with a point of another workstation (distance 0 between them, e.g.
                          a direct hand-over D1 -> P3). False: such layouts are forbidden and the
                          corresponding actions are masked.
        lookahead:        add to the observation, for every action, the loaded-cost reduction it
                          would produce (one vectorised table look-up, no extra LP). The policy
                          adds these values to its action scores, so it starts as a soft greedy
                          descent and learns when to deviate from it.
        """
        assert reward in REWARDS, reward
        self.n = n = number_of_machines
        self.episode_length = episode_length
        self.reward_type = reward
        self.canonical = canonical
        self.random_start = random_start
        self.legacy_bounds = legacy_bounds
        self.invalid_penalty = invalid_penalty
        self.allow_overlap = allow_overlap
        self.lookahead = lookahead
        self.rng = np.random.default_rng(seed)

        self.demands = [load_demand(d) if isinstance(d, str) else np.asarray(d, dtype=float)
                        for d in (demands or [])]
        for d in self.demands:
            assert d.shape == (n, n), f'demand shape {d.shape}, expected {(n, n)}'
        self.demand_sampler = demand_sampler
        self.sampler_probability = sampler_probability if self.demands else 1.0
        assert self.demands or demand_sampler is not None, 'give demands and/or a demand_sampler'

        self._build_tables()
        self._build_actions()

        self.n_actions = len(self.actions)
        self.n_lookahead = self.n_actions if lookahead else 0
        self.obs_size = 4 * n + 3 * n * n + 3 + self.n_lookahead

        self._lp_caches = {}
        self._ref_costs = {}
        self.cfg = None
        self.reset()

    # ------------------------------------------------------------------ tables
    def _build_tables(self):
        n = self.n
        graph = nx.read_gml(os.path.join(ROOT, 'Initial_layout', f'graph{n}M.gml'))
        with open(os.path.join(ROOT, 'sites_data', f'sites{n}machines.json')) as f:
            sites = {int(k): v for k, v in json.load(f).items()}
        node_pos = {node: data['pos'] for node, data in graph.nodes(data=True)}
        self.geometry = geo = LayoutGeometry(sites, node_pos)
        assert len(geo.site_corners) == n

        # configuration id = site * 20 + reference index * 5 + shape index
        n_cfg = n * N_CFG_PER_SITE
        point_id = {}
        self.cfg_triple = []
        p_pt, d_pt, xy = [], [], np.zeros((n_cfg, 4))
        for cfg in range(n_cfg):
            site, ref_i, shape_i = self.decode(cfg)
            ref = geo.site_corners[site][ref_i]
            self.cfg_triple.append((site, ref, SHAPES[shape_i]))
            p, d = geo.pd_coords(site, ref, SHAPES[shape_i])
            xy[cfg] = (p[0], p[1], d[0], d[1])
            p_pt.append(point_id.setdefault(_key(p), len(point_id)))
            d_pt.append(point_id.setdefault(_key(d), len(point_id)))
        self.triple_cfg = {t: c for c, t in enumerate(self.cfg_triple)}
        self.p_pt, self.d_pt = p_pt, d_pt                  # python lists (fast scalar access)
        self.p_pt_np, self.d_pt_np = np.array(p_pt), np.array(d_pt)
        self.n_points = len(point_id)
        self.xy = xy
        self.xy_norm = xy.copy()
        self.xy_norm[:, [0, 2]] = (xy[:, [0, 2]] - geo.xmin) / (geo.xmax - geo.xmin)
        self.xy_norm[:, [1, 3]] = (xy[:, [1, 3]] - geo.ymin) / (geo.ymax - geo.ymin)

        # shortest-path distance between every pair of candidate P/D points: the aisle graph with
        # all candidate points inserted on the aisles (subdividing an edge keeps the distances)
        nodes = {}
        for label in geo.grid_nodes:
            nodes[_key(geo.node_pos[label])] = len(nodes)
        offset = len(nodes)
        for key, pid in point_id.items():
            nodes[key] = offset + pid
        adj = lil_matrix((len(nodes), len(nodes)))
        grid_x = {_key(geo.node_pos[l])[0] for l in geo.grid_nodes}
        grid_y = {_key(geo.node_pos[l])[1] for l in geo.grid_nodes}
        for axis, lines in ((0, grid_x), (1, grid_y)):
            for line in lines:
                on_line = sorted(k for k in nodes if k[axis] == line)
                for a, b in zip(on_line[:-1], on_line[1:]):
                    w = (abs(a[0] - b[0]) + abs(a[1] - b[1])) / 2.0   # keys are 2 x coordinates
                    adj[nodes[a], nodes[b]] = w
                    adj[nodes[b], nodes[a]] = w
        full = shortest_path(adj.tocsr(), method='D', directed=False)
        self.dist = full[offset:, offset:].copy()
        assert np.all(np.isfinite(self.dist))
        self.dist_scale = float(self.dist.max())

        # grid symmetries: transform[g, cfg] -> cfg of the image; rank gives the canonical order
        self.transform = np.zeros((len(geo.transforms), n_cfg), dtype=np.int64)
        for g_i, g in enumerate(geo.transforms):
            for cfg in range(n_cfg):
                image, _ = geo.transform_layout(g, [self.cfg_triple[cfg]])
                self.transform[g_i, cfg] = self.triple_cfg[image[0]]
        order = sorted(range(n_cfg), key=lambda c: (c // N_CFG_PER_SITE, _key(xy[c, :2]), _key(xy[c, 2:])))
        self.rank = np.zeros(n_cfg, dtype=np.int64)
        self.rank[order] = np.arange(n_cfg)

        self.default_cfg = np.arange(n, dtype=np.int64) * N_CFG_PER_SITE   # site m, 1st corner, U-shape

    def _build_actions(self):
        """Same numbering as Env.generate_action_dict: 7 per workstation, then all pairs m<n."""
        self.actions, self.action_names = [], []
        for m in range(self.n):
            for s, shape in enumerate(SHAPES):
                self.actions.append(('shape', m, s))
                self.action_names.append(f'{m + 1}_{shape}')
            self.actions.append(('rot', m, 1))
            self.action_names.append(f'{m + 1}_rotation_clockwise')
            self.actions.append(('rot', m, -1))
            self.action_names.append(f'{m + 1}_rotation_counterclockwise')
        for m, k in combinations(range(self.n), 2):
            self.actions.append(('swap', m, k))
            self.action_names.append(f'{m + 1}_reimplant_machines_{m + 1}_to_{k + 1}')
        kinds = np.array([a[0] for a in self.actions])
        self._act_m = np.array([a[1] for a in self.actions])
        self._act_x = np.array([a[2] for a in self.actions])
        self._shape_idx = np.flatnonzero(kinds == 'shape')
        self._rot_idx = np.flatnonzero(kinds == 'rot')
        self._swap_idx = np.flatnonzero(kinds == 'swap')

    @staticmethod
    def decode(cfg):
        site, rest = divmod(int(cfg), N_CFG_PER_SITE)
        ref_i, shape_i = divmod(rest, N_SHAPES)
        return site, ref_i, shape_i

    # ----------------------------------------------------------------- layouts
    def layout(self, cfg=None):
        """[(site index, reference node, shape)] per workstation (format of canonical.py)."""
        return [self.cfg_triple[c] for c in (self.cfg if cfg is None else cfg)]

    def cfg_from_layout(self, layout):
        return np.array([self.triple_cfg[(int(s), str(r), sh)] for s, r, sh in layout], dtype=np.int64)

    def has_overlap(self, cfg):
        pts = [self.p_pt[c] for c in cfg] + [self.d_pt[c] for c in cfg]
        return len(set(pts)) != len(pts)

    def is_invalid(self, cfg):
        """A layout is invalid only when overlaps are forbidden and two P/D points coincide."""
        return (not self.allow_overlap) and self.has_overlap(cfg)

    def canonicalize(self, cfg):
        """Return (canonical cfg, index of the symmetry used; 0 = identity = already canonical)."""
        images = self.transform[:, cfg]
        keys = self.rank[images].tolist()
        best = min(range(len(keys)), key=keys.__getitem__)
        return images[best], best

    def random_layout(self):
        while True:
            sites = self.rng.permutation(self.n)
            cfg = sites * N_CFG_PER_SITE + self.rng.integers(0, N_CFG_PER_SITE, size=self.n)
            if not self.is_invalid(cfg):
                return cfg

    def _apply(self, cfg, action):
        """New cfg vector after an action (no validity check, no canonical mapping)."""
        kind, m, x = self.actions[action]
        new = cfg.copy()
        if kind == 'shape':
            new[m] = cfg[m] - cfg[m] % N_SHAPES + x
        elif kind == 'rot':
            site, ref_i, shape_i = self.decode(cfg[m])
            new[m] = site * N_CFG_PER_SITE + ((ref_i + x) % 4) * N_SHAPES + shape_i
        else:   # swap sites, each workstation keeps its orientation and shape
            sm, sx = cfg[m] // N_CFG_PER_SITE, cfg[x] // N_CFG_PER_SITE
            new[m] = sx * N_CFG_PER_SITE + cfg[m] % N_CFG_PER_SITE
            new[x] = sm * N_CFG_PER_SITE + cfg[x] % N_CFG_PER_SITE
        return new

    def next_cfgs(self, cfg):
        """(n_actions, N) array: the layout produced by each action (vectorised `_apply`)."""
        new = np.repeat(cfg[None, :], len(self.actions), axis=0)
        current = cfg[self._act_m]
        i = self._shape_idx
        new[i, self._act_m[i]] = current[i] - current[i] % N_SHAPES + self._act_x[i]
        i = self._rot_idx
        site, rest = np.divmod(current[i], N_CFG_PER_SITE)
        ref_i, shape_i = np.divmod(rest, N_SHAPES)
        new[i, self._act_m[i]] = site * N_CFG_PER_SITE + ((ref_i + self._act_x[i]) % 4) * N_SHAPES + shape_i
        i = self._swap_idx
        m, x = self._act_m[i], self._act_x[i]
        cm, cx = cfg[m], cfg[x]
        new[i, m] = (cx // N_CFG_PER_SITE) * N_CFG_PER_SITE + cm % N_CFG_PER_SITE
        new[i, x] = (cm // N_CFG_PER_SITE) * N_CFG_PER_SITE + cx % N_CFG_PER_SITE
        return new

    def loaded_cost_gain(self):
        """Loaded-cost reduction of every action, relative to C_ref (positive = improvement)."""
        nxt = self.next_cfgs(self.cfg)
        p, d = self.p_pt_np[nxt], self.d_pt_np[nxt]
        loaded = (self.dist[p[:, :, None], d[:, None, :]] * self.demand[None]).sum(axis=(1, 2))
        now = float((self.demand * self._loaded_dist).sum())
        return (now - loaded) / self.ref_cost

    def action_mask(self):
        """Boolean vector of allowed actions (all True when overlaps are allowed)."""
        if self.allow_overlap:
            return np.ones(self.n_actions, dtype=bool)
        p_pt, d_pt = self.p_pt, self.d_pt
        cfg = self.cfg.tolist()
        occ = [0] * self.n_points
        for c in cfg:
            occ[p_pt[c]] += 1
            occ[d_pt[c]] += 1
        mask = np.ones(self.n_actions, dtype=bool)
        for a, (kind, m, x) in enumerate(self.actions):
            c = cfg[m]
            if kind == 'shape':
                changed = ((c, c - c % N_SHAPES + x),)
            elif kind == 'rot':
                site, rest = divmod(c, N_CFG_PER_SITE)
                ref_i, shape_i = divmod(rest, N_SHAPES)
                changed = ((c, site * N_CFG_PER_SITE + ((ref_i + x) % 4) * N_SHAPES + shape_i),)
            else:
                c2 = cfg[x]
                changed = ((c, (c2 // N_CFG_PER_SITE) * N_CFG_PER_SITE + c % N_CFG_PER_SITE),
                           (c2, (c // N_CFG_PER_SITE) * N_CFG_PER_SITE + c2 % N_CFG_PER_SITE))
            for old, _ in changed:
                occ[p_pt[old]] -= 1
                occ[d_pt[old]] -= 1
            new_pts = []
            for _, new in changed:
                new_pts.append(p_pt[new])
                new_pts.append(d_pt[new])
            if any(occ[q] for q in new_pts) or len(set(new_pts)) != len(new_pts):
                mask[a] = False
            for old, _ in changed:
                occ[p_pt[old]] += 1
                occ[d_pt[old]] += 1
        return mask

    # -------------------------------------------------------------------- cost
    def set_demand(self, demand, demand_id=None):
        """demand_id (hashable) lets the empty-travel LP results be cached across episodes."""
        self.demand = np.asarray(demand, dtype=float)
        assert self.demand.shape == (self.n, self.n)
        self.demand_id = demand_id
        peak = self.demand.max()
        self.demand_norm = self.demand / peak if peak > 0 else self.demand
        row, col = self.demand.sum(1), self.demand.sum(0)
        self._b = np.flatnonzero(row > 0)      # pickups that need an empty vehicle
        self._w = np.flatnonzero(col > 0)      # deliveries that release an empty vehicle
        nb, nw = len(self._b), len(self._w)
        if nb and nw:
            a_eq = np.zeros((nw + nb, nw * nb))
            for i in range(nw):
                a_eq[i, i * nb:(i + 1) * nb] = 1
            for j in range(nb):
                a_eq[nw + j, j::nb] = 1
            self._a_eq = a_eq
            self._b_eq = np.concatenate((col[self._w], row[self._b]))
            self._supply = col[self._w].tolist()
            self._need = row[self._b].tolist()
        self._lp_cache = self._lp_caches.setdefault(demand_id, {}) if demand_id is not None else {}

    def _empty_cost(self, p, d):
        if not len(self._b) or not len(self._w):
            return 0.0
        dw, pb = d[self._w], p[self._b]
        key = dw.tobytes() + pb.tobytes()
        value = self._lp_cache.get(key)
        if value is None:
            c = self.dist[dw][:, pb]
            if c.size <= 400:       # small instance: dedicated exact solver, much faster
                value = transport_cost(c.tolist(), self._supply, self._need)
            else:
                res = linprog(c.ravel(), A_eq=self._a_eq, b_eq=self._b_eq, bounds=(0, None), method='highs')
                if res.status != 0:
                    raise RuntimeError(f'empty-travel LP failed: {res.message}')
                value = float(res.fun)
            if len(self._lp_cache) < 2_000_000:
                self._lp_cache[key] = value
        return value

    def cost_of(self, cfg):
        """Total transportation cost of a layout for the current demand."""
        p, d = self.p_pt_np[cfg], self.d_pt_np[cfg]
        loaded_dist = self.dist[p][:, d]                 # [i, j] = distance P_i -> D_j
        return float((self.demand * loaded_dist).sum()) + self._empty_cost(p, d), loaded_dist

    # --------------------------------------------------------------- gym-like API
    def reset(self, demand=None, demand_id=None, layout_cfg=None):
        """
        demand:     force a demand matrix (default: drawn from the list / the sampler).
        layout_cfg: force the starting layout (e.g. the layout of the previous period for a
                    dynamic FLP); default: the initial layout, or a random one with probability
                    `random_start`.
        """
        if demand is None:
            if self.demand_sampler is not None and self.rng.random() < self.sampler_probability:
                demand, demand_id = self.demand_sampler(self.rng), None
            else:
                demand_id = int(self.rng.integers(len(self.demands)))
                demand = self.demands[demand_id]
        self.set_demand(demand, demand_id)

        ref = self._ref_costs.get(demand_id) if demand_id is not None else None
        if ref is None:
            ref = max(self.cost_of(self.default_cfg)[0], 1e-9)
            if demand_id is not None:
                self._ref_costs[demand_id] = ref
        self.ref_cost = ref          # cost of the default initial layout: the reward scale

        if layout_cfg is not None:
            cfg = np.array(layout_cfg, dtype=np.int64)
        elif self.random_start > 0 and self.rng.random() < self.random_start:
            cfg = self.random_layout()
        else:
            cfg = self.default_cfg.copy()
        self.last_symmetry = 0
        if self.canonical:
            cfg, self.last_symmetry = self.canonicalize(cfg)
        self.cfg = cfg
        self.t = 0
        self.cost, self._loaded_dist = self.cost_of(cfg)
        self.start_cost = self.best_cost = self.cost
        self.best_cfg = cfg.copy()
        return self._obs()

    def step(self, action):
        """
        Rewards (C = cost, C_ref = cost of the default initial layout for the current demand):
          'improvement': (C_before - C_after) / C_ref. Summed over an episode it is exactly the
                         fraction of the reference cost removed, whatever the demand scale; a
                         temporary degradation costs what it degrades, no more.
          'best':        (best_before - best_after) / C_ref: only new best layouts are paid, so
                         passing through worse layouts is free.
          'legacy':      reward of the paper: scaled cost if the cost did not increase, else -1.
        """
        new = self._apply(self.cfg, action)
        self.t += 1
        done = self.t >= self.episode_length
        info = {'invalid': False}
        if self.is_invalid(new):
            info['invalid'] = True
            reward = -1.0 if self.reward_type == 'legacy' else -self.invalid_penalty
        else:
            self.last_symmetry = 0
            if self.canonical:
                new, self.last_symmetry = self.canonicalize(new)
            previous_cost, previous_best = self.cost, self.best_cost
            self.cfg = new
            self.cost, self._loaded_dist = self.cost_of(new)
            if self.cost < self.best_cost - 1e-9:
                self.best_cost, self.best_cfg = self.cost, new.copy()
            if self.reward_type == 'improvement':
                reward = (previous_cost - self.cost) / self.ref_cost
            elif self.reward_type == 'best':
                reward = (previous_best - self.best_cost) / self.ref_cost
            else:
                lo, hi = self.legacy_bounds or (0.0, self.ref_cost)
                if self.cost <= previous_cost:
                    reward = min(max((hi - self.cost) / (hi - lo), 0.0), 1.0)
                else:
                    reward = -1.0
        info.update(cost=self.cost, best_cost=self.best_cost, ref_cost=self.ref_cost,
                    start_cost=self.start_cost, symmetry=self.last_symmetry)
        return self._obs(), float(reward), done, info

    def _obs(self):
        """
        [P/D coordinates (4N) | distance P_i->D_j (N^2) | demand (N^2) | cost share of each flow
        (N^2) | current cost, best cost (both / C_ref), elapsed time fraction |
        with lookahead: scaled loaded-cost gain of each action (n_actions values, always last)]
        """
        flow_cost = self.demand * self._loaded_dist / self.ref_cost
        gain = (LOOKAHEAD_SCALE * self.loaded_cost_gain(),) if self.lookahead else ()
        return np.concatenate((
            self.xy_norm[self.cfg].ravel(),
            (self._loaded_dist / self.dist_scale).ravel(),
            self.demand_norm.ravel(),
            flow_cost.ravel(),
            (self.cost / self.ref_cost, self.best_cost / self.ref_cost, self.t / self.episode_length),
        ) + gain).astype(np.float32)

    # ------------------------------------------------------------------ helpers
    def pd_positions(self, cfg=None):
        cfg = self.cfg if cfg is None else cfg
        out = {}
        for m, c in enumerate(cfg):
            out[f'P{m + 1}'] = tuple(self.xy[c, :2])
            out[f'D{m + 1}'] = tuple(self.xy[c, 2:])
        return out

    def describe(self, cfg=None):
        cfg = self.cfg if cfg is None else cfg
        lines = []
        for m, c in enumerate(cfg):
            site, ref, shape = self.cfg_triple[c]
            x = self.xy[c]
            lines.append(f'  W{m + 1}: site {site + 1}, {shape:<13} P=({x[0]:g}, {x[1]:g}) D=({x[2]:g}, {x[3]:g})')
        return '\n'.join(lines)
