"""
Checks that the fast environment (flp_env.FLPEnv) is the same problem as the original one
(Environment.Env): same cost, same transitions, same canonical layouts, correct action mask.
Run from anywhere:  python tests/test_flp_env.py
"""
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import pandas as pd
from Environment import Env
from Environment_FLP.Configuration_cost import cost as legacy_cost
from flp_env import FLPEnv, load_demand, random_demand


def demand_path(n):
    return f'transportation_demand/transportation_demand_M{n}_1.csv'


def test_cost_matches_original():
    """Table-based distances + LP == Dijkstra on the rebuilt graph + PuLP of the original code."""
    for n, samples in ((4, 150), (9, 60), (16, 15)):
        legacy = Env(n, 10 ** 9, 10 ** 9, 'scaled_cost', [1], False, canonical=False)
        legacy.reset()
        # random layouts WITH overlapping P/D points included (allow_overlap=True)
        env = FLPEnv(n, demands=[demand_path(n)], canonical=False, seed=n)
        worst = 0.0
        for _ in range(samples):
            cfg = env.random_layout()
            fast, _ = env.cost_of(cfg)
            ref = legacy_cost(legacy.build_graph(env.layout(cfg)), legacy.df).Total_cost()
            worst = max(worst, abs(fast - ref))
            assert abs(fast - ref) < 1e-6, (n, fast, ref)
        print(f'  N={n}: {samples} random layouts, max |fast - original| = {worst:.2e}')
    # random demands too (the original cost needs a DataFrame)
    rng = np.random.default_rng(0)
    legacy = Env(9, 10 ** 9, 10 ** 9, 'scaled_cost', [1], False, canonical=False)
    legacy.reset()
    env = FLPEnv(9, demands=[demand_path(9)], canonical=False, seed=1)
    for _ in range(25):
        d = random_demand(9, rng)
        env.set_demand(d)
        cfg = env.random_layout()
        df = pd.DataFrame(d, index=legacy.df.index, columns=legacy.df.columns)
        ref = legacy_cost(legacy.build_graph(env.layout(cfg)), df).Total_cost()
        assert abs(env.cost_of(cfg)[0] - ref) < 1e-6
    print('cost identical to the original implementation: OK')


def test_transitions_match_original():
    """Same random action sequence in both environments -> same layouts and costs at each step."""
    for n, steps in ((4, 300), (9, 200)):
        for canonical, overlap in ((True, True), (False, True), (True, False), (False, False)):
            legacy = Env(n, 10 ** 9, 10 ** 9, 'scaled_cost', [1], False, canonical=canonical,
                         allow_overlap=overlap)
            state, c, _, _ = legacy.reset()
            env = FLPEnv(n, demands=[demand_path(n)], canonical=canonical, episode_length=10 ** 9, seed=0,
                         allow_overlap=overlap)
            env.reset()
            assert env.action_names == list(legacy.action_name_dict.values())
            assert abs(env.cost - c) < 1e-6
            rng = np.random.default_rng(n + canonical)
            invalid = 0
            for _ in range(steps):
                a = int(rng.integers(env.n_actions))
                mask = env.action_mask()
                _, state, _, _, c, _, _ = legacy.step(state, a, c)
                _, _, _, info = env.step(a)
                invalid += info['invalid']
                assert mask[a] == (not info['invalid'])
                assert legacy.state_to_layout(state) == env.layout(), (n, canonical)
                assert abs(env.cost - c) < 1e-6, (n, canonical, env.cost, c)
            print(f'  N={n} canonical={canonical} allow_overlap={overlap}: {steps} steps identical '
                  f'({invalid} cancelled)')
    print('transitions identical to the original implementation: OK')


def test_canonical_tables():
    for n in (4, 9, 16):
        env = FLPEnv(n, demands=[demand_path(n)], seed=3)      # overlapping layouts included
        assert len(env.geometry.transforms) == 8
        for _ in range(300):
            cfg = env.random_layout()
            canon, _ = env.canonicalize(cfg)
            assert env.layout(canon) == env.geometry.canonical_layout(env.layout(cfg))[0]
            assert env.canonicalize(canon)[1] == 0                      # idempotent
            c0 = env.cost_of(cfg)[0]
            for g in range(8):                                          # all images: same class, same cost
                image = env.transform[g, cfg]
                assert np.array_equal(env.canonicalize(image)[0], canon)
                assert abs(env.cost_of(image)[0] - c0) < 1e-6
    print('canonical tables == canonical.py, cost invariant on the 8 images: OK')


def test_shared_spot_is_zero_distance():
    """
    Two P/D points with the same coordinates are at distance 0 (and only those), in the fast
    environment and in the graph of the original environment.
    """
    import networkx as nx
    for n in (4, 9):
        env = FLPEnv(n, demands=[demand_path(n)], canonical=False, seed=7)      # allow_overlap=True
        assert env.allow_overlap and env.action_mask().all()
        assert np.all(np.diag(env.dist) == 0)                                   # same spot -> 0
        assert np.all(env.dist[~np.eye(env.n_points, dtype=bool)] > 0)          # different spots -> > 0
        assert np.allclose(env.dist, env.dist.T)
        legacy = Env(n, 10 ** 9, 10 ** 9, 'scaled_cost', [1], False, canonical=False)
        legacy.reset()
        checked = 0
        while checked < 40:
            cfg = env.random_layout()
            p, d = env.p_pt_np[cfg], env.d_pt_np[cfg]
            pairs = [(i, j) for i in range(n) for j in range(n) if i != j and p[i] == d[j]]
            if not pairs:
                continue
            graph = legacy.build_graph(env.layout(cfg))
            fast = env.dist[p][:, d]
            for i, j in pairs:                       # P_i and D_j share a spot
                assert tuple(env.xy[cfg[i], :2]) == tuple(env.xy[cfg[j], 2:])
                assert fast[i, j] == 0
                assert nx.dijkstra_path_length(graph, f'P{i + 1}', f'D{j + 1}', weight='weight') == 0
            checked += 1
        print(f'  N={n}: same coordinates <=> distance 0, in both environments ({checked} layouts with shared spots)')
    print('shared P/D spots have zero distance: OK')


def test_speed():
    for n in (4, 9, 16):
        env = FLPEnv(n, demand_sampler=lambda rng, n=n: random_demand(n, rng), episode_length=50, seed=0)
        rng = np.random.default_rng(0)
        t0, steps = time.time(), 3000
        for _ in range(steps):
            mask = env.action_mask()
            _, _, done, _ = env.step(int(rng.choice(np.flatnonzero(mask))))
            if done:
                env.reset()
        print(f'  N={n}: {(time.time() - t0) / steps * 1000:.2f} ms per step (mask + step, random demands)')


if __name__ == '__main__':
    test_cost_matches_original()
    test_transitions_match_original()
    test_canonical_tables()
    test_shared_spot_is_zero_distance()
    test_speed()
    print('ALL FAST-ENV TESTS PASSED')
