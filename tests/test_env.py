"""
Checks for the generalized swap actions and the canonical layout mapping.
Run from the repository root:  python tests/test_env.py
"""
import os
import random
import sys

import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from Environment import Env
from Environment_FLP.Actions import action
from Environment_FLP.canonical import SHAPES, _key
from Environment_FLP.Configuration_cost import cost

SIZES = [4, 9, 16]


def make_env(n, canonical=True):
    return Env(n, 40, 40, 'scaled_cost', [1], render=False, canonical=canonical)


def random_layout(env, rng):
    """Random valid layout: random permutation of sites, random reference and shape."""
    n = env.number_of_machines
    while True:
        sites = rng.permutation(n)
        layout = []
        for m in range(n):
            s = int(sites[m])
            ref = rng.choice(env.geometry.site_corners[s])
            layout.append((s, str(ref), str(rng.choice(SHAPES))))
        if not env.geometry.has_overlap(env.geometry.pd_positions(layout)):
            return layout


def test_geometry_matches_actions():
    for n in SIZES:
        env = make_env(n)
        assert len(env.geometry.transforms) == 8, (n, len(env.geometry.transforms))
        for site_index, corners in enumerate(env.geometry.site_corners):
            for ref in corners:
                for shape in SHAPES:
                    G = env.load_graph().copy()
                    a = action(G, env.sites)
                    G, *_ = getattr(a, shape)(int(ref), 1, site_index)
                    p, d = env.geometry.pd_coords(site_index, ref, shape)
                    assert _key(G.nodes['P1']['pos']) == _key(p), (n, site_index, ref, shape)
                    assert _key(G.nodes['D1']['pos']) == _key(d), (n, site_index, ref, shape)
        # 20 distinct P/D configurations per site
        assert len(env.geometry.config_lookup) == 20 * n
    print('geometry matches Actions.py: OK')


def test_action_space():
    for n in SIZES:
        env = make_env(n)
        names = list(env.action_name_dict.values())
        swaps = [a for a in names if 'reimplant' in a]
        assert len(swaps) == n * (n - 1) // 2
        pairs = {tuple(int(x) for x in a.split('_') if x.isdigit())[-2:] for a in swaps}
        assert all(m < k for m, k in pairs) and len(pairs) == len(swaps)
        assert len(names) == 7 * n + n * (n - 1) // 2
    print('action space (7N + N(N-1)/2, all pairs m<n): OK')


def test_swap_any_pair():
    env = make_env(9, canonical=False)
    for name_index, name in env.action_name_dict.items():
        if 'reimplant' not in name:
            continue
        m, k = [int(x) for x in name.split('_') if x.isdigit()][-2:]
        state, c, _, _ = env.reset()
        before = env.state_to_layout(state)
        _, new_state, _, _, _, _, _ = env.step(state, name_index, c)
        after = env.state_to_layout(new_state)
        assert after[m - 1][0] == before[k - 1][0] and after[k - 1][0] == before[m - 1][0], name
        for j in range(9):
            if j not in (m - 1, k - 1):
                assert after[j] == before[j], name
    print('swap of any pair m != n: OK')


def test_cost_invariance_and_canonical():
    rng = np.random.default_rng(0)
    for n in [4, 9]:
        env = make_env(n)
        env.reset()
        for _ in range(15 if n == 9 else 30):
            layout = random_layout(env, rng)
            base = cost(env.build_graph(layout), env.df).Total_cost()
            canon, _ = env.geometry.canonical_layout(layout)
            assert env.geometry.is_canonical(canon)
            for g in env.geometry.transforms:
                img, _ = env.geometry.transform_layout(g, layout)
                c = cost(env.build_graph(img), env.df).Total_cost()
                assert abs(c - base) < 1e-6, (n, c, base)
                assert env.geometry.canonical_layout(img)[0] == canon
    print('cost invariant under the 8 grid symmetries, canonical form unique: OK')


def test_initial_cost_unchanged():
    for n in SIZES:
        env = make_env(n, canonical=False)
        state, c, _, _ = env.reset()
        original = cost(env.load_graph(), env.df).Total_cost()
        env_c = make_env(n, canonical=True)
        _, c_canon, _, _ = env_c.reset()
        assert abs(c - original) < 1e-6 and abs(c_canon - original) < 1e-6, (n, c, c_canon, original)
        print(f'  N={n}: initial cost {original}')
    print('initial cost identical with/without canonical mapping: OK')


def test_random_rollouts():
    random.seed(1)
    env = make_env(9)
    state, c, _, _ = env.reset()
    shapes_seen = set()
    for step in range(120):
        a = random.randrange(len(env.action_name_dict))
        name = env.action_name_dict[a]
        shape_before = list(env.machine_shape)
        _, state, r, done, c, _, _ = env.step(state, a, c)
        layout = env.state_to_layout(state)
        assert env.geometry.is_canonical(layout)
        assert not env.geometry.has_overlap(env.geometry.pd_positions(layout))
        assert abs(cost(env.build_graph(layout), env.df).Total_cost() - c) < 1e-6
        assert -1 <= r <= 1
        shapes_seen.update(env.machine_shape)
        if done:
            state, c, _, _ = env.reset()
    print(f'random rollouts (N=9): OK, invalid (overlap) actions cancelled: {env.invalid_action_count}')


def test_rotation_keeps_shape():
    env = make_env(9, canonical=False)
    state, c, _, _ = env.reset()
    idx = {v: k for k, v in env.action_name_dict.items()}
    _, state, _, _, c, _, _ = env.step(state, idx['5_L_shape'], c)
    _, state, _, _, c, _, _ = env.step(state, idx['5_rotation_clockwise'], c)
    assert env.machine_shape[4] == 'L_shape'
    p, d = env.geometry.pd_coords(int(state[9 + 4]), str(int(state[4])), 'L_shape')
    assert _key(env.pos['P5']) == _key(p) and _key(env.pos['D5']) == _key(d)
    print('rotation keeps the current shape: OK')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    test_geometry_matches_actions()
    test_action_space()
    test_swap_any_pair()
    test_rotation_keeps_shape()
    test_initial_cost_unchanged()
    test_cost_invariance_and_canonical()
    test_random_rollouts()
    print('ALL TESTS PASSED')
