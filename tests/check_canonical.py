"""
Deep check of the canonical-layout mapping.

  1. N=4, exhaustive: all 24 x 20^4 = 3,840,000 layouts.
       - every layout has exactly 8 distinct symmetric images;
       - all 8 images get the same canonical layout, and it is one of the 8 images;
       - number of canonical layouts = number of layouts / 8.
  2. N=4, exhaustive reachability: breadth-first search with the real actions, with and without
     the canonical mapping: every valid (canonical) layout can be reached from the initial layout.
  3. Cost invariance on the 8 images for the 5 demands of the paper (N=4) and random demands (N=9).
  4. N=9, 200,000 random layouts: same structural checks as 1.
  5. How often the canonical mapping re-frames the layout during random episodes.

Run from anywhere:  python tests/check_canonical.py
"""
import glob
import os
import sys
from itertools import permutations, product

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
from flp_env import FLPEnv, N_CFG_PER_SITE, N_SHAPES, random_demand


def encode(env, cfgs):
    """One integer per layout (base = number of configurations), order = canonical order."""
    base = env.n * N_CFG_PER_SITE
    key = np.zeros(len(cfgs), dtype=np.int64)
    for m in range(env.n):
        key = key * base + env.rank[cfgs[:, m]]
    return key


def canonical_batch(env, cfgs):
    """Vectorised canonical mapping: returns (canonical cfgs, keys of the 8 images)."""
    images = env.transform[:, cfgs]                          # (8, M, n)
    keys = np.stack([encode(env, images[g]) for g in range(images.shape[0])])   # (8, M)
    best = keys.argmin(axis=0)
    return images[best, np.arange(len(cfgs))], keys


def overlap_batch(env, cfgs):
    pts = np.sort(np.concatenate((env.p_pt_np[cfgs], env.d_pt_np[cfgs]), axis=1), axis=1)
    return (pts[:, 1:] == pts[:, :-1]).any(axis=1)


def apply_batch(env, cfgs, action):
    kind, m, x = env.actions[action]
    new = cfgs.copy()
    if kind == 'shape':
        new[:, m] = cfgs[:, m] - cfgs[:, m] % N_SHAPES + x
    elif kind == 'rot':
        site, rest = np.divmod(cfgs[:, m], N_CFG_PER_SITE)
        ref_i, shape_i = np.divmod(rest, N_SHAPES)
        new[:, m] = site * N_CFG_PER_SITE + ((ref_i + x) % 4) * N_SHAPES + shape_i
    else:
        sm, sx = cfgs[:, m] // N_CFG_PER_SITE, cfgs[:, x] // N_CFG_PER_SITE
        new[:, m] = sx * N_CFG_PER_SITE + cfgs[:, m] % N_CFG_PER_SITE
        new[:, x] = sm * N_CFG_PER_SITE + cfgs[:, x] % N_CFG_PER_SITE
    return new


def structure_checks(env, cfgs, label):
    canon, keys = canonical_batch(env, cfgs)
    sorted_keys = np.sort(keys, axis=0)
    assert (sorted_keys[1:] != sorted_keys[:-1]).all(), 'a layout has fewer than 8 distinct images'
    canon_key = encode(env, canon)
    assert (canon_key == keys.min(axis=0)).all()
    for g in range(keys.shape[0]):                              # every image -> same canonical layout
        image_canon, _ = canonical_batch(env, env.transform[g][cfgs])
        assert (encode(env, image_canon) == canon_key).all()
    again, _ = canonical_batch(env, canon)                      # idempotent
    assert (again == canon).all()
    assert (overlap_batch(env, cfgs) == overlap_batch(env, canon)).all()   # validity is preserved
    print(f'  {label}: {len(cfgs):,} layouts, 8 distinct images each, one canonical layout per orbit, '
          f'idempotent, overlap status preserved')
    return canon_key


def exhaustive_n4(env):
    per_site = np.array(list(product(range(N_CFG_PER_SITE), repeat=4)), dtype=np.int64)   # 160,000
    blocks = []
    for perm in permutations(range(4)):
        blocks.append(per_site + np.array(perm, dtype=np.int64) * N_CFG_PER_SITE)
    cfgs = np.concatenate(blocks)
    canon_key = structure_checks(env, cfgs, 'N=4 exhaustive')
    n_classes = len(np.unique(canon_key))
    assert n_classes * 8 == len(cfgs), (n_classes, len(cfgs))
    valid = ~overlap_batch(env, cfgs)
    n_valid_classes = len(np.unique(canon_key[valid]))
    assert n_valid_classes * 8 == valid.sum()
    print(f'  all layouts: {len(cfgs):,} -> {n_classes:,} canonical (exactly 8x fewer)')
    print(f'  valid (no overlapping P/D): {valid.sum():,} -> {n_valid_classes:,} canonical (exactly 8x fewer)')
    return int(valid.sum()), n_valid_classes


def reachability(env, canonical, allow_overlap):
    """BFS over layouts with the environment's actions; returns (#reached, depth of farthest)."""
    base = env.n * N_CFG_PER_SITE
    start = env.default_cfg[None, :].copy()
    if canonical:
        start, _ = canonical_batch(env, start)
    visited = np.zeros(base ** env.n, dtype=bool)
    visited[encode(env, start)] = True
    frontier, depth, total = start, 0, 1
    while len(frontier):
        nxt = []
        for a in range(env.n_actions):
            new = apply_batch(env, frontier, a)
            if not allow_overlap:
                new = new[~overlap_batch(env, new)]
            if canonical and len(new):
                new, _ = canonical_batch(env, new)
            if len(new):
                key = encode(env, new)
                key, first = np.unique(key, return_index=True)
                fresh = ~visited[key]
                visited[key[fresh]] = True
                nxt.append(new[first[fresh]])
        frontier = np.concatenate(nxt) if nxt else np.zeros((0, env.n), dtype=np.int64)
        if len(frontier):
            depth += 1
            total += len(frontier)
    return total, depth


def cost_invariance(env, demands, samples, label):
    worst = 0.0
    for d_i, demand in enumerate(demands):
        env.set_demand(demand, ('check', label, d_i))
        for _ in range(samples):
            cfg = env.random_layout()
            c0 = env.cost_of(cfg)[0]
            for g in range(env.transform.shape[0]):
                worst = max(worst, abs(env.cost_of(env.transform[g, cfg])[0] - c0))
    assert worst < 1e-6, worst
    print(f'  {label}: {len(demands)} demands x {samples} layouts x 8 images, max cost difference = {worst:.1e}')


def reframing_stats(n, steps=20000):
    env = FLPEnv(n, demand_sampler=lambda rng: random_demand(n, rng), episode_length=50, seed=0)
    rng = np.random.default_rng(0)
    counts = {True: [0, 0], False: [0, 0]}      # involves workstation 1 -> [steps, re-framed]
    for _ in range(steps):
        a = int(rng.choice(np.flatnonzero(env.action_mask())))
        kind, m, x = env.actions[a]
        involves_first = m == 0          # workstation 1 is machine index 0 (for swaps m < x)
        _, _, done, info = env.step(a)
        counts[involves_first][0] += 1
        counts[involves_first][1] += info['symmetry'] != 0
        if done:
            env.reset()
    a, b = counts[True], counts[False]
    print(f'  N={n}: layout re-framed on {a[1]}/{a[0]} steps that act on workstation 1 '
          f'({100 * a[1] / max(a[0], 1):.0f}%), and on {b[1]}/{b[0]} of the other steps '
          f'({100 * b[1] / max(b[0], 1):.1f}%)')


if __name__ == '__main__':
    paper = sorted(glob.glob('transportation_demand/paper_M4_train/*.csv'))
    env4 = FLPEnv(4, demands=paper, seed=0)

    print('1. Structure, N=4 (exhaustive)')
    n_valid, n_valid_canon = exhaustive_n4(env4)

    print('2. Reachability with the real actions, N=4 (exhaustive breadth-first search)')
    for allow, n_all, n_canon, label in ((True, 3_840_000, 480_000, 'shared P/D spots allowed (paper)'),
                                         (False, n_valid, n_valid_canon, 'shared P/D spots forbidden')):
        reached, depth = reachability(env4, canonical=False, allow_overlap=allow)
        assert reached == n_all, (reached, n_all)
        reached_c, depth_c = reachability(env4, canonical=True, allow_overlap=allow)
        assert reached_c == n_canon, (reached_c, n_canon)
        print(f'  {label}:')
        print(f'    without canonical mapping: all {reached:,} layouts reachable; farthest needs {depth} actions')
        print(f'    with canonical mapping:    all {reached_c:,} canonical layouts reachable; farthest needs {depth_c} actions')

    print('3. Cost invariance on the 8 images')
    cost_invariance(env4, env4.demands, 400, 'N=4, paper demands T1..T5')
    env9 = FLPEnv(9, demand_sampler=lambda rng: random_demand(9, rng), seed=1)
    rng = np.random.default_rng(5)
    cost_invariance(env9, [random_demand(9, rng) for _ in range(20)], 25, 'N=9, random demands')

    print('4. Structure, N=9 (random sample)')
    sample = np.stack([env9.random_layout() for _ in range(200_000)])
    structure_checks(env9, sample, 'N=9 sample')

    print('5. Re-framing frequency during random episodes')
    reframing_stats(4)
    reframing_stats(9)
    print('CANONICAL MAPPING: ALL CHECKS PASSED')
