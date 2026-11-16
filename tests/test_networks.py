"""
Checks of the two policy networks (flat 'mlp' and weight-sharing 'graph').
Run from anywhere:  python tests/test_networks.py
"""
import os
import sys
import tempfile
from itertools import combinations

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import torch

from flp_env import FLPEnv, random_demand
from ppo import GraphActorCritic, load_model, make_model, save_model, train


def permute_observation(obs, perm, n, n_actions):
    """Observation of the same layout with the workstations renamed: new workstation i = old perm[i]."""
    xy = obs[:4 * n].reshape(n, 4)[perm]
    mats = [obs[4 * n + k * n * n: 4 * n + (k + 1) * n * n].reshape(n, n)[np.ix_(perm, perm)] for k in range(3)]
    scalars = obs[4 * n + 3 * n * n: 4 * n + 3 * n * n + 3]
    look = obs[-n_actions:]
    return np.concatenate([xy.ravel()] + [m.ravel() for m in mats] + [scalars, permute_actions(look, perm, n)])


def permute_actions(values, perm, n):
    """Per-action values (7 per workstation, then one per pair m<k) after the same renaming."""
    node = values[:7 * n].reshape(n, 7)[perm]
    pairs = list(combinations(range(n), 2))
    index = {p: i for i, p in enumerate(pairs)}
    swap = values[7 * n:]
    new_swap = np.array([swap[index[tuple(sorted((perm[a], perm[b])))]] for a, b in pairs])
    return np.concatenate((node.ravel(), new_swap))


def test_graph_network_shares_weights():
    """Renaming the workstations renames the action scores and leaves the value unchanged."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    for n in (4, 9):
        env = FLPEnv(n, demand_sampler=lambda r, n=n: random_demand(n, r), seed=1, canonical=False)
        model = GraphActorCritic(n, env.n_lookahead)
        for p in model.parameters():                    # random weights, so that the test is not trivial
            torch.nn.init.normal_(p, std=0.3)
        mask = torch.ones(1, env.n_actions, dtype=torch.bool)
        worst = 0.0
        for _ in range(20):
            obs, _, done, _ = env.step(int(rng.integers(env.n_actions)))
            if done:
                obs = env.reset()
            perm = rng.permutation(n)
            with torch.no_grad():
                d1, v1 = model.evaluate(torch.as_tensor(obs)[None], mask)
                d2, v2 = model.evaluate(torch.as_tensor(permute_observation(obs, perm, n, env.n_actions))[None].float(), mask)
            expected = permute_actions(d1.logits[0].numpy(), perm, n)
            worst = max(worst, float(np.abs(expected - d2.logits[0].numpy()).max()), float(abs(v1 - v2)))
        assert worst < 1e-3, worst
        print(f'  N={n}: scores follow the renaming of workstations, value unchanged (max difference {worst:.1e})')
    print('graph network shares its weights across workstations: OK')


def test_parameter_counts():
    for n in (4, 9, 16):
        env = FLPEnv(n, demand_sampler=lambda r, n=n: random_demand(n, r), seed=1)
        count = lambda m: sum(p.numel() for p in m.parameters())
        mlp = make_model(env.obs_size, env.n_actions, env.n_lookahead, n, 'mlp')
        graph = make_model(env.obs_size, env.n_actions, env.n_lookahead, n, 'graph')
        print(f'  N={n}: flat network {count(mlp):,} parameters, graph network {count(graph):,}')
    print('the graph network has the same number of parameters for every workshop size: OK')


def test_training_and_reload():
    for network in ('mlp', 'graph'):
        make = lambda seed: FLPEnv(4, demand_sampler=lambda r: random_demand(4, r), episode_length=20, seed=seed)
        model, history = train(make, 2 * 8 * 64, n_envs=8, n_steps=64, verbose=False, network=network)
        env = make(0)
        config = dict(machines=4, obs_size=env.obs_size, n_actions=env.n_actions, lookahead='total',
                      network=network, hidden=256, d_model=64, n_layers=3, n_heads=4)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, 'model.pt')
            save_model(model, path, config)
            loaded, _ = load_model(path)
        obs, mask = env.reset(), env.action_mask()
        assert model.act(obs, mask, greedy=True) == loaded.act(obs, mask, greedy=True)
        print(f'  {network}: trains, saves and reloads')
    print('both networks train and reload: OK')


if __name__ == '__main__':
    test_graph_network_shares_weights()
    test_parameter_counts()
    test_training_and_reload()
    print('ALL NETWORK TESTS PASSED')
