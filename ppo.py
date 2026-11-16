"""
PPO with invalid-action masking for flp_env.FLPEnv (PyTorch, single file, CPU is enough).

The policy sees the layout AND the transportation demand, so one network is trained on many
demands (a new one at every episode) and is then used on any demand, including unseen ones.
"""
import copy
import csv
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn


def _layer(n_in, n_out, gain=np.sqrt(2)):
    layer = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_size, n_actions, hidden=256, n_lookahead=0):
        """
        n_lookahead: number of per-action lookahead values at the end of the observation
        (0 or n_actions). They are added to the action scores through a learned weight, so the
        untrained policy already prefers actions that reduce the loaded cost.
        """
        super().__init__()
        self.n_lookahead = n_lookahead
        self.lookahead_weight = nn.Parameter(torch.ones(1))
        self.actor = nn.Sequential(_layer(obs_size, hidden), nn.Tanh(), _layer(hidden, hidden), nn.Tanh(),
                                   _layer(hidden, n_actions, gain=0.01))
        self.critic = nn.Sequential(_layer(obs_size, hidden), nn.Tanh(), _layer(hidden, hidden), nn.Tanh(),
                                    _layer(hidden, 1, gain=1.0))

    def distribution(self, obs, mask):
        logits = self.actor(obs)
        if self.n_lookahead:
            logits = logits + self.lookahead_weight * obs[..., -self.n_lookahead:]
        logits = logits.masked_fill(~mask, -1e9)               # masked actions get probability 0
        return torch.distributions.Categorical(logits=logits)

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def evaluate(self, obs, mask):
        """(action distribution, state value)."""
        return self.distribution(obs, mask), self.value(obs)

    @torch.no_grad()
    def act(self, obs, mask, greedy=False):
        obs = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        mask = torch.as_tensor(np.asarray(mask), dtype=torch.bool)
        single = obs.dim() == 1
        if single:
            obs, mask = obs[None], mask[None]
        dist = self.distribution(obs, mask)
        action = dist.probs.argmax(-1) if greedy else dist.sample()
        return int(action[0]) if single else action.numpy()


class _AttentionBlock(nn.Module):
    """Self-attention between workstations; the pair features (demand, distance) bias the attention."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.Tanh(), nn.Linear(2 * d_model, d_model))

    def forward(self, h, bias):
        """h: (batch, n, d_model) tokens; bias: (batch, heads, n, n) attention bias from the pair features."""
        batch, n, d_model = h.shape
        q, k, v = self.qkv(self.norm1(h)).reshape(batch, n, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        attention = (q @ k.transpose(-1, -2)) / self.d_head ** 0.5 + bias
        # softmax written out: several times faster than torch.softmax on these small tensors (CPU)
        attention = (attention - attention.max(-1, keepdim=True).values).exp()
        attention = attention / attention.sum(-1, keepdim=True)
        y = (attention @ v).transpose(1, 2).reshape(batch, n, d_model)
        h = h + self.proj(y)
        return h + self.mlp(self.norm2(h))


class GraphActorCritic(ActorCritic):
    """
    Policy and value network that shares its weights across workstations.

    The flat observation of FLPEnv is read as a graph: one token per workstation (its P/D
    coordinates, the lookahead of its 7 own actions, its total in- and out-flows) and one feature
    vector per ordered pair (distances, demands, flow costs, lookahead of their swap). A few
    attention layers mix the tokens. The scores of the 7 actions of a workstation come from its
    token, the score of a swap from the two tokens and their pair features, always with the same
    weights. "Swap W2 and W5" and "swap W3 and W7" are therefore learned once, and the number of
    parameters does not depend on the number of workstations.
    """
    NODE_FEATURES, EDGE_FEATURES, NODE_ACTIONS = 18, 8, 7

    def __init__(self, n_machines, n_lookahead, d_model=64, n_layers=3, n_heads=4):
        nn.Module.__init__(self)
        n = self.n = n_machines
        self.n_lookahead = n_lookahead
        d_edge = self.EDGE_FEATURES
        upper = torch.triu_indices(n, n, 1)            # pairs m < k, same order as the swap actions
        self.register_buffer('pair_i', upper[0])
        self.register_buffer('pair_j', upper[1])
        self.register_buffer('eye', torch.eye(n))
        self.n_layers, self.n_heads = n_layers, n_heads
        self.lookahead_weight = nn.Parameter(torch.ones(1))
        self.node_in = _layer(self.NODE_FEATURES, d_model)
        # attention bias of every layer and head, straight from the pair features
        self.edge_bias = nn.Linear(self.EDGE_FEATURES, n_layers * n_heads)
        self.blocks = nn.ModuleList([_AttentionBlock(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.node_head = _layer(d_model, self.NODE_ACTIONS, gain=0.01)
        self.swap_head = nn.Sequential(_layer(d_model + d_edge, d_model), nn.Tanh(), _layer(d_model, 1, gain=0.01))
        self.value_head = nn.Sequential(_layer(2 * d_model + 3, d_model), nn.Tanh(), _layer(d_model, 1, gain=1.0))

    def _encode(self, obs):
        batch, n = obs.shape[0], self.n
        xy = obs[:, :4 * n].reshape(batch, n, 4)
        start = 4 * n
        dist = obs[:, start:start + n * n].reshape(batch, n, n)
        demand = obs[:, start + n * n:start + 2 * n * n].reshape(batch, n, n)
        flow = obs[:, start + 2 * n * n:start + 3 * n * n].reshape(batch, n, n)
        start += 3 * n * n
        scalars = obs[:, start:start + 3]
        node_look = obs.new_zeros(batch, n, self.NODE_ACTIONS)
        pair_look = obs.new_zeros(batch, n, n)
        if self.n_lookahead:
            look = (obs[:, start + 3:] / 5.0).clamp(-3.0, 3.0)        # same scale as the other inputs
            node_look = look[:, :self.NODE_ACTIONS * n].reshape(batch, n, self.NODE_ACTIONS)
            pair_look[:, self.pair_i, self.pair_j] = look[:, self.NODE_ACTIONS * n:]
            pair_look = pair_look + pair_look.transpose(1, 2)
        node = torch.cat((xy, node_look,
                          demand.sum(2, keepdim=True), demand.sum(1).unsqueeze(-1),
                          flow.sum(2, keepdim=True), flow.sum(1).unsqueeze(-1),
                          scalars.unsqueeze(1).expand(batch, n, 3)), dim=-1)
        edge = torch.stack((dist, dist.transpose(1, 2), demand, demand.transpose(1, 2),
                            flow, flow.transpose(1, 2), pair_look, self.eye.expand(batch, n, n)), dim=-1)
        h = torch.tanh(self.node_in(node))
        bias = self.edge_bias(edge).reshape(batch, n, n, self.n_layers, self.n_heads).permute(3, 0, 4, 1, 2)
        for layer, block in enumerate(self.blocks):
            h = block(h, bias[layer])
        return self.norm(h), edge, scalars

    def _logits(self, obs, mask, h, e):
        node_logits = self.node_head(h).reshape(obs.shape[0], -1)          # workstation by workstation
        pair = torch.cat((h[:, self.pair_i] + h[:, self.pair_j],
                          e[:, self.pair_i, self.pair_j] + e[:, self.pair_j, self.pair_i]), dim=-1)
        logits = torch.cat((node_logits, self.swap_head(pair).squeeze(-1)), dim=1)
        if self.n_lookahead:
            logits = logits + self.lookahead_weight * obs[..., -self.n_lookahead:]
        return logits.masked_fill(~mask, -1e9)

    def _value(self, h, scalars):
        return self.value_head(torch.cat((h.mean(1), h.max(1).values, scalars), dim=-1)).squeeze(-1)

    def distribution(self, obs, mask):
        h, e, _ = self._encode(obs)
        return torch.distributions.Categorical(logits=self._logits(obs, mask, h, e))

    def value(self, obs):
        h, _, scalars = self._encode(obs)
        return self._value(h, scalars)

    def evaluate(self, obs, mask):
        h, e, scalars = self._encode(obs)          # one pass for both outputs
        return torch.distributions.Categorical(logits=self._logits(obs, mask, h, e)), self._value(h, scalars)


def make_model(obs_size, n_actions, n_lookahead, n_machines, network='mlp', hidden=256,
               d_model=64, n_layers=3, n_heads=4):
    """network: 'mlp' (flat network) or 'graph' (weights shared across workstations)."""
    if network == 'graph':
        return GraphActorCritic(n_machines, n_lookahead, d_model, n_layers, n_heads)
    assert network == 'mlp', network
    return ActorCritic(obs_size, n_actions, hidden, n_lookahead)


def save_model(model, path, config):
    torch.save({'state_dict': model.state_dict(), 'config': config}, path)


def load_model(path):
    data = torch.load(path, map_location='cpu')
    cfg = data['config']
    model = make_model(cfg['obs_size'], cfg['n_actions'], cfg['n_actions'] if cfg.get('lookahead') else 0,
                       cfg['machines'], cfg.get('network', 'mlp'), cfg.get('hidden', 256),
                       cfg.get('d_model', 64), cfg.get('n_layers', 3), cfg.get('n_heads', 4))
    model.load_state_dict(data['state_dict'])
    model.eval()
    return model, cfg


def train(make_env, total_steps, n_envs=16, n_steps=128, epochs=4, n_minibatches=8, lr=3e-4,
          gamma=0.99, gae_lambda=0.95, clip=0.2, entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5,
          hidden=256, seed=0, log_path=None, verbose=True, checkpoint=None, checkpoint_every=25,
          network='mlp', d_model=64, n_layers=3, n_heads=4):
    """
    make_env: function(seed) -> FLPEnv.   Returns (model, history) where history is a list of
    dicts (one per update) with the mean statistics of the last 200 finished episodes.
    checkpoint: optional function(model, steps_done), called every `checkpoint_every` updates
    (the training log is written at the same moments), so a long run can be stopped or inspected.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    envs = [make_env(seed * 1000 + i) for i in range(n_envs)]
    obs_size, n_actions = envs[0].obs_size, envs[0].n_actions
    model = make_model(obs_size, n_actions, envs[0].n_lookahead, envs[0].n, network, hidden,
                       d_model, n_layers, n_heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    obs_buf = torch.zeros((n_steps, n_envs, obs_size))
    mask_buf = torch.zeros((n_steps, n_envs, n_actions), dtype=torch.bool)
    act_buf = torch.zeros((n_steps, n_envs), dtype=torch.long)
    logp_buf = torch.zeros((n_steps, n_envs))
    rew_buf = torch.zeros((n_steps, n_envs))
    done_buf = torch.zeros((n_steps, n_envs))
    val_buf = torch.zeros((n_steps, n_envs))

    obs = torch.as_tensor(np.stack([e.reset() for e in envs]))
    mask = torch.as_tensor(np.stack([e.action_mask() for e in envs]))
    ep_return = np.zeros(n_envs)
    recent = deque(maxlen=200)          # (return, best/ref, final/ref) of finished episodes
    history, steps_done, start = [], 0, time.time()
    n_updates = max(1, total_steps // (n_steps * n_envs))
    batch_size = n_steps * n_envs
    mb_size = batch_size // n_minibatches

    def write_log():
        if log_path and history:
            os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
            with open(log_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)

    for update in range(n_updates):
        optimizer.param_groups[0]['lr'] = lr * (1.0 - update / n_updates)      # linear decay
        for t in range(n_steps):
            obs_buf[t], mask_buf[t] = obs, mask
            with torch.no_grad():
                dist, value = model.evaluate(obs, mask)
                action = dist.sample()
                logp_buf[t] = dist.log_prob(action)
                val_buf[t] = value
            act_buf[t] = action
            next_obs, next_mask = [], []
            for i, env in enumerate(envs):
                o, r, done, info = env.step(int(action[i]))
                rew_buf[t, i] = r
                done_buf[t, i] = float(done)
                ep_return[i] += r
                if done:
                    recent.append((ep_return[i], info['best_cost'] / info['ref_cost'],
                                   info['cost'] / info['ref_cost']))
                    ep_return[i] = 0.0
                    o = env.reset()
                next_obs.append(o)
                next_mask.append(env.action_mask())
            obs = torch.as_tensor(np.stack(next_obs))
            mask = torch.as_tensor(np.stack(next_mask))
        steps_done += batch_size

        # generalised advantage estimation; an episode end is a true terminal state because the
        # elapsed time is part of the observation
        with torch.no_grad():
            next_value = model.value(obs)
        adv = torch.zeros_like(rew_buf)
        last = torch.zeros(n_envs)
        for t in reversed(range(n_steps)):
            nonterminal = 1.0 - done_buf[t]
            next_v = next_value if t == n_steps - 1 else val_buf[t + 1]
            delta = rew_buf[t] + gamma * next_v * nonterminal - val_buf[t]
            last = delta + gamma * gae_lambda * nonterminal * last
            adv[t] = last
        returns = adv + val_buf

        b_obs = obs_buf.reshape(batch_size, obs_size)
        b_mask = mask_buf.reshape(batch_size, n_actions)
        b_act, b_logp = act_buf.reshape(-1), logp_buf.reshape(-1)
        b_adv, b_ret = adv.reshape(-1), returns.reshape(-1)
        for _ in range(epochs):
            perm = torch.randperm(batch_size)
            for s in range(0, batch_size, mb_size):
                idx = perm[s:s + mb_size]
                dist, value = model.evaluate(b_obs[idx], b_mask[idx])
                ratio = torch.exp(dist.log_prob(b_act[idx]) - b_logp[idx])
                a = b_adv[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                policy_loss = -torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a).mean()
                value_loss = 0.5 * (value - b_ret[idx]).pow(2).mean()
                loss = policy_loss + value_coef * value_loss - entropy_coef * dist.entropy().mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        if recent:
            arr = np.array(recent)
            row = {'steps': steps_done, 'episode_return': arr[:, 0].mean(),
                   'best_cost_ratio': arr[:, 1].mean(), 'final_cost_ratio': arr[:, 2].mean(),
                   'seconds': time.time() - start}
            history.append(row)
            if verbose and (update % max(1, n_updates // 25) == 0 or update == n_updates - 1):
                print(f"steps {steps_done:>8,}  return {row['episode_return']:+.3f}  "
                      f"best cost / initial cost {row['best_cost_ratio']:.3f}  "
                      f"final cost / initial cost {row['final_cost_ratio']:.3f}  "
                      f"{steps_done / row['seconds']:.0f} steps/s", flush=True)

        if checkpoint is not None and (update + 1) % checkpoint_every == 0:
            checkpoint(model, steps_done)
            write_log()

    write_log()
    model.eval()
    return model, history


def rollout(model, env, demand, demand_id=None, greedy=True, layout_cfg=None):
    """One episode on a given demand. Returns (best cost, best cfg, start cost, final cost)."""
    obs = env.reset(demand=demand, demand_id=demand_id, layout_cfg=layout_cfg)
    done = False
    while not done:
        obs, _, done, _ = env.step(model.act(obs, env.action_mask(), greedy=greedy))
    return env.best_cost, env.best_cfg.copy(), env.start_cost, env.cost


def solve(model, env, demand, demand_id=None, samples=0, layout_cfg=None):
    """
    Greedy episode, plus `samples` stochastic episodes; the best layout found is returned
    (cost, cfg, greedy cost). Sampling is a cheap way to trade computing time for quality.
    """
    if samples == 0:
        best, cfg, _, _ = rollout(model, env, demand, demand_id, True, layout_cfg)
        return best, cfg, best
    # all episodes run side by side: one network call per step for the whole group.
    # Copies of the environment share its tables and caches and keep their own layout.
    envs = [env] + [copy.copy(env) for _ in range(samples)]
    obs = np.stack([e.reset(demand=demand, demand_id=demand_id, layout_cfg=layout_cfg) for e in envs])
    done = False
    while not done:
        masks = np.stack([e.action_mask() for e in envs])
        with torch.no_grad():
            dist = model.distribution(torch.as_tensor(obs), torch.as_tensor(masks))
            actions = dist.sample()
            actions[0] = dist.probs[0].argmax()          # the first episode is the greedy one
        steps = [e.step(int(a)) for e, a in zip(envs, actions)]
        obs, done = np.stack([s[0] for s in steps]), steps[0][2]
    costs = [e.best_cost for e in envs]
    winner = int(np.argmin(costs))
    return costs[winner], envs[winner].best_cfg.copy(), envs[0].best_cost
