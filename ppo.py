"""
PPO with invalid-action masking for flp_env.FLPEnv (PyTorch, single file, CPU is enough).

The policy sees the layout AND the transportation demand, so one network is trained on many
demands (a new one at every episode) and is then used on any demand, including unseen ones.
"""
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


def save_model(model, path, config):
    torch.save({'state_dict': model.state_dict(), 'config': config}, path)


def load_model(path):
    data = torch.load(path, map_location='cpu')
    cfg = data['config']
    model = ActorCritic(cfg['obs_size'], cfg['n_actions'], cfg['hidden'],
                        cfg['n_actions'] if cfg.get('lookahead') else 0)
    model.load_state_dict(data['state_dict'])
    model.eval()
    return model, cfg


def train(make_env, total_steps, n_envs=16, n_steps=128, epochs=4, n_minibatches=8, lr=3e-4,
          gamma=0.99, gae_lambda=0.95, clip=0.2, entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5,
          hidden=256, seed=0, log_path=None, verbose=True):
    """
    make_env: function(seed) -> FLPEnv.   Returns (model, history) where history is a list of
    dicts (one per update) with the mean statistics of the last 200 finished episodes.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    envs = [make_env(seed * 1000 + i) for i in range(n_envs)]
    obs_size, n_actions = envs[0].obs_size, envs[0].n_actions
    model = ActorCritic(obs_size, n_actions, hidden, envs[0].n_lookahead)
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

    for update in range(n_updates):
        optimizer.param_groups[0]['lr'] = lr * (1.0 - update / n_updates)      # linear decay
        for t in range(n_steps):
            obs_buf[t], mask_buf[t] = obs, mask
            with torch.no_grad():
                dist = model.distribution(obs, mask)
                action = dist.sample()
                logp_buf[t] = dist.log_prob(action)
                val_buf[t] = model.value(obs)
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
                dist = model.distribution(b_obs[idx], b_mask[idx])
                ratio = torch.exp(dist.log_prob(b_act[idx]) - b_logp[idx])
                a = b_adv[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                policy_loss = -torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a).mean()
                value_loss = 0.5 * (model.value(b_obs[idx]) - b_ret[idx]).pow(2).mean()
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

    if log_path and history:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        with open(log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
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
    best, cfg, _, _ = rollout(model, env, demand, demand_id, True, layout_cfg)
    greedy_cost = best
    for _ in range(samples):
        c, k, _, _ = rollout(model, env, demand, demand_id, False, layout_cfg)
        if c < best - 1e-9:
            best, cfg = c, k
    return best, cfg, greedy_cost
