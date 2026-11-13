"""
Train one PPO policy on several transportation demands.

Examples (from the project folder):
  # the 5 demands of the paper, 4 workstations
  python train_ppo.py --machines 4 --demands transportation_demand/paper_M4_train --steps 400000 --out runs/m4

  # 9 workstations, a new random demand at every episode (stochastic FLP)
  python train_ppo.py --machines 9 --random-demands --steps 3000000 --out runs/m9

  # your own demands: put the CSV files in a folder
  python train_ppo.py --machines 9 --demands my_demands_folder --out runs/mine
"""
import argparse
import os
import time

import torch

from evaluate_ppo import build_env, demand_files, evaluation_table
from flp_env import REWARDS, ROOT, load_demand
from ppo import save_model, train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--machines', type=int, default=4)
    parser.add_argument('--demands', help='folder of CSV demand files, or one CSV file')
    parser.add_argument('--random-demands', action='store_true',
                        help='draw a new random demand at each episode (alone, or mixed with --demands)')
    parser.add_argument('--sampler-probability', type=float, default=0.5,
                        help='with both --demands and --random-demands: share of random episodes')
    parser.add_argument('--reward', choices=REWARDS, default='improvement')
    parser.add_argument('--no-canonical', action='store_true')
    parser.add_argument('--forbid-overlap', action='store_true',
                        help='forbid two P/D points on the same spot (the paper allows it)')
    parser.add_argument('--no-lookahead', action='store_true',
                        help='do not give the policy the loaded-cost gain of each action')
    parser.add_argument('--random-start', type=float, default=0.0,
                        help='probability of starting an episode from a random layout (useful for dynamic FLPs)')
    parser.add_argument('--episode-length', type=int, default=None)
    parser.add_argument('--steps', type=int, default=400000)
    parser.add_argument('--n-envs', type=int, default=16)
    parser.add_argument('--n-steps', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--entropy', type=float, default=0.01)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--out', default='runs/ppo')
    parser.add_argument('--no-eval', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    files = demand_files(args.demands) if args.demands else []
    assert files or args.random_demands, 'give --demands and/or --random-demands'
    config = dict(machines=args.machines,
                  demand_files=[os.path.relpath(f, ROOT) for f in files],
                  random_demands=args.random_demands,
                  sampler_probability=args.sampler_probability if files else 1.0,
                  reward=args.reward, canonical=not args.no_canonical, allow_overlap=not args.forbid_overlap,
                  random_start=args.random_start, lookahead=not args.no_lookahead,
                  episode_length=args.episode_length or (40 if args.machines == 4 else 8 * args.machines),
                  legacy_bounds=(400, 1750) if args.machines == 4 else None, hidden=args.hidden)
    demands = [load_demand(f) for f in files]
    out = args.out if os.path.isabs(args.out) else os.path.join(os.getcwd(), args.out)
    os.makedirs(out, exist_ok=True)
    print(config, flush=True)

    t0 = time.time()
    model, history = train(lambda seed: build_env(config, demands, seed), args.steps, n_envs=args.n_envs,
                           n_steps=args.n_steps, lr=args.lr, gamma=args.gamma, entropy_coef=args.entropy,
                           hidden=args.hidden, seed=args.seed, log_path=os.path.join(out, 'training_log.csv'))
    env = build_env(config, demands, seed=args.seed + 999)
    config.update(obs_size=env.obs_size, n_actions=env.n_actions)
    save_model(model, os.path.join(out, 'model.pt'), config)
    try:
        from flp_plot import plot_training
        plot_training(history, os.path.join(out, 'training_curve.png'),
                      f"{args.machines} workstations, reward '{args.reward}'")
    except Exception as error:      # plotting must never lose a trained model
        print('could not draw the training curve:', error)
    print(f'trained in {(time.time() - t0) / 60:.1f} min; saved to {out}', flush=True)

    if not args.no_eval and files:
        names = [os.path.splitext(os.path.basename(f))[0] for f in files]
        eval_env = build_env(config, demands, seed=1, random_demands=False)
        evaluation_table(model, eval_env, demands, names, plot_dir=os.path.join(out, 'layouts'))
