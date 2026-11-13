"""
Evaluate a trained policy on a set of transportation demands.

Examples (from the project folder):
  python evaluate_ppo.py --model runs/m4/model.pt
  python evaluate_ppo.py --model runs/m4/model.pt --demands transportation_demand/paper_M4_perturbed --plots
  python evaluate_ppo.py --model runs/m9/model.pt --random 20 --samples 16
  python evaluate_ppo.py --model runs/m4/model.pt --dynamic      (layout carried from one demand to the next)
"""
import argparse
import glob
import os

import numpy as np

from baselines import exact_optimum, simulated_annealing, steepest_descent
from flp_env import ROOT, FLPEnv, load_demand, random_demand
from ppo import load_model, rollout, solve


def demand_files(path):
    path = path if os.path.isabs(path) else os.path.join(ROOT, path)
    return sorted(glob.glob(os.path.join(path, '*.csv'))) if os.path.isdir(path) else [path]


def build_env(config, demands=None, seed=0, random_demands=None):
    n = config['machines']
    use_sampler = config['random_demands'] if random_demands is None else random_demands
    sampler = (lambda rng: random_demand(n, rng)) if use_sampler or not demands else None
    return FLPEnv(n, demands=demands, demand_sampler=sampler,
                  sampler_probability=config.get('sampler_probability', 1.0),
                  episode_length=config['episode_length'], reward=config['reward'],
                  canonical=config['canonical'], random_start=config.get('random_start', 0.0),
                  legacy_bounds=config.get('legacy_bounds'), allow_overlap=config['allow_overlap'],
                  lookahead=config.get('lookahead', False), seed=seed)


def reference_cost(env, demand, sa_iterations=20000):
    """True optimum for 4 workstations, simulated annealing (best of 2 runs) otherwise."""
    if env.n == 4:
        return exact_optimum(env, demand)[0], 'optimum'
    return min(simulated_annealing(env, demand, sa_iterations, seed=s)[0] for s in range(2)), 'SA'


def evaluation_table(model, env, demands, names, samples=0, sa_iterations=20000, plot_dir=None, verbose=True):
    """
    One row per demand. `achieved` = share of the possible cost reduction obtained by the agent:
    100 * (initial - agent) / (initial - reference); 100% means the reference cost is reached.
    """
    rows = []
    env.random_start = 0.0          # evaluation always starts from the default initial layout
    for name, demand in zip(names, demands):
        ref, ref_name = reference_cost(env, demand, sa_iterations)
        local, _, _ = steepest_descent(env, demand)
        best, cfg, greedy = solve(model, env, demand, ('eval', name), samples)
        initial = env.ref_cost
        span = max(initial - ref, 1e-9)
        rows.append(dict(name=name, initial=initial, greedy=greedy, best=best, local_search=local,
                         reference=ref, achieved_greedy=100 * (initial - greedy) / span,
                         achieved=100 * (initial - best) / span,
                         achieved_local=100 * (initial - local) / span))
        if plot_dir:
            from flp_plot import plot_layout
            os.makedirs(plot_dir, exist_ok=True)
            plot_layout(env, cfg, os.path.join(plot_dir, f'{name}.png'), f'{name}: cost {best:g} ({ref_name} {ref:g})')
    if verbose:
        sampled = f'best of {samples + 1}'
        print(f"{'demand':<10}{'initial':>9}{'agent':>9}" + (f"{sampled:>12}" if samples else '') +
              f"{'local search':>14}{ref_name:>10}{'achieved':>10}")
        for r in rows:
            print(f"{r['name']:<10}{r['initial']:>9.1f}{r['greedy']:>9.1f}" +
                  (f"{r['best']:>12.1f}" if samples else '') +
                  f"{r['local_search']:>14.1f}{r['reference']:>10.1f}{r['achieved']:>9.1f}%")
        print(f"mean cost: agent {np.mean([r['best'] for r in rows]):.1f}, "
              f"local search {np.mean([r['local_search'] for r in rows]):.1f}, "
              f"{ref_name} {np.mean([r['reference'] for r in rows]):.1f};  share of the possible reduction: "
              f"agent {np.mean([r['achieved'] for r in rows]):.1f}%, "
              f"local search {np.mean([r['achieved_local'] for r in rows]):.1f}%")
    return rows


def dynamic_run(model, env, demands, names):
    """Weak-form dynamic FLP: the layout found for one period is the starting layout of the next."""
    print('Dynamic run (each period starts from the layout of the previous one):')
    env.random_start = 0.0
    cfg = None
    for name, demand in zip(names, demands):
        best, best_cfg, start, _ = rollout(model, env, demand, ('eval', name), True, layout_cfg=cfg)
        ref, ref_name = reference_cost(env, demand)
        moved = '' if cfg is None else f', {int((best_cfg != env.canonicalize(cfg)[0]).sum()) if env.canonical else int((best_cfg != cfg).sum())} workstations changed'
        print(f'  {name}: cost when the period starts {start:.1f} -> {best:.1f} ({ref_name} {ref:.1f}){moved}')
        cfg = best_cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--demands', help='folder of CSV files or one CSV (default: the training demands)')
    parser.add_argument('--random', type=int, default=0, help='evaluate on this many new random demands')
    parser.add_argument('--samples', type=int, default=0, help='extra stochastic episodes per demand')
    parser.add_argument('--sa-iterations', type=int, default=20000)
    parser.add_argument('--plots', action='store_true', help='save a picture of each layout next to the model')
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--seed', type=int, default=12345)
    args = parser.parse_args()

    model, config = load_model(args.model)
    if args.random:
        rng = np.random.default_rng(args.seed)
        demands = [random_demand(config['machines'], rng) for _ in range(args.random)]
        names = [f'random{i + 1}' for i in range(args.random)]
    else:
        files = demand_files(args.demands) if args.demands else [os.path.join(ROOT, f) for f in config['demand_files']]
        assert files, 'no demand file: use --demands or --random'
        demands = [load_demand(f) for f in files]
        names = [os.path.splitext(os.path.basename(f))[0] for f in files]
    env = build_env(config, demands, seed=args.seed, random_demands=False)
    plot_dir = os.path.join(os.path.dirname(os.path.abspath(args.model)), 'layouts') if args.plots else None
    evaluation_table(model, env, demands, names, args.samples, args.sa_iterations, plot_dir)
    if args.dynamic:
        dynamic_run(model, env, demands, names)
