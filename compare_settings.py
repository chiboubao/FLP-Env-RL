"""
Controlled comparison on the 4-workstation case of the paper: reward functions and canonical
mapping on/off, several seeds each, same PPO settings and same number of steps.

Each policy is trained on T1..T5 and evaluated (greedy, one episode per demand) on
  * the 5 training demands, and
  * the 5 perturbed demands T1_P..T5_P, never seen during training,
against the true optimum. The score is the share of the possible cost reduction that is obtained.

  python compare_settings.py --steps 400000 --seeds 3 --workers 6
"""
import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np

from flp_env import ROOT, load_demand


def run(job):
    import torch
    torch.set_num_threads(1)
    from evaluate_ppo import build_env, evaluation_table
    from ppo import train
    label, config, seed, steps = job
    train_files = sorted(glob.glob(os.path.join(ROOT, 'transportation_demand', 'paper_M4_train', '*.csv')))
    test_files = sorted(glob.glob(os.path.join(ROOT, 'transportation_demand', 'paper_M4_perturbed', '*.csv')))
    train_demands = [load_demand(f) for f in train_files]
    test_demands = [load_demand(f) for f in test_files]
    model, history = train(lambda s: build_env(config, train_demands, s), steps, seed=seed, verbose=False)
    env = build_env(config, train_demands, seed=1, random_demands=False)
    names = [os.path.basename(f)[:-4] for f in train_files]
    seen = evaluation_table(model, env, train_demands, names, verbose=False)
    unseen = evaluation_table(model, env, test_demands, [os.path.basename(f)[:-4] for f in test_files], verbose=False)
    return dict(label=label, seed=seed,
                seen=float(np.mean([r['achieved'] for r in seen])),
                unseen=float(np.mean([r['achieved'] for r in unseen])),
                seen_optimal=sum(abs(r['best'] - r['reference']) < 1e-6 for r in seen),
                unseen_optimal=sum(abs(r['best'] - r['reference']) < 1e-6 for r in unseen),
                seen_cost=float(np.mean([r['best'] for r in seen])),
                train_curve=[(h['steps'], h['best_cost_ratio']) for h in history[::max(1, len(history) // 8)]])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=400000)
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--settings', nargs='*', default=['improvement', 'best', 'legacy', 'improvement-nocanon'])
    args = parser.parse_args()

    base = dict(machines=4, random_demands=False, sampler_probability=1.0, canonical=True, allow_overlap=True,
                random_start=0.0, episode_length=40, legacy_bounds=(400, 1750), hidden=256, lookahead=True)
    settings = {
        'improvement': dict(base, reward='improvement'),
        'best': dict(base, reward='best'),
        'legacy': dict(base, reward='legacy'),
        'improvement-nocanon': dict(base, reward='improvement', canonical=False),
        'best-nocanon': dict(base, reward='best', canonical=False),
        'improvement-nolookahead': dict(base, reward='improvement', lookahead=False),
        'legacy-nolookahead': dict(base, reward='legacy', lookahead=False),
    }
    jobs = [(name, settings[name], seed, args.steps) for name in args.settings for seed in range(args.seeds)]
    with Pool(args.workers) as pool:
        results = pool.map(run, jobs, chunksize=1)

    print(f'\n{args.steps:,} steps per run, {args.seeds} seeds; score = % of the possible cost reduction obtained')
    print(f"{'setting':<22}{'training demands':>20}{'optimal':>9}{'unseen demands':>18}{'optimal':>9}")
    for name in args.settings:
        rs = [r for r in results if r['label'] == name]
        seen, unseen = [r['seen'] for r in rs], [r['unseen'] for r in rs]
        print(f"{name:<22}{np.mean(seen):>12.1f}% +-{np.std(seen):>4.1f}"
              f"{sum(r['seen_optimal'] for r in rs):>5}/{5 * len(rs)}"
              f"{np.mean(unseen):>10.1f}% +-{np.std(unseen):>4.1f}"
              f"{sum(r['unseen_optimal'] for r in rs):>5}/{5 * len(rs)}")
    print('\nper run:')
    for r in results:
        print(f"  {r['label']:<22} seed {r['seed']}: seen {r['seen']:.1f}% ({r['seen_optimal']}/5 optimal), "
              f"unseen {r['unseen']:.1f}% ({r['unseen_optimal']}/5 optimal)")
