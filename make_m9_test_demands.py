"""
Writes to disk the 9-workstation demands used in the report:
  * transportation_demand/random_M9_test/random1.csv .. random12.csv : the 12 test demands
    (same as `python evaluate_ppo.py --model ... --random 12`, seed 12345);
  * transportation_demand/random_M9_examples/example1.csv .. example3.csv : three examples of the
    demands drawn during training (generator seeds 0, 1, 2).
Run once:  python make_m9_test_demands.py
"""
import os

import numpy as np
import pandas as pd

from flp_env import ROOT, random_demand

N = 9


def save(demand, folder, name):
    out = os.path.join(ROOT, 'transportation_demand', folder)
    os.makedirs(out, exist_ok=True)
    df = pd.DataFrame(demand.astype(int), index=[f'P{i}' for i in range(1, N + 1)],
                      columns=[f'D{i}' for i in range(1, N + 1)])
    df.to_csv(os.path.join(out, f'{name}.csv'))


if __name__ == '__main__':
    rng = np.random.default_rng(12345)
    for i in range(12):
        save(random_demand(N, rng, n_products=(N // 2, N)), 'random_M9_test', f'random{i + 1}')
    for seed in range(3):
        save(random_demand(N, np.random.default_rng(seed), n_products=(N // 2, N)),
             'random_M9_examples', f'example{seed + 1}')
    print('written transportation_demand/random_M9_test and transportation_demand/random_M9_examples')
