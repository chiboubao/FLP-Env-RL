# Training one PPO agent on several transportation demands

Everything below runs locally from the project folder. Trained models are written to `runs/`,
which is not stored in Git; the commands of section 3 recreate them.

## 1. Install

Python 3.11, then:

```bash
pip install -r requirements_ppo.txt
```

TensorFlow and gym are not needed for this part (they are only used by `Agent_Environment.py`).

## 2. Check that everything works on your machine

```bash
python tests/test_env.py
python tests/test_flp_env.py
python tests/check_canonical.py
```

* `test_env.py`: the original environment (`Environment.py`).
* `test_flp_env.py`: the fast environment gives exactly the same cost and the same transitions as the original one.
* `check_canonical.py`: exhaustive check of the canonical mapping on the 4-workstation case (about 2 minutes).

## 3. Train

```bash
python make_paper_demands.py
python train_ppo.py --machines 4 --demands transportation_demand/paper_M4_train --steps 400000 --out runs/m4
```

The first command writes the demands T1..T5 of the CIE 2025 paper and their perturbed versions.
The second trains one policy on the five demands (a new demand at each episode), then prints a
table that compares the agent with a local search and with the true optimum. It writes in `runs/m4`:
`model.pt`, `training_log.csv`, `training_curve.png` and one picture per demand in `layouts/`.

Other cases:

```bash
# 9 workstations, a new random demand at every episode
python train_ppo.py --machines 9 --random-demands --steps 1200000 --out runs/m9

# your own demands: one CSV per demand in a folder (same format as transportation_demand/*.csv)
python train_ppo.py --machines 9 --demands my_folder --steps 1200000 --out runs/mine

# your demands mixed with random ones (helps on demands the agent has never seen)
python train_ppo.py --machines 9 --demands my_folder --random-demands --out runs/mixed
```

Useful options: `--reward improvement|best|legacy`, `--no-canonical`, `--forbid-overlap`,
`--no-lookahead`, `--random-start 0.5`, `--episode-length`, `--seed`. `python train_ppo.py -h` lists all.

## 4. Evaluate

```bash
python evaluate_ppo.py --model runs/m4/model.pt
python evaluate_ppo.py --model runs/m4/model.pt --demands transportation_demand/paper_M4_perturbed --plots
python evaluate_ppo.py --model runs/m9/model.pt --random 12 --samples 15
python evaluate_ppo.py --model runs/m4/model.pt --dynamic
```

* `--demands`: demands the agent has not seen (folder or file).
* `--random 12`: twelve new random demands.
* `--samples 15`: besides the greedy episode, 15 stochastic episodes; the best layout is kept.
* `--dynamic`: the layout found for one demand is the starting layout for the next one.
* The reference is the true optimum for 4 workstations and simulated annealing otherwise.

## 5. Reproduce the comparison of rewards and of the canonical mapping

```bash
python compare_settings.py --steps 400000 --seeds 3 --workers 6
```

## Results measured with this code

Score = share of the possible cost reduction that the agent obtains,
`100 * (initial - agent) / (initial - reference)`. Shared P/D spots are allowed, as in the paper.

**4 workstations, trained on T1..T5 of the paper, 400,000 steps, 3 seeds, one greedy episode per demand.**
The reference is the true optimum (enumeration). "Unseen" = perturbed demands T1_P..T5_P.

| Setting | Training demands | Optimal | Unseen demands | Optimal |
|---|---|---|---|---|
| `improvement` reward (default) | 99.5% ± 0.1 | 12/15 | 99.3% ± 0.2 | 12/15 |
| `best` reward | 99.4% ± 0.0 | 12/15 | 99.1% ± 0.1 | 12/15 |
| `legacy` reward (paper) | 68.7% ± 6.3 | 0/15 | 69.4% ± 5.0 | 0/15 |
| `improvement`, no canonical mapping | 99.0% ± 0.1 | 12/15 | 98.8% ± 0.1 | 12/15 |
| `improvement`, no lookahead | 97.5% ± 0.6 | 5/15 | 97.6% ± 0.7 | 5/15 |

Local search (best single action until stuck) obtains 72%. True optima: T1 80, T2 40, T3 240,
T4 40, T5 44. A model trained with the command of section 3 found 80, 40, 260, 40, 44.

**9 workstations, trained on random demands only, 1,200,000 steps (25 minutes), one run.**
Tested on 12 random demands never seen in training; the reference is simulated annealing
(2 runs of 20,000 iterations).

| Method | Mean cost | Score |
|---|---|---|
| Initial layout | 2573 | 0% |
| Local search | 1229 | 75.7% |
| Agent, one greedy episode | 1184 | 74.4% |
| Agent, best of 16 episodes (`--samples 15`) | 937 | 92.5% |
| Simulated annealing (reference) | 805 | 100% |

On the repository demand `transportation_demand_M9_1.csv` (never seen): initial 1750, agent 460
(best of 16), simulated annealing 425, local search 730.
Without the lookahead feature the same training reached only 57.5%.

## Known limits

* 9 workstations: one greedy episode is only as good as local search; the gain comes from the
  16 episodes. The training curve was still improving at 1.2 million steps.
* Dynamic FLP (restart from the previous layout, `--dynamic`): not solved. A policy trained from
  the initial layout stalls when it starts elsewhere, and one test with `--random-start 0.5`
  (600,000 steps) was worse on the normal case (87.7%) without fixing it. Rearrangement costs are
  not modelled.
* Cost values differ from the paper except T1 (optimum 80 in both): the initial layout of
  `graph4M.gml` is not the one of the paper's figure, and the paper reports 130 for T3 while the
  true optimum of this model is 240.
* 16 workstations and more were not trained here. The lookahead costs 1.8 ms per step at 16
  workstations; use `--no-lookahead` if it becomes too slow for larger workshops.
* The 9-workstation numbers come from a single training run.

## What is in the folder

| File | Role |
|---|---|
| `flp_env.py` | Fast environment for training: several demands, rewards, action mask, canonical mapping |
| `ppo.py` | PPO with action masking (PyTorch) |
| `train_ppo.py`, `evaluate_ppo.py` | Command-line training and evaluation |
| `baselines.py` | True optimum (4 workstations), simulated annealing, local search |
| `compare_settings.py` | Controlled comparison (rewards, canonical on/off, lookahead on/off) |
| `Environment_FLP/canonical.py` | Geometry and canonical layouts |
| `Environment_FLP/transport.py` | Exact solver for the empty-travel problem |
| `Environment.py` | Original environment, kept working; new reward `improvement_cost`, new option `allow_overlap` |
| `tests/` | The three checks of section 2 |

## The reward

`C` is the transportation cost and `C_ref` the cost of the initial layout for the demand of the episode.

* `improvement` (default): `r = (C_before - C_after) / C_ref`. The sum over an episode is the
  fraction of the initial cost that was removed, so the scale is the same for every demand. A
  step that degrades the layout costs exactly what it degrades, so the agent can cross a worse
  layout to reach a better one.
* `best`: `r = (best_before - best_after) / C_ref`. Only a new best layout is paid.
* `legacy`: the reward of the paper (scaled cost if the cost did not increase, otherwise -1).

The same `improvement` reward is available in the original environment:
`Env(..., reward='improvement_cost', ...)`.

## The observation

Per workstation the P/D coordinates, the matrix of distances P_i -> D_j, the demand matrix, the
cost share of each flow, the current and best cost, the elapsed time and, with the lookahead
option, the loaded-cost reduction of each action. The demand is part of the observation: this is
what lets one network handle several demands.
