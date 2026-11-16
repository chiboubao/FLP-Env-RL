# PPO on facility layout problems: 4 and 9 workstations

Report of 13 November 2026, updated on 14 November with the second training (section 8b) and the weight-sharing network (section 8c). Training, evaluation and testing of one PPO policy per workshop size, with the data used in each phase.

## 1. Summary

One PPO policy per workshop size learns to solve several transportation demands. On 4 workstations it obtains 99.5% of the possible cost reduction and reaches the true optimum on 4 of the 5 demands. On 9 workstations it obtains 92.5% on demands it never saw, but only when it runs 16 episodes per demand. A second training with three changes raises it by about 2.5 points (91.9% to 94.4% over 10 evaluation runs). A third training with a network that shares its weights across workstations reaches 99.3% with 16 episodes and 101.3% with 64, that is, cheaper layouts than the simulated annealing reference; for the same computing effort it is about 2 points ahead of simulated annealing at 16 episodes and level with it beyond. On 4 workstations, 20 demands with new flow patterns lower the score to 84.2% in one episode and 94.0% with the best of 16.

| Case | Trained on | Tested on | Result |
|---|---|---|---|
| 4 workstations | 5 demands of the paper (T1 to T5), 400,000 steps | 5 perturbed demands, never seen | 99.3% of the possible reduction, one episode per demand |
| 9 workstations | Random demands, a new one each episode, 1,200,000 steps | 12 random demands, never seen | 92.5% with the best of 16 episodes; 74.4% with one episode |
| 9 workstations, weight-sharing network (section 8c) | Random demands, a new one each episode, 1,500,000 steps, 3 seeds | The same 12 demands | 99.3% with the best of 16 episodes; 82.1% with one episode; 101.3% with the best of 64 |

The score is `100 x (initial cost - agent cost) / (initial cost - reference cost)`. The reference is the true optimum for 4 workstations and simulated annealing for 9. Two pickup or delivery points at the same coordinates are at distance zero, as in the paper.

Three things are not shown by these results: the first 9-workstation numbers (sections 6 to 8) come from a single training run, the dynamic case (restart from the previous layout) does not work yet, and only the T1 optimum (80) matches the cost reported in the paper.

## 2. Setup common to both cases

Both cases use the same environment, reward, observation and PPO settings; only the workshop size, the demands and the number of steps differ.

**Environment.** `flp_env.py`, a fast version of the original `Environment.py`. The tests check that it gives exactly the same cost and the same transitions as the original.

- Cost: loaded travel (demand x shortest path from pickup to delivery) plus empty travel (transportation problem between delivery and pickup points).
- Actions per workstation: 5 shapes (U, U', L, L', I) and 2 rotations. Plus one swap for every pair of workstations.
- Canonical mapping: on. Every layout is replaced by its representative under the 8 rotations and mirrors of the grid.
- Shared spots: allowed. Two points with the same coordinates are at distance zero.
- Start of an episode: the default initial layout (workstation m on site m, U-shape).

**Reward.** Cost reduction of the step divided by the cost of the initial layout for the demand of the episode: `r = (C_before - C_after) / C_ref`. Summed over an episode, it equals the fraction of the initial cost that was removed, so every demand has the same scale.

**Observation.** Per workstation the pickup and delivery coordinates; the matrix of distances from pickup i to delivery j; the demand matrix; the cost share of each flow; current cost, best cost and elapsed time; and for every action the loaded-cost change it would produce (lookahead).

**PPO settings.**

| Setting | Value |
|---|---|
| Network | Actor and critic, each 2 hidden layers of 256 units (tanh) |
| Parallel environments | 16 |
| Steps per update | 128 per environment (2,048 per update) |
| Epochs per update, minibatches | 4, 8 |
| Learning rate | 0.0003, linear decay to 0 |
| Discount, GAE lambda | 0.99, 0.95 |
| Clip range | 0.2 |
| Entropy coefficient, value coefficient | 0.01, 0.5 |
| Gradient clipping | 0.5 |
| Action masking | On (no action is masked when shared spots are allowed) |
| Seed | 0 for the models reported in detail |

**The three phases.**

1. Training: the policy is updated; a new demand is drawn at each episode.
2. Evaluation: the trained policy is run on the training data, with no learning.
3. Testing: the trained policy is run on demands it never saw, with no learning.

**Baselines.** Local search applies the best single action until no action improves. Simulated annealing uses the same actions for 20,000 iterations. For 4 workstations the true optimum is found by enumerating all 480,000 canonical layouts.

## 3. 4 workstations: training phase

The policy was trained for 399,360 steps (9,984 episodes) in 187 seconds. The mean cost at the end of an episode fell from 24.5% to 11.6% of the initial cost.

| Item | Value |
|---|---|
| Training data | 5 demand matrices T1 to T5 (Appendix A); one drawn at random at each episode, about 2,000 episodes per demand |
| Workshop | 2 x 2 sites, 34 actions (28 shape and rotation actions, 6 swaps) |
| Observation size | 101 values |
| Episode | 40 steps from the default initial layout |
| Steps and updates | 399,360 steps, 195 updates |
| Duration | 187 seconds on one CPU core |
| Mean episode return | 0.755 after the first update, 0.884 at the end |

![Training curve, 4 workstations](runs/m4/training_curve.png)

Training curve, 12 points along the run (mean of the last 200 episodes). The full log, 195 rows, is `runs/m4/training_log.csv`.

| Steps | Mean episode return | Best layout of the episode, cost / initial | Last layout of the episode, cost / initial |
|---|---|---|---|
| 2,048 | 0.755 | 13.7% | 24.5% |
| 34,816 | 0.815 | 12.7% | 18.5% |
| 69,632 | 0.858 | 11.1% | 14.2% |
| 104,448 | 0.859 | 11.8% | 14.1% |
| 139,264 | 0.890 | 10.4% | 11.0% |
| 174,080 | 0.882 | 11.4% | 11.8% |
| 208,896 | 0.895 | 10.1% | 10.5% |
| 243,712 | 0.895 | 10.0% | 10.5% |
| 276,480 | 0.907 | 9.2% | 9.3% |
| 311,296 | 0.893 | 10.5% | 10.7% |
| 346,112 | 0.889 | 11.0% | 11.1% |
| 399,360 | 0.884 | 11.5% | 11.6% |

The first point is already at 24.5% because the untrained policy follows the lookahead input, which makes it a soft greedy descent. Learning closes most of the remaining gap by about 125,000 steps. The mean optimum for an equal mix of the five demands is 10.3%; a window of 200 episodes with few T3 episodes can fall below it, because T3 has the highest optimum. Actions are sampled during training, so these values are slightly worse than the greedy evaluation in the next section.

## 4. 4 workstations: evaluation phase

On the five training demands, one greedy episode per demand, the trained policy reaches the true optimum on T1, T2, T4 and T5 and misses it on T3 (260 against 240). The mean score is 99.4%.

**Per demand (model of section 3, seed 0).**

| Demand | Total flow | Initial cost | PPO agent | Steps to its best layout | Local search | True optimum | Score |
|---|---|---|---|---|---|---|---|
| T1 | 120 | 820 | 80 | 4 | 80 | 80 | 100.0% |
| T2 | 100 | 780 | 40 | 5 | 360 | 40 | 100.0% |
| T3 | 120 | 940 | 260 | 5 | 300 | 240 | 97.1% |
| T4 | 110 | 820 | 40 | 5 | 280 | 40 | 100.0% |
| T5 | 92 | 686 | 44 | 5 | 412 | 44 | 100.0% |
| Mean | 108.4 | 809.2 | 92.8 | 4.8 | 286.4 | 88.8 | 99.4% |

The agent reaches its best layout in 4 or 5 actions and then keeps it for the rest of the 40 steps: the last layout of each episode equals the best one. Local search stops in a local optimum on four of the five demands and obtains 72.0%.

**Actions chosen by the agent.**

| Demand | Actions, in order |
|---|---|
| T1 | Rotate W1 counter-clockwise twice, rotate W2 counter-clockwise twice |
| T2, T4, T5 | Swap W1 and W4, rotate W1 counter-clockwise, rotate W3 counter-clockwise, rotate W2 counter-clockwise, rotate W4 clockwise |
| T3 | Rotate W2 clockwise, rotate W1 counter-clockwise, set W2 to L', swap W3 and W4, set W4 to L' |

Workstation names are those of the canonical frame, so a swap that involves W1 can re-frame the whole layout.

**Layouts found.** Coordinates are (x, y); sites 1 to 4 are top-left, top-right, bottom-right, bottom-left; the middle aisle is the line y = 6. Pictures are in `runs/m4/layouts/`.

| Demand | W1 | W2 | W3 | W4 | Shared spots |
|---|---|---|---|---|---|
| T1 | Site 1, U: P (2, 6), D (1, 6) | Site 2, U: P (5, 6), D (4, 6) | Site 3, U: P (4, 6), D (5, 6) | Site 4, U: P (1, 6), D (2, 6) | P1 = D4, D1 = P4, P2 = D3, D2 = P3 |
| T2, T4, T5 | Site 1, U: P (2, 6), D (1, 6) | Site 3, U: P (4, 6), D (5, 6) | Site 4, U: P (1, 6), D (2, 6) | Site 2, U: P (5, 6), D (4, 6) | P1 = D3, D1 = P3, P2 = D4, D2 = P4 |
| T3 | Site 1, U': P (1, 6), D (2, 6) | Site 4, L': P (3, 4.5), D (1.5, 6) | Site 2, U': P (3, 8), D (3, 7) | Site 3, L': P (4.5, 6), D (3, 4.5) | P2 = D4 |

In T1 the flows W1 to W4, W2 to W3 and W4 to W1 travel a distance of zero; only W1 to W2 and W3 to W4 travel 2 units each, which gives 20 x 2 + 20 x 2 = 80.

![Layout found for T2](runs/m4/layouts/T2.png)

**Robustness over seeds and settings.** Three independent trainings per setting, 400,000 steps each, one greedy episode per demand.

| Setting | Score on T1 to T5 | Optimal cases | Score on perturbed demands | Optimal cases |
|---|---|---|---|---|
| New reward, default setup | 99.5% ± 0.1 | 12 of 15 | 99.3% ± 0.2 | 12 of 15 |
| Best-so-far reward | 99.4% ± 0.0 | 12 of 15 | 99.1% ± 0.1 | 12 of 15 |
| Reward of the paper | 68.7% ± 6.3 | 0 of 15 | 69.4% ± 5.0 | 0 of 15 |
| New reward, no canonical mapping | 99.0% ± 0.1 | 12 of 15 | 98.8% ± 0.1 | 12 of 15 |
| New reward, no lookahead input | 97.5% ± 0.6 | 5 of 15 | 97.6% ± 0.7 | 5 of 15 |

The value after ± is the standard deviation over the three seeds. In every seed of the default setup the missed case is one demand out of five.

## 5. 4 workstations: testing phase

On the five perturbed demands, which were never used in training, the same policy reaches the true optimum on four and misses it on T3_P (227 against 210). The mean score is 99.4%, the same as on the training demands.

**Test data.** T1_P to T5_P are the perturbed scenarios of the paper (Appendix A). Each keeps the flow pattern of its training demand with different quantities. The largest change is W3 to W4 in T1 (20 to 9); in T5 the flow W4 to W2 disappears (8 to 0).

**Per demand (same model, one greedy episode).**

| Demand | Total flow | Initial cost | PPO agent | Steps to its best layout | Local search | True optimum | Score |
|---|---|---|---|---|---|---|---|
| T1_P | 114 | 858 | 96 | 4 | 96 | 96 | 100.0% |
| T2_P | 87 | 694 | 44 | 5 | 304 | 44 | 100.0% |
| T3_P | 105 | 824 | 227 | 6 | 258 | 210 | 97.2% |
| T4_P | 107 | 804 | 36 | 5 | 264 | 36 | 100.0% |
| T5_P | 79 | 610 | 44 | 5 | 360 | 44 | 100.0% |
| Mean | 98.4 | 758.0 | 89.4 | 5.0 | 256.4 | 86.0 | 99.4% |

The agent applies the same action sequence as on the matching training demand. On T3_P it adds one action (rotate W3 counter-clockwise), which moves W3 to P (4, 6), D (5, 6). Running 15 extra sampled episodes per demand did not improve any of the five results. Local search obtains 73.3% on these demands.

## 5b. 4 workstations: demands with new flow patterns

On 20 random demands whose flow patterns the policy never saw, the score falls from 99% to 84.2% with one greedy episode and to 94.0% with the best of 16 episodes. This test was added on 14 November.

**Test data.** 20 demands from the generator (2 to 4 products, each with a random route through 2 to 4 workstations and a quantity from 1 to 40), seed 12345. `python evaluate_ppo.py --model runs/m4/model.pt --random 20 --samples 15` recreates them. The reference is the true optimum.

**Model of the previous sections (trained on T1 to T5 only).**

| Test set | One greedy episode | Best of 16 episodes | Local search |
|---|---|---|---|
| Training demands T1 to T5 | 99.4%, 4 of 5 optimal | 99.7%, 4 of 5 optimal | 72.0% |
| Perturbed demands | 99.4%, 4 of 5 optimal | 99.4%, 4 of 5 optimal | 73.3% |
| 20 demands with new flow patterns | 84.2%, 2 of 20 optimal | 94.0%, 6 of 20 optimal | 77.7% |

On the 20 new demands the mean costs are 833.1 for the initial layout, 252.6 for one greedy episode, 180.6 for the best of 16, 293.2 for local search and 149.3 for the optimum.

The perturbed demands keep the flow pattern of a training demand, so the policy replays a known action sequence. On a new pattern it cannot, and it finds the optimum on 2 demands out of 20 in one episode.

**Does training on random demands help?** Two more models were trained for 600,000 steps on a half-and-half mix of T1 to T5 and random demands. Scores on the 20 new demands:

| Model | One greedy episode | Best of 16 episodes |
|---|---|---|
| Trained on T1 to T5 only | 84.2% | 94.0%, 6 of 20 optimal |
| Mixed training, loaded-cost lookahead (`runs/m4_mixed`) | 85.4% | 94.1%, 8 of 20 optimal |
| Mixed training, total-cost lookahead | 79.7% | 93.7%, 8 of 20 optimal |

Mixed training did not close the gap. All three models keep 99.7% on T1 to T5 with the best of 16. Each row is a single training run, so the differences between rows are not conclusive.

## 6. 9 workstations: training phase

The policy was trained for 1,198,080 steps (16,640 episodes) in 24.6 minutes, on random demands only. The mean cost at the end of an episode fell from 60.5% to 42.4% of the initial cost and was still decreasing slowly when training stopped.

| Item | Value |
|---|---|
| Training data | A new random demand at each episode, so 16,640 different demands; none is reused |
| Demand generator | 4 to 9 products; each has a route through 2 to 4 distinct workstations and a quantity from 1 to 40; flows add up along the routes |
| Generated demands, measured on 5,000 draws | 12.1 flows on average (3 to 23), total flow 269 on average, 8.2 of the 9 workstations involved |
| Workshop | 3 x 3 sites, 99 actions (63 shape and rotation actions, 36 swaps) |
| Observation size | 381 values |
| Episode | 72 steps from the default initial layout |
| Steps and updates | 1,198,080 steps, 585 updates |
| Duration | 1,479 seconds on one CPU core, with other jobs running on the machine |
| Mean episode return | 0.395 after the first update, 0.576 at the end |

![Training curve, 9 workstations](runs/m9/training_curve.png)

Training curve, 12 points along the run (mean of the last 200 episodes). The full log, 585 rows, is `runs/m9/training_log.csv`.

| Steps | Mean episode return | Best layout of the episode, cost / initial | Last layout of the episode, cost / initial |
|---|---|---|---|
| 2,048 | 0.395 | 48.4% | 60.5% |
| 104,448 | 0.432 | 46.3% | 56.8% |
| 208,896 | 0.497 | 42.1% | 50.3% |
| 313,344 | 0.519 | 41.5% | 48.1% |
| 417,792 | 0.531 | 40.0% | 46.9% |
| 520,192 | 0.559 | 38.3% | 44.1% |
| 624,640 | 0.552 | 39.8% | 44.8% |
| 729,088 | 0.575 | 37.7% | 42.5% |
| 833,536 | 0.575 | 38.4% | 42.5% |
| 937,984 | 0.582 | 37.9% | 41.8% |
| 1,040,384 | 0.571 | 38.3% | 42.9% |
| 1,198,080 | 0.576 | 38.3% | 42.4% |

Unlike the 4-workstation case, the curve has not flattened: it goes from 44.1% at 520,000 steps to 42.4% at the end. No optimum is given because the true optimum is unknown at this size. Three example training demands are listed in Appendix B.

## 7. 9 workstations: evaluation phase

No demand is ever repeated in training, so there is no fixed training set to replay. The evaluation is the behaviour of the final policy on the training distribution, plus two comparisons made with the same test protocol as section 8.

**Final policy on training demands** (last 200 training episodes, actions sampled):

| Measure | Value |
|---|---|
| Best layout of the episode, cost / initial cost | 38.3% |
| Last layout of the episode, cost / initial cost | 42.4% |
| Mean episode return | 0.576 |

The gap between the best and the last layout (4.1 points) shows that the policy still leaves its best layout before the episode ends, which did not happen at 4 workstations.

**Effect of the lookahead input and of the reward** (each: one training run, score on the 12 test demands of section 8, best of 16 episodes):

| Training | Steps | Mean test cost | Score |
|---|---|---|---|
| New reward with lookahead (the model of this report) | 1,198,080 | 936.8 | 92.5% |
| Best-so-far reward with lookahead | 1,198,080 | 949.2 | 92.2% |
| New reward without lookahead | 1,998,848 | 1,545.3 | 57.5% |

Without the lookahead input the policy stayed at about 60% of the initial cost on training episodes and did worse than local search on the test demands.

## 8. 9 workstations: testing phase

On 12 random demands never seen in training, the policy obtains 92.5% of the possible cost reduction when the best of 16 episodes is kept, and 74.4% with a single greedy episode. Local search obtains 75.7%.

**Test data.** 12 demands drawn from the same generator with a seed not used in training (12345). They are stored in `transportation_demand/random_M9_test/` and listed in Appendix B. The reference is simulated annealing, best of 2 runs of 20,000 iterations.

**Per demand.**

| Demand | Flows | Total flow | Initial cost | PPO, one greedy episode | PPO, best of 16 | Local search | Simulated annealing | Score, one episode | Score, best of 16 |
|---|---|---|---|---|---|---|---|---|---|
| random1 | 13 | 298 | 2,768 | 1,118 | 928 | 1,119 | 835 | 85.4% | 95.2% |
| random2 | 9 | 203 | 1,588 | 1,078 | 641 | 790 | 441 | 44.5% | 82.6% |
| random3 | 17 | 353 | 3,204 | 1,298 | 1,204 | 1,476 | 989 | 86.0% | 90.3% |
| random4 | 9 | 183 | 1,500 | 830 | 324 | 776 | 216 | 52.2% | 91.6% |
| random5 | 18 | 355 | 3,116 | 1,456 | 1,282 | 1,597 | 1,146 | 84.3% | 93.1% |
| random6 | 16 | 527 | 5,460 | 2,110 | 1,766 | 2,370 | 1,472 | 84.0% | 92.6% |
| random7 | 14 | 350 | 3,128 | 1,390 | 1,338 | 1,486 | 1,000 | 81.7% | 84.1% |
| random8 | 6 | 62 | 652 | 324 | 138 | 165 | 112 | 60.7% | 95.2% |
| random9 | 21 | 512 | 3,648 | 1,852 | 1,812 | 2,052 | 1,856 | 100.2% | 102.5% |
| random10 | 10 | 281 | 2,254 | 1,244 | 930 | 1,307 | 765 | 67.8% | 88.9% |
| random11 | 10 | 303 | 2,526 | 1,120 | 764 | 1,326 | 821 | 82.5% | 103.3% |
| random12 | 6 | 94 | 1,036 | 388 | 114 | 280 | 12 | 63.3% | 90.0% |
| Mean | 12.4 | 293.4 | 2,573.3 | 1,184.0 | 936.8 | 1,228.7 | 805.4 | 74.4% | 92.5% |

Reading:

- A single greedy episode is at the level of local search. It is weakest on demands with few flows (random2, random4, random8, random12), where it scores 44% to 63%.
- Keeping the best of 16 episodes (1 greedy and 15 sampled, 1,152 steps in total) brings every demand to at least 82.6%.
- On random9 and random11 the agent finds a better layout than simulated annealing, hence scores above 100%.
- The 15 sampled episodes are random: a rerun gives slightly different values in the "best of 16" column. The greedy column is deterministic.

**Solve time per demand.** Measured on the 12 test demands, one CPU thread, with no result reused between methods.

| Method | Mean time | Range | Mean cost |
|---|---|---|---|
| PPO, one greedy episode (72 steps) | 0.04 s | 0.03 to 0.12 s | 1,184.0 |
| Local search | 0.22 s | 0.04 to 0.40 s | 1,228.7 |
| PPO, best of 16 episodes (1,152 steps) | 0.83 s | 0.63 to 1.08 s | 940.0 |
| Simulated annealing, 6,577 iterations (same time as PPO best of 16) | 0.83 s | | 931.8 |
| Simulated annealing, one run of 20,000 iterations | 2.52 s | 0.70 to 4.69 s | 847.5 |
| Simulated annealing, best of 2 runs (the reference) | 5.12 s | 1.40 to 8.83 s | 805.4 |

Given the same time, simulated annealing is as good as the best of 16 PPO episodes (931.8 against 940.0). PPO has a time advantage only for a single greedy episode, which is 5 times faster than local search for a slightly lower cost. The best-of-16 cost is 940.0 in this timing run and 936.8 in the table above because the sampled episodes are random.

**Demand of the repository** (`transportation_demand_M9_1.csv`, 14 flows, total flow 210, never seen in training):

| Initial cost | PPO, one greedy episode | PPO, best of 16 | Local search | Simulated annealing | Score, best of 16 |
|---|---|---|---|---|---|
| 1,750 | 640 | 460 | 730 | 425 | 97.4% |

## 8b. 9 workstations: second training with three changes

A second training, run on 14 November with three changes and three seeds, raises the best-of-16 score on the 12 test demands from 91.9% to 94.4%, each averaged over 10 evaluation runs. The single greedy episode does not improve, and simulated annealing is better than PPO in the same time.

**What changed.**

| Change | First training | Second training |
|---|---|---|
| Lookahead input | Loaded-cost change of each action | Total-cost change (loaded and empty travel, empty vehicles kept on their current routes) |
| Parallel environments, minibatches | 16 and 8 | 64 and 16 |
| Demand generator | 4 to 9 products | 2 to 9 products, so sparse demands are as common as dense ones |
| Steps | 1,198,080 | 5,996,544 |
| Seeds | 1 | 3 |

The total-cost lookahead never overstates a gain. On 2,400 layouts, its top-ranked action is the truly best action 87.3% of the time, against 60.8% for the loaded-cost lookahead.

**Training.** Each run took 62.5 minutes with the three runs in parallel (732 updates of 8,192 steps). The curve is flat from about 1.5 million steps: the cost of the last layout of an episode stays between about 35% and 38% of the initial cost. More steps did not help. Full logs: `runs/m9_v2_seed0` to `runs/m9_v2_seed2`.

| Steps | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| 8,192 | 48.9% | 48.3% | 47.9% |
| 753,664 | 39.2% | 39.1% | 39.5% |
| 1,507,328 | 36.7% | 38.1% | 37.2% |
| 3,006,464 | 36.3% | 36.1% | 37.6% |
| 4,505,600 | 36.5% | 35.3% | 35.5% |
| 5,996,544 | 35.8% | 36.3% | 36.4% |

The gap between the best and the last layout of an episode is now about 1 point (it was 4.1 points), so the policy holds its layout. These values cannot be compared with the first training curve because the demand generator changed.

**Testing on the same 12 demands.** All models were evaluated in one run; the reference is the same simulated annealing as before.

| Model | One greedy episode | One sampled episode (mean of 15) | Best of 16 episodes | Mean cost, best of 16 |
|---|---|---|---|---|
| First training | 74.4% | 83.2% | 91.4% | 940.0 |
| Second training, seed 0 | 75.9% | 85.0% | 95.2% | 893.4 |
| Second training, seed 1 | 73.7% | 83.7% | 94.6% | 907.6 |
| Second training, seed 2 | 72.0% | 83.7% | 94.9% | 901.8 |
| Second training, mean ± standard deviation | 73.9% ± 1.6 | 84.1% ± 0.6 | 94.9% ± 0.2 | 900.9 |

- A sampled episode is 9 to 12 points better than the greedy one for every model: the greedy episode gets stuck, and a random choice now and then gets it out.
- The gain of the second training is in the best of 16 only. One episode, greedy or sampled, is where it was.
- Each second-training model matches or beats the simulated annealing reference on 2 of the 12 demands.

**Sampling noise.** The table above is a single evaluation run with the same random numbers for every model. Because the 15 sampled episodes are random, each model was evaluated 10 more times with different random numbers:

| Model | Best of 16, mean of 10 runs | Lowest and highest run | Mean cost |
|---|---|---|---|
| First training | 91.9% ± 0.8 | 90.1% to 93.4% | 940.1 |
| Second training, seed 0 | 95.4% ± 0.8 | 93.5% to 96.5% | 886.6 |
| Second training, seed 1 | 93.5% ± 0.5 | 92.6% to 94.4% | 922.0 |
| Second training, seed 2 | 94.4% ± 1.1 | 92.1% to 95.8% | 908.7 |
| Second training, mean of the three seeds | 94.4% ± 0.8 | | 905.8 |

A single best-of-16 evaluation moves by about 1 point from run to run, and the three training seeds differ by up to 1.9 points. The gain of the second training, about 2.5 points, is larger than both but not by much.

**Best of 16 per demand** (first training against second training, seed 0).

| Demand | First training | Second training, seed 0 | Simulated annealing reference |
|---|---|---|---|
| random1 | 856 | 856 | 835 |
| random2 | 692 | 558 | 441 |
| random3 | 1,202 | 1,257 | 989 |
| random4 | 496 | 324 | 216 |
| random5 | 1,274 | 1,184 | 1,146 |
| random6 | 1,578 | 1,609 | 1,472 |
| random7 | 1,340 | 1,208 | 1,000 |
| random8 | 130 | 108 | 112 |
| random9 | 1,852 | 1,796 | 1,856 |
| random10 | 950 | 849 | 765 |
| random11 | 794 | 926 | 821 |
| random12 | 116 | 46 | 12 |

The largest relative gains are on the sparse demands (random2, random4, random12), which the wider generator was meant to cover. Three demands got worse (random3, random6, random11).

**Solve time, measured again.** The empty-travel solver is now compiled, which speeds up simulated annealing and local search much more than PPO, whose time is mostly spent in the network.

| Method | Mean time per demand | Mean cost | Score |
|---|---|---|---|
| Local search | 0.02 s | 1,228.7 | 75.7% |
| Simulated annealing, 20,000 iterations | 0.35 to 0.40 s | 847.5 | 97.8% |
| PPO second training, best of 16 | 0.59 to 0.61 s | 893.4 to 907.6 | 94.6% to 95.2% |
| Simulated annealing in the same time as PPO (about 32,000 iterations, mean of 3 runs) | 0.58 to 0.62 s | 800.7 to 838.8 | 98.4% to 99.9% |

In the same time, simulated annealing reaches a lower cost than the best of 16 PPO episodes on these 12 demands. The timing table of section 8 was measured before the solver was compiled and is kept for the record.

**What this says about the next step.** More steps and better inputs did not move the single episode, and the curve is flat. The remaining lever is the network itself: one that shares its weights across workstations instead of reading the layout as a flat list. Section 8c reports that test.

## 8c. 9 workstations: weight-sharing network

A network that shares its weights across workstations, trained on 14 November with three seeds, raises the best of 16 episodes on the 12 test demands from about 95% to 99.3% and the single greedy episode from about 75% to 82%, with four times fewer training steps. With 64 episodes it finds cheaper layouts than the simulated annealing reference (101.3%). For the same computing effort it is about 2 points ahead of simulated annealing at 16 episodes and level with it at 64 and 256 episodes.

**What changed.** Only the network. The environment, reward, observation, lookahead, demand generator and PPO settings are those of the second training.

| | Flat network (sections 6 to 8b) | Weight-sharing network |
|---|---|---|
| How it reads the layout | One long list of numbers; every workstation and every pair has its own weights | One token per workstation and one feature vector per pair; the same weights for all |
| Layers | 2 hidden layers of 256 units | 3 attention layers of width 64 with 4 heads |
| Parameters | 352,869 at 9 workstations; grows with the workshop size (738,282 at 16) | 115,574 for any workshop size |
| Training steps | 5,996,544 | 1,499,136 |
| Command-line option | `--network mlp` (default) | `--network graph` |

- Workstation token: its P/D coordinates, the lookahead of its 7 own actions, its total outgoing and incoming demand and flow cost, and the 3 global values (current cost, best cost, time).
- Pair features: the distances, demands and flow costs in both directions and the lookahead of their swap. They steer the attention between two workstations and enter the score of their swap.
- Action scores: the 7 actions of a workstation are scored from its token, a swap from the two tokens and their pair features. "Swap W2 and W5" and "swap W3 and W7" are therefore learned once.

**Check.** `tests/test_networks.py` renames the workstations at random and verifies, for 4 and 9 workstations, that the action scores follow the renaming and that the value is unchanged (largest difference 0.000002, tolerance 0.001). It also verifies that the network has the same number of parameters for 4, 9 and 16 workstations and that both networks train, save and reload.

**Training.** Each run took 70 minutes with the three runs in parallel on 2 processor threads each (183 updates of 8,192 steps, 356 steps per second). The table gives the cost of the last layout of an episode as a share of the initial cost, on the random training demands. Full logs: `runs/m9_graph_seed0` to `runs/m9_graph_seed2`.

| Steps | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| 8,192 | 47.5% | 49.4% | 48.3% |
| 204,800 | 40.9% | 38.4% | 37.0% |
| 401,408 | 34.6% | 35.9% | 34.1% |
| 753,664 | 31.7% | 33.6% | 33.2% |
| 999,424 | 32.6% | 30.8% | 30.6% |
| 1,253,376 | 30.7% | 31.1% | 31.1% |
| 1,499,136 | 30.8% | 31.1% | 30.3% |

The flat network ended at 35.8% to 36.4% after 6 million steps (section 8b, same demand generator, so the values are comparable). The weight-sharing network passes that level at about 400,000 steps and ends 5 points lower. The curve flattens from about 1 million steps.

**Testing on the same 12 demands.** All models were evaluated in one run with the same simulated annealing reference as before. Best of 16 is the mean of 10 evaluation runs with different random numbers, best of 64 the mean of 3; "±" is the spread over those runs. "Reference matched" counts the demands, out of 12, where the agent's layout is at least as cheap as the reference.

| Model | One greedy episode | One sampled episode (mean of 5) | Best of 16 | Mean cost | Reference matched | Best of 64 | Mean cost | Reference matched |
|---|---|---|---|---|---|---|---|---|
| First training (flat) | 74.4% | 83.0% | 92.5% ± 1.0 | 934.0 | 1.8 | 95.2% ± 0.2 | 885.1 | 2.7 |
| Second training, seed 0 (flat) | 75.9% | 84.6% | 95.0% ± 1.2 | 894.5 | 2.9 | 97.9% ± 0.4 | 837.1 | 5.7 |
| Weight-sharing, seed 0 | 83.5% | 89.5% | 99.2% ± 0.8 | 825.6 | 5.8 | 101.2% ± 0.1 | 787.7 | 8.3 |
| Weight-sharing, seed 1 | 84.9% | 90.4% | 99.4% ± 0.4 | 822.1 | 6.3 | 101.8% ± 0.5 | 776.6 | 9.3 |
| Weight-sharing, seed 2 | 77.8% | 88.2% | 99.3% ± 0.4 | 825.6 | 6.3 | 101.0% ± 0.4 | 791.2 | 8.3 |
| Weight-sharing, mean of the three seeds | 82.1% ± 3.1 | 89.4% ± 0.9 | 99.3% ± 0.1 | 824.4 | 6.1 | 101.3% ± 0.3 | 785.2 | 8.6 |

- Best of 16: 99.3% for all three seeds, about 4 points above the flat network. The three seeds agree to 0.2 points, where the flat seeds differed by 1.9.
- Single episode: the greedy episode gains 2 to 9 points and is now above local search (75.7%) for every seed. It is the least stable number: seed 2 is 7 points below seed 1.
- Above 100%: with 64 episodes the agent's layouts are on average cheaper than the reference, and at least as cheap on 8 to 9 of the 12 demands.
- The flat-network rows differ slightly from section 8b (92.5% against 91.9%, 95.0% against 95.4%) because the episodes are now run in one batch, which changes the random numbers. The differences are within the run-to-run spread.

**Best layout cost per demand** (one evaluation run).

| Demand | Flat, second training, seed 0, best of 16 | Weight-sharing, seed 0, best of 16 | Weight-sharing, seed 0, best of 256 | Simulated annealing reference |
|---|---|---|---|---|
| random1 | 856 | 808 | 768 | 835 |
| random2 | 558 | 441 | 441 | 441 |
| random3 | 1,257 | 1,210 | 1,005 | 989 |
| random4 | 324 | 244 | 216 | 216 |
| random5 | 1,184 | 1,152 | 1,108 | 1,146 |
| random6 | 1,609 | 1,404 | 1,261 | 1,472 |
| random7 | 1,208 | 1,047 | 991 | 1,000 |
| random8 | 108 | 104 | 104 | 112 |
| random9 | 1,796 | 1,808 | 1,796 | 1,856 |
| random10 | 849 | 676 | 570 | 765 |
| random11 | 926 | 752 | 716 | 821 |
| random12 | 46 | 24 | 12 | 12 |

The weight-sharing network is better than the flat one on 11 of the 12 demands with 16 episodes. With 256 episodes it is at least as cheap as the reference on 11 of 12, and far cheaper on random6 and random10, which shows how far the reference is from the optimum on some demands.

**Against simulated annealing for the same computing effort.** The processor of this machine ran at two speeds during the measurements (the same work took 0.45 s in one run and 1.07 s in another), so times in seconds are not stable. The comparison below was made inside one run, with simulated annealing given the wall-clock time PPO had just used.

| Budget | Equivalent simulated annealing iterations | Time per demand (fast to slow run) | PPO weight-sharing, 3 seeds | Simulated annealing, same effort |
|---|---|---|---|---|
| Best of 16 | About 22,000 | 0.45 to 1.1 s | 99.3% ± 0.1, cost 824.4 | 97.6% ± 1.0, cost 845.3 (6 runs) |
| Best of 64 | About 71,000 | 1.4 to 3.3 s | 101.3% ± 0.3, cost 785.2 | 100.7% ± 0.9, cost 787.0 (6 runs) |
| Best of 256 | Not timed, about 4 times the line above | Not timed | 102.7% ± 0.3, cost 754.4 (6 runs) | 103.2%, cost 745.9 at 234,000 iterations (3 runs) |

Simulated annealing at other budgets (mean of 3 runs): 94.3% at 11,700 iterations, 97.2% at 23,400, 98.1% at 35,100, 99.0% at 58,500, 102.0% at 117,000 and 103.2% at 234,000.

- Short budget: with 16 episodes PPO is ahead. Simulated annealing needs about 60,000 iterations, 2.5 times the effort, to reach 99%.
- Longer budgets: at 64 and 256 episodes the two are level within the spread of the runs. Simulated annealing improves with more iterations about as fast as PPO with more episodes.
- The flat network takes about 0.6 times the time of the weight-sharing one for the same number of episodes (0.55 to 0.65 s against 1.07 to 1.10 s for 16 episodes in the slow runs), but stays below simulated annealing at equal time, as in section 8b.
- PPO needs 70 minutes of training before the first demand; simulated annealing needs none. The trained policy then serves any new 9-workstation demand without retraining.

## 9. Limits

- 9 workstations, single episode: one greedy episode is only as good as local search with the flat network. With the weight-sharing network it reaches 78% to 85% depending on the seed, still far from the 99% of 16 episodes: the policy needs several tries per demand.
- 9 workstations, seeds: the first training is a single run (seed 0). The second training has three seeds, whose best-of-16 scores range from 93.5% to 95.4% over 10 evaluation runs. The weight-sharing network has three seeds, from 99.2% to 99.4%.
- Against simulated annealing for equal effort: the flat network stays below simulated annealing (95% against 98% to 100%). The weight-sharing network is ahead at 16 episodes (99.3% against 97.6%) and level at 64 and 256 episodes; it does not beat simulated annealing when both are given a long time.
- Solve times in seconds: the processor ran at two speeds during the measurements of section 8c (a factor of about 2.4 between runs), so the comparison there is in equal effort inside one run, not in absolute seconds.
- Test set: all three 9-workstation trainings are compared on the same 12 demands, and the network design was chosen after seeing pilot results on them. The scores are not from a set kept aside until the end.
- Weight-sharing network, other sizes: it was trained on 9 workstations only. Whether a model trained on one size works on another was not tested (the saved model is tied to its number of workstations).
- New flow patterns at 4 workstations: the 99% scores hold for the training demands and their perturbed versions only. On 20 demands with new flow patterns the score is 84.2% in one episode, and training on random demands did not improve it.
- Plateau: the second 9-workstation training is flat from about 1.5 million steps and the weight-sharing one from about 1 million, so more training of the same networks is unlikely to help.
- Separate effect of each change: the three changes of the second 9-workstation training were applied together, so their individual effects are not known.
- Reference at 9 workstations: simulated annealing is not the optimum. With 256 episodes the weight-sharing network finds a cheaper layout on 8 of the 12 demands (for example 570 against 765 on random10), so scores above 100% are possible and the true optima are unknown.
- Dynamic case: restarting from the layout of the previous demand does not work. A policy trained from the initial layout stalls when it starts elsewhere; one test with random starting layouts (600,000 steps) scored 87.7% on the normal case and did not fix it. Rearrangement costs are not modelled.
- Paper comparison: only the T1 optimum (80) matches the paper. The paper reports 130 for T3, while the true optimum in this code is 240. Initial costs also differ because `graph4M.gml` is not the layout of the paper's figure.
- Larger workshops: 16 workstations and more were not trained.

## 10. How to reproduce

From the project folder, with the packages of `requirements_ppo.txt`:

```bash
python make_paper_demands.py
python make_m9_test_demands.py
python train_ppo.py --machines 4 --demands transportation_demand/paper_M4_train --steps 400000 --out runs/m4
python evaluate_ppo.py --model runs/m4/model.pt --demands transportation_demand/paper_M4_perturbed --plots
python compare_settings.py --steps 400000 --seeds 3 --workers 6
python train_ppo.py --machines 9 --random-demands --steps 1200000 --out runs/m9
python evaluate_ppo.py --model runs/m9/model.pt --demands transportation_demand/random_M9_test --samples 15
python evaluate_ppo.py --model runs/m9/model.pt --demands transportation_demand/transportation_demand_M9_1.csv --samples 15
```

The two trained models of the first trainings are in `runs/m4` and `runs/m9` (local only, not in Git).

The second 9-workstation training is `python train_ppo.py --machines 9 --random-demands --n-envs 64 --n-minibatches 16 --steps 6000000 --seed 0 --out runs/m9_v2_seed0`, repeated with seeds 1 and 2. Its three models are in `runs/m9_v2_seed0` to `runs/m9_v2_seed2`.

The weight-sharing training of section 8c is `python train_ppo.py --machines 9 --random-demands --network graph --n-envs 64 --n-minibatches 16 --steps 1500000 --threads 2 --seed 0 --out runs/m9_graph_seed0`, repeated with seeds 1 and 2. Its three models are in `runs/m9_graph_seed0` to `runs/m9_graph_seed2`; `python evaluate_ppo.py --model runs/m9_graph_seed1/model.pt --demands transportation_demand/random_M9_test --samples 15` evaluates one of them, and `--samples 63` gives the best of 64.

Since 14 November the default lookahead is the total-cost one and the default generator draws 2 to 9 products. To repeat the first trainings, add `--lookahead loaded` to their commands. The first 9-workstation training also used the earlier generator (4 to 9 products), so the current code does not reproduce it exactly; its model is kept in `runs/m9`.

## Appendix A. Demands for 4 workstations

Quantities to transport from the pickup of one workstation to the delivery of another. Files: `transportation_demand/paper_M4_train/` and `transportation_demand/paper_M4_perturbed/`.

**Training demands.**

| Demand | From W1 | From W2 | From W3 | From W4 | Total |
|---|---|---|---|---|---|
| T1 | W2: 20, W4: 30 | W3: 20 | W4: 20 | W1: 30 | 120 |
| T2 | W3: 40 | W3: 10, W4: 10 | W1: 40 | none | 100 |
| T3 | W2: 20, W4: 10 | W3: 20, W4: 40 | W4: 20 | W1: 10 | 120 |
| T4 | W3: 30 | W3: 10, W4: 20 | W1: 30 | W2: 20 | 110 |
| T5 | W2: 3, W3: 24, W4: 2 | W3: 3, W4: 23 | W1: 24, W4: 3 | W1: 2, W2: 8 | 92 |

**Testing demands (perturbed).**

| Demand | From W1 | From W2 | From W3 | From W4 | Total |
|---|---|---|---|---|---|
| T1_P | W2: 24, W4: 31 | W3: 22 | W4: 9 | W1: 28 | 114 |
| T2_P | W3: 33 | W3: 11, W4: 10 | W1: 33 | none | 87 |
| T3_P | W2: 16, W4: 11 | W3: 16, W4: 35 | W4: 16 | W1: 11 | 105 |
| T4_P | W3: 29 | W3: 9, W4: 21 | W1: 28 | W2: 20 | 107 |
| T5_P | W2: 3, W3: 23, W4: 2 | W3: 3, W4: 20 | W1: 23, W4: 3 | W1: 2 | 79 |

## Appendix B. Demands for 9 workstations

`A>B: q` means q units from workstation A to workstation B.

**Testing demands** (files in `transportation_demand/random_M9_test/`).

| Demand | Flows |
|---|---|
| random1 | W1>W5: 14, W2>W8: 49, W3>W7: 32, W3>W8: 20, W4>W2: 14, W4>W3: 20, W5>W4: 23, W7>W6: 29, W7>W9: 11, W8>W1: 14, W8>W3: 38, W8>W5: 23, W9>W2: 11 |
| random2 | W1>W2: 5, W2>W8: 11, W4>W6: 28, W5>W2: 11, W5>W4: 28, W6>W7: 38, W6>W8: 22, W8>W9: 22, W9>W6: 38 |
| random3 | W1>W8: 12, W2>W3: 34, W2>W4: 25, W2>W8: 31, W2>W9: 5, W3>W1: 31, W3>W2: 25, W4>W2: 39, W5>W6: 16, W5>W7: 14, W6>W1: 16, W6>W3: 25, W7>W2: 14, W8>W3: 42, W8>W4: 7, W9>W1: 12, W9>W6: 5 |
| random4 | W1>W4: 5, W1>W6: 27, W2>W1: 28, W2>W6: 1, W2>W8: 35, W4>W6: 5, W7>W1: 27, W8>W7: 27, W9>W2: 28 |
| random5 | W1>W2: 12, W1>W7: 10, W1>W9: 2, W2>W4: 38, W2>W5: 12, W4>W1: 10, W4>W3: 14, W4>W5: 28, W4>W8: 14, W5>W3: 16, W5>W4: 14, W6>W1: 31, W6>W2: 38, W7>W6: 31, W8>W1: 2, W8>W3: 14, W9>W4: 38, W9>W7: 31 |
| random6 | W1>W4: 71, W1>W5: 14, W1>W9: 29, W2>W4: 29, W2>W5: 18, W3>W2: 2, W5>W1: 66, W5>W7: 47, W5>W8: 14, W7>W1: 78, W7>W3: 29, W7>W4: 18, W8>W5: 31, W8>W7: 38, W8>W9: 14, W9>W2: 29 |
| random7 | W1>W6: 6, W2>W4: 36, W2>W5: 9, W3>W1: 22, W3>W4: 31, W3>W5: 6, W3>W7: 5, W4>W8: 67, W5>W3: 22, W6>W3: 31, W7>W3: 31, W8>W1: 36, W8>W5: 24, W9>W8: 24 |
| random8 | W1>W7: 9, W4>W1: 9, W4>W2: 17, W7>W4: 17, W7>W9: 6, W8>W7: 4 |
| random9 | W1>W5: 51, W1>W6: 14, W2>W1: 14, W2>W4: 27, W2>W5: 19, W2>W6: 22, W3>W2: 27, W4>W9: 27, W5>W2: 26, W5>W4: 25, W5>W7: 19, W6>W2: 19, W6>W9: 22, W7>W1: 21, W7>W9: 15, W8>W1: 26, W8>W7: 40, W9>W2: 15, W9>W4: 22, W9>W7: 21, W9>W8: 40 |
| random10 | W1>W5: 33, W1>W9: 31, W2>W1: 31, W4>W1: 33, W4>W9: 27, W5>W6: 33, W6>W3: 8, W8>W2: 31, W8>W5: 27, W9>W8: 27 |
| random11 | W1>W4: 40, W1>W5: 15, W3>W4: 24, W4>W6: 40, W4>W9: 24, W5>W1: 40, W5>W8: 15, W7>W1: 33, W8>W4: 15, W9>W7: 57 |
| random12 | W4>W1: 30, W5>W4: 2, W6>W2: 12, W7>W8: 10, W8>W5: 10, W9>W4: 30 |
| M9_1 (repository) | W1>W2: 20, W1>W4: 30, W2>W5: 10, W2>W9: 5, W3>W4: 10, W3>W6: 5, W4>W1: 30, W4>W3: 10, W5>W7: 25, W6>W2: 15, W7>W5: 20, W8>W1: 10, W9>W3: 5, W9>W8: 15 |

**Examples of training demands** (generator seeds 0, 1 and 2; files in `transportation_demand/random_M9_examples/`). The 16,640 demands actually used in training were drawn on the fly and not stored.

| Example | Flows |
|---|---|
| example1 | W1>W5: 20, W1>W7: 35, W1>W9: 27, W3>W5: 19, W3>W9: 1, W4>W7: 28, W4>W9: 35, W5>W4: 28, W5>W6: 19, W7>W4: 35, W7>W6: 21, W7>W9: 66, W9>W3: 27, W9>W4: 1, W9>W6: 38, W9>W7: 16 |
| example2 | W1>W7: 20, W1>W8: 38, W3>W4: 5, W3>W5: 31, W4>W2: 9, W4>W5: 5, W5>W2: 31, W6>W3: 36, W7>W2: 20, W7>W4: 9, W8>W3: 11, W8>W6: 38 |
| example3 | W1>W3: 61, W2>W6: 16, W3>W1: 33, W3>W2: 37, W3>W6: 40, W6>W3: 11, W6>W4: 27, W6>W7: 5, W6>W9: 25, W7>W4: 5, W8>W1: 11, W8>W7: 9, W9>W8: 9 |
