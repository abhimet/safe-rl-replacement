# Count-Aware FABR — Controlled Safety-Credit Study (22 Nov 2026)

**Project:** Learning-Aware Action Replacement for Reducing Shield Dependence in Reinforcement Learning.

**Status:** Implemented, tested, and measured in three paired seeds. **Not** an established SOTA improvement or statistically confirmed finding. Do not mix these runs with historical model-free or planner-assisted experiments.

## What actually changed

The original FABR samples from safe actions with a feedback-adaptive Boltzmann distribution, where the temperature depends on the recent greedy unsafe-action rate. Its weakness is that high-Q safe actions can become overused when early Q estimates are unreliable, so shield interventions repeatedly collect similar transitions.

The **count-aware FABR** variant adds a training-only state-action visit bonus to the action-selection logits:

```
score(s,a) = beta(u) * z(Q(s,a)) + c * u / sqrt(1 + N(s,a))
beta(u)   = 4 * (1 - u)^2
P(a | s, safe) = softmax(score(s,a))  over safe actions only
```

- `u`: fraction of the last 100 **greedy** action proposals that were unsafe.
- `N(s,a)`: number of times the action actually executed at that training state, including non-interventions; distinguished by training layout and location.
- `z(Q(s,a))`: normalized estimated Q value among available safe actions (0 for a nearly constant vector).
- `c=3`: fixed novelty weight in this pilot (not tuned to Clutter).
- When the greedy policy is often unsafe, unexplored safe transitions receive larger probability. As unsafe proposal rate falls, the count bonus shrinks and Q estimates dominate.
- The replacement table and training shield are not loaded or used for shield-OFF evaluation. The deployed artifact is a regular DQN checkpoint.

**Training support, identical for all three shielded comparators:** `storage_mode=counterfactual` adds a **known, model-derived** terminal collision transition for rejected actions in the deterministic GridWorld; `backup_mode=shield_safe` avoids maximizing over unsafe future actions in Double DQN targets. These features use additional knowledge of the training environment's dynamics. This protocol is **not the same** as the original model-free-only experiment or the separately planned shortest-path-teacher experiment.

## Controlled experiment

- 10×10 Rooms and Zigzag as *only* training layouts; Clutter excluded from training. It was consulted during development, so a separate fixed holdout suite is used for an independent generalization check.
- Paired seeds 42, 43, 44; **5,000 environment interactions per model**; 9 independent training runs.
- Spatial DQN, learning rate 1e-4, discount 0.99, potential shaping scale 4, all-reachable starting positions, same epsilon decay.
- No shortest-path expert, no imitation training, no new procedural maps, no auxiliary safety ranking loss.
- Baselines: Highest-Q replacement and original adaptive Boltzmann FABR with the identical counterfactual learning, architecture and step budget.
- Evaluations: every reachable start, greedy action selection, **shield OFF**, no replacement. Rooms 84 starts, Zigzag 71, Clutter 71 per model.

### Shield-OFF results — full start-position sweeps

| Method | Rooms (252 starts) | Zigzag (213 starts) | Seen total (465 starts) | Held-out Clutter (213 starts) | Collisions |
|---|---:|---:|---:|---:|---:|
| Highest-Q | 49/252 (19.4%) | 34/213 (16.0%) | **83/465 (17.8%)** | **43/213 (20.2%)** | 0 |
| Original FABR | 128/252 (50.8%) | 30/213 (14.1%) | **158/465 (34.0%)** | 25/213 (11.7%) | 0 |
| **Count-aware FABR** | **138/252 (54.8%)** | **56/213 (26.3%)** | **194/465 (41.7%)** | 34/213 (16.0%) | 0 |

The count-aware variant was **+7.7 percentage points** over original FABR and **+23.9 points** over Highest-Q on seen layouts, averaged across seeds. On Clutter it improved on original FABR by **+4.2 points** but trailed Highest-Q by **−4.2 points**. All these differences are pilot descriptive statistics, not statistically significant claims.

### Variation across training seeds (Rooms + Zigzag combined)

| Seed | Highest-Q | Original FABR | Count-aware FABR |
|---|---:|---:|---:|
| 42 | 78/155 (50.3%) | 98/155 (63.2%) | 98/155 (63.2%) |
| 43 | 0/155 (0%) | 1/155 (0.6%) | **91/155 (58.7%)** |
| 44 | 5/155 (3.2%) | **59/155 (38.1%)** | 5/155 (3.2%) |

**Critical interpretation:** Gains are not consistent seed by seed. The count-aware method tied original FABR in seed 42, beat it substantially in seed 43, and underperformed in seed 44. Three paired seeds are too few to demonstrate a robust positive effect, and the observed variance is large. Do not hide this table behind the three-seed mean.

### Shield-OFF Clutter results by seed

| Seed | Highest-Q | Original FABR | Count-aware FABR |
|---|---:|---:|---:|
| 42 | 20/71 | 6/71 | 6/71 |
| 43 | 7/71 | 2/71 | 20/71 |
| 44 | 16/71 | 17/71 | 8/71 |

Most failed episodes exhausted the 100-step limit, rather than colliding. **Zero collisions is not the same as good navigation.**

## Additional fixed holdout suite: 3 NEW unseen mazes

The three new 10×10 maps were **generated and fixed after the algorithm was frozen**, before their model scores were inspected. They use seeds and fixed wall probabilities recorded in `create_holdout_suite.py` and file hashes in `configs/holdout_suite/manifest.json`. **No models were retrained or tuned** using these maps. Each contains 89, 77 and 61 reachable non-goal starts respectively (227 evaluations per seed, 681 per method). This is a stricter generalization check because Clutter was consulted repeatedly during the earlier development process.

| Method | Unseen suite success (681 episodes) | Collision rate | By-seed successes (227 per seed) |
|---|---:|---:|---|
| Highest-Q | 49/681 (**7.2%**) | 0% | 41, 1, 7 |
| Original FABR | 37/681 (**5.4%**) | 0% | 14, 0, 23 |
| **Count-aware FABR** | **62/681 (9.1%)** | 0% | 14, 30, 18 |

The count-aware method has a small descriptive advantage on this fresh suite (+1.9 percentage points vs Highest-Q), but absolute success is **very low**. It cannot support a claim of robust novel-maze navigation or SOTA superiority. The three maps are related random-obstacle environments; they are not independent samples from a broad benchmark distribution.

Raw data: `results/fabr_safety_credit_5k/holdout_suite_results.csv` and each checkpoint's `off_heldout_*.csv`. The fixed map files are under `configs/holdout_suite/`; the runner is `evaluate_holdout_suite.py`.

## What these results support

- The modified algorithm is materially different from the original Boltzmann replacement: it uses feedback-dependent state-action novelty, not merely a renamed Q-greedy action.
- Under a **model-derived safety-credit protocol**, count-aware FABR can improve route learning compared with Highest-Q and the original FABR at shorter training budgets, on average across the three seeds tested.
- The model-free-only historical study had weaker shield-OFF behavior, but its different architecture/supervision/step budgets prevent direct causal attribution.
- **Unproven:** consistency at larger budgets and more seeds; reliable unseen-maze navigation; superiority over published SOTA; transfer to stochastic or partially observed environments.
- **Not reproduced:** the previously uploaded 20,000-step, planner-assisted runs; those are a separate experiment in which Highest-Q and FABR both achieved 155/155 on seen layouts, largely because of the teacher.

## Exactly what to do next (team, not another debugging loop)

1. **Abhiram:** Use `results/fabr_safety_credit_5k/comparison.png` and CSVs for a results/ablation subsection. Explicitly report the weak Clutter performance and strong seed variation. Freeze the new algorithm code. If computational budget permits, repeat the *pre-registered same* protocol at 20k steps; do not tune to the held-out Clutter set.
2. **Sriram:** Run the original model-free six-method suite as a **separate** experiment. The 5k safety-credit protocol is model-derived and cannot silently replace that promised study.
3. **Sai:** Document a literature comparison to ADVICE and tempestpy, then establish if there is a truly shared task where the published implementation can be run. Do not compare published MiniGridSafe percentages numerically with the custom Rooms/Zigzag/Clutter percentages.
4. **All:** Report both shielded training interventions and shield-OFF success/collision/timeout results; clearly state the safety oracle assumptions, model-based counterfactual assumptions, seeds, interactions, and held-out split.

## Reproduction

From the repo root (will **create new results**, not needed if you only want the existing evidence):

```bash
python run_comparison.py --track safety_credit --methods value_based par par_count --steps 5000 --seeds 42 43 44 --output results/reproduction_5k
```

Outputs: `results/reproduction_5k/safety_credit/{per_seed_layout.csv,aggregate.csv}` and method/seed checkpoint folders. All nine already-run checkpoints, training CSVs and shield-off evaluation CSVs are included under `results/fabr_safety_credit_5k/`.

### Research references for literature comparison

- ADVICE: Bethell, Gerasimou, Calinescu, Imrie, *Safe Reinforcement Learning in Black-Box Environments via Adaptive Shielding* (accepted ECAI 2025), https://arxiv.org/abs/2405.18180 . Uses a learned contrastive-autoencoder safety representation and high-value safe replacements; not implemented here.
- Pranger and Könighofer, *tempestpy: Easy-to-Use Shielded Reinforcement Learning* (2026), https://arxiv.org/abs/2606.03804 ; framework with formal shield synthesis and MiniGridSafe, using MaskablePPO in its experiments. We have **not** run a shared benchmark.
- Dawood et al., *A Dynamic Safety Shield for Safe and Efficient Reinforcement Learning of Navigation Tasks* (L4DC 2025), https://proceedings.mlr.press/v283/dawood25a.html . Useful related work on navigation-specific shielding, **not** reproduced.
