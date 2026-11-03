# MSML635 Week 1 prototype

Minimal reference implementation for the team's integration milestone. It is **not** the completed research project and has not been benchmarked across seeds.

## Setup

```bash
python -m pip install -r requirements.txt
python test_environment.py
```

## Training

```bash
python train.py --method unshielded --episodes 300 --seed 42
python train.py --method random_safe --episodes 300 --seed 42
```

Outputs are CSV training logs and `.pt` model weights in `results/`.

## Design notes

- Observations contain the agent's normalized row/column, not a map of obstacles. The fixed layout is therefore part of the experiment's assumed known environment configuration, not an explicit model input.
- Collision terminates the episode; a shielded action should avoid collisions whenever at least one safe move exists.
- A shielded transition stores the **executed** action in replay, while unsafe proposal counts are logged separately. This is a methodological choice to analyze, not proof that the agent learns unsafe proposals are bad.
- The random-safe replacement is only the Week 1 integration baseline.
- Reward values, layouts, and hyperparameters are initial test settings, not final tuned choices.
- For the final project, add validation for safe-action edge cases, multiple layouts, fixed-mixture/adaptive methods, shield-off evaluation and multi-seed comparisons.

## Abhiram's Week 1 evaluation tools

After installing requirements, run:

```bash
python test_evaluation.py
python train.py --method random_safe --episodes 300 --seed 42
python evaluate.py --model results/random_safe_seed42.pt --episodes 100 --output results/shield_off.csv
python plot_results.py --csv results/random_safe_seed42.csv --output results/training_plot.png
```

`evaluate.py` runs greedy (epsilon=0) actions with **no shield**. It uses the
same fixed starting position and layout as training, so these results test
shield dependence on the same task, **not** generalization to new layouts.
To test generalization later, add alternate layouts and starts explicitly.

`adaptive.py` is a standalone Week 1 helper. It is **not yet integrated**
into `train.py`. Its rule raises the random-safe probability as the recent
unsafe-proposal rate increases. Test it before integrating it in Week 3.

Metrics distinguish collision rate per episode from unsafe proposal rate per
executed step. With a deterministic fixed start and greedy policy, many
shield-off evaluation episodes may be identical; later vary starts/layouts
and use multiple independently trained seeds for meaningful comparisons.
