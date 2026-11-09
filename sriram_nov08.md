# Sriram: November 8 deliverables

## Implementation

The existing DQN has a 2-64-64-4 ReLU network, bounded replay, an Adam
optimizer, Huber loss, gradient clipping, epsilon-greedy exploration, and a
target network synchronized every 100 optimizer updates. Tests now verify
exploration, replay ownership/capacity, terminal Bellman targets, parameter
updates, target synchronization, checkpoint loading, and seeded repeatability.

All six training strategies are supported. Training tests exercise each on 6x6
and 10x10 grids, check shielded collision prevention, and verify log accounting.
The value-penalty strategy now correctly counts its value-based replacements.
A dedicated test verifies that replay stores the executed action and penalized
training reward, while the CSV reports the unmodified environment reward.

`train.py --env-config PATH` accepts JSON GridWorld constructor settings.
`evaluate.py` accepts the same option so evaluation can reproduce the training
map. The example `configs/integration_10x10.json` has an unobstructed path along
the top row and right column, ten obstacles, and a 200-step limit. This provides
initial integration; multiple layouts and varied starts remain Sai's work.

The observation remains normalized row/column (two float32 values); actions
remain up/right/down/left (0/1/2/3). Environment interfaces, rewards, and
executed-action replay semantics are unchanged. Truncation continues to end
the replay transition, as in the starter. Policies are specific to their fixed
map and goal: two coordinates do not encode different layouts or goals.

Training saves online-network weights every 50 episodes and at the end.
These are inference checkpoints, not resumable optimizer/replay/RNG snapshots.
All six required intermediate checkpoints come from one uninterrupted run.
Use a separate output directory for each run to avoid overwriting artifacts.

## Reproduction

From the repository directory, install `requirements.txt` in a virtual
environment, then run:

```powershell
python -m pytest -q
python validate_training.py --output results/nov08_sriram
```

The local verification environment is `.venv` with system packages inherited;
use `.\.venv\Scripts\python` in place of `python` to run it on this machine.

The validation command creates:

- `smoke/6x6` and `smoke/10x10`: five episodes for every method, seed 42,
  with final weights, episode CSVs, and settings/metric summaries.
- `adaptive_seed42.csv`, `.pt`, and `_summary.json`: the continuous 300-episode
  6x6 training run, with resolved configuration and library versions.
- `adaptive_seed42_ep50.pt` through `adaptive_seed42_ep300.pt`: six checkpoints.
- `adaptive_seed42.png`: the training curve with a 20-episode moving average.
- `reload_evaluation/`: ten shield-off evaluation episodes per checkpoint.
- `validation.json`: smoke-run and checkpoint-reload results, including a check
  that final weights equal episode-300 weights.

Every training CSV logs episode, environment reward, steps, success, collision,
unsafe proposals, interventions, random/value replacements, and adaptive rate.
The summary additionally records DQN hyperparameters and the environment.

## Interpretation

Local verification on November 8, 2026: **31 tests passed**. All twelve smoke
runs completed (six methods on each grid size, 60 episodes total). All shielded
smoke runs recorded zero collisions. All six adaptive checkpoints loaded and
completed shield-off evaluation; final weights exactly matched episode 300.

The fresh 300-episode adaptive run had 39.33% training success, zero collisions,
average environment reward -3.1593, average length 71.32 steps, and a 55.23%
unsafe-proposal/intervention rate. Every checkpoint had 0% shield-off success
and 100% collision rate on the fixed evaluation map. The final checkpoint
collided after six steps. Loading works, but the learned policy still depends
on the shield; these results do not demonstrate the project's intended
reduction in shield dependence.

These runs verify implementation and artifact generation. Five-episode baseline
smoke runs and one 300-episode seed do not establish comparative performance
or convergence. Checkpoint evaluation uses a fixed map, fixed start, greedy
actions, and no shield, so repeated episodes are identical and are not
independent trials. Multi-seed/multi-layout experiments remain later work.
