# FABR: Feedback-Adaptive Boltzmann Replacement for shielded RL

MSML635 final project. Team: Abhiram Metuku, Sai Malkireddy, Sriram Vema.

**Question.** When a safety shield blocks an RL agent's unsafe action, it has to execute *some*
safe action instead. Does the choice of replacement change what the agent learns, and whether it
can navigate safely once the shield is removed?

**FABR (our method).** FABR samples the replacement from a softmax over the Q-values of the safe
actions, with inverse temperature β = 4·(1 − r)², where r is the recent rate at which the agent's
greedy action is unsafe. While the agent still proposes unsafe moves, replacements explore the
safe alternatives. As it learns, they focus on the best safe action. See
[docs/METHODS.md](docs/METHODS.md) for the full definition and
[docs/RELATED_WORK.md](docs/RELATED_WORK.md) for how it differs from shielding and ADVICE.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q tests                      # ~30 s

# 1. Train + evaluate all 7 methods (resumable; rerun the same command after an interruption)
python run_experiments.py --steps 50000 --seeds 42 43 44 45 46 --output results/final

# 2. Figures -> results/final/figures/
python plots.py --results results/final

# 3. Animated demo: the same agents with the shield ON vs OFF on an unseen map
python demo.py --results results/final --methods fabr unshielded --layout pillars --start 9 0
```

To train or evaluate one model:

```bash
python train.py --method fabr --steps 50000 --seed 42 --output results/one
python evaluate.py --model results/one/fabr_seed42.pt
```

## Methods compared

All methods use Double DQN with the same network, budget, seeds and training maps. Only the
replacement rule differs (`replacement.py`).

| Method | When the proposal is unsafe, execute… |
|---|---|
| `unshielded` | the unsafe action (no shield) |
| `random_safe` | a random safe action |
| `highest_q` | the highest-Q safe action |
| `value_penalty` | the highest-Q safe action, with a −1 training penalty per intervention |
| `adaptive_mix` | random or highest-Q, with P(random) = recent unsafe-proposal rate |
| **`fabr`** | **a Boltzmann sample over safe Q-values, with β adapted to the greedy unsafe rate** |
| `fabr_fixed` | ablation: the same sampling with β fixed at FABR's average β |

**Training** uses rooms, zigzag and 100 generated random maps, with every reachable start, so the
agent cannot memorise one map. **Evaluation** is shield-off and greedy, run from every reachable
start of 8 frozen maps (581 episodes). The maps are grouped as familiar (rooms, zigzag), clutter
(development), unseen (pillars, heldout_1–3) and spiral (unseen, reported separately). See
[docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md).

## Results

_Filled in from `results/final/aggregate.csv` after the final run._

## Repository layout

```
environment.py        GridWorld, the named maps, generated training pool
replacement.py        all replacement strategies (incl. FABR)
dqn.py                Double DQN and the Q-networks
train.py              training loop (one configuration for every method)
evaluate.py           shield-off evaluation on the frozen scenario set
run_experiments.py    train + evaluate + aggregate all methods and seeds
plots.py, demo.py     figures and the shield ON/OFF animation
make_eval_scenarios.py, evaluation_scenarios*.json, configs/holdout_suite/   frozen evaluation set
tests/                unit and end-to-end tests
docs/                 methods, evaluation protocol, related work
archive/              earlier experiments, scripts, reports and results (kept for history; not used)
```
