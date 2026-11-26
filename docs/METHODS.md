# Methods

## Problem

A Double DQN agent learns to navigate 10×10 grids to a goal. During training a **shield**
blocks any move that would leave the grid or enter a wall, and executes a safe move instead.
After training the shield is removed. The question is how the *choice of replacement action*
affects what the agent learns, and whether it can then navigate safely on its own.

## Environment (`environment.py`)

- Actions: up, right, down, left. Rewards: −10 for a collision (episode ends), +10 for reaching
  the goal (episode ends), −0.1 for any other step. Episodes are limited to 100 steps (a timeout).
- Observation (104 numbers): agent row and column, goal row and column (scaled to [0, 1]), and the
  10×10 wall grid. The agent sees the map, so it can in principle generalise to new maps.
- The shield is exact and identical for all methods: `GridWorld.safe_actions()`.

## Training maps (to avoid memorisation)

| Maps | Count | Notes |
|---|---:|---|
| `rooms`, `zigzag` | 2 | Hand-designed layouts (`environment.LAYOUTS`) |
| Generated pool | 100 | `generate_training_pool(seed=7)`: 10–30% random walls, random goal, ≥30 reachable cells |

A map is drawn uniformly each episode. Within each map, starts cycle through every reachable cell
in shuffled order. A generated map is rejected if its walls overlap any evaluation map's walls by
more than 50% (Jaccard), so no evaluation geometry leaks into training.

## Shared training configuration (`train.py`)

Every method uses exactly this configuration, budget and seeds. Only the replacement rule differs.

| Setting | Value |
|---|---|
| Algorithm | Double DQN, Huber loss, Adam lr 1e-4, γ = 0.99, batch 64, replay 10,000, target sync every 100 updates, gradient clip 10 |
| Network | `SpatialQNetwork`: the 104 inputs plus one-hot agent and goal planes, the 5×5 wall patch around the agent and the goal offset; two hidden layers of 128 |
| Exploration | ε-greedy, linear from 1.0 to 0.05 over the first 80% of the step budget |
| Reward shaping | Potential-based, Φ(s) = −4 · d(s)/d_max with shortest-path distance d. It leaves the optimal policy unchanged and is used for training only; evaluation uses the raw reward |
| Time limit | A timeout is *not* treated as terminal (the target still bootstraps) |
| Shield-derived signals (shielded methods only) | **Collision counterfactual:** a blocked proposal `a` is stored as `(s, a, −10, terminal)`, which is exactly what would have happened in this deterministic grid. **Shield-safe bootstrap:** the next-state action in the target is chosen only among safe actions |

The unshielded baseline gets neither shield-derived signal, because it has no shield. All shielded
methods share the same privileged information: the exact shield, the collision counterfactual and
the distance-based shaping. Comparisons between them are therefore fair, but the setup is not
purely model-free.

## Replacement strategies (`replacement.py`)

When the ε-greedy proposal is unsafe:

| Method | Executed action |
|---|---|
| `unshielded` | The proposal itself (no shield; collisions end training episodes) |
| `random_safe` | A uniformly random safe action |
| `highest_q` | The safe action with the highest Q-value |
| `value_penalty` | As `highest_q`, and the training reward is reduced by 1 whenever the shield intervenes |
| `adaptive_mix` | `random_safe` with probability *p*, otherwise `highest_q`; *p* = fraction of the last 100 proposals that were unsafe |
| **`fabr`** | **Feedback-Adaptive Boltzmann Replacement** (below) |
| `fabr_fixed` | Ablation: the same Boltzmann sampling with β fixed at the mean β the `fabr` runs used |

### FABR

Sample the replacement from a softmax over the z-scored Q-values of the safe actions:

    P(a) ∝ exp(β · (Q(s,a) − mean_safe Q) / std_safe Q),   a ∈ safe(s)

with the inverse temperature set by feedback from the agent's own behaviour:

    r = fraction of the last 100 steps on which the greedy action argmax_a Q(s,a) was unsafe
    β = β_min + (β_max − β_min) · (1 − r)²,     β_min = 0, β_max = 4

- While the policy still prefers unsafe moves (r high), β is low: replacements spread over the safe
  alternatives, so the agent gets experience with actions it would not choose itself.
- As the policy learns (r → 0), β rises: replacement concentrates on the best safe action, as
  `highest_q` does.
- The two ends of the range are existing baselines. β = 0 is `random_safe`, and β → ∞ is `highest_q`.

**Ablation logic.** `fabr_fixed` uses the same sampling rule at the *average* β that FABR actually
used. Any difference between `fabr` and `fabr_fixed` is therefore due to adapting β over time
(the feedback), not to the overall amount of exploration among safe actions.

## Evaluation (`evaluate.py`, `docs/EVALUATION_PROTOCOL.md`)

Each final checkpoint is run greedily with the **shield off**, once from every reachable start of
8 frozen maps (581 episodes). Every episode ends in exactly one of success, collision or timeout.

| Group | Maps | Episodes |
|---|---|---:|
| Familiar | rooms, zigzag | 155 |
| Development (inspected during earlier development) | clutter | 71 |
| Unseen | pillars, heldout_1, heldout_2, heldout_3 | 296 |
| Unseen, long paths (reported separately) | spiral | 59 |

`run_experiments.py` reports success, collision, timeout and unsafe-proposal rates for each group:
the mean over seeds with a 95% t-interval, plus FABR minus each method paired by seed. Training
records the intervention rate and the training collision rate.
