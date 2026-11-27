# Methods

## Problem

A Double DQN agent learns to navigate 10×10 grids to a goal. During training a **shield**
blocks any move that would leave the grid or enter a wall, and executes a safe move instead.
After training the shield is removed. Two design choices decide what the agent learns while
shielded:

1. the **replacement rule**: which safe action is executed instead of the blocked proposal;
2. the **feedback mode**: what the agent learns about the blocked proposal itself.

The question is how these choices affect whether the agent can navigate safely on its own.

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

## Shared training configuration (`train.py`, `configs/final_protocol.json`)

Every method uses the same configuration, budget and seeds. Final values come only from the
frozen protocol file (`configs/final_protocol.json`, frozen 2026-11-26). The table shows the final
values; [TUNING.md](TUNING.md) lists every alternative tried.

| Setting | Final value (v1 value if different) |
|---|---|
| Algorithm | Double DQN, Huber loss, Adam lr 1e-4, γ = 0.99, batch 64, replay 10,000, target sync every 100 updates, gradient clip 10 |
| Budget | 100,000 environment steps (v1: 50,000) |
| Returns | **n-step, n = 3** (v1: 1). An n-step transition stores Σ γ^i r_i and bootstraps with γ^k from the state k ≤ n steps later. Partial returns are flushed at episode end; a timeout still bootstraps |
| Network | `SpatialQNetwork`: the 104 inputs plus one-hot agent and goal planes, the 5×5 wall patch around the agent and the goal offset; two hidden layers of 128. (A `ConvQNetwork` option, below, was tried and not kept) |
| Exploration | ε-greedy, linear from 1.0 to 0.05 over the first 80% of the step budget |
| Reward shaping | Potential-based, F = γΦ(s′) − Φ(s), Φ(terminal) = 0, with **Φ(s) = −4 · d(s)/d_max**. d is the BFS shortest-path distance to the goal around walls (`GridWorld.distances_to_goal`), and d_max is the largest d on that map. It leaves the optimal policy unchanged and is used for training only; evaluation uses the raw reward. (Φ = −d was tried and not kept) |
| Shield-safe bootstrap (shielded methods) | The next-state action in the target is chosen only among safe actions |

The unshielded baseline has no shield, so it gets no shield-derived signal. All shielded methods
share the exact shield and the distance-based shaping. Only `cf` adds privileged outcome
information (below). The setup is therefore not purely model-free.

## Feedback modes (`train.py --feedback-mode`)

When the ε-greedy proposal a_prop is unsafe and the shield executes a_exec:

| Mode | Stored in replay | Source |
|---|---|---|
| `none` | (s, a_exec, r, s') | the agent never observes the outcome of an unsafe action |
| `cf` | (s, a_exec, r, s') **and** (s, a_prop, −10, terminal) | Alshiekh et al. (2018), post-posed option 1. **ORACLE:** it uses the outcome of an action that was never executed. That outcome is exact here only because the grid is deterministic |
| `credit` | (s, a_prop, r, s') **instead of** the executed transition; r and s' come from executing a_exec | Alshiekh et al. (2018), option 2 (the proposal's "credit leak"). Q(s, a_prop) learns the value of the replacement, so the replacement rule directly shapes what the blocked action is worth |
| `margin` | (s, a_exec, r, s'), with a_prop, an "intervened" flag and the safe-action mask of s | FABR v2's margin loss (below) |

Defaults: `cf` for the original methods (the v1 default), `margin` for `fabr_v2*`, and `none`
for `unshielded`. The legacy `--no-collision-feedback` / `collision_feedback=False` is an alias
for `none`; v1 commands reproduce byte for byte.

## Replacement rules (`replacement.py`)

| Method | Executed action |
|---|---|
| `unshielded` | The proposal itself (no shield; collisions end training episodes) |
| `random_safe` | A uniformly random safe action |
| `highest_q` | The safe action with the highest Q-value |
| `value_penalty` | As `highest_q`, and the training reward is reduced by a penalty whenever the shield intervenes (tuned over {0.25, 0.5, 1, 2, 5, 10}) |
| `adaptive_mix` | `random_safe` with probability *p*, otherwise `highest_q`; *p* = fraction of the last 100 proposals that were unsafe |
| `fabr` | FABR v1 (below) |
| `fabr_fixed` | Ablation: FABR v1's sampling with β fixed at the mean β of the `fabr` runs in the same feedback mode |
| `fabr_v2`, `fabr_v2_fixedtarget` | As `highest_q`; they differ in learning (below) |

### FABR v1 (motivating negative result)

Sample the replacement from a softmax over the z-scored Q-values of the safe actions:

    P(a) ∝ exp(β · (Q(s,a) − mean_safe Q) / std_safe Q),   a ∈ safe(s)
    r = fraction of the last 100 steps on which the greedy action argmax_a Q(s,a) was unsafe
    β = β_min + (β_max − β_min) · (1 − r)²,     β_min = 0, β_max = 4

β = 0 is `random_safe` and β → ∞ is `highest_q`. `fabr_fixed` uses the same rule at the
*average* β, so a difference between the two isolates the adaptation. In v1, no difference was
significant (README, v1 results).

### FABR v2: large-margin loss on blocked proposals

Execution is `highest_q`. Learning adds a margin term to the TD loss, computed on the replay
samples whose proposal was blocked:

    L = L_TD + λ · mean_{intervened samples} relu( Q(s, a_prop) − max_{a ∈ safe(s)} Q(s, a) + m )

- It uses only what the shield knows ("a_prop was blocked in s"), never the outcome of a_prop.
  It is therefore not an oracle.
- The hinge switches itself off once the blocked action is valued at least m below the best safe
  action. After that, the TD loss alone shapes Q, including the values of the safe actions.
- Unsafe actions are never executed in `margin` mode, so Q(s, a_prop) receives no TD target.
  The margin term is its only training signal.
- Gradient flows through both terms by default. `margin_stop_grad` treats the max as a constant,
  and is tuned as a variant.
- Tuned on dev maps: λ ∈ {0.1, 0.5, 1} at m = 1, then m ∈ {0.5, 2}, then stop-grad on the best
  (6 configs, the same budget as `value_penalty`).
- Related to DQfD's large-margin loss (Hester et al., 2018), which pushes non-demonstrated
  actions below the demonstrated one. Here the blocked action is pushed below the best *safe*
  action.

**Ablation `fabr_v2_fixedtarget`.** It has the same target, m, λ and stop-grad as the tuned
`fabr_v2`, but a two-sided squared loss:

    (Q(s, a_prop) − (max_{a ∈ safe(s)} Q(s, a) − m))²

It never switches off: it keeps pulling Q(s, a_prop) to *exactly* m below the best safe action,
pushing it up as well as down. The loss is bounded, so the ablation isolates the hinge's
switch-off behaviour instead of comparing against a diverging loss.

### Options tried in tuning round 2 (not used in the final runs)

- **`ConvQNetwork`** (`--architecture conv`): the observation becomes three 10×10 planes (walls,
  agent, goal). The wall plane is padded with walls so the grid edge looks like a wall. Then come
  two 3×3 convolutions (16 and 32 channels, ReLU) and an MLP head (128) over the 4 actions. The
  safe-action mask is unchanged. On dev maps it lowered highest_q@cf dev success from 29.4% to 8.4%,
  and it trains about 5–8× slower.
- **Φ = −d** (`--shaping distance`): the same BFS distance, without the 4/d_max scaling, so one
  step toward the goal is worth about 1 instead of about 0.2–0.4. On dev maps it lowered dev
  success to 1.8%.

## Tuning and the frozen protocol

- `tune.py` tunes on **development maps only**: the train split (rooms, zigzag = *familiar*)
  and dev_inspected (clutter). It uses seeds 0–1, disjoint from the final seeds 42–51.
  `tune.dev_evaluate` refuses any other split.
- The selection rules are fixed in `tune.py` before any result is seen:
  - **navigation**: highest_q@cf, one change at a time, kept if familiar success rises ≥ 2 pp;
  - **budget**: 100k steps, or 150k if 100k gives < 50% familiar success and 150k adds ≥ 5 pp;
  - **method settings**: the best mean (success − collision) over dev episodes.
- Every config tried is listed in [TUNING.md](TUNING.md).
- **Round 1** (MacBook): navigation (familiar success), budget, and the value_penalty and fabr_v2
  grids.
- **Round 2** (Zaratan): one navigation-improvement round on top of round 1. It tried the conv
  network, then Φ = −d on the winner, keeping a change only if dev success rose ≥ 2 pp. Dev success
  is round 1's definition: familiar + clutter pooled, 226 episodes per seed, mean over seeds.
  Neither change was kept. Round 1's method grids had already run at exactly these settings, so
  they were not re-run.
- The chosen values are written to `configs/final_protocol.json` together with the tiers, the
  seeds, the scenario file's sha256 and the metric list. The file is then frozen.
- `run_experiments.py --protocol` takes no training flags. Each run records the protocol's sha256,
  and a run made under a different protocol is never reused.

## Evaluation (`evaluate.py`, `docs/EVALUATION_PROTOCOL.md`)

Each final checkpoint is run greedily with the **shield off**, once from every reachable start of
8 frozen maps (581 episodes). Every episode ends in exactly one of success, collision or timeout.

| Group | Maps | Episodes | Role |
|---|---|---:|---|
| **Unseen** (headline) | pillars, heldout_1, heldout_2, heldout_3 | 296 | held-out evidence |
| **Unseen, long paths** | spiral | 59 | held-out; reported separately |
| Familiar † | rooms, zigzag | 155 | training maps; used for tuning |
| Clutter † | clutter | 71 | development map; used for tuning |

† Used for tuning: not held-out evidence.

**Reproducibility.** The final runs were trained on UMD's Zaratan cluster.
- Retraining a run there gives a bit-identical CSV, checkpoint and evaluation
  (`cluster/check_determinism.py`).
- `reproduce.py verify` re-evaluates the shipped checkpoints on any machine. All 63,910 episodes
  matched exactly on an M1 Mac.

`run_experiments.py` reports, per group:

- success, collision and timeout rate;
- **mean episode reward** (raw, unshaped);
- **greedy unsafe-state rate**: the fraction of reachable states (= the scenario starts) where
  the greedy action, shield off, would hit a wall or leave the grid;
- the per-step unsafe-proposal rate;
- **training goal-reached rate**: the share of training episodes that reach the goal (shield on,
  ε-greedy, for shielded methods), over all of training and over the last 10% of steps. Comparing
  it with shield-off success separates "never learned to reach the goal" from "learned it, but
  only with the shield".

Each is the mean over seeds with a 95% t-interval. `paired.csv` gives `fabr_v2@margin` minus every
config, and `fabr@<mode>` minus every config of that mode. Differences are paired by seed, with
95% CIs, for success, collision, timeout, reward and greedy unsafe-state rate.
