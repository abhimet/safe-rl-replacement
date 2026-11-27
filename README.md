# FABR: replacement and feedback for shielded RL

MSML635 final project. Team: Abhiram Metuku, Sai Malkireddy, Sriram Vema.

**Question.** When a safety shield blocks an RL agent's unsafe action, it has to execute *some*
safe action instead, and the agent has to learn *something* from the blocked proposal. Do these
choices change what the agent learns, and whether it can navigate safely once the shield is
removed?

- **FABR v1** (the motivating negative result) samples the replacement from a softmax over
  safe Q-values, with β = 4·(1 − r)² set by the greedy unsafe rate r. At 50k steps it does not
  beat simple baselines ([v1 results](#v1-results-50k-steps-motivating)).
- **FABR v2** (`fabr_v2`) executes the highest-Q safe action and adds a large-margin loss on
  the blocked proposal: relu(Q(s,a_prop) − max_safe Q(s,·) + m). It uses only "this action was
  blocked", never the outcome of an action that was not executed.

See [docs/METHODS.md](docs/METHODS.md) for definitions and
[docs/RELATED_WORK.md](docs/RELATED_WORK.md) for how this relates to shielding, DQfD and ADVICE.

## For graders: reproduce in about 35 minutes

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py smoke verify quick
```

| Command | What it does | Time* |
|---|---|---|
| `smoke` | the full test suite (unit + tiny end-to-end) | ~1 min |
| `verify` | re-evaluates every shipped checkpoint in `docs/results/v2/` and re-aggregates; every reported number must match exactly | ~3 min |
| `quick` | retrains the 5 headline configs × 3 seeds from scratch, using the frozen protocol | ~30 min |

\*Measured on an 8-core M1 laptop with 6 parallel jobs. Every command resumes after an
interruption: rerun it.

## Full pipeline

All final settings come from **one frozen file**, [`configs/final_protocol.json`](configs/final_protocol.json).
`run_experiments.py --protocol` rejects any training flag, and every run records the protocol's
sha256.

```bash
python reproduce.py tune                          # dev-map tuning → docs/TUNING.md     (~1–1.5 h)
python reproduce.py core --seeds 42 43 44 45 46   # reported results, batch 1           (~1.5 h)
python reproduce.py core --seeds 47 48 49 50 51   # batch 2 + all fabr_fixed@credit     (~1.9 h)
python reproduce.py full                          # optional: all 21 configs            (~6.5 h)
python reproduce.py figures --results results/core
python reproduce.py ship core                     # copy into docs/results/v2/core
```

One run, by hand:

```bash
python train.py --method fabr_v2 --n-step 3 --steps 100000 --seed 42 --output results/one
python train.py --method highest_q --feedback-mode credit --output results/one
python evaluate.py --model results/one/fabr_v2_seed42.pt
```

## What is compared

**Feedback mode:** what a shielded agent stores about a blocked proposal a_prop (`train.py`).

| Mode | Stored | Notes |
|---|---|---|
| `none` | (s, a_exec, r, s') only | the agent never sees the outcome of an unsafe action |
| `cf` | also (s, a_prop, −10, terminal) | **ORACLE**: uses the outcome of an action that was never executed (Alshiekh et al. 2018, option 1) |
| `credit` | (s, a_prop, r, s') *instead*, with r, s' from the replacement | Alshiekh et al. option 2. Here the replacement rule decides what Q(s, a_prop) learns |
| `margin` | (s, a_exec, r, s') + (s, a_prop) for the margin loss | FABR v2 only |

**Replacement rule:** what the shield executes instead (`replacement.py`).

| Method | When the proposal is unsafe, execute… |
|---|---|
| `unshielded` | the unsafe action (no shield) |
| `random_safe` | a random safe action |
| `highest_q` | the highest-Q safe action |
| `value_penalty` | highest-Q, with a tuned training penalty per intervention |
| `adaptive_mix` | random or highest-Q, with P(random) = recent unsafe-proposal rate |
| `fabr` | FABR v1: a Boltzmann sample over safe Q-values, β adapted to the greedy unsafe rate |
| `fabr_fixed` | ablation of v1: the same sampling with β fixed at FABR's mean β |
| **`fabr_v2`** | **highest-Q, plus the hinge margin loss on the blocked proposal** |
| `fabr_v2_fixedtarget` | ablation of v2: same target and settings, but a two-sided loss (Q(s,a_prop) − (max_safe Q − m))² that never switches off |

**Core tier (the reported results, 11 configs × seeds 42–51).** Each config answers one question:

| Configs | Question |
|---|---|
| `fabr_v2@margin` vs `highest_q@none` | Does the margin loss remove shield dependence? Only the loss differs |
| `fabr_v2@margin` vs `highest_q@cf` | How close does it get to the oracle counterfactual? |
| `value_penalty@none` | The standard non-oracle alternative (penalty tuned with the same budget as fabr_v2) |
| `highest_q`, `random_safe`, `fabr`, `fabr_fixed` `@credit` | The replacement-rule question, where the rule actually changes what is learned |
| `fabr@none` | FABR v1 where its β adapts |
| `fabr_v2_fixedtarget@margin` | Does the hinge's switching off matter? |
| `unshielded` | Lower bound / reference |

The `quick` tier is unshielded, highest_q@none, highest_q@cf, value_penalty@none and
fabr_v2@margin (3 seeds). The `full` tier is all 3 non-margin modes × 6 rules + 2 margin
methods + unshielded (21 configs).

**Training:** Double DQN on rooms, zigzag and 100 generated maps, with every reachable start.
**Evaluation:** shield off, greedy, from every reachable start of 8 frozen maps (581 episodes).
**Tuning:** navigation settings, value_penalty's penalty and fabr_v2's m / λ / stop-grad are
tuned on development maps only: familiar (rooms, zigzag) and clutter, using seeds 0–1. They are
then frozen ([docs/TUNING.md](docs/TUNING.md)). Held-out maps are never evaluated while tuning.
See [docs/EVALUATION_PROTOCOL.md](docs/EVALUATION_PROTOCOL.md).

## Results (v2, frozen protocol)

*Pending: filled in from `docs/results/v2/core/` after the frozen core run.* Each cell will show
success / collision / timeout, mean episode reward and the greedy unsafe-state rate. Unseen
maps are the headline. Familiar and clutter are marked † because they were used for tuning.

| Config | **Unseen** (pillars, heldout 1–3) | **Spiral** (unseen, long paths) | Familiar† | Clutter† |
|---|---|---|---|---|
| … | | | | |

† Used for tuning (navigation settings and method hyperparameters); not held-out evidence.

## v1 results (50k steps, motivating)

7 methods × 10 seeds (42–51) × 50,000 steps, with the pre-tuning settings, in two feedback
settings: `cf` (oracle counterfactual) and `none` (called `no_cf` in these folders). Shield-off,
greedy, 581 frozen scenarios. Cells show **success / collision**, as the mean over 10 seeds.
CIs, per-layout results and paired differences are in [`docs/results/`](docs/results/). No
tuning was done for v1. Clutter was inspected during earlier FABR development.

**With the collision counterfactual (`cf`)**

| Method | **Unseen** | **Spiral** | Familiar | Clutter (dev) |
|---|---|---|---|---|
| Unshielded | 6% / 0% | 3% / 0% | 29% / 0% | 27% / 0% |
| Random safe | 8% / 0% | 1% / 0% | 16% / 0% | 11% / 0% |
| Highest-Q | 8% / 0% | 2% / 0% | 19% / 0% | 19% / 0% |
| Value + penalty | 6% / 0% | 3% / 0% | 13% / 0% | 18% / 0% |
| Adaptive mix | 6% / 0% | 3% / 0% | 13% / 0% | 21% / 0% |
| FABR v1 | 8% / 0% | 2% / 0% | 20% / 0% | 16% / 0% |
| FABR fixed-β | 9% / 0% | 3% / 0% | 19% / 0% | 19% / 0% |

**Without feedback (`none` / `no_cf`)**

| Method | **Unseen** | **Spiral** | Familiar | Clutter (dev) |
|---|---|---|---|---|
| Unshielded | 6% / 0% | 3% / 0% | 29% / 0% | 27% / 0% |
| Random safe | 5% / 71% | 2% / 87% | 14% / 68% | 5% / 80% |
| Highest-Q | 1% / 65% | 1% / 78% | 18% / 69% | 2% / 80% |
| Value + penalty | 1% / 62% | 0% / 86% | 4% / 44% | 2% / 89% |
| Adaptive mix | 2% / 66% | 0% / 88% | 19% / 64% | 6% / 91% |
| FABR v1 | 2% / 75% | 1% / 89% | 10% / 73% | 4% / 84% |
| FABR fixed-β | 1% / 69% | 1% / 83% | 11% / 65% | 3% / 88% |

Any episode that is neither a success nor a collision is a timeout.

What v1 showed, and why v2 exists:

1. **The replacement rule made no significant difference.** Every paired 95% CI on FABR v1's
   success difference against another shielded method contains zero
   ([`paired.csv`](docs/results/no_cf/paired.csv)). Under `cf` and `none`, Q(s, a_prop) never
   learns from the replacement, so the rule has little to act on. The v2 core therefore tests the
   rules under `credit`, where it does.
2. **Feedback about blocked actions decides shield dependence.** Without it, shielded agents hit
   walls in 62–91% of shield-off episodes. With the oracle counterfactual, none do. v2 asks
   whether a non-oracle signal (the margin loss) can close that gap.
3. **β barely adapts under `cf`.** With the counterfactual, the greedy unsafe rate drops to zero
   within about 2,000 steps, so FABR v1's β saturates at 4
   ([figure](docs/results/cf/figures/fabr_beta.png)).
4. **Navigation was weak** (≤ 29% familiar, ≤ 9% unseen success). v2 tunes n-step returns,
   learning rate, replay size, ε schedule and budget on dev maps, then applies one setting to
   every method.

| | |
|---|---|
| ![shield-off outcomes, no_cf](docs/results/no_cf/figures/outcomes.png) | ![demo](docs/results/no_cf/figures/demo_pillars.gif) |

## Limitations

- One environment family (10×10 grids with an exact, deterministic shield). The planned
  stochastic-hazard variant, where `cf`'s "always −10, terminal" would be miscalibrated, was not
  run in this round. It is future work.
- The budget is capped at 100k steps (150k only if tuning shows a clear gain).
- Only navigation, `value_penalty` and `fabr_v2` are tuned. `adaptive_mix` and FABR v1 use
  their v1 settings.
- Tuning uses 2 seeds, so small differences between tuning configs are noise. The selection
  rules are fixed in `tune.py` before any result is seen.

## Repository layout

```
reproduce.py          one entry point: smoke, verify, quick, tune, core, full, figures, ship
configs/final_protocol.json   the frozen final settings, tiers and seeds (the only source for final runs)
environment.py        GridWorld, the named maps, generated training pool
replacement.py        replacement rules, feedback modes
dqn.py                Double DQN, replay buffer, margin loss (FABR v2)
train.py              training loop (feedback modes, n-step returns)
evaluate.py           shield-off evaluation on the frozen scenario set
run_experiments.py    protocol runner: train + evaluate + aggregate + paired comparisons
tune.py               dev-map tuning with fixed selection rules → docs/TUNING.md
plots.py, demo.py     figures and the shield ON/OFF animation
make_eval_scenarios.py, evaluation_scenarios*.json, configs/holdout_suite/   frozen evaluation set
tests/                unit and end-to-end tests
docs/                 methods, evaluation protocol, related work, tuning log; docs/results/ = results
archive/              earlier experiments, scripts, reports and results (kept for history; not used)
```
