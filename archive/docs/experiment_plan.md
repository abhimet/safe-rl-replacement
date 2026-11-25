
# MSML635 — Final Experiment Plan

## Project
Learning-Aware Action Replacement for Reducing Shield
Dependence in Reinforcement Learning

## 1. Research Question

Can adaptive action replacement reduce an RL agent's
dependence on a safety shield while maintaining task
success and minimizing unsafe behavior?

## 2. Methods

We will compare six approaches:

1. Unshielded DQN
   - Executes the agent's proposed actions directly.

2. Random-safe replacement
   - Replaces unsafe proposals with randomly selected
     safe actions.

3. Value-based replacement
   - Replaces unsafe proposals with the safe action
     having the highest predicted Q-value.

4. Value-based replacement with intervention penalty
   - Uses value-based replacement and subtracts a fixed
     penalty from the learning reward when intervention
     occurs.

5. Fixed-mixture replacement
   - Uses random-safe or value-based replacement with
     a fixed 50/50 probability.

6. Adaptive replacement (proposed method)
   - Tracks the recent unsafe-action proposal rate.
   - Uses this rate as the probability of selecting
     random-safe replacement.
   - Otherwise uses value-based replacement.

All shielded methods execute the original action
when it is already safe.

## 3. Environment

Preliminary environment:
- 6×6 deterministic GridWorld
- Fixed start and goal
- Maximum 100 steps per episode

Final target environment:
- 10×10 configurable GridWorld
- At least 2–3 reproducible obstacle layouts
- Multiple valid starting positions
- Solvable paths from supported starts to the goal
- Consistent collision, goal, and termination rules

Final layout definitions and obstacle coordinates will
be recorded after environment integration.

## 4. Training Configuration

- Algorithm: Deep Q-Network (DQN)
- Equal training budget across all six methods
- Preliminary budget: 300 episodes
- Final budget: to be frozen by November 15
- Replay buffer and target network: shared DQN settings
- Exploration schedule: identical across methods
- Maximum episode length: 100 steps, unless revised
  consistently for all methods
- Adaptive window: 100 recent action proposals
- Fixed-mixture random replacement probability: 0.5
- Intervention penalty: 1.0 (subject to final review)

Random seeds:
- Initial seed: 42
- Proposed final seeds: 42, 43, 44, 45, 46

These configurations must be finalized before the
main experimental runs.

## 5. Evaluation Protocol

For each method and training seed:

1. Train using identical environment configurations.
2. Save the final model and selected intermediate
   checkpoints.
3. Evaluate the greedy learned policy with shielding.
4. Evaluate the same learned policy without shielding.
5. Use common evaluation scenarios across methods.
6. Use evaluation random seeds disjoint from training.
7. Record per-episode performance.
8. Aggregate metrics across independent training seeds.

The shield-on evaluation must reproduce each method's
replacement strategy. If adaptive controller state is
not restored, a standardized tracker initialization
must be documented.

## 6. Evaluation Metrics

Primary:
- Task success rate
- Collision rate
- Unsafe-action proposal rate
- Shield intervention rate
- Shield-off task success rate

Secondary:
- Mean episode reward
- Mean steps to goal, computed for successful episodes
- Overall episode length
- Training stability across seeds

Unsafe proposal rate and intervention rate will be
computed per action proposal. Denominators and
aggregation conventions must be documented.

## 7. Comparison and Reporting

For each method, report:
- Mean performance across training seeds
- Standard deviation across training seeds
- Shield-on versus shield-off success
- Unsafe proposals and shield interventions
- Learning curves over training
- Performance by environment layout

We will distinguish between:
- Safety achieved through external shield interventions
- Safety learned by the underlying DQN policy

Results from repeated deterministic episodes will not
be treated as independent environment scenarios.

## 8. Reproducibility

Record:
- Git commit hash
- Python and package versions
- Training seed
- Evaluation seed
- Environment layout and starting state
- Hyperparameters
- Checkpoint episode
- Shield configuration
- Executed training and evaluation commands

## 9. Schedule

November 9–12:
- Integrate harder 10×10 environments.

November 13–15:
- Freeze implementation and experiment settings.

November 16–20:
- Run six-method, multi-seed experiments.

November 21–23:
- Complete evaluation, analysis, and figures.

November 24–27:
- Prepare final report.

November 28–29:
- Verify reproducibility and submission package.

## 10. Current Limitations and Risks

- Preliminary shield-off success is 0%.
- Preliminary shield-on performance is unstable.
- Current reset behavior produces identical starts.
- Checkpoints currently store model weights only.
- Shielded training may not produce independently
  safe policies.
- Larger environments may require additional
  training or hyperparameter adjustments.

Any changes to observations, rewards, action encoding,
or replay-learning behavior must be coordinated across
the team before final experiments.
