
# Preliminary Findings — Adaptive Action Replacement

## Experiment Setup

- Environment: deterministic 6×6 GridWorld
- Training method: adaptive action replacement
- Training seed: 42
- Training duration: 300 episodes
- Checkpoints: 50, 100, 150, 200, 250, 300
- Evaluation: 100 episodes per checkpoint and shield condition
- Evaluation policy: greedy DQN (no epsilon exploration)
- Adaptive window: 100 recent action proposals

## Preliminary Training Results

- Success rate: 45.67%
- Collision rate: 0%
- Average reward: -2.28
- Average episode length: 68.92 steps
- Unsafe-action proposal rate: 56.61%
- Shield intervention rate: 56.61%

These results are aggregated over the training run and
should not be interpreted as final policy performance.

## Checkpoint Evaluation Results

| Episode | Shield-ON Success | Shield-OFF Success |
|---------|-------------------|--------------------|
| 50      | 0%                | 0%                 |
| 100     | 100%              | 0%                 |
| 150     | 100%              | 0%                 |
| 200     | 0%                | 0%                 |
| 250     | 45%               | 0%                 |
| 300     | 45%               | 0%                 |

## Key Observations

1. Shield-on success fluctuates substantially between
   checkpoints, indicating non-monotonic performance.

2. Shield-off success remains 0% at every checkpoint.
   The learned policies have not demonstrated successful
   independent navigation from the evaluated starting state.

3. The shield prevents collisions during shield-on evaluation,
   but the agents continue proposing unsafe actions frequently.

4. High shield-on success does not necessarily indicate that
   the underlying DQN policy has learned safe behavior.

## Limitations

- The environment uses one fixed starting position and layout.
- Reset seeds currently do not change the environment.
- Shield-off evaluation repeats deterministic trajectories.
- Shield-on evaluation initializes a new adaptive tracker
  each episode rather than restoring training-time history.
- Only one training seed has been evaluated.
- No comparative baseline results have been analyzed yet.
- Results may not generalize to larger or harder environments.

## Preliminary Interpretation

The adaptive replacement method allows the DQN agent
to train without observed collisions. However, the
current results do not establish reduced shield dependence.

The large variation between checkpoint performances,
together with persistent shield-off failures, motivates
further investigation into training stability and
action-replacement learning behavior.

## Next Steps

- Integrate configurable 10×10 environments.
- Add varied, reproducible starting positions.
- Finalize all six experimental methods.
- Evaluate with multiple training seeds.
- Compare adaptive replacement against five baselines.
- Investigate whether replaying executed actions instead
  of originally proposed actions contributes to dependence.
- Report aggregated results and variability across seeds.
