"""Train-only planning teacher for a model-assisted DQN navigation experiment.

This is privileged model-based training: shortest-path distances from the
training layouts are used to supervise every reachable state/action pair.
The deployed student gets only its ordinary observation; no shield or planner.
Do not conflate this variant with model-free DQN baselines.
"""

import numpy as np


def discounted_optimal_return(distance, gamma):
    """Return for taking distance shortest-path moves to goal.

    Intermediate steps give -0.1; the final goal step gives +10.
    """
    if distance < 1:
        raise ValueError("Only non-goal states have a shortest-path return")
    return -0.1 * (1 - gamma ** (distance - 1)) / (1 - gamma) + 10.0 * gamma ** (distance - 1)


def training_teacher(environments, gamma=0.99, shaping_scale=4.0, reward_shaping="none"):
    from train import potential
    observations, targets = [], []
    for env in environments:
        if env.observation_mode != "map":
            raise ValueError("Planner teacher requires map observations")
        distances = env.distances_to_goal()
        for position in sorted(set(distances) - {env.goal}):
            env.position = position
            observation = env._observation()
            values = []
            shift = (-potential(distances, position, shaping_scale)
                     if reward_shaping == "potential" else 0.0)
            for dr, dc in env.ACTIONS:
                nxt = (position[0] + dr, position[1] + dc)
                if not env._inside(nxt) or nxt in env.obstacles:
                    value = -10.0
                elif nxt == env.goal:
                    value = 10.0
                else:
                    # The grid is undirected and all reachable neighboring
                    # states remain in the goal-connected component.
                    value = -0.1 + gamma * discounted_optimal_return(distances[nxt], gamma)
                values.append(value + shift)
            observations.append(observation)
            targets.append(values)
        env.reset()
    return np.asarray(observations, dtype=np.float32), np.asarray(targets, dtype=np.float32)
