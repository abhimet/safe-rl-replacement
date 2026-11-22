"""Full-map shortest-path controller: non-learning upper bound, NOT an RL baseline.

The planner has exact knowledge of the occupancy map at inference, which is
stronger information-processing capability than the Q-networks. Use only as an
environment sanity check and an explicit model-based reference.
"""

import argparse
import json
from collections import deque

from environment import GridWorld, LAYOUTS
from metrics import summarize


def shortest_path_action(env):
    # Deliberately derives the route from the CURRENT observation/map,
    # without querying env.safe_actions or a safety shield.
    obs = env._observation()
    if env.observation_mode != "map":
        raise ValueError("Reference planner needs full map observations")
    size = env.size
    row = round(float(obs[0]) * (size - 1))
    col = round(float(obs[1]) * (size - 1))
    goal = (round(float(obs[2]) * (size - 1)), round(float(obs[3]) * (size - 1)))
    occupied = obs[4:].reshape(size, size)
    distances = {goal: 0}
    queue = deque([goal])
    directions = GridWorld.ACTIONS
    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            cell = (nr, nc)
            if 0 <= nr < size and 0 <= nc < size and not occupied[nr, nc] and cell not in distances:
                distances[cell] = distances[(r, c)] + 1
                queue.append(cell)
    candidates = []
    for action, (dr, dc) in enumerate(directions):
        neighbor = (row + dr, col + dc)
        if neighbor in distances:
            candidates.append((distances[neighbor], action))
    if not candidates:
        raise RuntimeError("No path from observation to goal")
    return min(candidates)[1]


def evaluate_planner(layout="clutter", size=10, all_starts=True, episodes=100):
    env = GridWorld(layout=layout, size=size, observation_mode="map")
    starts = (sorted(set(env.distances_to_goal()) - {env.goal})
              if all_starts else [None] * episodes)
    rows = []
    for index, start in enumerate(starts):
        env.reset(seed=index, options={"start": start} if start is not None else None)
        total = 0.
        info = {"success": False, "collision": False}
        for _ in range(env.max_steps):
            action = shortest_path_action(env)
            _, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        rows.append({"steps": env.steps, "reward": total,
                     "success": int(info["success"]), "collision": int(info["collision"]),
                     "unsafe_proposals": 0, "interventions": 0})
    result = summarize(rows)
    print(json.dumps({"layout": layout, "reference": "exact_full_map_planner_not_RL", **result}, indent=2))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layout", choices=tuple(LAYOUTS), default="clutter")
    p.add_argument("--size", type=int, default=10)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--all-starts", action="store_true")
    evaluate_planner(**vars(p.parse_args()))
