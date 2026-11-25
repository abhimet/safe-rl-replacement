"""Train one method with Double DQN on the 10x10 training maps.

Every method uses the same configuration; only the shield's replacement rule
(replacement.py) differs:

* Training maps: the designed ``rooms`` and ``zigzag`` layouts plus a seeded
  pool of random-obstacle maps (``environment.generate_training_pool``) that
  cannot resemble any evaluation map. A random map is drawn each episode, and
  starts cycle through every reachable cell of that map.
* Observation: agent position, goal position and the 10x10 wall grid (104 inputs).
* Network: SpatialQNetwork, Double DQN, one gradient update per environment step.
* Shielded methods additionally get two training signals from the shield:
  - a collision counterfactual: a blocked proposal is stored as
    (s, proposed, -10, terminal), because in this deterministic grid that is
    exactly what would have happened;
  - a shield-safe bootstrap: the target's next action is chosen among safe actions.
* Potential-based reward shaping from the map's shortest-path distances.
  Shaping preserves the optimal policy and is used for training only.

Usage:
    python train.py --method fabr --steps 50000 --seed 42 --output results/runs/fabr_seed42
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from dqn import DQNAgent
from environment import GridWorld, generate_training_pool
from metrics import summarize
from replacement import METHODS, Replacer

TRAIN_LAYOUTS = ("rooms", "zigzag")
COLLISION_REWARD = -10.0


def potential(distances, position, scale):
    """Phi(s) = -scale * distance_to_goal / max_distance, in [-scale, 0]."""
    return -float(scale) * distances[tuple(position)] / max(distances.values())


def shaping_bonus(distances, position, next_position, done, gamma, scale):
    """F = gamma * Phi(s') - Phi(s), with Phi(terminal) = 0."""
    after = 0.0 if done else potential(distances, next_position, scale)
    return gamma * after - potential(distances, position, scale)


def training_maps(pool_size=100, pool_seed=7):
    """[(name, GridWorld)]: rooms, zigzag and the generated pool."""
    from make_eval_scenarios import evaluation_maps
    avoid = [env.obstacles for env in evaluation_maps().values()]
    maps = [(name, GridWorld(layout=name, observation_mode="map")) for name in TRAIN_LAYOUTS]
    if pool_size:
        maps += generate_training_pool(n=pool_size, seed=pool_seed, avoid=avoid)
    return maps


def train(method="fabr", steps=50000, seed=42, output="results/run", pool_size=100, pool_seed=7,
          window=100, penalty=1.0, lr=1e-4, beta_min=0.0, beta_max=4.0, fixed_beta=None,
          shaping_scale=4.0, epsilon_fraction=0.8, verbose=True):
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")
    if steps < 1:
        raise ValueError("steps must be positive")
    shielded = method != "unshielded"

    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    replace_rng = random.Random(seed + 1)
    map_rng = random.Random(seed + 2)
    start_rng = random.Random(seed + 3)

    maps = training_maps(pool_size, pool_seed)
    distances = [env.distances_to_goal() for _, env in maps]
    start_queues = [[] for _ in maps]

    agent = DQNAgent(input_dim=maps[0][1].observation_dim, seed=seed, lr=lr,
                     backup_mode="shield_safe" if shielded else "standard")
    replacer = Replacer(method, replace_rng, window=window, beta_min=beta_min,
                        beta_max=beta_max, fixed_beta=fixed_beta)

    rows, betas = [], []
    total = 0
    while total < steps:
        index = map_rng.randrange(len(maps))
        name, env = maps[index]
        if not start_queues[index]:
            start_queues[index] = sorted(c for c in distances[index] if c != env.goal)
            start_rng.shuffle(start_queues[index])
        obs, _ = env.reset(options={"start": start_queues[index].pop()})

        ep = dict(reward=0.0, unsafe_proposals=0, greedy_unsafe=0, interventions=0)
        ep_betas = []
        info, terminated, truncated = {"success": False, "collision": False}, False, False
        while not (terminated or truncated) and total < steps:
            epsilon = max(0.05, 1.0 - 0.95 * total / max(1, int(steps * epsilon_fraction)))
            proposed = agent.act(obs, epsilon)
            q = agent.q_values(obs)
            safe = env.safe_actions()
            if method in ("fabr", "fabr_fixed"):
                ep_betas.append(replacer.beta)
            executed, intervened = replacer.choose(proposed, safe, q)

            position = env.position
            next_obs, reward, terminated, truncated, info = env.step(executed)
            total += 1
            # Truncation (time limit) is not a terminal MDP state: keep bootstrapping.
            train_reward = reward + shaping_bonus(distances[index], position, env.position,
                                                  terminated, agent.gamma, shaping_scale)
            if method == "value_penalty" and intervened:
                train_reward -= penalty
            next_safe = env.safe_actions() if shielded and not terminated else None
            agent.remember(obs, executed, train_reward, next_obs, terminated, next_safe)
            if intervened:
                # Known outcome of the blocked proposal in this deterministic grid.
                cf_reward = COLLISION_REWARD + shaping_bonus(
                    distances[index], position, position, True, agent.gamma, shaping_scale)
                agent.remember(obs, proposed, cf_reward, obs, True, None)
            agent.learn()

            ep["reward"] += reward
            ep["unsafe_proposals"] += int(proposed not in safe)
            ep["greedy_unsafe"] += int(int(np.argmax(q)) not in safe)
            ep["interventions"] += int(intervened)
            obs = next_obs

        betas += ep_betas
        rows.append({
            "episode": len(rows) + 1, "env_steps": total, "map": name, "start": str(env.start),
            "reward": round(ep["reward"], 4), "steps": env.steps,
            "success": int(info["success"]), "collision": int(info["collision"]),
            "timeout": int(truncated), "budget_cut": int(not (terminated or truncated)),
            "unsafe_proposals": ep["unsafe_proposals"], "greedy_unsafe": ep["greedy_unsafe"],
            "interventions": ep["interventions"], "epsilon": round(epsilon, 4),
            "beta": round(float(np.mean(ep_betas)), 4) if ep_betas else "",
        })

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    prefix = f"{method}_seed{seed}"
    with (out / f"{prefix}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    torch.save(agent.online.state_dict(), out / f"{prefix}.pt")

    completed = [r for r in rows if not r["budget_cut"]]
    summary = {
        "method": method, "seed": seed, "steps": total, "episodes": len(completed),
        "config": dict(pool_size=pool_size, pool_seed=pool_seed, window=window, penalty=penalty,
                       lr=lr, beta_min=beta_min, beta_max=beta_max, fixed_beta=fixed_beta,
                       shaping_scale=shaping_scale, epsilon_fraction=epsilon_fraction,
                       training_maps=len(maps)),
        "mean_beta": float(np.mean(betas)) if betas else None,
        "training": summarize(completed) if completed else None,
        "torch_version": torch.__version__,
    }
    (out / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    if verbose:
        t = summary["training"] or {}
        print(f"[{method} seed={seed}] {total} steps, {len(completed)} episodes, "
              f"train success={t.get('success_rate', 0):.3f} collision={t.get('collision_rate', 0):.3f} "
              f"interventions/step={t.get('intervention_rate', 0):.3f}"
              + (f" mean beta={summary['mean_beta']:.2f}" if betas else ""), flush=True)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=METHODS, default="fabr")
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results/run")
    p.add_argument("--pool-size", type=int, default=100)
    p.add_argument("--fixed-beta", type=float, default=None, help="Required for fabr_fixed")
    p.add_argument("--beta-max", type=float, default=4.0)
    p.add_argument("--penalty", type=float, default=1.0, help="value_penalty intervention penalty")
    train(**vars(p.parse_args()))
