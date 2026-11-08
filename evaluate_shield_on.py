
"""Greedy evaluation using the adaptive safety shield."""
import argparse
import csv
import json
import random
from pathlib import Path

import torch

from adaptive import UnsafeRateTracker
from dqn import QNetwork
from environment import GridWorld
from metrics import summarize
from shield import choose_action


def evaluate(model, episodes=100, output="results/shield_on.csv",
             size=6, seed=10000, window=100):
    network = QNetwork()
    network.load_state_dict(
        torch.load(model, map_location="cpu", weights_only=True)
    )
    network.eval()

    env = GridWorld(size=size)
    rows = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        rng = random.Random(seed + episode)
        tracker = UnsafeRateTracker(window=window)

        reward_sum = 0.0
        unsafe_count = 0
        interventions = 0

        for _ in range(env.max_steps):
            with torch.no_grad():
                q = network(
                    torch.as_tensor(obs, dtype=torch.float32)
                ).flatten().numpy()

            proposed = int(q.argmax())
            unsafe = proposed not in env.safe_actions()

            executed, intervened, _ = choose_action(
                env,
                proposed,
                "adaptive",
                q_values=q,
                rng=rng,
                random_probability=tracker.rate,
            )

            tracker.record(unsafe)
            unsafe_count += int(unsafe)
            interventions += int(intervened)

            obs, reward, terminated, truncated, info = env.step(
                executed
            )
            reward_sum += reward

            if terminated or truncated:
                break

        rows.append({
            "episode": episode + 1,
            "reward": round(reward_sum, 4),
            "steps": env.steps,
            "success": int(info["success"]),
            "collision": int(info["collision"]),
            "unsafe_proposals": unsafe_count,
            "interventions": interventions,
        })

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    print(json.dumps(summary, indent=2))
    print(f"Shield ON results saved to {out}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--output", default="results/shield_on.csv")
    p.add_argument("--size", type=int, default=6)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--window", type=int, default=100)
    evaluate(**vars(p.parse_args()))
