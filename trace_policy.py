
"""Trace a trained DQN's decisions with or without shielding."""
import argparse
import json
import random

import torch

from adaptive import UnsafeRateTracker
from dqn import QNetwork
from environment import GridWorld
from shield import choose_action, METHODS

NAMES = ("UP", "RIGHT", "DOWN", "LEFT")


def trace(
    model,
    method="adaptive",
    shield="off",
    size=6,
    env_config=None,
    seed=10000,
    window=100,
    max_trace_steps=100,
):
    settings = {"size": size}

    if env_config:
        with open(env_config, encoding="utf-8") as f:
            settings.update(json.load(f))

    env = GridWorld(**settings)
    obs, _ = env.reset(seed=seed)

    network = QNetwork()
    network.load_state_dict(
        torch.load(model, map_location="cpu", weights_only=True)
    )
    network.eval()

    rng = random.Random(seed)
    tracker = UnsafeRateTracker(window=window)

    print(f"start={env.start} goal={env.goal}")
    print(f"method={method} shield={shield}")
    print(
        "step | position | proposed | safe_actions | "
        "executed | Q-values | result"
    )

    for t in range(min(env.max_steps, max_trace_steps)):
        pos = env.position

        with torch.no_grad():
            q = network(
                torch.as_tensor(obs, dtype=torch.float32)
            ).flatten().tolist()

        proposed = max(range(4), key=lambda a: q[a])
        safe = env.safe_actions()

        if shield == "on":
            executed, _, _ = choose_action(
                env,
                proposed,
                method,
                q_values=q,
                rng=rng,
                random_probability=tracker.rate,
            )
        else:
            executed = proposed

        tracker.record(proposed not in safe)

        obs, reward, terminated, truncated, info = env.step(
            executed
        )

        outcome = (
            "COLLISION" if info["collision"]
            else "GOAL" if info["success"]
            else "TIMEOUT" if truncated
            else "continue"
        )

        print(
            f"{t+1:4d} | {str(pos):>8} | "
            f"{NAMES[proposed]:>8} | "
            f"{','.join(NAMES[a] for a in safe):>19} | "
            f"{NAMES[executed]:>8} | "
            f"{[round(v, 2) for v in q]} | {outcome}"
        )

        if terminated or truncated:
            break


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--method", choices=METHODS, default="adaptive")
    p.add_argument("--shield", choices=("on", "off"), default="off")
    p.add_argument("--size", type=int, default=6)
    p.add_argument("--env-config")
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--max-trace-steps", type=int, default=100)

    args = p.parse_args()

    if args.shield == "on" and args.method == "unshielded":
        p.error("Unshielded method cannot have shield on")

    trace(**vars(args))
