"""Evaluate greedy learned policy with its training-time shield restored.

This is a separate diagnostic from evaluate.py, which never intervenes.
FABR's adaptive statistics are rebuilt from evaluation trajectories; its
initial rate is not copied from training. No policy optimization occurs here.
"""

import argparse
import csv
import json
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch

from adaptive import UnsafeRateTracker
from dqn import network_from_weights
from environment import GridWorld, LAYOUTS
from metrics import summarize
from shield import choose_action, METHODS

SHIELDED_METHODS = tuple(m for m in METHODS if m != "unshielded") + ("par", "worst_safe")


def evaluate(model, episodes=100, output="results/shield_on.csv", size=6,
             seed=10000, window=100, method="value_based", env_config=None,
             all_starts=False, layout=None, observation_mode="position",
             par_beta_min=-4.0, par_beta_max=4.0, par_schedule="relative",
             par_fixed_beta=None):
    if method not in SHIELDED_METHODS:
        raise ValueError(f"Unsupported shielded method: {method}")
    if episodes < 1 or window < 1:
        raise ValueError("episodes and window must be positive")
    if par_schedule not in ("relative", "absolute"):
        raise ValueError("par_schedule must be relative or absolute")
    if par_beta_min >= par_beta_max:
        raise ValueError("par_beta_min must be smaller than par_beta_max")
    if env_config is not None and layout is not None:
        raise ValueError("Use --layout OR --env-config")

    network = network_from_weights(torch.load(model, map_location="cpu", weights_only=True))
    input_dim = getattr(network, "input_dim", network.net[0].in_features)
    settings = {"size": size, "observation_mode": observation_mode}
    if layout is not None:
        settings["layout"] = layout
    if env_config is not None:
        with open(env_config, encoding="utf-8") as stream:
            settings.update(json.load(stream))
    env = GridWorld(**settings)
    if input_dim != env.observation_dim:
        raise ValueError(f"Model expects {input_dim} features; evaluation environment provides {env.observation_dim}")

    starts = (sorted(set(env.distances_to_goal()) - {env.goal})
              if all_starts else [None] * episodes)
    rows = []
    rng = random.Random(seed)
    tracker = UnsafeRateTracker(window=window)
    greedy_history = deque(maxlen=window)
    initial_rate = None

    for index, start in enumerate(starts):
        obs, _ = (env.reset(seed=seed + index, options={"start": start})
                  if start is not None else env.reset(seed=seed + index))
        reward_sum = 0.0
        unsafe_count = 0
        interventions = 0
        last_beta = None
        info = {"collision": False, "success": False}
        for _ in range(env.max_steps):
            with torch.no_grad():
                q = network(torch.as_tensor(obs, dtype=torch.float32)).flatten().cpu().numpy()
            proposed = int(np.argmax(q))
            safe = env.safe_actions()
            greedy_unsafe = proposed not in safe
            greedy_history.append(greedy_unsafe)
            if initial_rate is None and len(greedy_history) == window:
                initial_rate = float(np.mean(greedy_history))
            if method == "par":
                rate = float(np.mean(greedy_history))
                if par_schedule == "absolute":
                    beta = par_beta_min + (par_beta_max - par_beta_min) * (1 - rate) ** 2
                elif initial_rate is None:
                    beta = par_beta_min
                elif initial_rate == 0:
                    beta = par_beta_max if rate == 0 else par_beta_min
                else:
                    beta = par_beta_min + (par_beta_max - par_beta_min) * (1 - min(rate / initial_rate, 1.0))
                if par_fixed_beta is not None:
                    beta = par_fixed_beta
                last_beta = beta
                if greedy_unsafe:
                    q_safe = np.asarray([q[a] for a in safe], dtype=float)
                    std = q_safe.std()
                    normalized = (q_safe - q_safe.mean()) / std if std >= 1e-6 else np.zeros_like(q_safe)
                    logits = beta * normalized
                    logits -= logits.max()
                    executed = rng.choices(safe, weights=np.exp(logits).tolist(), k=1)[0]
                    intervened = True
                else:
                    executed, intervened = proposed, False
            elif method == "worst_safe":
                executed, intervened = (min(safe, key=lambda a: q[a]), True) if greedy_unsafe else (proposed, False)
            else:
                executed, intervened, _ = choose_action(
                    env, proposed, method, q_values=q, rng=rng,
                    random_probability=tracker.rate,
                )
            tracker.record(greedy_unsafe)
            unsafe_count += int(greedy_unsafe)
            interventions += int(intervened)
            obs, reward, terminated, truncated, info = env.step(executed)
            reward_sum += reward
            if terminated or truncated:
                break
        rows.append({
            "episode": index + 1, "layout": env.layout or "custom",
            "start": str(start if start is not None else env.start),
            "reward": round(reward_sum, 4), "steps": env.steps,
            "success": int(info["success"]), "collision": int(info["collision"]),
            "unsafe_proposals": unsafe_count, "interventions": interventions,
            "par_beta": last_beta,
        })
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    print(json.dumps(summary, indent=2))
    print(f"Shield ON results saved to {target}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--output", default="results/shield_on.csv")
    p.add_argument("--size", type=int, default=6)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--method", choices=SHIELDED_METHODS, default="value_based")
    p.add_argument("--env-config")
    p.add_argument("--all-starts", action="store_true")
    p.add_argument("--layout", choices=tuple(LAYOUTS))
    p.add_argument("--observation-mode", choices=("position", "map"), default="position")
    p.add_argument("--par-beta-min", type=float, default=-4.)
    p.add_argument("--par-beta-max", type=float, default=4.)
    p.add_argument("--par-schedule", choices=("relative", "absolute"), default="relative")
    p.add_argument("--par-fixed-beta", type=float)
    evaluate(**vars(p.parse_args()))
