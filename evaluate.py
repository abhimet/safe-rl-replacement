import argparse
import csv
import json
from pathlib import Path

import torch

from dqn import network_from_weights
from environment import LAYOUTS, GridWorld
from metrics import summarize


def evaluate(
    model,
    episodes=100,
    output="results/shield_off.csv",
    size=6,
    env_config=None,
    all_starts=False,
    seed=10000,
    layout=None,
    observation_mode="position",
    scenarios=None,
    split=None,
):
    if scenarios is not None:
        return evaluate_scenarios(model, scenarios, output=output, split=split)
    if split is not None:
        raise ValueError("--split requires --scenarios")
    if episodes < 1:
        raise ValueError("episodes must be positive")

    # Small single-state networks are faster with one CPU thread.
    torch.set_num_threads(1)
    weights = torch.load(model, map_location="cpu", weights_only=True)
    network = network_from_weights(weights)
    input_dim = getattr(network, "input_dim", None)
    if input_dim is None:
        input_dim = weights["net.0.weight"].shape[1]

    settings = {"size": size, "observation_mode": observation_mode}
    if layout is not None:
        if env_config is not None:
            raise ValueError("Use --layout OR --env-config, not both")
        settings["layout"] = layout
    if env_config is not None:
        with open(env_config, encoding="utf-8") as f:
            settings.update(json.load(f))

    env = GridWorld(**settings)
    if env.observation_dim != input_dim:
        raise ValueError(
            f"Model expects {input_dim} features, but environment provides "
            f"{env.observation_dim}. Check --observation-mode and --layout."
        )
    rows = []

    starts = (
        sorted(cell for cell in env.distances_to_goal()
           if cell != env.goal)
        if all_starts
        else [None] * episodes
    )

    for episode, start in enumerate(starts):
        if start is None:
            obs, _ = env.reset(seed=seed + episode)
        else:
            obs, _ = env.reset(options={"start": start})
        reward_sum = 0.0
        unsafe = 0

        for _ in range(env.max_steps):
            with torch.no_grad():
                action = int(
                    network(
                        torch.as_tensor(obs, dtype=torch.float32)
                    ).argmax().item()
                )

            unsafe += int(action not in env.safe_actions())

            obs, reward, terminated, truncated, info = env.step(
                action
            )
            reward_sum += reward

            if terminated or truncated:
                break

        rows.append({
            "episode": episode + 1,
            "layout": env.layout or "custom",
            "start": str(start if start is not None else env.start),
            "reward": round(reward_sum, 4),
            "steps": env.steps,
            "success": int(info["success"]),
            "collision": int(info["collision"]),
            "unsafe_proposals": unsafe,
            "interventions": 0,
        })

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize(rows)
    print(json.dumps(summary, indent=2))
    print(f"Shield OFF results saved to {out}")
    return summary


def run_greedy_episode(network, env, start):
    """One shield-off, exploration-free episode from a forced start."""
    obs, _ = env.reset(options={"start": start})
    reward_sum = 0.0
    unsafe = 0
    info = {"success": False, "collision": False}
    truncated = False
    for _ in range(env.max_steps):
        with torch.no_grad():
            action = int(network(torch.as_tensor(obs, dtype=torch.float32)).argmax().item())
        unsafe += int(action not in env.safe_actions())
        obs, reward, terminated, truncated, info = env.step(action)
        reward_sum += reward
        if terminated or truncated:
            break
    return {
        "reward": round(reward_sum, 4),
        "steps": env.steps,
        "success": int(info["success"]),
        "collision": int(info["collision"]),
        "timeout": int(truncated),
        "unsafe_proposals": unsafe,
        "interventions": 0,
    }


def load_scenarios(path, split=None):
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    if split not in (None, "train", "heldout"):
        raise ValueError("split must be train or heldout")
    chosen = [s for s in spec["scenarios"] if split is None or s["split"] == split]
    if not chosen:
        raise ValueError(f"No scenarios for split {split!r} in {path}")
    return spec, chosen


def evaluate_scenarios(model, scenarios, output="results/shield_off.csv", split=None):
    """Evaluate a checkpoint on the frozen scenario set (shield off, greedy)."""
    torch.set_num_threads(1)
    weights = torch.load(model, map_location="cpu", weights_only=True)
    network = network_from_weights(weights)
    network.eval()
    input_dim = getattr(network, "input_dim", None)
    if input_dim is None:
      input_dim = weights["net.0.weight"].shape[1]

    spec, chosen = load_scenarios(scenarios, split)
    envs = {}
    rows = []
    for index, scenario in enumerate(chosen):
        name = scenario["layout"]
        if name not in envs:
            envs[name] = GridWorld(layout=name, max_steps=spec["max_steps"],
                                   observation_mode=spec["observation_mode"])
            if envs[name].observation_dim != input_dim:
                raise ValueError(
                    f"Model expects {input_dim} features, but {name} provides "
                    f"{envs[name].observation_dim}. Scenarios need a {spec['observation_mode']}-mode model.")
        result = run_greedy_episode(network, envs[name], tuple(scenario["start"]))
        rows.append({
            "episode": index + 1,
            "scenario_id": scenario["id"],
            "split": scenario["split"],
            "layout": name,
            "start": str(tuple(scenario["start"])),
            "marked_start": int(scenario["marked_start"]),
            "shortest_path": scenario["shortest_path"],
            **result,
        })

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {"model": str(model), "scenarios": str(scenarios),
               "scenario_version": spec["version"], "overall": summarize(rows)}
    for key in ("split", "layout"):
        for value in dict.fromkeys(row[key] for row in rows):
            summary[f"{key}={value}"] = summarize(r for r in rows if r[key] == value)
    print(json.dumps(summary, indent=2))
    print(f"Shield OFF scenario results saved to {out}")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--output", default="results/shield_off.csv")
    p.add_argument("--size", type=int, default=6)
    p.add_argument("--env-config")
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--all-starts", action="store_true")
    p.add_argument("--layout", choices=tuple(LAYOUTS))
    p.add_argument("--scenarios", help="Frozen evaluation_scenarios.json; overrides layout/start options")
    p.add_argument("--split", choices=("train", "heldout"), help="Only with --scenarios")
    p.add_argument("--observation-mode", choices=("position", "map"), default="position")
    evaluate(**vars(p.parse_args()))
