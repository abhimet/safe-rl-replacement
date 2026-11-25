"""Evaluate a trained checkpoint on the frozen scenario set (docs/EVALUATION_PROTOCOL.md).

Evaluation is greedy (argmax Q), with no exploration and the SHIELD OFF: the
agent executes its own proposals, so a collision ends the episode. There is one
episode per scenario, each from a forced start cell.

    python evaluate.py --model results/runs/fabr_seed42/fabr_seed42.pt --output results/eval.csv
"""
import argparse
import csv
import json
from pathlib import Path

import torch

from dqn import network_from_weights
from environment import GridWorld
from make_eval_scenarios import V2_PATH
from metrics import summarize

SPLITS = ("train", "dev_inspected", "heldout")
FAMILIES = ("designed", "random")


def load_network(path):
    torch.set_num_threads(1)
    return network_from_weights(torch.load(path, map_location="cpu", weights_only=True))


def rollout(network, env, start, shield=False):
    """Run one greedy episode. Returns the summary row and the trajectory.

    With shield=True an unsafe greedy action is replaced by the highest-Q safe
    action (used only for the demo; the official evaluation is shield-off).
    Trajectory entries are (position, proposed, executed, intervened).
    """
    obs, _ = env.reset(options={"start": start})
    trajectory, reward_sum, unsafe, interventions = [], 0.0, 0, 0
    info, truncated = {"success": False, "collision": False}, False
    for _ in range(env.max_steps):
        with torch.no_grad():
            q = network(torch.as_tensor(obs, dtype=torch.float32))
        proposed = int(q.argmax().item())
        safe = env.safe_actions()
        executed = proposed
        if shield and proposed not in safe and safe:
            executed = max(safe, key=lambda a: float(q[a]))
        unsafe += int(proposed not in safe)
        interventions += int(executed != proposed)
        trajectory.append((env.position, proposed, executed, executed != proposed))
        obs, reward, terminated, truncated, info = env.step(executed)
        reward_sum += reward
        if terminated or truncated:
            break
    trajectory.append((env.position, None, None, False))
    row = {"reward": round(reward_sum, 4), "steps": env.steps, "success": int(info["success"]),
           "collision": int(info["collision"]), "timeout": int(truncated),
           "unsafe_proposals": unsafe, "interventions": interventions}
    return row, trajectory


def load_scenarios(path=V2_PATH, split=None, family=None):
    with open(path, encoding="utf-8") as f:
        spec = json.load(f)
    if split not in (None, *SPLITS):
        raise ValueError(f"split must be one of {SPLITS}")
    if family not in (None, *FAMILIES):
        raise ValueError(f"family must be one of {FAMILIES}")
    chosen = [s for s in spec["scenarios"]
              if (split is None or s["split"] == split)
              and (family is None or s.get("family") == family)]
    if not chosen:
        raise ValueError(f"No scenarios for split={split!r}, family={family!r} in {path}")
    return spec, chosen


def scenario_env(spec, name, scenario_path=V2_PATH):
    """Environment for one scenario layout: a named map, or a hash-checked config file."""
    info = spec.get("layouts", {}).get(name, {})
    if "config" not in info:
        return GridWorld(layout=name, max_steps=spec["max_steps"],
                         observation_mode=spec["observation_mode"])
    from make_eval_scenarios import config_env, sha256
    config_path = Path(scenario_path).parent / info["config"]
    if sha256(config_path) != info["sha256"]:
        raise ValueError(f"{config_path} does not match the sha256 frozen in {scenario_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config_env(config, spec["max_steps"], spec["observation_mode"])


def evaluate(model, scenarios=V2_PATH, output=None, split=None, family=None, verbose=True):
    """Shield-off evaluation on every chosen scenario.

    Returns {"overall": ..., "split=<s>": ..., "family=<f>": ..., "layout=<name>": ...}.
    """
    network = load_network(model)
    spec, chosen = load_scenarios(scenarios, split, family)
    envs, rows = {}, []
    for index, scenario in enumerate(chosen):
        name = scenario["layout"]
        if name not in envs:
            envs[name] = scenario_env(spec, name, scenarios)
            if envs[name].observation_dim != network.input_dim:
                raise ValueError(f"Model expects {network.input_dim} features, but {name} provides "
                                 f"{envs[name].observation_dim}; scenarios need a map-observation model.")
        result, _ = rollout(network, envs[name], tuple(scenario["start"]))
        rows.append({"episode": index + 1, "scenario_id": scenario["id"], "split": scenario["split"],
                     "family": scenario.get("family", ""), "layout": name,
                     "start": str(tuple(scenario["start"])),
                     "marked_start": int(scenario["marked_start"]),
                     "shortest_path": scenario["shortest_path"], **result})

    if output:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    summary = {"model": str(model), "scenarios": str(scenarios),
               "scenario_version": spec["version"], "overall": summarize(rows)}
    for key in ("split", "family", "layout"):
        for value in dict.fromkeys(row[key] for row in rows if row[key]):
            summary[f"{key}={value}"] = summarize(r for r in rows if r[key] == value)
    if verbose:
        print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--scenarios", default=str(V2_PATH))
    p.add_argument("--output", default=None, help="Per-episode CSV (optional)")
    p.add_argument("--split", choices=SPLITS)
    p.add_argument("--family", choices=FAMILIES)
    evaluate(**vars(p.parse_args()))
