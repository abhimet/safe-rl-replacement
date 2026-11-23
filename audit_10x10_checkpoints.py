"""Audit existing 10x10 calibration checkpoints WITHOUT retraining.

Place this file beside dqn.py and environment.py. Then run:
    python audit_10x10_checkpoints.py

Do not evaluate the held-out clutter map here.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import torch

from dqn import QNetwork
from environment import GridWorld


def load_policy(path):
    weights = torch.load(str(path), map_location="cpu", weights_only=True)
    input_dim = weights["net.0.weight"].shape[1]
    if input_dim != 104:
        raise ValueError(f"Expected 104-feature map observations; got {input_dim} in {path}")
    network = QNetwork(input_dim=input_dim)
    network.load_state_dict(weights)
    network.eval()
    return network, weights


def parameter_delta(previous, current):
    if previous is None:
        return None
    squared = sum(torch.sum((current[k].float() - previous[k].float()) ** 2).item()
                  for k in current)
    return math.sqrt(squared)


def audit_layout(network, layout):
    env = GridWorld(layout=layout, observation_mode="map")
    distances = env.distances_to_goal()
    starts = sorted(cell for cell in distances if cell != env.goal)
    states = []
    for start in starts:
        env.reset(options={"start": start})
        states.append(env._observation().copy())

    # One Q forward pass for all states. Rollouts use this greedy policy.
    with torch.no_grad():
        q_table = network(torch.as_tensor(__import__("numpy").stack(states),
                                                 dtype=torch.float32)).numpy()

    actions = {cell: int(q.argmax()) for cell, q in zip(starts, q_table)}
    unsafe_states = 0
    improving_states = 0
    gaps = []
    for cell, q in zip(starts, q_table):
        env.reset(options={"start": cell})
        safe = env.safe_actions()
        unsafe_states += actions[cell] not in safe
        a = actions[cell]
        if a in safe:
            dr, dc = env.ACTIONS[a]
            next_cell = (cell[0] + dr, cell[1] + dc)
            improving_states += distances[next_cell] < distances[cell]
        bad = [a for a in range(4) if a not in safe]
        if bad:
            gaps.append(float(q[bad].max() - q[safe].max()))

    successes = collisions = timeouts = 0
    for start in starts:
        env.reset(options={"start": start})
        result = None
        for _ in range(env.max_steps):
            a = actions[env.position]
            _, _, terminated, truncated, result = env.step(a)
            if terminated or truncated:
                if result["success"]:
                    successes += 1
                elif result["collision"]:
                    collisions += 1
                else:
                    timeouts += 1
                break
        else:
            raise RuntimeError("Rollout exceeded max_steps without termination")

    return {
        "layout": layout,
        "starts": len(starts),
        "shield_off_successes": successes,
        "shield_off_success_pct": round(100.0 * successes / len(starts), 2),
        "collisions": collisions,
        "timeouts": timeouts,
        "greedy_unsafe_state_pct": round(100.0 * unsafe_states / len(starts), 2),
        "greedy_shortest_path_progress_state_pct": round(100.0 * improving_states / len(starts), 2),
        "mean_unsafe_minus_safe_q_gap": round(sum(gaps) / len(gaps), 4) if gaps else None,
    }


def oracle_check(layout):
    """Check environment solvability independently of the learned agent."""
    env = GridWorld(layout=layout, observation_mode="map")
    distances = env.distances_to_goal()
    starts = [cell for cell in distances if cell != env.goal]
    for start in starts:
        env.reset(options={"start": start})
        for _ in range(env.max_steps):
            if env.position == env.goal:
                break
            safe = env.safe_actions()
            a = next((a for a in safe if (
                (env.position[0] + env.ACTIONS[a][0],
                 env.position[1] + env.ACTIONS[a][1]) in distances and
                distances[(env.position[0] + env.ACTIONS[a][0],
                           env.position[1] + env.ACTIONS[a][1])] < distances[env.position]
            )), None)
            if a is None:
                raise AssertionError(f"Oracle cannot progress: {layout} {env.position}")
            _, _, terminated, truncated, info = env.step(a)
            if info["collision"] or truncated:
                raise AssertionError(f"Oracle rollout failed: {layout} from {start}")
            if terminated:
                break
        if env.position != env.goal:
            raise AssertionError(f"Oracle failed to reach goal: {layout} from {start}")
    return len(starts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="results/10x10_calibration/neutral_seed42_ep300",
                        help="Path to the completed 300-episode calibration directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    folder = Path(args.folder)
    prefix = f"par_seed{args.seed}_both_allstarts_fixed_beta_pos0"
    final_model = folder / (prefix + ".pt")
    checkpoints = sorted(folder.glob(prefix + "_ep*.pt"),
                         key=lambda p: int(p.stem.rsplit("_ep", 1)[1]))
    if not final_model.exists():
        raise FileNotFoundError(f"Missing final model: {final_model.resolve()}")
    paths = checkpoints + [final_model]
    print("Auditing existing models. NO TRAINING. NO held-out clutter evaluation.", flush=True)
    for layout in ("rooms", "zigzag"):
        print(f"Oracle sanity check {layout}: {oracle_check(layout)} / "
              f"{len(GridWorld(layout=layout).distances_to_goal()) - 1} starts solved", flush=True)
    previous = None
    records = []
    for path in paths:
        stage = "final" if path == final_model else path.stem.rsplit("_ep", 1)[1]
        network, weights = load_policy(path)
        delta = parameter_delta(previous, weights)
        for layout in ("rooms", "zigzag"):
            row = {"checkpoint": stage, "layout": layout,
                   "weights_changed_l2": round(delta, 5) if delta is not None else ""}
            row.update(audit_layout(network, layout))
            records.append(row)
            print(f"{str(stage):>5} | {layout:<6} | success "
                  f"{row['shield_off_successes']}/{row['starts']} "
                  f"({row['shield_off_success_pct']:.1f}%) | greedy-unsafe states "
                  f"{row['greedy_unsafe_state_pct']:.1f}% | Q gap "
                  f"{row['mean_unsafe_minus_safe_q_gap']}", flush=True)
        previous = weights
    out = folder / "checkpoint_audit.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    print(f"\nAUDIT COMPLETE — {out.resolve()}")
    print("Send the table above or checkpoint_audit.csv. Do not retrain yet.")


if __name__ == "__main__":
    main()
