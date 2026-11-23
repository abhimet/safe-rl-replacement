"""Diagnostic: isolate shield-mediated replay attribution in the 10x10 neutral control.

Run beside train.py, evaluate.py, dqn.py, and environment.py:
    python run_10x10_replay_diagnostic.py

The existing both-replay calibration model is never retrained/modified.
The held-out clutter layout is NEVER evaluated.
"""
import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from dqn import QNetwork
from environment import GridWorld

HERE = Path(__file__).resolve().parent
LAYOUTS = ("rooms", "zigzag")


def invoke(script, *args):
    command = [sys.executable, str(HERE / script), *map(str, args)]
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=HERE, check=True)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_summary(model, storage, episodes, seed):
    summary_file = model.with_name(model.stem + "_summary.json")
    training_file = model.with_suffix(".csv")
    if not (model.is_file() and summary_file.is_file() and training_file.is_file()):
        raise FileNotFoundError(f"Missing model, summary, or training CSV: {model}")
    meta = json.loads(summary_file.read_text(encoding="utf-8"))
    expected = {
        "method": "par", "storage_mode": storage, "episodes": episodes,
        "seed": seed, "backup_mode": "standard", "par_fixed_beta": 0.0,
        "train_start_policy": "all_reachable", "input_dim": 104,
    }
    for key, val in expected.items():
        if meta.get(key) != val:
            raise ValueError(f"Reference isn't matched: {key}={meta.get(key)!r}, expected {val!r}")
    env = meta.get("environment", {})
    if env.get("size") != 10 or env.get("observation_mode") != "map" or env.get("train_layouts") != list(LAYOUTS):
        raise ValueError(f"Wrong training environments or observation mode in {summary_file}")
    rows = read_csv(training_file)
    if len(rows) != episodes:
        raise ValueError(f"Expected {episodes} training rows in {training_file}, found {len(rows)}")
    return meta, rows


def policy_audit(model, layout):
    weights = torch.load(str(model), map_location="cpu", weights_only=True)
    if weights["net.0.weight"].shape[1] != 104:
        raise ValueError(f"Unexpected model input features: {model}")
    policy = QNetwork(input_dim=104)
    policy.load_state_dict(weights)
    policy.eval()
    env = GridWorld(layout=layout, observation_mode="map")
    distances = env.distances_to_goal()
    starts = sorted(s for s in distances if s != env.goal)
    obs = []
    for s in starts:
        state, _ = env.reset(options={"start": s})
        obs.append(state)
    with torch.no_grad():
        q = policy(torch.as_tensor(np.stack(obs), dtype=torch.float32)).numpy()
    actions = {s: int(qvalues.argmax()) for s, qvalues in zip(starts, q)}
    unsafe_states = sum(actions[s] not in _safe_at(env, s) for s in starts)
    gaps = []
    for s, values in zip(starts, q):
        safe = _safe_at(env, s)
        unsafe = [a for a in range(4) if a not in safe]
        if unsafe:
            gaps.append(float(values[unsafe].max() - values[safe].max()))
    counts = Counter()
    success = collision = timeout = 0
    for start in starts:
        env.reset(options={"start": start})
        for _ in range(env.max_steps):
            collision_state = env.position
            action = actions[collision_state]
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                if info["success"]:
                    success += 1
                elif info["collision"]:
                    collision += 1
                    counts[collision_state] += 1
                else:
                    timeout += 1
                break
    return {
        "layout": layout,
        "starts": len(starts), "successes": success, "collisions": collision,
        "timeouts": timeout, "success_rate": round(success / len(starts), 4),
        "greedy_unsafe_state_rate": round(unsafe_states / len(starts), 4),
        "mean_unsafe_minus_safe_q_gap": round(float(np.mean(gaps)), 4) if gaps else None,
        "top_collision_states": [
            {"state": list(pos), "failed_starts": n}
            for pos, n in counts.most_common(5)
        ],
    }


def _safe_at(env, cell):
    env.reset(options={"start": cell})
    return env.safe_actions()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-folder", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--retrain-executed", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    reference = (args.reference_folder or
                 HERE / "results" / "10x10_calibration" / f"neutral_seed{args.seed}_ep{args.episodes}")
    root = args.output_root or HERE / "results" / "10x10_replay_diagnostic" / f"seed{args.seed}_ep{args.episodes}"
    root.mkdir(parents=True, exist_ok=True)
    reference_model = reference / f"par_seed{args.seed}_both_allstarts_fixed_beta_pos0.pt"
    print("Comparing executed-only replay against EXISTING both-replay calibration.", flush=True)
    print("Only rooms and zigzag. Held-out clutter is untouched.", flush=True)
    both_meta, both_rows = verify_summary(reference_model, "both", args.episodes, args.seed)
    executed_dir = root / "executed"
    executed_model = executed_dir / f"par_seed{args.seed}_allstarts_fixed_beta_pos0.pt"
    if args.retrain_executed or not executed_model.exists():
        invoke("train.py", "--method", "par", "--storage-mode", "executed",
               "--backup-mode", "standard", "--size", 10, "--observation-mode", "map",
               "--train-layouts", *LAYOUTS, "--train-start-policy", "all_reachable",
               "--episodes", args.episodes, "--window", 100, "--seed", args.seed,
               "--par-fixed-beta", 0, "--output", executed_dir)
    else:
        print(f"Reusing executed-only model: {executed_model}", flush=True)
    executed_meta, executed_rows = verify_summary(executed_model, "executed", args.episodes, args.seed)

    records = []
    for replay, model, meta, rows in (
        ("both", reference_model, both_meta, both_rows),
        ("executed", executed_model, executed_meta, executed_rows),
    ):
        total_steps = sum(int(row["steps"]) for row in rows)
        num_starts = {layout: len({row["train_start"] for row in rows if row["layout"] == layout})
                      for layout in LAYOUTS}
        for layout in LAYOUTS:
            # Standard evaluation produces a separately verifiable CSV; our audit additionally
            # measures greedy unsafe states and exact collision hotspots.
            output_csv = root / f"{replay}_{layout}_shield_off.csv"
            invoke("evaluate.py", "--model", model, "--size", 10,
                   "--observation-mode", "map", "--layout", layout,
                   "--all-starts", "--output", output_csv)
            ev = read_csv(output_csv)
            stats = policy_audit(model, layout)
            if stats["successes"] != sum(int(r["success"]) for r in ev):
                raise RuntimeError(f"Evaluation / independent policy audit mismatch: {replay}, {layout}")
            row = {
                "replay": replay, "layout": layout, "seed": args.seed,
                "episodes_per_training_run": args.episodes,
                "training_environment_steps": total_steps,
                "training_starts_seen_this_layout": num_starts[layout],
                "credited_unsafe_proposals": meta.get("credited_proposals", 0),
                **stats,
            }
            records.append(row)
            hot = ", ".join(f"{x['state']}({x['failed_starts']})" for x in stats["top_collision_states"][:3])
            print(f"{replay:>8} | {layout:<6} | success {stats['successes']:>2}/{stats['starts']} "
                  f"({stats['success_rate']*100:5.1f}%) | unsafe-state "
                  f"{stats['greedy_unsafe_state_rate']*100:5.1f}% | "
                  f"steps {total_steps} | crash hotspots: {hot}", flush=True)

    report = {
        "purpose": "Single-seed replay-attribution diagnostic, not final statistical evidence",
        "train_layouts": list(LAYOUTS), "test_holdout": "clutter (NOT evaluated)",
        "controlled_parameters": "same architecture, layouts, start policy, beta=0, seed, episode count, backup mode, epsilon schedule",
        "limitation": "Same episode count may yield different environment steps; both mode adds extra replay tuples and changes the learned trajectory distribution. Neither arm receives ground-truth rejected-action collision transitions.",
        "results": records,
    }
    out = root / "replay_diagnostic_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nREPLAY DIAGNOSTIC COMPLETE — {out}", flush=True)
    print("No algorithmic superiority claims from one seed. Do NOT test clutter yet.", flush=True)


if __name__ == "__main__":
    main()
