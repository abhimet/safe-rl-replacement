"""Controlled training-reward diagnostic for 10x10 GridWorld.

One command: python run_10x10_reward_check.py

1) Validates potential shaping math on training layouts.
2) Runs a 5-episode smoke test if needed.
3) Trains ONE new shaped 300-episode model with matching seed/algorithm.
4) Compares it against the previously trained sparse-reward counterfactual model.
Only rooms/zigzag are used. Clutter is deliberately untouched.
"""
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEED = 42
EPISODES = 300
LAYOUTS = ("rooms", "zigzag")
SCALE = 4.0


def run(script, *args):
    args = [sys.executable, str(HERE / script), *[str(a) for a in args]]
    print("\n$ " + " ".join(args), flush=True)
    subprocess.run(args, cwd=HERE, check=True)


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def check_model(path, shaping):
    for item in (path, path.with_suffix(".csv"), path.with_name(path.stem + "_summary.json")):
        if not item.is_file():
            raise FileNotFoundError(f"Missing model artifact: {item}")
    info = json.loads(path.with_name(path.stem + "_summary.json").read_text())
    expected = {"method": "par", "storage_mode": "counterfactual", "seed": SEED,
                "episodes": EPISODES, "backup_mode": "standard", "par_fixed_beta": 0.0,
                "train_start_policy": "all_reachable", "input_dim": 104}
    for key, expected_value in expected.items():
        if info.get(key) != expected_value:
            raise ValueError(f"Metadata mismatch {key}: {info.get(key)!r} != {expected_value!r}")
    env = info.get("environment", {})
    if env.get("size") != 10 or env.get("observation_mode") != "map" or env.get("train_layouts") != list(LAYOUTS):
        raise ValueError(f"Wrong environment/holdout split in {path}")
    if info.get("reward_shaping", "none") != shaping:
        raise ValueError(f"Wrong reward shaping in {path}")
    if shaping == "potential" and (not math.isclose(info["shaping_scale"], SCALE) or not math.isclose(info["discount_gamma"], 0.99)):
        raise ValueError("Wrong shaping scale or discount factor")
    rows = read_rows(path.with_suffix(".csv"))
    if len(rows) != EPISODES:
        raise ValueError(f"Expected {EPISODES} episode rows; found {len(rows)}")
    return info, rows


def validate_shaping():
    from environment import GridWorld
    from train import shaping_bonus, potential

    for layout in LAYOUTS:
        env = GridWorld(layout=layout, observation_mode="map")
        dmap = env.distances_to_goal()
        # A guaranteed successful trajectory: take strictly decreasing-distance actions.
        start = max((p for p in dmap if p != env.goal), key=lambda p: dmap[p])
        env.reset(options={"start": start})
        trajectory = []
        for _ in range(env.max_steps):
            old = tuple(env.position)
            a = next(a for a in env.safe_actions()
                     if dmap[(old[0]+env.ACTIONS[a][0], old[1]+env.ACTIONS[a][1])] == dmap[old]-1)
            _, _, terminated, truncated, _ = env.step(a)
            f = shaping_bonus(dmap, old, env.position, terminated or truncated, .99, SCALE)
            trajectory.append(f)
            if terminated or truncated:
                break
        assert env.position == env.goal
        telescoped = sum((.99**t)*f for t, f in enumerate(trajectory))
        expected = -potential(dmap, start, SCALE)
        if not math.isclose(telescoped, expected, abs_tol=1e-6):
            raise AssertionError(f"Success potential shaping isn't telescoping ({layout})")

        # Collision and time-limit termination should have zero final potential.
        for termination in ("collision", "timeout"):
            if termination == "collision":
                env = GridWorld(layout=layout, observation_mode="map")
                env.reset(options={"start": (0, 0)})
                action = next(a for a in range(4) if a not in env.safe_actions())
                start_pos = tuple(env.position)
                _, _, term, trunc, _ = env.step(action)
                assert term and not trunc
                f = shaping_bonus(dmap, start_pos, env.position, term or trunc, .99, SCALE)
                assert math.isclose(f, -potential(dmap, start_pos, SCALE), abs_tol=1e-7)
            else:
                env = GridWorld(layout=layout, observation_mode="map", max_steps=1)
                env.reset(options={"start": (0, 0)})
                action = next(a for a in env.safe_actions()
                              if (env.ACTIONS[a][0], env.ACTIONS[a][1]) != (env.goal[0], env.goal[1]))
                start_pos = tuple(env.position)
                _, _, term, trunc, _ = env.step(action)
                assert trunc and not term
                f = shaping_bonus(dmap, start_pos, env.position, term or trunc, .99, SCALE)
                assert math.isclose(f, -potential(dmap, start_pos, SCALE), abs_tol=1e-7)
        print(f"PASS shaping math ({layout}): success, collision and timeout", flush=True)


def main():
    print("10×10 REWARD CHECK — only rooms + zigzag; clutter remains held out.", flush=True)
    validate_shaping()

    existing = (HERE / "results/10x10_safety_signal/seed42_ep300/"
                "par_seed42_cf_collision_allstarts_fixed_beta_pos0.pt")
    original_info, original_rows = check_model(existing, "none")

    output = HERE / "results/10x10_reward_check/seed42_ep300"
    output.mkdir(parents=True, exist_ok=True)
    shaped = output / "par_seed42_cf_collision_allstarts_phi4_fixed_beta_pos0.pt"
    # First validate the real training/save path, not just the reward equation.
    smoke = HERE / "results/10x10_reward_check/smoke"
    smoke_model = smoke / "par_seed42_cf_collision_allstarts_phi4_fixed_beta_pos0.pt"
    if not smoke_model.is_file():
        run("train.py", "--method", "par", "--storage-mode", "counterfactual",
            "--backup-mode", "standard", "--size", 10, "--observation-mode", "map",
            "--train-layouts", *LAYOUTS, "--train-start-policy", "all_reachable",
            "--episodes", 5, "--window", 20, "--seed", SEED,
            "--par-fixed-beta", 0, "--reward-shaping", "potential",
            "--shaping-scale", SCALE, "--output", smoke)
    if not smoke_model.is_file():
        raise RuntimeError("Smoke train did not save model")
    print("PASS 5-episode smoke train and model save", flush=True)

    if not shaped.is_file():
        run("train.py", "--method", "par", "--storage-mode", "counterfactual",
            "--backup-mode", "standard", "--size", 10, "--observation-mode", "map",
            "--train-layouts", *LAYOUTS, "--train-start-policy", "all_reachable",
            "--episodes", EPISODES, "--window", 100, "--seed", SEED,
            "--par-fixed-beta", 0, "--reward-shaping", "potential",
            "--shaping-scale", SCALE, "--output", output)
    else:
        print(f"Reusing shaped model {shaped}", flush=True)
    shaped_info, shaped_rows = check_model(shaped, "potential")

    results = []
    for label, model, meta, rows in (
        ("original", existing, original_info, original_rows),
        ("potential", shaped, shaped_info, shaped_rows),
    ):
        train_steps = sum(int(row["steps"]) for row in rows)
        for layout in LAYOUTS:
            evaluated = output / f"{label}_{layout}_shield_off.csv"
            run("evaluate.py", "--model", model, "--size", 10,
                "--observation-mode", "map", "--layout", layout,
                "--all-starts", "--output", evaluated)
            data = read_rows(evaluated)
            n = len(data)
            successes = sum(int(row["success"]) for row in data)
            collisions = sum(int(row["collision"]) for row in data)
            timeout = n - successes - collisions
            record = {"reward": label, "layout": layout, "starts": n,
                      "successes": successes, "collisions": collisions, "timeouts": timeout,
                      "success_pct": round(100 * successes / n, 2),
                      "collision_pct": round(100 * collisions / n, 2),
                      "timeout_pct": round(100 * timeout / n, 2),
                      "training_steps": train_steps,
                      "train_success_pct": round(100 * meta["success_rate"], 2),
                      "train_timeout_pct": round(100 * meta.get("timeout_rate", 1-meta["success_rate"]-meta["collision_rate"]), 2)}
            results.append(record)
            print(f"{label:>9} | {layout:>6} | success {successes:>2}/{n} | "
                  f"collisions {collisions:>2} | timeouts {timeout:>2}", flush=True)

    report = {"status": "single-seed diagnostic, NOT final evidence",
              "train_only_layouts": list(LAYOUTS), "holdout_evaluated": False,
              "comparison": "counterfactual replay, fixed beta=0, seed=42, 300 episodes",
              "reward_note": "Potential shaping uses oracle shortest-path distance for TRAINING REWARD only; no change to observations/evaluation",
              "limits": ["different environment-step counts", "single training seed", "only two training layouts"],
              "results": results}
    (output / "reward_diagnostic_report.json").write_text(json.dumps(report, indent=2))
    with (output / "reward_diagnostic_comparison.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print("\nREWARD DIAGNOSTIC COMPLETE", flush=True)
    print(json.dumps(results, indent=2), flush=True)
    print("Send REWARD DIAGNOSTIC COMPLETE. Do not evaluate clutter.")


if __name__ == "__main__":
    main()
