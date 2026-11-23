"""10x10 research foundation smoke test, not a final research comparison.
Run: python run_10x10_foundation.py
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from environment import GridWorld

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "results" / "10x10_foundation"
MODEL = OUTPUT / "par_seed42_both.pt"


def run(*arguments):
    command = [sys.executable, str(HERE / arguments[0]), *map(str, arguments[1:])]
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=HERE, check=True)


def preflight():
    environments = {name: GridWorld(layout=name, observation_mode="map")
                    for name in ("rooms", "zigzag", "clutter")}
    for name, env in environments.items():
        assert env.observation_dim == 104, (name, env.observation_dim)
        assert env.distances_to_goal(), name
        for start in env.starts:
            assert start in env.distances_to_goal(), (name, start)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (104,), (name, obs.shape)
        assert np.isfinite(obs).all(), name
        print(f"PASS {name}: {len(env.distances_to_goal())-1} reachable non-goal starts, input 104")
    rooms = environments["rooms"]
    zigzag = environments["zigzag"]
    rooms.reset(options={"start": (0, 0)})
    zigzag.reset(options={"start": (0, 0)})
    assert not np.array_equal(rooms._observation(), zigzag._observation()), \
        "Map observations must distinguish layouts at the same position"
    # Original 6x6 feature contract remains unchanged.
    assert GridWorld(size=6).reset(seed=42)[0].shape == (2,)
    print("PASS no observation aliasing between named maps; 6x6 legacy input intact")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    preflight()
    run("train.py", "--method", "par", "--storage-mode", "both",
        "--backup-mode", "standard", "--size", "10",
        "--observation-mode", "map", "--train-layouts", "rooms", "zigzag",
        "--episodes", "20", "--window", "20", "--seed", "42",
        "--output", OUTPUT)
    if not MODEL.is_file():
        raise RuntimeError(f"Training failed to save model: {MODEL}")
    summary = {}
    # Do NOT evaluate on clutter here: reserve it for the final held-out test.
    for layout in ("rooms", "zigzag"):
        csv_path = OUTPUT / f"{layout}_shield_off.csv"
        run("evaluate.py", "--model", MODEL, "--size", "10",
            "--observation-mode", "map", "--layout", layout,
            "--all-starts", "--output", csv_path)
        with csv_path.open(newline="") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            raise RuntimeError(f"No evaluation rows for {layout}")
        summary[layout] = {"starts": len(rows), "successes": sum(int(r["success"]) for r in rows),
                           "collisions": sum(int(r["collision"]) for r in rows),
                           "role": "train_layout"}
    (OUTPUT / "foundation_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nPASS: 10x10 multi-layout training, checkpoint, and shield-off evaluation all completed")
    print(json.dumps(summary, indent=2))
    print("HOLDOUT: clutter was not evaluated; reserve it for the final test.")
    print("CAUTION: 20 episodes are only a code smoke test. Do not report these as research results.")

if __name__ == "__main__":
    main()
