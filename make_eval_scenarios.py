"""Generate (or check) the frozen shield-off evaluation scenario set.

One deterministic scenario per reachable, non-goal free cell of every train and
held-out layout. Evaluation is greedy and GridWorld is deterministic, so each
scenario is one episode; `seed` is recorded only as metadata for reproducibility.

    python make_eval_scenarios.py            # write evaluation_scenarios.json
    python make_eval_scenarios.py --check    # fail if the committed file drifted
"""
import argparse
import json
import sys
from pathlib import Path

from environment import HELDOUT_LAYOUTS, LAYOUTS, TRAIN_LAYOUTS, GridWorld

VERSION = 1
FROZEN = "2026-11-22"
MAX_STEPS = 100
OBSERVATION_MODE = "map"
BASE_SEED = 10000
DEFAULT_PATH = Path(__file__).with_name("evaluation_scenarios.json")


def build():
    scenarios = []
    for split, names in (("train", TRAIN_LAYOUTS), ("heldout", HELDOUT_LAYOUTS)):
        for name in names:
            env = GridWorld(layout=name, max_steps=MAX_STEPS, observation_mode=OBSERVATION_MODE)
            distances = env.distances_to_goal()
            marked = set(env.starts)
            for cell in sorted(c for c in distances if c != env.goal):
                scenarios.append({
                    "id": f"{name}-{cell[0]}-{cell[1]}",
                    "layout": name,
                    "split": split,
                    "start": list(cell),
                    "marked_start": cell in marked,
                    "seed": BASE_SEED + len(scenarios),
                    "shortest_path": distances[cell],
                })
    layouts = {}
    for name in (*TRAIN_LAYOUTS, *HELDOUT_LAYOUTS):
        env = GridWorld(layout=name)
        layouts[name] = {"rows": list(LAYOUTS[name]), "goal": list(env.goal),
                         "marked_starts": [list(s) for s in env.starts],
                         "obstacles": len(env.obstacles)}
    return {
        "version": VERSION,
        "frozen": FROZEN,
        "max_steps": MAX_STEPS,
        "observation_mode": OBSERVATION_MODE,
        "train_layouts": list(TRAIN_LAYOUTS),
        "heldout_layouts": list(HELDOUT_LAYOUTS),
        "protocol": "shield off, greedy argmax, one episode per scenario, forced start",
        "layouts": layouts,
        "scenarios": scenarios,
    }


def render(spec):
    return json.dumps(spec, indent=1) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=DEFAULT_PATH)
    p.add_argument("--check", action="store_true", help="compare with the existing file instead of writing")
    args = p.parse_args()
    text = render(build())
    if args.check:
        if not args.output.exists() or args.output.read_text() != text:
            sys.exit(f"{args.output} differs from a fresh generation; scenarios are not frozen as committed.")
        print(f"{args.output} matches a fresh generation.")
        return
    args.output.write_text(text)
    spec = json.loads(text)
    counts = {}
    for s in spec["scenarios"]:
        counts[s["layout"]] = counts.get(s["layout"], 0) + 1
    print(f"Wrote {len(spec['scenarios'])} scenarios to {args.output}: {counts}")


if __name__ == "__main__":
    main()
