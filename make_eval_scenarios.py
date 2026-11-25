"""Generate (or check) the frozen shield-off evaluation scenario sets.

One deterministic scenario per reachable, non-goal free cell of every layout.
Evaluation is greedy and GridWorld is deterministic, so each scenario is one
episode; `seed` is recorded only as metadata for reproducibility.

Version 1 (frozen 2026-11-22) is kept byte-for-byte. Version 2 (frozen
2026-11-23) is the official evaluation set: it adds the fixed random-obstacle
holdout suite (configs/holdout_suite) and reclassifies `clutter` as
development-inspected. See EVALUATION_PROTOCOL.md.

    python make_eval_scenarios.py --version 2            # write evaluation_scenarios_v2.json
    python make_eval_scenarios.py --version 2 --check    # fail if the committed file drifted
    python make_eval_scenarios.py --check                # same for v1
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from environment import HELDOUT_LAYOUTS, LAYOUTS, TRAIN_LAYOUTS, GridWorld

VERSION = 1
FROZEN = "2026-11-22"
MAX_STEPS = 100
OBSERVATION_MODE = "map"
BASE_SEED = 10000
ROOT = Path(__file__).parent
DEFAULT_PATH = ROOT / "evaluation_scenarios.json"
V2_PATH = ROOT / "evaluation_scenarios_v2.json"
PATHS = {1: DEFAULT_PATH, 2: V2_PATH}

# Version 2 grouping. `split` decides how a map may be reported; `family` says how it was built.
V2_FROZEN = "2026-11-23"
V2_DEV_INSPECTED = ("clutter",)
V2_HELDOUT_DESIGNED = ("pillars", "spiral")
HOLDOUT_SUITE = "configs/holdout_suite"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def holdout_suite(root=ROOT, suite=HOLDOUT_SUITE):
    """Yield (name, relative config path, config dict); refuse if a file differs from its manifest hash."""
    manifest = json.loads((Path(root) / suite / "manifest.json").read_text())
    for entry in manifest:
        relative = f"{suite}/{entry['layout']}.json"
        actual = sha256(Path(root) / relative)
        if actual != entry["sha256"]:
            raise ValueError(f"{relative} sha256 {actual} does not match manifest {entry['sha256']}")
        yield entry["layout"], relative, json.loads((Path(root) / relative).read_text())


def config_env(config, max_steps=MAX_STEPS, observation_mode=OBSERVATION_MODE):
    settings = {k: v for k, v in config.items() if k not in ("max_steps", "observation_mode")}
    return GridWorld(**settings, max_steps=max_steps, observation_mode=observation_mode)


def evaluation_maps():
    """{name: GridWorld} for every non-training map in the official v2 evaluation set."""
    maps = {name: GridWorld(layout=name, max_steps=MAX_STEPS, observation_mode=OBSERVATION_MODE)
            for name in (*V2_DEV_INSPECTED, *V2_HELDOUT_DESIGNED)}
    for name, _, config in holdout_suite():
        maps[name] = config_env(config)
    return maps


def build(version=1):
    if version == 2:
        return build_v2()
    if version != 1:
        raise ValueError("version must be 1 or 2")
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


def build_v2():
    groups = [(name, "train", "designed") for name in TRAIN_LAYOUTS]
    groups += [(name, "dev_inspected", "designed") for name in V2_DEV_INSPECTED]
    groups += [(name, "heldout", "designed") for name in V2_HELDOUT_DESIGNED]
    layouts, envs = {}, {}
    for name, split, family in groups:
        envs[name] = GridWorld(layout=name, max_steps=MAX_STEPS, observation_mode=OBSERVATION_MODE)
        layouts[name] = {"split": split, "family": family, "source": "environment.py",
                         "rows": list(LAYOUTS[name])}
    for name, relative, config in holdout_suite():
        groups.append((name, "heldout", "random"))
        envs[name] = config_env(config)
        layouts[name] = {"split": "heldout", "family": "random", "source": relative,
                         "config": relative, "sha256": sha256(ROOT / relative)}

    scenarios = []
    for name, split, family in groups:
        env = envs[name]
        distances = env.distances_to_goal()
        marked = set(env.starts)
        layouts[name].update(goal=list(env.goal), marked_starts=[list(c) for c in env.starts],
                             obstacles=len(env.obstacles))
        for cell in sorted(c for c in distances if c != env.goal):
            scenarios.append({
                "id": f"{name}-{cell[0]}-{cell[1]}",
                "layout": name,
                "split": split,
                "family": family,
                "start": list(cell),
                "marked_start": cell in marked,
                "seed": BASE_SEED + len(scenarios),
                "shortest_path": distances[cell],
            })
    by_split = lambda split: [n for n, s, _ in groups if s == split]
    return {
        "version": 2,
        "frozen": V2_FROZEN,
        "parent": {"file": DEFAULT_PATH.name, "version": VERSION, "frozen": FROZEN,
                   "sha256": sha256(DEFAULT_PATH)},
        "changes_from_parent": [
            "clutter moved from heldout to dev_inspected: FABR_V3_STUDY.md records that it was "
            "consulted during development, so it is no longer an untouched held-out map.",
            "Added heldout_1..3 from configs/holdout_suite (commit 5271a81), verified against "
            "manifest.json sha256 hashes. They are random-obstacle maps (family=random).",
            "rooms, zigzag, pillars and spiral are unchanged from v1.",
        ],
        "max_steps": MAX_STEPS,
        "observation_mode": OBSERVATION_MODE,
        "train_layouts": by_split("train"),
        "dev_inspected_layouts": by_split("dev_inspected"),
        "heldout_layouts": by_split("heldout"),
        "protocol": "shield off, greedy argmax, one episode per scenario, forced start",
        "reporting": "Report every layout separately and never pool splits. Report spiral on its own "
                     "(paths ~3x longer). A combined held-out number must name exactly which layouts it pools.",
        "layouts": layouts,
        "scenarios": scenarios,
    }


def render(spec):
    return json.dumps(spec, indent=1) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", type=int, choices=(1, 2), default=1)
    p.add_argument("--output", type=Path, help="default: the committed file for --version")
    p.add_argument("--check", action="store_true", help="compare with the existing file instead of writing")
    args = p.parse_args()
    args.output = args.output or PATHS[args.version]
    text = render(build(args.version))
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
