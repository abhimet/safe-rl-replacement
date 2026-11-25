"""Create immutable 10x10 external-to-training evaluation layouts.

Seed and density settings were fixed before checking model scores. These
three scenarios are not among Rooms/Zigzag/Clutter. Do NOT use them for model
selection and then describe them as independent final tests.
"""
import hashlib
import json
import random
from pathlib import Path

from environment import GridWorld


def create_suite(directory="configs/holdout_suite"):
    rng = random.Random(20261122)
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, density in enumerate((0.13, 0.23, 0.32), start=1):
        for _ in range(10000):
            walls = {(r, c) for r in range(10) for c in range(10)
                     if (r,c) not in ((9,9),(9,8)) and rng.random() < density}
            env = GridWorld(size=10, obstacles=walls, start=(9,8),
                            goal=(9,9), observation_mode="map")
            reachable = env.distances_to_goal()
            if len(reachable) >= 55 and max(reachable.values()) >= 15:
                break
        else:
            raise RuntimeError("Could not construct held-out connected layout")
        config = {"size":10, "obstacles":[list(c) for c in sorted(walls)],
                  "start":[9,8], "goal":[9,9], "observation_mode":"map"}
        label = f"heldout_{index}"
        path=output/f"{label}.json"
        path.write_text(json.dumps(config,indent=2)+"\n")
        sig=hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.append({"layout":label,"density":density,"reachable_starts":len(reachable)-1,
                         "sha256":sig,"config_file":str(path)})
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


if __name__ == "__main__":
    print(json.dumps(create_suite(),indent=2))
