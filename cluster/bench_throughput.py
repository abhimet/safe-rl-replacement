"""Measure training throughput: N parallel highest_q runs of STEPS steps (outputs go to $TMPDIR).

    python cluster/bench_throughput.py 16 8000 [spatial|conv]
Used only to estimate job time; these runs are not results.
"""
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import train  # noqa: E402


def one(args):
    i, steps, out, architecture = args
    t = time.time()
    train(method="highest_q", feedback_mode="none", n_step=3, steps=steps, seed=1000 + i,
          output=f"{out}/bench{i}", architecture=architecture, verbose=False)
    return time.time() - t


if __name__ == "__main__":
    jobs, steps = int(sys.argv[1]), int(sys.argv[2])
    architecture = sys.argv[3] if len(sys.argv) > 3 else "spatial"
    out = tempfile.mkdtemp(prefix="bench_", dir=os.environ.get("TMPDIR"))
    t = time.time()
    with ProcessPoolExecutor(jobs) as pool:
        times = list(pool.map(one, [(i, steps, out, architecture) for i in range(jobs)]))
    wall = time.time() - t
    print(f"{architecture} jobs={jobs} steps={steps}: per-run {min(times):.1f}-{max(times):.1f}s, wall {wall:.1f}s, "
          f"per-process {steps / (sum(times) / len(times)):.0f} steps/s, total {jobs * steps / wall:.0f} steps/s")
