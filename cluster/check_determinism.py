"""Retrain finished protocol runs from scratch and check they are bit-identical to the originals.

    python cluster/check_determinism.py --results results/core --tier core \
        --runs highest_q@none:42 fabr_v2@margin:42 --jobs 2

Each listed run is retrained with exactly the protocol's settings into $TMPDIR (nothing under
--results is touched). The check compares the training CSV byte for byte, every checkpoint tensor
exactly, and the shield-off evaluation numbers exactly. Exits non-zero on any difference.
"""
import argparse
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import evaluate  # noqa: E402
from make_eval_scenarios import ROOT  # noqa: E402
from run_experiments import load_protocol, parse_config, protocol_kwargs, run_paths  # noqa: E402
from train import train  # noqa: E402


def retrain(protocol_path, results, name, seed, scratch):
    protocol, digest = load_protocol(protocol_path)
    method, mode = parse_config(name)
    original, summary_path, eval_path = run_paths(Path(results) / mode, method, seed)
    if not eval_path.exists():
        raise FileNotFoundError(f"{original} is not a finished run")
    fixed_beta = json.loads(summary_path.read_text())["config"]["fixed_beta"] if method == "fabr_fixed" else None
    kwargs = protocol_kwargs(protocol, method, mode, fixed_beta)
    copy = Path(scratch) / mode / f"{method}_seed{seed}"
    train(method=method, seed=seed, output=copy, protocol_sha256=digest, verbose=False, **kwargs)
    fresh = evaluate(copy / f"{method}_seed{seed}.pt", ROOT / protocol["scenarios"]["file"], verbose=False)

    problems = []
    stem = f"{method}_seed{seed}"
    if (original / f"{stem}.csv").read_bytes() != (copy / f"{stem}.csv").read_bytes():
        problems.append("training CSV differs")
    a, b = torch.load(original / f"{stem}.pt"), torch.load(copy / f"{stem}.pt")
    if a.keys() != b.keys() or any(not torch.equal(a[k], b[k]) for k in a):
        problems.append("checkpoint tensors differ")
    stored = json.loads(eval_path.read_text())
    bad = [k for k in stored if k not in ("model", "scenarios") and stored[k] != fresh.get(k)]
    if bad:
        problems.append(f"evaluation differs in {bad[:5]}")
    return name, seed, problems


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--protocol", default=str(ROOT / "configs" / "final_protocol.json"))
    p.add_argument("--results", required=True, help="tier output directory, e.g. results/core")
    p.add_argument("--tier", default="core")
    p.add_argument("--runs", nargs="+", required=True, metavar="CONFIG:SEED")
    p.add_argument("--jobs", type=int, default=1)
    args = p.parse_args(argv)
    runs = [(c, int(s)) for c, s in (r.rsplit(":", 1) for r in args.runs)]
    scratch = tempfile.mkdtemp(prefix="determinism_", dir=os.environ.get("TMPDIR"))
    tasks = [(args.protocol, args.results, c, s, scratch) for c, s in runs]
    if args.jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(min(args.jobs, len(tasks))) as pool:
            results = list(pool.map(retrain, *zip(*tasks)))
    else:
        results = [retrain(*t) for t in tasks]
    failed = 0
    for name, seed, problems in results:
        print(f"{name} seed {seed}: " + ("IDENTICAL (CSV, checkpoint, evaluation)" if not problems
                                          else "DIFFERENT: " + "; ".join(problems)))
        failed += bool(problems)
    print(f"determinism: {len(results) - failed}/{len(results)} runs bit-identical")
    return int(failed > 0)


if __name__ == "__main__":
    sys.exit(main())
