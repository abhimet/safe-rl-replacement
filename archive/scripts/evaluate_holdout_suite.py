"""Evaluate *already trained* checkpoints on the fixed holdout suite.

Superseded: the holdout suite is part of evaluation_scenarios_v2.json (family=random),
and `evaluate.py --scenarios evaluation_scenarios_v2.json` gives identical numbers
(see test_v2_random_family_equals_legacy_holdout_evaluation). Kept for old result folders.
"""
import argparse
import contextlib
import csv
import io
import json
from pathlib import Path

from evaluate import evaluate


def evaluate_suite(methods=("value_based", "par", "par_count"),
                   result_dir="results/fabr_safety_credit_5k",
                   suite_dir="configs/holdout_suite"):
    root=Path(result_dir)
    manifest=json.loads((Path(suite_dir)/"manifest.json").read_text())
    results=[]
    for method in methods:
        for seed in (42,43,44):
            directory=root/method/f"seed{seed}"
            checkpoints=list(directory.glob("*.pt"))
            if len(checkpoints)!=1:
                raise RuntimeError(f"Expected one checkpoint in {directory}")
            for layout in manifest:
                name=layout["layout"]
                config=Path(suite_dir)/f"{name}.json"
                with contextlib.redirect_stdout(io.StringIO()):
                    d=evaluate(model=str(checkpoints[0]),env_config=str(config),
                               size=10,observation_mode="map",all_starts=True,
                               output=str(directory/f"off_{name}.csv"))
                n=d["episodes"]
                result={"method":method,"seed":seed,"layout":name,
                        "n_starts":n,"successes":round(d["success_rate"]*n),
                        "collisions":round(d["collision_rate"]*n),
                        "success_rate":d["success_rate"],
                        "collision_rate":d["collision_rate"]}
                results.append(result)
                print(f"{method} seed{seed} {name}: {result['successes']}/{n} successes")
    output=root/"holdout_suite_results.csv"
    old=[]
    if output.exists():
        old=list(csv.DictReader(output.open()))
    # Merge the new results into previous results without duplicated method/seed/layout.
    merged={(r['method'],int(r['seed']),r['layout']):r for r in old}
    for r in results:merged[(r['method'],r['seed'],r['layout'])]=r
    with output.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=results[0])
        writer.writeheader()
        writer.writerows(sorted(merged.values(),key=lambda r:(r['method'],int(r['seed']),r['layout'])))
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--methods',nargs='+',default=['value_based','par','par_count'])
    p.add_argument('--result-dir',default='results/fabr_safety_credit_5k')
    p.add_argument('--suite-dir',default='configs/holdout_suite')
    evaluate_suite(**vars(p.parse_args()))
