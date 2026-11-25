"""Train every method on every seed with the same budget, evaluate each shield-off, and aggregate.

    python run_experiments.py --steps 50000 --seeds 42 43 44 45 46 --output results/final

Outputs (under --output):
    runs/<method>_seed<k>/        checkpoint, training CSV, summary, shield-off episode CSV
    per_seed.csv                  one row per (method, seed, group), groups = layouts + pooled groups
    aggregate.csv                 mean, std and 95% CI over seeds per (method, group)
    paired.csv                    FABR minus each other method, paired by seed

A finished run (summary + evaluation present) is reused, so an interrupted
experiment can be resumed with the same command. fabr_fixed runs after fabr and
uses the mean beta those fabr runs actually used, unless --fixed-beta is given.
"""
import argparse
import csv
import json
import statistics
from pathlib import Path

from evaluate import evaluate
from make_eval_scenarios import V2_PATH
from replacement import METHODS
from train import train

# Pooled reporting groups (docs/EVALUATION_PROTOCOL.md): never pool splits,
# and keep Spiral separate because its paths are about 3x longer.
GROUPS = {
    "familiar": ("rooms", "zigzag"),
    "dev_clutter": ("clutter",),
    "unseen": ("pillars", "heldout_1", "heldout_2", "heldout_3"),
    "unseen_spiral": ("spiral",),
}
LAYOUTS = tuple(layout for names in GROUPS.values() for layout in names)
METRICS = ("success_rate", "collision_rate", "timeout_rate", "unsafe_proposal_rate", "average_steps")
TRAINING_METRICS = ("success_rate", "collision_rate", "intervention_rate", "unsafe_proposal_rate")
# Two-sided 95% Student-t critical values by degrees of freedom.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def ci95(values):
    if len(values) < 2:
        return 0.0
    return T95.get(len(values) - 1, 1.96) * statistics.stdev(values) / len(values) ** 0.5


def run_one(root, method, seed, steps, fixed_beta=None, pool_size=100):
    directory = Path(root) / "runs" / f"{method}_seed{seed}"
    summary_path = directory / f"{method}_seed{seed}_summary.json"
    eval_path = directory / "shield_off_eval.json"
    if summary_path.exists() and eval_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary["steps"] == steps and summary["config"]["fixed_beta"] == fixed_beta:
            print(f"[{method} seed={seed}] reusing finished run", flush=True)
            return summary, json.loads(eval_path.read_text())
        raise FileExistsError(f"{directory} holds a run with different settings; use another --output")
    summary = train(method=method, steps=steps, seed=seed, output=directory,
                    fixed_beta=fixed_beta, pool_size=pool_size)
    result = evaluate(directory / f"{method}_seed{seed}.pt", V2_PATH,
                      output=directory / "shield_off_episodes.csv", verbose=False)
    eval_path.write_text(json.dumps(result, indent=2))
    return summary, result


def group_rows(method, seed, summary, result):
    """One row per layout and per pooled group, weighting layouts by scenario count."""
    training = {f"train_{k}": summary["training"][k] for k in TRAINING_METRICS}
    rows = []
    for group, layouts in [*((name, (name,)) for name in LAYOUTS), *GROUPS.items()]:
        parts = [result[f"layout={name}"] for name in layouts]
        n = sum(p["episodes"] for p in parts)
        row = {"method": method, "seed": seed, "group": group, "episodes": n}
        for metric in METRICS:
            row[metric] = sum(p[metric] * p["episodes"] for p in parts) / n
        row["mean_beta"] = summary["mean_beta"]
        rows.append({**row, **training})
    return rows


def aggregate(rows):
    out = []
    for method in dict.fromkeys(r["method"] for r in rows):
        for group in dict.fromkeys(r["group"] for r in rows):
            sel = [r for r in rows if r["method"] == method and r["group"] == group]
            entry = {"method": method, "group": group, "seeds": len(sel), "episodes": sel[0]["episodes"]}
            for metric in (*METRICS, *(f"train_{m}" for m in TRAINING_METRICS)):
                values = [r[metric] for r in sel]
                entry[f"{metric}_mean"] = statistics.mean(values)
                entry[f"{metric}_ci95"] = ci95(values)
            out.append(entry)
    return out


def paired(rows, reference="fabr"):
    """FABR minus each other method, paired by seed."""
    index = {(r["method"], r["seed"], r["group"]): r for r in rows}
    out = []
    for method in dict.fromkeys(r["method"] for r in rows):
        if method == reference:
            continue
        for group in dict.fromkeys(r["group"] for r in rows):
            seeds = [s for (m, s, g) in index if m == method and g == group and (reference, s, g) in index]
            if not seeds:
                continue
            for metric in ("success_rate", "collision_rate"):
                diffs = [index[(reference, s, group)][metric] - index[(method, s, group)][metric] for s in seeds]
                out.append({"comparison": f"{reference} - {method}", "group": group, "metric": metric,
                            "seeds": len(seeds), "mean_diff": statistics.mean(diffs), "ci95": ci95(diffs),
                            "fabr_better_seeds": sum(d > 0 for d in diffs)})
    return out


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run_experiments(output="results/final", steps=50000, seeds=(42, 43, 44), methods=METHODS,
                    fixed_beta=None, pool_size=100):
    methods = tuple(methods)
    if any(m not in METHODS for m in methods) or len(set(methods)) != len(methods):
        raise ValueError(f"methods must be distinct names from {METHODS}")
    if "fabr_fixed" in methods and fixed_beta is None and "fabr" not in methods:
        raise ValueError("fabr_fixed needs fabr in the same run or an explicit --fixed-beta")
    # fabr must finish before fabr_fixed can copy its mean beta.
    methods = tuple(sorted(methods, key=lambda m: m == "fabr_fixed"))
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    rows, fabr_betas = [], []
    for method in methods:
        beta = None
        if method == "fabr_fixed":
            beta = fixed_beta if fixed_beta is not None else round(statistics.mean(fabr_betas), 4)
            (root / "fabr_fixed_beta.json").write_text(json.dumps(
                {"fixed_beta": beta, "source": "command line" if fixed_beta is not None
                 else f"mean beta of fabr seeds {list(seeds)}"}, indent=2))
        for seed in seeds:
            summary, result = run_one(root, method, seed, steps, beta, pool_size)
            if method == "fabr":
                fabr_betas.append(summary["mean_beta"])
            rows += group_rows(method, seed, summary, result)
    write_csv(root / "per_seed.csv", rows)
    agg = aggregate(rows)
    write_csv(root / "aggregate.csv", agg)
    if "fabr" in methods and len(methods) > 1:
        write_csv(root / "paired.csv", paired(rows))
    print_table(agg)
    print(f"Results written to {root}/")
    return agg


def print_table(agg):
    groups = list(GROUPS)
    print("\nShield-OFF success rate (mean ± 95% CI over seeds)")
    print(f"{'method':<15}" + "".join(f"{g:>20}" for g in groups))
    for method in dict.fromkeys(r["method"] for r in agg):
        cells = {r["group"]: r for r in agg if r["method"] == method}
        print(f"{method:<15}" + "".join(
            f"{cells[g]['success_rate_mean']:>12.3f} ± {cells[g]['success_rate_ci95']:.3f}" for g in groups))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", default="results/final")
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    p.add_argument("--fixed-beta", type=float, default=None,
                   help="beta for fabr_fixed (default: mean beta of this run's fabr seeds)")
    p.add_argument("--pool-size", type=int, default=100)
    run_experiments(**vars(p.parse_args()))
