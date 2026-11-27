"""Train every config on every seed, evaluate each shield-off, and aggregate.

Final runs read EVERYTHING from a frozen protocol file (configs/final_protocol.json):

    python run_experiments.py --protocol configs/final_protocol.json --tier core --jobs 6
    python run_experiments.py --protocol configs/final_protocol.json --tier core --seeds 42 43 44 45 46

A tier lists configs ("method@feedback_mode", or "unshielded") and seeds. --seeds runs a
batch (a subset of the tier's seeds); rerunning resumes, and the tables always cover every
finished run. Training flags cannot be combined with --protocol.

Outputs (under --output, default results/<tier>):
    <mode>/runs/<method>_seed<k>/   checkpoint, training CSV, summary, shield-off evaluation
    per_seed.csv                    one row per (config, seed, group)
    aggregate.csv                   mean and 95% CI over seeds per (config, group)
    paired.csv                      fabr_v2@margin minus every config, and fabr@<mode> minus
                                    every config of that mode, paired by seed

fabr_fixed@<mode> uses the mean beta of ALL the tier's fabr@<mode> seeds, so it runs (for every
tier seed) once those are finished.

Legacy single-mode runs (the v1 results in docs/results/cf and no_cf) still work:

    python run_experiments.py --steps 50000 --seeds 42 43 44 45 46 47 48 49 50 51 --jobs 4 --output results/final_cf
    python run_experiments.py ... --no-collision-feedback --output results/final_no_cf
"""
import argparse
import csv
import hashlib
import inspect
import json
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from evaluate import evaluate
from make_eval_scenarios import ROOT, V2_PATH, sha256
from replacement import FEEDBACK_MODES, MARGIN_METHODS, METHODS
from train import resolve_feedback_mode, train

# Pooled reporting groups (docs/EVALUATION_PROTOCOL.md): never pool splits, and keep Spiral
# separate because its paths are about 3x longer. Unseen maps are the headline; familiar and
# clutter were used for tuning (TUNING_GROUPS).
GROUPS = {
    "unseen": ("pillars", "heldout_1", "heldout_2", "heldout_3"),
    "unseen_spiral": ("spiral",),
    "familiar": ("rooms", "zigzag"),
    "dev_clutter": ("clutter",),
}
TUNING_GROUPS = ("familiar", "dev_clutter")
LAYOUTS = tuple(layout for names in GROUPS.values() for layout in names)
METRICS = ("success_rate", "collision_rate", "timeout_rate", "average_reward",
           "greedy_unsafe_state_rate", "unsafe_proposal_rate", "average_steps")
TRAINING_METRICS = ("success_rate", "collision_rate", "intervention_rate", "unsafe_proposal_rate")
# Paired metrics and their sign: +1 = higher is better.
PAIRED_METRICS = (("success_rate", 1), ("collision_rate", -1), ("timeout_rate", -1),
                  ("average_reward", 1), ("greedy_unsafe_state_rate", -1))
LEGACY_METHODS = METHODS[:7]  # the methods of the v1 (single feedback setting) experiments
# Two-sided 95% Student-t critical values by degrees of freedom.
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}

TRAIN_DEFAULTS = {name: p.default for name, p in inspect.signature(train).parameters.items()
                  if p.default is not inspect.Parameter.empty}
# Settings that define a run (method and seed are in its directory name).
RUN_KEYS = tuple(k for k in TRAIN_DEFAULTS
                 if k not in ("method", "seed", "output", "verbose", "protocol_sha256",
                              "collision_feedback"))
FIXED_BETA_FROM_FABR = "mean_beta_of_fabr_same_mode"


def ci95(values):
    if len(values) < 2:
        return 0.0
    return T95.get(len(values) - 1, 1.96) * statistics.stdev(values) / len(values) ** 0.5


# ---- one run ----

def run_matches(summary, method, kwargs, protocol_sha256=None):
    """Does a finished run's summary match these settings? Old summaries count as defaults."""
    config = summary["config"]
    found = {**config, "steps": summary["steps"]}
    found.setdefault("feedback_mode", "cf" if config.get("collision_feedback", True) else "none")
    if method == "unshielded":
        found["feedback_mode"] = "none"
    expected = {k: kwargs.get(k, TRAIN_DEFAULTS[k]) for k in RUN_KEYS}
    expected["feedback_mode"] = resolve_feedback_mode(method, kwargs.get("feedback_mode"))
    return (all(found.get(k, TRAIN_DEFAULTS[k]) == v for k, v in expected.items())
            and summary.get("protocol_sha256") == protocol_sha256)


def run_paths(root, method, seed):
    directory = Path(root) / "runs" / f"{method}_seed{seed}"
    return directory, directory / f"{method}_seed{seed}_summary.json", directory / "shield_off_eval.json"


def finished_run(root, method, seed, kwargs, protocol_sha256=None):
    """(summary, evaluation) of a matching finished run, None if there is none."""
    directory, summary_path, eval_path = run_paths(root, method, seed)
    if not (summary_path.exists() and eval_path.exists()):
        return None
    summary = json.loads(summary_path.read_text())
    if not run_matches(summary, method, kwargs, protocol_sha256):
        raise FileExistsError(f"{directory} holds a run with different settings; use another --output")
    return summary, json.loads(eval_path.read_text())


def run_one(root, method, seed, kwargs, protocol_sha256=None, scenarios=V2_PATH):
    done = finished_run(root, method, seed, kwargs, protocol_sha256)
    if done:
        print(f"[{method} seed={seed}] reusing finished run", flush=True)
        return done
    directory, _, eval_path = run_paths(root, method, seed)
    summary = train(method=method, seed=seed, output=directory, protocol_sha256=protocol_sha256, **kwargs)
    result = evaluate(directory / f"{method}_seed{seed}.pt", scenarios,
                      output=directory / "shield_off_episodes.csv", verbose=False)
    eval_path.write_text(json.dumps(result, indent=2))
    return summary, result


def run_all(tasks, jobs=1):
    """tasks: [(root, method, seed, kwargs, protocol_sha256, scenarios)]; results keep task order."""
    if jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            return list(pool.map(run_one, *zip(*tasks)))
    return [run_one(*t) for t in tasks]


# ---- tables ----

def group_rows(identity, seed, summary, result):
    """One row per layout and per pooled group, weighting layouts by scenario count.

    ``identity`` holds the columns that name the run, e.g. {"method": ...}; a plain string
    is taken as the method.
    """
    if isinstance(identity, str):
        identity = {"method": identity}
    training = {f"train_{k}": summary["training"][k] for k in TRAINING_METRICS}
    rows = []
    for group, layouts in [*((name, (name,)) for name in LAYOUTS), *GROUPS.items()]:
        parts = [result[f"layout={name}"] for name in layouts]
        n = sum(p["episodes"] for p in parts)
        row = {**identity, "seed": seed, "group": group, "episodes": n}
        for metric in METRICS:
            known = all(metric in p for p in parts)
            row[metric] = sum(p[metric] * p["episodes"] for p in parts) / n if known else ""
        row["mean_beta"] = summary["mean_beta"]
        rows.append({**row, **training})
    return rows


def identity_columns(rows):
    return ("config", "mode", "method") if "config" in rows[0] else ("method",)


def aggregate(rows):
    ident = identity_columns(rows)
    key = ident[0]
    out = []
    for name in dict.fromkeys(r[key] for r in rows):
        for group in dict.fromkeys(r["group"] for r in rows):
            sel = [r for r in rows if r[key] == name and r["group"] == group]
            entry = {c: sel[0][c] for c in ident}
            entry.update(group=group, seeds=len(sel), episodes=sel[0]["episodes"])
            for metric in (*METRICS, *(f"train_{m}" for m in TRAINING_METRICS)):
                values = [r[metric] for r in sel if r.get(metric, "") != ""]
                entry[f"{metric}_mean"] = statistics.mean(values) if values else ""
                entry[f"{metric}_ci95"] = ci95(values) if values else ""
            out.append(entry)
    return out


def paired(rows, reference="fabr", key="method"):
    """reference minus every other value of ``key``, paired by seed."""
    index = {(r[key], r["seed"], r["group"]): r for r in rows}
    out = []
    for name in dict.fromkeys(r[key] for r in rows):
        if name == reference:
            continue
        for group in dict.fromkeys(r["group"] for r in rows):
            seeds = [s for (m, s, g) in index if m == name and g == group and (reference, s, g) in index]
            if not seeds:
                continue
            for metric, sign in PAIRED_METRICS:
                if any(index[(n, s, group)].get(metric, "") == "" for s in seeds for n in (name, reference)):
                    continue
                diffs = [index[(reference, s, group)][metric] - index[(name, s, group)][metric] for s in seeds]
                out.append({"comparison": f"{reference} - {name}", "group": group, "metric": metric,
                            "seeds": len(seeds), "mean_diff": statistics.mean(diffs), "ci95": ci95(diffs),
                            "reference_better_seeds": sum(sign * d > 0 for d in diffs)})
    return out


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def print_table(agg):
    key = identity_columns(agg)[0]
    groups = list(GROUPS)
    print("\nShield-OFF success / collision (mean over seeds; * = used for tuning)")
    print(f"{key:<26}{'seeds':>6}" + "".join(
        f"{g + ('*' if g in TUNING_GROUPS else ''):>22}" for g in groups))
    for name in dict.fromkeys(r[key] for r in agg):
        cells = {r["group"]: r for r in agg if r[key] == name}
        print(f"{name:<26}{cells[groups[0]]['seeds']:>6}" + "".join(
            f"{cells[g]['success_rate_mean']:>13.3f} / {cells[g]['collision_rate_mean']:.3f}" for g in groups))


def write_tables(root, rows, references):
    """per_seed / aggregate / paired CSVs. references: [(reference, rows subset)]."""
    write_csv(root / "per_seed.csv", rows)
    agg = aggregate(rows)
    write_csv(root / "aggregate.csv", agg)
    key = identity_columns(rows)[0]
    comparisons = [c for ref, subset in references for c in paired(subset, ref, key)]
    if comparisons:
        write_csv(root / "paired.csv", comparisons)
    print_table(agg)
    print(f"Results written to {root}/")
    return agg


# ---- protocol (final runs) ----

def parse_config(name):
    """'method@mode' (or 'unshielded') -> (method, mode)."""
    method, _, mode = name.partition("@")
    if method not in METHODS:
        raise ValueError(f"Unknown method in config {name!r}")
    mode = mode or None
    if method == "unshielded" and mode not in (None, "none"):
        raise ValueError("write the unshielded config as 'unshielded'")
    if method != "unshielded" and mode not in FEEDBACK_MODES:
        raise ValueError(f"Config {name!r} needs a feedback mode from {FEEDBACK_MODES}")
    if resolve_feedback_mode(method, mode) != (mode or "none"):
        raise ValueError(f"Invalid config {name!r}")
    return method, mode or "none"


def config_name(method, mode):
    return "unshielded" if method == "unshielded" else f"{method}@{mode}"


def load_protocol(path):
    """Read and validate a protocol file. Returns (protocol, sha256 of the file)."""
    raw = Path(path).read_bytes()
    protocol = json.loads(raw)
    for field in ("frozen", "scenarios", "train", "methods", "tiers", "metrics"):
        if field not in protocol:
            raise ValueError(f"protocol is missing {field!r}")
    scenario_path = ROOT / protocol["scenarios"]["file"]
    if sha256(scenario_path) != protocol["scenarios"]["sha256"]:
        raise ValueError(f"{scenario_path} does not match the sha256 frozen in {path}")
    allowed = set(RUN_KEYS) - {"feedback_mode"}
    for where, settings in [("train", protocol["train"]), *protocol["methods"].items()]:
        unknown = set(settings) - allowed - {"fixed_beta"}
        if unknown:
            raise ValueError(f"protocol {where!r} has unknown settings {sorted(unknown)}")
    if (set(MARGIN_METHODS) - {"fabr_v2"}) & set(protocol["methods"]):
        raise ValueError("fabr_v2_fixedtarget always uses fabr_v2's settings; do not list it")
    if not set(protocol["metrics"]) <= set(METRICS):
        raise ValueError(f"protocol metrics must be from {METRICS}")
    for tier, spec in protocol["tiers"].items():
        configs = [parse_config(c) for c in spec["configs"]]
        if not spec["seeds"] or len(set(spec["seeds"])) != len(spec["seeds"]):
            raise ValueError(f"tier {tier!r} needs distinct seeds")
        for method, mode in configs:
            if method == "fabr_fixed" and ("fabr", mode) not in configs:
                raise ValueError(f"tier {tier!r}: fabr_fixed@{mode} needs fabr@{mode} in the same tier")
    return protocol, hashlib.sha256(raw).hexdigest()


def protocol_kwargs(protocol, method, mode, fixed_beta=None):
    """train() settings for one config, taken only from the protocol."""
    kwargs = dict(protocol["train"])
    params = protocol["methods"].get("fabr_v2" if method in MARGIN_METHODS else method, {})
    kwargs.update({k: v for k, v in params.items() if k != "fixed_beta"})
    kwargs["feedback_mode"] = mode
    if method == "fabr_fixed":
        if params.get("fixed_beta", FIXED_BETA_FROM_FABR) != FIXED_BETA_FROM_FABR:
            kwargs["fixed_beta"] = float(params["fixed_beta"])
        elif fixed_beta is None:
            raise ValueError("fabr_fixed needs the mean beta of the fabr runs")
        else:
            kwargs["fixed_beta"] = fixed_beta
    return kwargs


def run_protocol(protocol_path, tier, output=None, seeds=None, jobs=1):
    """Run one tier of a protocol file. The protocol is the only source of training settings."""
    protocol, digest = load_protocol(protocol_path)
    if not protocol["frozen"]:
        print(f"WARNING: {protocol_path} is a DRAFT (frozen=false); these are not final results.",
              flush=True)
    if tier not in protocol["tiers"]:
        raise ValueError(f"Unknown tier {tier!r}; choose from {sorted(protocol['tiers'])}")
    spec = protocol["tiers"][tier]
    tier_seeds = list(spec["seeds"])
    seeds = list(seeds) if seeds else tier_seeds
    if not set(seeds) <= set(tier_seeds):
        raise ValueError(f"--seeds must be a subset of tier {tier!r} seeds {tier_seeds}")
    configs = [parse_config(c) for c in spec["configs"]]
    root = Path(output or Path("results") / tier)
    root.mkdir(parents=True, exist_ok=True)
    scenarios = ROOT / protocol["scenarios"]["file"]

    def task(method, mode, seed, fixed_beta=None):
        return (root / mode, method, seed, protocol_kwargs(protocol, method, mode, fixed_beta),
                digest, scenarios)

    first = [task(m, mode, s) for m, mode in configs if m != "fabr_fixed" for s in seeds]
    run_all(first, jobs)

    fixed_betas = {}
    for method, mode in configs:
        if method != "fabr_fixed":
            continue
        fabr = [finished_run(root / mode, "fabr", s, protocol_kwargs(protocol, "fabr", mode), digest)
                for s in tier_seeds]
        missing = [s for s, r in zip(tier_seeds, fabr) if r is None]
        if missing:
            print(f"fabr_fixed@{mode} waits for fabr@{mode} seeds {missing} (its beta is their mean)")
            continue
        beta = round(statistics.mean(r[0]["mean_beta"] for r in fabr), 4)
        fixed_betas[mode] = beta
        (root / mode / "fabr_fixed_beta.json").write_text(json.dumps(
            {"fixed_beta": beta, "source": f"mean beta of fabr@{mode} seeds {tier_seeds}"}, indent=2))
        run_all([task("fabr_fixed", mode, s, beta) for s in tier_seeds], jobs)

    rows, done = [], {}
    for method, mode in configs:
        if method == "fabr_fixed" and mode not in fixed_betas:
            continue
        name = config_name(method, mode)
        kwargs = protocol_kwargs(protocol, method, mode, fixed_betas.get(mode))
        for seed in tier_seeds:
            result = finished_run(root / mode, method, seed, kwargs, digest)
            if result:
                done.setdefault(name, []).append(seed)
                rows += group_rows({"config": name, "mode": mode, "method": method}, seed, *result)
    (root / "experiment_config.json").write_text(json.dumps(
        {"protocol": str(protocol_path), "protocol_sha256": digest, "tier": tier,
         "frozen": protocol["frozen"], "finished_seeds": done, "fixed_betas": fixed_betas}, indent=2))
    if not rows:
        print("No finished runs yet.")
        return []
    names = list(dict.fromkeys(r["config"] for r in rows))
    references = [("fabr_v2@margin", rows)] if "fabr_v2@margin" in names else []
    for mode in FEEDBACK_MODES:
        if f"fabr@{mode}" in names:
            references.append((f"fabr@{mode}", [r for r in rows if r["mode"] == mode]))
    return write_tables(root, rows, references)


# ---- legacy single-setting experiments ----

def run_experiments(output="results/final", steps=50000, seeds=(42, 43, 44), methods=LEGACY_METHODS,
                    fixed_beta=None, pool_size=100, collision_feedback=True, jobs=1, feedback_mode=None):
    """One feedback setting for every method (how the v1 results were produced)."""
    methods = tuple(methods)
    if any(m not in METHODS for m in methods) or len(set(methods)) != len(methods):
        raise ValueError(f"methods must be distinct names from {METHODS}")
    if "fabr_fixed" in methods and fixed_beta is None and "fabr" not in methods:
        raise ValueError("fabr_fixed needs fabr in the same run or an explicit --fixed-beta")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)

    def kwargs(method, beta=None):
        legacy = None if method in MARGIN_METHODS else collision_feedback
        return dict(steps=steps, pool_size=pool_size, fixed_beta=beta,
                    feedback_mode=resolve_feedback_mode(method, feedback_mode, legacy))

    # Phase 1: everything except fabr_fixed. Phase 2: fabr_fixed, which needs fabr's mean beta.
    first = [(m, s) for m in methods if m != "fabr_fixed" for s in seeds]
    results = dict(zip(first, run_all([(root, m, s, kwargs(m)) for m, s in first], jobs)))
    if "fabr_fixed" in methods:
        if fixed_beta is None:
            fixed_beta = round(statistics.mean(results[("fabr", s)][0]["mean_beta"] for s in seeds), 4)
            source = f"mean beta of fabr seeds {list(seeds)}"
        else:
            source = "command line"
        (root / "fabr_fixed_beta.json").write_text(json.dumps(
            {"fixed_beta": fixed_beta, "source": source}, indent=2))
        second = [("fabr_fixed", s) for s in seeds]
        results.update(zip(second, run_all([(root, m, s, kwargs(m, fixed_beta)) for m, s in second], jobs)))
    (root / "experiment_config.json").write_text(json.dumps(
        {"steps": steps, "pool_size": pool_size, "collision_feedback": collision_feedback,
         "feedback_mode": feedback_mode, "seeds": list(seeds), "methods": list(methods)}, indent=2))
    rows = []
    for method in methods:
        for seed in seeds:
            rows += group_rows({"method": method}, seed, *results[(method, seed)])
    references = [("fabr", rows)] if "fabr" in methods and len(methods) > 1 else []
    return write_tables(root, rows, references)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--protocol", default=None, help="Frozen protocol file (final runs)")
    p.add_argument("--tier", default="core", help="Protocol tier to run")
    p.add_argument("--output", default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=None)
    p.add_argument("--jobs", type=int, default=1, help="Parallel training processes")
    legacy = p.add_argument_group("legacy single-setting runs (not allowed with --protocol)")
    legacy.add_argument("--steps", type=int, default=None)
    legacy.add_argument("--methods", nargs="+", default=None, choices=METHODS)
    legacy.add_argument("--fixed-beta", type=float, default=None,
                        help="beta for fabr_fixed (default: mean beta of this run's fabr seeds)")
    legacy.add_argument("--pool-size", type=int, default=None)
    legacy.add_argument("--feedback-mode", choices=FEEDBACK_MODES, default=None)
    legacy.add_argument("--no-collision-feedback", dest="collision_feedback", action="store_const",
                        const=False, default=None, help="Legacy alias for --feedback-mode none")
    args = p.parse_args()
    flags = {"--steps": args.steps, "--methods": args.methods, "--fixed-beta": args.fixed_beta,
             "--pool-size": args.pool_size, "--feedback-mode": args.feedback_mode,
             "--no-collision-feedback": args.collision_feedback}
    if args.protocol:
        given = [flag for flag, value in flags.items() if value is not None]
        if given:
            p.error(f"--protocol is the only source of training settings; remove {', '.join(given)}")
        run_protocol(args.protocol, args.tier, args.output, args.seeds, args.jobs)
    else:
        run_experiments(output=args.output or "results/final", steps=args.steps or 50000,
                        seeds=args.seeds or [42, 43, 44], methods=args.methods or LEGACY_METHODS,
                        fixed_beta=args.fixed_beta, pool_size=args.pool_size or 100,
                        collision_feedback=args.collision_feedback is not False,
                        jobs=args.jobs, feedback_mode=args.feedback_mode)
