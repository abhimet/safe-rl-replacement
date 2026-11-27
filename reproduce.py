"""One entry point for everything: tests, verification, experiments, tuning and figures.

    python reproduce.py smoke                 # unit + end-to-end tests                     (~1 min)
    python reproduce.py verify                # re-evaluate the shipped checkpoints: every
                                              # episode must match exactly                  (~8 min)
    python reproduce.py quick                 # SANITY CHECK of direction: retrain 5 headline
                                              # configs x 3 seeds; numbers will differ    (~30 min)
    python reproduce.py smoke verify quick    # several commands run in order (grader path)

    python reproduce.py tune                  # dev-map tuning that produced the protocol   (~1-1.5 h)
    python reproduce.py core --seeds 42 43 44 45 46   # the reported results, in seed batches
    python reproduce.py core                  # all 10 seeds                                (~3.4 h)
    python reproduce.py full                  # complete 21-config grid (optional)          (~6.5 h)
    python reproduce.py figures --results results/core
    python reproduce.py ship core             # copy results/core into docs/results/v2/core

Every experiment reads its settings from configs/final_protocol.json and resumes after an
interruption: rerun the same command. Results go to <results root>/<tier>/, where the
results root is --results-root, else $SHIELDED_RL_RESULTS, else ./results. Time estimates assume
~900 training steps/s in total, measured over a long run on an 8-core M1 laptop with 6 jobs.
"""
import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # one thread per process; parallelism comes from --jobs

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
PROTOCOL = HERE / "configs" / "final_protocol.json"
SHIPPED = HERE / "docs" / "results" / "v2"
TIERS = ("quick", "core", "full")
COMMANDS = ("smoke", "verify", "tune", *TIERS, "figures", "ship")
EVAL_SECONDS = 8  # one shield-off evaluation (581 episodes)


def results_root(args):
    """Where tier outputs go: --results-root, else $SHIELDED_RL_RESULTS, else ./results."""
    root = getattr(args, "results_root", None) or os.environ.get("SHIELDED_RL_RESULTS") or HERE / "results"
    return Path(root).resolve()


def default_jobs():
    return max(1, min(6, (os.cpu_count() or 2) - 2))


def throughput(jobs):
    """Rough total training steps/s (sustained, M1 laptop)."""
    return {1: 600, 2: 800}.get(jobs, 900)


def minutes(seconds):
    return f"{seconds / 60:.0f} min" if seconds < 5400 else f"{seconds / 3600:.1f} h"


def smoke(args):
    print("== smoke: running the test suite ==", flush=True)
    return subprocess.run([sys.executable, "-m", "pytest", "-q", str(HERE / "tests")], cwd=HERE).returncode


def load_frozen(args):
    from run_experiments import load_protocol
    protocol, digest = load_protocol(PROTOCOL)
    if not protocol["frozen"] and not args.allow_draft:
        sys.exit(f"{PROTOCOL.name} is still a DRAFT (frozen=false): its hyperparameters are not final. "
                 "Run `reproduce.py tune`, approve and freeze it, or pass --allow-draft for a test run.")
    return protocol, digest


def pending_runs(protocol, digest, tier, seeds, root):
    """How many (config, seed) runs of a tier are not finished yet."""
    from run_experiments import finished_run, parse_config, protocol_kwargs, run_paths
    count = 0
    for name in protocol["tiers"][tier]["configs"]:
        method, mode = parse_config(name)
        for seed in (protocol["tiers"][tier]["seeds"] if method == "fabr_fixed" else seeds):
            if method == "fabr_fixed":
                count += not run_paths(root / mode, method, seed)[1].exists()
            else:
                count += finished_run(root / mode, method, seed, protocol_kwargs(protocol, method, mode),
                                      digest) is None
    return count


def experiment(tier):
    def run(args):
        from run_experiments import run_protocol
        protocol, digest = load_frozen(args)
        spec = protocol["tiers"][tier]
        seeds = args.seeds or spec["seeds"]
        root = results_root(args) / tier
        todo = pending_runs(protocol, digest, tier, seeds, root)
        steps = protocol["train"]["steps"]
        estimate = todo * steps / throughput(args.jobs) + todo * EVAL_SECONDS / args.jobs
        if tier == "quick":
            print("== quick is a SANITY CHECK OF DIRECTION, not a reproduction: training is bit-"
                  "reproducible only on the same CPU and software stack, so these numbers will differ "
                  "from the reported ones. `verify` is the exact reproduction. ==", flush=True)
        print(f"== {tier}: {len(spec['configs'])} configs x seeds {seeds}; {todo} runs to train at "
              f"{steps:,} steps, {args.jobs} jobs. Rough estimate: {minutes(estimate)} ==", flush=True)
        run_protocol(PROTOCOL, tier, root, seeds, args.jobs)
        figures(argparse.Namespace(results=root, results_root=None))
        return 0
    return run


def tune(args):
    """Re-run both tuning rounds (dev maps only) and compare the choices with the protocol."""
    import tune as tuning
    from run_experiments import load_protocol
    print("== tune: dev maps only (train + dev_inspected); round 1 then round 2. "
          "Rough estimate: 2-3 h on a laptop ==", flush=True)
    for number in (1, 2):
        tuning.main("all", args.jobs, write=False, round=number)
    selected = tuning.Tuner(round=2).selected
    protocol, _ = load_protocol(PROTOCOL)
    same = ({**selected["train"], "steps": selected["steps"]} == protocol["train"]
            and selected["value_penalty"] == protocol["methods"]["value_penalty"]
            and selected["fabr_v2"] == protocol["methods"]["fabr_v2"])
    print("Tuning choices " + ("match" if same else "DIFFER from") + f" {PROTOCOL.name}.")
    return 0


def figures(args):
    import plots
    root = Path(args.results or results_root(args) / "core")
    if not (root / "aggregate.csv").exists():
        print(f"No aggregate.csv in {root}; nothing to plot.")
        return 0
    plots.main(root)
    return 0


def shipped_runs(tier_dir):
    return sorted(p.parent for p in Path(tier_dir).glob("*/runs/*/*.pt"))


FLOAT_TOLERANCE = 1e-9  # averaged floats only; see verify()


def summary_mismatches(stored, fresh):
    """Keys of an evaluation summary that differ. Every metric must match exactly, except the
    float average ``average_reward``, which may differ in its last bits: Python >= 3.12 sums floats
    with compensated summation, so the same episode rewards can average differently by ~1e-15."""
    bad = []
    for key, metrics in stored.items():
        if key in ("model", "scenarios"):
            continue
        if not isinstance(metrics, dict):  # e.g. scenario_version
            if fresh.get(key) != metrics:
                bad.append(key)
            continue
        other = fresh.get(key, {})
        for metric, value in metrics.items():
            if metric == "average_reward":
                ok = metric in other and abs(value - other[metric]) <= FLOAT_TOLERANCE
            else:
                ok = other.get(metric) == value
            if not ok:
                bad.append(f"{key}:{metric}")
    return bad


def verify(args):
    """Re-evaluate every shipped checkpoint and re-aggregate.

    Exact: every episode (outcome, steps, reward, unsafe proposals, start state) must equal the
    shipped per-episode CSV, and every outcome rate must match. Float averages (mean reward and the
    aggregate means / CIs) must match to 1e-9, which allows only last-bit summation differences.
    """
    import csv
    import tempfile
    from evaluate import evaluate
    from run_experiments import aggregate, group_rows
    tiers = [d for d in sorted(SHIPPED.glob("*")) if shipped_runs(d)] if SHIPPED.exists() else []
    if not tiers:
        print(f"No shipped checkpoints in {SHIPPED.relative_to(HERE)} yet: they are added after the "
              "final runs. Nothing to verify.")
        return 0
    failures, episodes = 0, 0
    scratch = Path(tempfile.mkdtemp(prefix="verify_"))
    for tier_dir in tiers:
        rows = []
        runs = shipped_runs(tier_dir)
        print(f"== verify {tier_dir.name}: re-evaluating {len(runs)} checkpoints ==", flush=True)
        for run in runs:
            name = run.name
            method, seed = name.rsplit("_seed", 1)
            mode = run.parents[1].name
            stored = json.loads((run / "shield_off_eval.json").read_text())
            fresh_csv = scratch / f"{mode}_{name}.csv"
            fresh = evaluate(run / f"{name}.pt", output=fresh_csv, verbose=False)
            problems = summary_mismatches(stored, fresh)
            shipped_csv = run / "shield_off_episodes.csv"
            if shipped_csv.exists():
                a, b = shipped_csv.read_text().splitlines(), fresh_csv.read_text().splitlines()
                episodes += len(a) - 1
                if a != b:
                    differing = sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
                    problems.append(f"{differing} episode rows differ")
            else:
                problems.append("no shipped per-episode CSV")
            if problems:
                failures += 1
                print(f"  MISMATCH {mode}/{name}: {problems[:6]}")
            summary = json.loads((run / f"{name}_summary.json").read_text())
            config = method if method == "unshielded" else f"{method}@{mode}"
            rows += group_rows({"config": config, "mode": mode, "method": method}, int(seed), summary, fresh)
        shipped = {(r["config"], r["group"]): r for r in csv.DictReader((tier_dir / "aggregate.csv").open())}
        for entry in aggregate(rows):
            ref = shipped.get((entry["config"], entry["group"]))
            bad = ref is None or any(
                ref[k] != "" and abs(float(ref[k]) - float(v)) > FLOAT_TOLERANCE
                for k, v in entry.items() if k.endswith(("_mean", "_ci95")) and v != "")
            if bad:
                failures += 1
                print(f"  MISMATCH aggregate {entry['config']} / {entry['group']}")
    shutil.rmtree(scratch, ignore_errors=True)
    if failures:
        print(f"verify: {failures} mismatches")
    else:
        print(f"verify: all {episodes} shipped episodes identical; every outcome rate exact; "
              f"float averages within {FLOAT_TOLERANCE:g}")
    return int(failures > 0)


def ship(args):
    """Copy a finished tier into docs/results/v2/<tier> (checkpoints, summaries, tables, figures)."""
    tier = args.tier_name or "core"
    src, dst = results_root(args) / tier, SHIPPED / tier
    if not (src / "aggregate.csv").exists():
        sys.exit(f"{src} has no finished results")
    for name in ("per_seed.csv", "aggregate.csv", "paired.csv", "experiment_config.json"):
        if (src / name).exists():
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / name, dst / name)
    if (src / "figures").exists():
        shutil.copytree(src / "figures", dst / "figures", dirs_exist_ok=True)
    for beta in src.glob("*/fabr_fixed_beta.json"):
        (dst / beta.parent.name).mkdir(parents=True, exist_ok=True)
        shutil.copy2(beta, dst / beta.parent.name / beta.name)
    for run in shipped_runs(src):
        out = dst / run.relative_to(src)
        out.mkdir(parents=True, exist_ok=True)
        for f in run.iterdir():
            if f.suffix in (".pt", ".json", ".csv"):  # incl. per-episode CSVs, used by verify
                shutil.copy2(f, out / f.name)
    print(f"Shipped {tier} to {dst.relative_to(HERE)}")
    return 0


HANDLERS = {"smoke": smoke, "verify": verify, "tune": tune, "figures": figures, "ship": ship,
            **{t: experiment(t) for t in TIERS}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("commands", nargs="+", help=f"one or more of {COMMANDS} (ship takes a tier name)")
    p.add_argument("--jobs", type=int, default=default_jobs(), help="parallel training processes")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="run a batch of the tier's seeds")
    p.add_argument("--results", default=None, help="figures: results directory")
    p.add_argument("--results-root", default=None,
                   help="where tier outputs go (default: $SHIELDED_RL_RESULTS or ./results)")
    p.add_argument("--allow-draft", action="store_true", help="run a tier with an unfrozen protocol")
    args = p.parse_args(argv)
    commands, args.tier_name = list(args.commands), None
    if "ship" in commands:
        i = commands.index("ship")
        if i + 1 < len(commands) and commands[i + 1] in TIERS:
            args.tier_name = commands.pop(i + 1)
    unknown = [c for c in commands if c not in COMMANDS]
    if unknown:
        p.error(f"unknown command(s) {unknown}; choose from {COMMANDS}")
    os.chdir(HERE)
    for command in commands:
        code = HANDLERS[command](args)
        if code:
            print(f"`{command}` failed (exit code {code}); stopping.")
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
