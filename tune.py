"""Tune on development maps only: the train split (rooms, zigzag) and dev_inspected (clutter).

Held-out maps (pillars, spiral, heldout_1..3) are never evaluated here: ``dev_evaluate``
refuses any other split. Tuning seeds (0, 1) are disjoint from the final seeds (42-51).

    python tune.py --jobs 6                     # every stage, in order (resumable)
    python tune.py --stage value_penalty        # one stage (needs the earlier stages' choices)
    python tune.py --write-protocol             # rerun (reusing finished trials) and copy the
                                                # choices into configs/final_protocol.json

The selection rules are fixed here, before any result is seen:

  nav            highest_q@cf at 50k steps. Starting from the current config, try one change at a
                 time in this order: n_step 3, lr 3e-4, replay capacity 50k, epsilon_fraction 0.5.
                 Keep a change only if mean familiar (train-split) success rises by >= 2 pp.
  budget         the chosen config at 100k steps. If familiar success is below 50%, also try 150k
                 and use it only if it adds >= 5 pp.
  value_penalty  value_penalty@none, penalty in {0.25, 0.5, 1, 2, 5, 10}.
  fabr_v2        fabr_v2@margin: margin_lambda in {0.1, 0.5, 1} at margin 1; then margin in
                 {0.5, 2} at the best lambda; then stop-gradient on the best. 6 configs, the same
                 number as value_penalty.
  Method settings are chosen by the highest mean (success - collision) over all dev episodes
  (train + dev_inspected, 226 per seed); ties keep the earlier config.

Outputs: results/tuning/<stage>/<trial>/<method>_seed<k>/, results/tuning/selected.json and
docs/TUNING.md (every config tried, with its scores).
"""
import argparse
import json
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from evaluate import evaluate
from make_eval_scenarios import V2_PATH
from run_experiments import load_protocol, run_matches
from train import train

DEV_SPLITS = ("train", "dev_inspected")
TUNING_SEEDS = (0, 1)
NAV_STEPS = 50000
# The pre-tuning configuration (the v1 settings). Fixed here so that tuning is reproducible.
NAV_BASE = dict(lr=1e-4, n_step=1, capacity=10000, epsilon_fraction=0.8, pool_size=100, pool_seed=7,
                shaping_scale=4.0, window=100)
NAV_MIN_GAIN = 0.02
BUDGET_TARGET, BUDGET_MIN_GAIN = 0.50, 0.05
NAV_CHANGES = (("n_step", {"n_step": 3}), ("lr", {"lr": 3e-4}), ("capacity", {"capacity": 50000}),
               ("epsilon", {"epsilon_fraction": 0.5}))
PENALTIES = (0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
LAMBDAS, MARGINS = (0.1, 0.5, 1.0), (0.5, 2.0)
STAGES = ("nav", "budget", "value_penalty", "fabr_v2")
PROTOCOL = Path("configs/final_protocol.json")
ROOT = Path("results/tuning")
REPORT = Path("docs/TUNING.md")


def dev_evaluate(model, output=None, splits=DEV_SPLITS):
    """Shield-off evaluation restricted to development maps."""
    if not splits or any(s not in DEV_SPLITS for s in splits):
        raise ValueError(f"tuning may only evaluate the {DEV_SPLITS} splits, got {splits}")
    result = evaluate(model, V2_PATH, output=output, splits=splits, verbose=False)
    seen = {k.split("=", 1)[1] for k in result if k.startswith("split=")}
    if not seen <= set(DEV_SPLITS):
        raise RuntimeError(f"held-out scenarios were evaluated during tuning: {seen}")
    return result


def run_trial(directory, method, seed, kwargs):
    """Train one tuning run and evaluate it on dev maps (reused if already finished)."""
    directory = Path(directory) / f"{method}_seed{seed}"
    summary_path = directory / f"{method}_seed{seed}_summary.json"
    eval_path = directory / "dev_eval.json"
    if summary_path.exists() and eval_path.exists():
        summary = json.loads(summary_path.read_text())
        if not run_matches(summary, method, kwargs):
            raise FileExistsError(f"{directory} holds a tuning run with different settings")
        return json.loads(eval_path.read_text())
    train(method=method, seed=seed, output=directory, **kwargs)
    result = dev_evaluate(directory / f"{method}_seed{seed}.pt", directory / "dev_episodes.csv")
    eval_path.write_text(json.dumps(result, indent=2))
    return result


def scores(results):
    """Mean over seeds of the selection scores."""
    def mean(f):
        return statistics.mean(f(r) for r in results)
    return {"familiar_success": mean(lambda r: r["split=train"]["success_rate"]),
            "clutter_success": mean(lambda r: r["split=dev_inspected"]["success_rate"]),
            "dev_success": mean(lambda r: r["overall"]["success_rate"]),
            "dev_collision": mean(lambda r: r["overall"]["collision_rate"]),
            "dev_score": mean(lambda r: r["overall"]["success_rate"] - r["overall"]["collision_rate"])}


class Tuner:
    def __init__(self, jobs=1, root=ROOT, report=REPORT):
        self.jobs, self.root, self.report = jobs, Path(root), Path(report)
        self.root.mkdir(parents=True, exist_ok=True)
        self.selected_path = self.root / "selected.json"
        self.log_path = self.root / "log.json"
        self.selected = json.loads(self.selected_path.read_text()) if self.selected_path.exists() else {}
        self.log = json.loads(self.log_path.read_text()) if self.log_path.exists() else []

    def trials(self, stage, entries):
        """Run [(trial name, method, kwargs)] on every tuning seed, in parallel. Returns scores."""
        tasks = [(self.root / stage / name, method, seed, kwargs)
                 for name, method, kwargs in entries for seed in TUNING_SEEDS]
        if self.jobs > 1 and len(tasks) > 1:
            with ProcessPoolExecutor(max_workers=min(self.jobs, len(tasks))) as pool:
                results = list(pool.map(run_trial, *zip(*tasks)))
        else:
            results = [run_trial(*t) for t in tasks]
        n = len(TUNING_SEEDS)
        out = []
        for i, (name, method, kwargs) in enumerate(entries):
            score = scores(results[i * n:(i + 1) * n])
            record = {"stage": stage, "trial": name, "method": method, "settings": kwargs, **score}
            self.log = [r for r in self.log if (r["stage"], r["trial"]) != (stage, name)] + [record]
            out.append(score)
            print(f"[tune {stage}] {name}: familiar {score['familiar_success']:.3f}, "
                  f"dev success-collision {score['dev_score']:.3f}", flush=True)
        self.save()
        return out

    def choose(self, key, value):
        self.selected[key] = value
        self.save()

    def mark(self, stage, trial):
        for r in self.log:
            if r["stage"] == stage:
                r["chosen"] = r["trial"] == trial
        self.save()

    def save(self):
        self.selected_path.write_text(json.dumps(self.selected, indent=2))
        self.log_path.write_text(json.dumps(self.log, indent=2))
        write_report(self.log, self.selected, self.report)

    def need(self, *keys):
        missing = [k for k in keys if k not in self.selected]
        if missing:
            raise RuntimeError(f"run the earlier tuning stages first (missing {missing})")

    # ---- stages ----

    def nav(self):
        current = {**NAV_BASE, "steps": NAV_STEPS, "feedback_mode": "cf"}
        best_name = "baseline"
        best = self.trials("nav", [(best_name, "highest_q", dict(current))])[0]["familiar_success"]
        for name, change in NAV_CHANGES:
            candidate = {**current, **change}
            trial = f"{best_name}+{name}" if best_name != "baseline" else name
            score = self.trials("nav", [(trial, "highest_q", candidate)])[0]["familiar_success"]
            if score >= best + NAV_MIN_GAIN:
                current, best, best_name = candidate, score, trial
        self.mark("nav", best_name)
        train_settings = {k: v for k, v in current.items() if k not in ("steps", "feedback_mode")}
        self.choose("train", train_settings)
        return train_settings

    def budget(self):
        self.need("train")
        base = {**self.selected["train"], "feedback_mode": "cf"}
        steps, best = 100000, self.trials("budget", [("100k", "highest_q", {**base, "steps": 100000})])[0]
        chosen = "100k"
        if best["familiar_success"] < BUDGET_TARGET:
            longer = self.trials("budget", [("150k", "highest_q", {**base, "steps": 150000})])[0]
            if longer["familiar_success"] >= best["familiar_success"] + BUDGET_MIN_GAIN:
                steps, chosen = 150000, "150k"
        self.mark("budget", chosen)
        self.choose("steps", steps)
        return steps

    def base(self, mode):
        self.need("train", "steps")
        return {**self.selected["train"], "steps": self.selected["steps"], "feedback_mode": mode}

    def value_penalty(self):
        base = self.base("none")
        entries = [(f"penalty={p:g}", "value_penalty", {**base, "penalty": p}) for p in PENALTIES]
        results = self.trials("value_penalty", entries)
        i = max(range(len(entries)), key=lambda j: (results[j]["dev_score"], -j))
        self.mark("value_penalty", entries[i][0])
        self.choose("value_penalty", {"penalty": PENALTIES[i]})
        return PENALTIES[i]

    def fabr_v2(self):
        base = self.base("margin")

        def entry(lam, m, stop):
            name = f"lambda={lam:g},m={m:g}" + (",stop_grad" if stop else "")
            return name, "fabr_v2", {**base, "margin_lambda": lam, "margin": m, "margin_stop_grad": stop}

        tried = []
        first = [entry(lam, 1.0, False) for lam in LAMBDAS]
        tried += list(zip(first, self.trials("fabr_v2", first)))
        lam = max(tried, key=lambda t: t[1]["dev_score"])[0][2]["margin_lambda"]  # first max wins
        second = [entry(lam, m, False) for m in MARGINS]
        tried += list(zip(second, self.trials("fabr_v2", second)))
        best = max(tried, key=lambda t: t[1]["dev_score"])
        settings = best[0][2]
        third = entry(settings["margin_lambda"], settings["margin"], True)
        stop_score = self.trials("fabr_v2", [third])[0]
        if stop_score["dev_score"] > best[1]["dev_score"]:
            best = (third, stop_score)
        name, _, settings = best[0]
        self.mark("fabr_v2", name)
        chosen = {k: settings[k] for k in ("margin", "margin_lambda", "margin_stop_grad")}
        self.choose("fabr_v2", chosen)
        return chosen


def write_report(log, selected, path=REPORT):
    lines = ["# Tuning log (development maps only)", "",
             "Generated by `tune.py`. Every config tried is listed. Scores are means over tuning seeds "
             f"{list(TUNING_SEEDS)} on the train split (rooms, zigzag = *familiar*) and dev_inspected "
             "(clutter). Held-out maps were never evaluated. Selection rules are in `tune.py`'s docstring.",
             "", "**Selected:** `" + json.dumps(selected) + "`", ""]
    for stage in STAGES:
        rows = [r for r in log if r["stage"] == stage]
        if not rows:
            continue
        lines += [f"## {stage}", "", "| trial | familiar success | clutter success | dev success | "
                  "dev collision | dev success − collision | chosen |", "|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| `{r['trial']}` | {r['familiar_success']:.3f} | {r['clutter_success']:.3f} | "
                         f"{r['dev_success']:.3f} | {r['dev_collision']:.3f} | {r['dev_score']:.3f} | "
                         f"{'✔' if r.get('chosen') else ''} |")
        lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines))


def write_protocol(selected, path=PROTOCOL):
    """Copy the tuned values into the protocol file. It stays a draft until approved and frozen."""
    for key in ("train", "steps", "value_penalty", "fabr_v2"):
        if key not in selected:
            raise RuntimeError(f"tuning is incomplete (missing {key!r})")
    protocol = json.loads(Path(path).read_text())
    if protocol["frozen"]:
        raise RuntimeError(f"{path} is frozen; it cannot be changed")
    protocol["train"] = {**selected["train"], "steps": selected["steps"]}
    protocol["methods"]["value_penalty"] = selected["value_penalty"]
    protocol["methods"]["fabr_v2"] = selected["fabr_v2"]
    protocol["note"] = ("DRAFT with tuned values from tune.py (see docs/TUNING.md); "
                        "set frozen=true once approved. Final runs read ONLY from this file.")
    Path(path).write_text(json.dumps(protocol, indent=2) + "\n")
    load_protocol(path)  # still valid
    print(f"Wrote tuned values to {path} (still a draft)")


def main(stage="all", jobs=1, write=False):
    tuner = Tuner(jobs)
    for name in (STAGES if stage == "all" else (stage,)):
        getattr(tuner, name)()
    print(f"Selected: {json.dumps(tuner.selected)}\nReport: {REPORT}")
    if write:
        write_protocol(tuner.selected)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=("all", *STAGES), default="all")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--write-protocol", dest="write", action="store_true",
                   help="Copy the selected values into configs/final_protocol.json (stays a draft)")
    args = p.parse_args()
    main(args.stage, args.jobs, args.write)  # finished trials are reused, so reruns are cheap
