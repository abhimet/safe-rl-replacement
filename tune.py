"""Tune on development maps only: the train split (rooms, zigzag) and dev_inspected (clutter).

Held-out maps (pillars, spiral, heldout_1..3) are never evaluated here: ``dev_evaluate``
refuses any other split. Tuning seeds (0, 1) are disjoint from the final seeds (42-51).

    python tune.py --jobs 6                     # round 1: every stage, in order (resumable)
    python tune.py --round 2 --jobs 16 --write-protocol   # round 2 (Zaratan) -> proposed protocol
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

Round 2 (``--round 2``, run on Zaratan) is one navigation-improvement round on top of round 1's
choices (read from docs/tuning/round1/selected.json, so it does not depend on rerunning round 1):

  nav_r2         highest_q@cf at round 1's settings and budget. Changes, one at a time and each on
                 top of what was kept: the convolutional Q-network, then shaping with Phi = -d
                 (instead of -4 d / d_max; both use BFS shortest-path distance). Keep a change only
                 if dev success rises by >= 2 pp, where dev success is defined exactly as round 1's
                 `dev_success`: the success rate over all 226 dev episodes of a seed (familiar +
                 clutter pooled), averaged over the tuning seeds.
  value_penalty, fabr_v2   the same grids and rules as round 1, at the round-2 navigation settings.
                 If nav_r2 keeps nothing, the navigation settings equal round 1's and round 1's grids
                 already ran at exactly those settings, so they are not re-run: round 1's values are
                 carried over (``carry_round1_methods``).
  Nothing is selected by comparing methods with each other.

Outputs: results/tuning[_r2]/<stage>/<trial>/<method>_seed<k>/, selected.json and log.json there,
and docs/TUNING.md (every config of every round, with its scores). Round 2's --write-protocol
writes a proposal to configs/final_protocol.proposed.json; the frozen protocol is not touched.
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
# Round 2: one change at a time from round 1's frozen choices.
# Applied in this order, each on top of whatever was kept before (so Phi = -d is tested on the
# architecture that won the conv decision).
NAV_R2_CHANGES = (("conv", {"architecture": "conv"}), ("phi_distance", {"shaping": "distance"}))
NAV_R2_MIN_GAIN = 0.02
ROUND_STAGES = {1: STAGES, 2: ("nav_r2", "value_penalty", "fabr_v2")}
PROTOCOL = Path("configs/final_protocol.json")
PROPOSED = Path("configs/final_protocol.proposed.json")
ROOT = Path("results/tuning")
ROOTS = {1: ROOT, 2: Path("results/tuning_r2")}
ROUND1_RECORD = Path("docs/tuning/round1")  # committed copy of round 1's log.json and selected.json
REPORT = Path("docs/TUNING.md")
ROUND_NOTES = {
    1: "Round 1 (MacBook, M1, torch 2.15): navigation and budget for the spatial-feature network, "
       "then the method grids. Navigation kept a change if mean **familiar** success rose >= 2 pp.",
    2: "Round 2 (Zaratan, torch 2.14.1+cpu): one navigation-improvement round on top of round 1's "
       "choices, applied identically to every method: the convolutional Q-network, then (on the "
       "winning architecture) shaping with Φ = −d instead of −4·d/d_max. A change is kept if **dev "
       "success** (round 1's definition: familiar + clutter pooled, 226 episodes per seed, mean over "
       "seeds) rises >= 2 pp for highest_q@cf. If a change is kept, the same value_penalty and fabr_v2 "
       "grids are re-run at the new settings; otherwise round 1's values stand. Shaping already used "
       "BFS shortest-path distance around walls.",
}


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
        return {**json.loads(eval_path.read_text()), "training": summary["training"]}
    summary = train(method=method, seed=seed, output=directory, **kwargs)
    result = dev_evaluate(directory / f"{method}_seed{seed}.pt", directory / "dev_episodes.csv")
    eval_path.write_text(json.dumps(result, indent=2))
    return {**result, "training": summary["training"]}


def scores(results):
    """Mean over seeds of the selection scores."""
    def mean(f):
        return statistics.mean(f(r) for r in results)
    return {"familiar_success": mean(lambda r: r["split=train"]["success_rate"]),
            "clutter_success": mean(lambda r: r["split=dev_inspected"]["success_rate"]),
            "dev_success": mean(lambda r: r["overall"]["success_rate"]),
            "dev_collision": mean(lambda r: r["overall"]["collision_rate"]),
            "dev_score": mean(lambda r: r["overall"]["success_rate"] - r["overall"]["collision_rate"]),
            "train_success": mean(lambda r: r["training"]["success_rate"]),
            "train_final_success": mean(lambda r: r["training"].get("final_success_rate") or 0.0)}


class Tuner:
    def __init__(self, jobs=1, root=None, report=REPORT, round=1, round1=ROUND1_RECORD):
        if round not in ROUND_STAGES:
            raise ValueError(f"round must be one of {sorted(ROUND_STAGES)}")
        self.round, self.round1 = round, Path(round1)
        self.jobs, self.root, self.report = jobs, Path(root or ROOTS[round]), Path(report)
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
        rounds = {self.round: (self.log, self.selected)}
        other = self.round1 if self.round == 2 else ROOTS[2]  # keep the other round in the report
        if (other / "log.json").exists() and (other / "selected.json").exists():
            rounds[3 - self.round] = (json.loads((other / "log.json").read_text()),
                                      json.loads((other / "selected.json").read_text()))
        write_report(rounds, self.report)

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

    def nav_r2(self):
        """One navigation-improvement round on top of round 1's choices.

        The baseline and the first change are independent, so they run together; every later
        change is applied on top of whatever was kept. The decisions are the same as running
        them one after another.
        """
        first = json.loads((self.round1 / "selected.json").read_text())
        current = {**first["train"], "steps": first["steps"], "feedback_mode": "cf"}
        best_name = "round1"
        (name0, change0), *rest = NAV_R2_CHANGES
        both = self.trials("nav_r2", [(best_name, "highest_q", dict(current)),
                                      (f"{best_name}+{name0}", "highest_q", {**current, **change0})])
        best = both[0]["dev_success"]
        if both[1]["dev_success"] >= best + NAV_R2_MIN_GAIN:
            current, best, best_name = {**current, **change0}, both[1]["dev_success"], f"{best_name}+{name0}"
        for name, change in rest:
            candidate = {**current, **change}
            trial = f"{best_name}+{name}"
            score = self.trials("nav_r2", [(trial, "highest_q", candidate)])[0]["dev_success"]
            if score >= best + NAV_R2_MIN_GAIN:
                current, best, best_name = candidate, score, trial
        self.mark("nav_r2", best_name)
        self.choose("train", {k: v for k, v in current.items() if k not in ("steps", "feedback_mode")})
        self.choose("steps", current["steps"])
        return current

    def navigation_unchanged(self):
        first = json.loads((self.round1 / "selected.json").read_text())
        return (self.selected.get("train") == first["train"]
                and self.selected.get("steps") == first["steps"])

    def carry_round1_methods(self):
        """Round 2 kept no navigation change: keep round 1's method settings without re-running."""
        if not self.navigation_unchanged():
            raise RuntimeError("navigation changed in round 2; the method grids must be re-run")
        first = json.loads((self.round1 / "selected.json").read_text())
        self.selected["value_penalty"], self.selected["fabr_v2"] = first["value_penalty"], first["fabr_v2"]
        self.choose("carried_from_round1", ["value_penalty", "fabr_v2"])
        print("[tune round 2] navigation unchanged: round 1's value_penalty and fabr_v2 grids ran at "
              "exactly these settings, so they are not re-run; round 1's values are kept.", flush=True)

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


def write_report(rounds, path=REPORT):
    """rounds: {round number: (log, selected)}. Lists every config tried in every round."""
    lines = ["# Tuning log (development maps only)", "",
             "Generated by `tune.py`. Every config tried is listed. Scores are means over tuning seeds "
             f"{list(TUNING_SEEDS)} on the train split (rooms, zigzag = *familiar*) and dev_inspected "
             "(clutter). *dev* pools both (226 episodes). Held-out maps were never evaluated. "
             "*train goal* is the goal-reached rate during training (shield on for shielded methods), "
             "over all episodes / the last 10% of steps. Selection rules are in `tune.py`'s docstring.",
             ""]

    def pct(r, key):
        return "–" if r.get(key) is None else f"{r[key]:.3f}"

    for number in sorted(rounds):
        log, selected = rounds[number]
        lines += [f"# Round {number}", "", ROUND_NOTES.get(number, ""), "",
                  "**Selected:** `" + json.dumps(selected) + "`", ""]
        if selected.get("carried_from_round1"):
            lines += ["**Method grids not re-run.** Round 2 left navigation unchanged (neither the conv "
                      "network nor Φ = −d passed the 2 pp rule). Round 1's value_penalty and fabr_v2 grids "
                      "already ran at exactly these settings (n-step 3, 100k steps, spatial network, "
                      "Φ = −4·d/d_max; checked against all 24 run summaries), so round 1's values are kept "
                      "and the frozen protocol is unchanged. (Round 1 ran on a MacBook, round 2 on Zaratan.)",
                      ""]
        for stage in ROUND_STAGES[number]:
            rows = [r for r in log if r["stage"] == stage]
            if not rows:
                continue
            lines += [f"## {stage} (round {number})", "",
                      "| trial | familiar success | clutter success | dev success | dev collision | "
                      "dev success − collision | train goal (all / last 10%) | chosen |",
                      "|---|---|---|---|---|---|---|---|"]
            for r in rows:
                lines.append(f"| `{r['trial']}` | {r['familiar_success']:.3f} | {r['clutter_success']:.3f} | "
                             f"{r['dev_success']:.3f} | {r['dev_collision']:.3f} | {r['dev_score']:.3f} | "
                             f"{pct(r, 'train_success')} / {pct(r, 'train_final_success')} | "
                             f"{'✔' if r.get('chosen') else ''} |")
            lines.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines))


def write_protocol(selected, path=PROTOCOL, source=None, version=None):
    """Copy the tuned values into a protocol file, which stays a draft until approved and frozen.

    ``source`` (default: ``path``) is the protocol to start from, e.g. the frozen round-1 protocol
    when writing a round-2 proposal to a new file.
    """
    for key in ("train", "steps", "value_penalty", "fabr_v2"):
        if key not in selected:
            raise RuntimeError(f"tuning is incomplete (missing {key!r})")
    protocol = json.loads(Path(source or path).read_text())
    if Path(path).exists() and json.loads(Path(path).read_text())["frozen"]:
        raise RuntimeError(f"{path} is frozen; it cannot be changed")
    protocol.update(frozen=False, frozen_on=None)
    if version is not None:
        protocol["version"] = version
    protocol["train"] = {**selected["train"], "steps": selected["steps"]}
    protocol["methods"]["value_penalty"] = selected["value_penalty"]
    protocol["methods"]["fabr_v2"] = selected["fabr_v2"]
    protocol["note"] = ("DRAFT with tuned values from tune.py (see docs/TUNING.md); "
                        "set frozen=true once approved. Final runs read ONLY from this file.")
    Path(path).write_text(json.dumps(protocol, indent=2) + "\n")
    load_protocol(path)  # still valid
    print(f"Wrote tuned values to {path} (still a draft)")


def main(stage="all", jobs=1, write=False, round=1):
    tuner = Tuner(jobs, round=round)
    stages = ROUND_STAGES[round]
    if stage != "all" and stage not in stages:
        raise ValueError(f"round {round} has stages {stages}")
    for name in (stages if stage == "all" else (stage,)):
        if round == 2 and name in ("value_penalty", "fabr_v2") and tuner.navigation_unchanged():
            if "carried_from_round1" not in tuner.selected:
                tuner.carry_round1_methods()
            continue
        getattr(tuner, name)()
    print(f"Selected: {json.dumps(tuner.selected)}\nReport: {REPORT}")
    if write and round == 1:
        write_protocol(tuner.selected)
    elif write and "carried_from_round1" in tuner.selected:
        print(f"Round 2 changed nothing: {PROTOCOL} stays as frozen; no proposal written.")
    elif write:
        write_protocol(tuner.selected, PROPOSED, source=PROTOCOL, version=round)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--round", type=int, choices=sorted(ROUND_STAGES), default=1)
    p.add_argument("--stage", choices=("all", *dict.fromkeys(s for v in ROUND_STAGES.values() for s in v)),
                   default="all")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--write-protocol", dest="write", action="store_true",
                   help="round 1: copy the choices into configs/final_protocol.json (stays a draft); "
                        "round 2: write configs/final_protocol.proposed.json")
    args = p.parse_args()
    main(args.stage, args.jobs, args.write, args.round)  # finished trials are reused: reruns are cheap
