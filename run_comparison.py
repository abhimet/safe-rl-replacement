"""Reproducible paired-seed comparison; no manual checkpoint filename guessing.

Three distinct tracks are available: historical model-free, model-based
collision safety-credit, and planner-assisted. Never pool them as though
they share training supervision.

Every final checkpoint is evaluated shield-OFF on the frozen scenario file
(default evaluation_scenarios_v2.json), so all methods share one evaluation.
Results are reported per layout with their split; see EVALUATION_PROTOCOL.md.
"""

import argparse
import csv
import json
from pathlib import Path

from evaluate import evaluate
from make_eval_scenarios import V2_PATH
from train import train

MODEL_FREE_METHODS = ("unshielded", "random_safe", "value_based", "value_penalty", "adaptive", "par")
ASSISTED_METHODS = ("unshielded", "random_safe", "value_based", "adaptive", "par")
SAFETY_CREDIT_METHODS = ("value_based", "par", "par_count", "random_safe", "adaptive", "unshielded")


def run_comparison(output="results/final_comparison", track="model_free",
                   steps=20000, seeds=(42, 43, 44), methods=None,
                   expert_updates=2000, expert_weight=0.3,
                   par_schedule="absolute", overwrite=False, scenarios=V2_PATH):
    if steps < 1 or not seeds:
        raise ValueError("steps must be positive and seeds nonempty")
    if track not in ("model_free", "assisted", "safety_credit"):
        raise ValueError("track must be model_free, safety_credit or assisted")
    permitted = (MODEL_FREE_METHODS if track == "model_free" else
                 ASSISTED_METHODS if track == "assisted" else SAFETY_CREDIT_METHODS)
    methods = tuple(methods) if methods else permitted
    if not methods or len(set(methods)) != len(methods) or any(m not in permitted for m in methods):
        raise ValueError(f"Unsupported methods for {track}: {methods}")
    root = Path(output) / track
    root.mkdir(parents=True, exist_ok=True)
    experiments = []
    for method in methods:
        for seed in seeds:
            directory = root / f"{method}_seed{seed}_steps{steps}"
            if directory.exists() and any(directory.iterdir()) and not overwrite:
                raise FileExistsError(f"Experiment directory exists: {directory}. Use a different --output.")
            directory.mkdir(parents=True, exist_ok=True)
            actual_method = "par" if method == "par_count" else method
            kwargs = dict(method=actual_method, max_env_steps=steps, seed=seed,
                          size=10, observation_mode="map", train_layouts=("rooms", "zigzag"),
                          train_start_policy="all_reachable", storage_mode="executed",
                          reward_shaping="potential", shaping_scale=4.,
                          output=str(directory))
            if track == "safety_credit":
                # The unshielded baseline cannot use a training shield.
                if method != "unshielded":
                    kwargs.update(storage_mode="counterfactual",
                                  backup_mode="shield_safe")
                kwargs["architecture"] = "spatial"
            if method in ("par", "par_count"):
                kwargs.update(par_schedule=par_schedule,
                              par_beta_min=0. if par_schedule == "absolute" else -4.,
                              par_beta_max=4.)
            if method == "par_count":
                kwargs.update(par_selector="count_aware",
                              par_novelty_weight=3.)
            if track == "assisted":
                kwargs.update(architecture="spatial",
                              expert_pretrain_updates=expert_updates,
                              expert_loss_weight=expert_weight)
            train(**kwargs)
            checkpoints = sorted(directory.glob("*.pt"))
            if len(checkpoints) != 1:
                raise RuntimeError(f"Expected one final checkpoint in {directory}, found {checkpoints}")
            experiments += evaluate_checkpoint(checkpoints[0], scenarios,
                                               directory / "shield_off_scenarios.csv",
                                               track=track, method=method, seed=seed,
                                               training_steps=steps)
            with (directory / "run_config.json").open("w") as stream:
                json.dump(kwargs, stream, indent=2)

    path = root / "per_seed_layout.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=experiments[0].keys())
        writer.writeheader()
        writer.writerows(experiments)
    grouped = {}
    for row in experiments:
        key = (row["method"], row["layout"])
        grouped.setdefault(key, []).append(row)
    aggregate = []
    for (method, layout), group in sorted(grouped.items()):
        aggregate.append({"method": method, "layout": layout, "split": group[0]["split"],
                          "seeds": len(group),
                          "mean_success_rate": sum(r["success_rate"] for r in group)/len(group),
                          "mean_collision_rate": sum(r["collision_rate"] for r in group)/len(group),
                          "mean_unsafe_proposal_rate": sum(r["unsafe_proposal_rate"] for r in group)/len(group)})
    with (root / "aggregate.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=aggregate[0].keys())
        writer.writeheader()
        writer.writerows(aggregate)
    print(f"Saved {len(experiments)} shield-off evaluations to {path}")
    return aggregate


def evaluate_checkpoint(checkpoint, scenarios, output, **labels):
    """Shield-OFF evaluation on every frozen scenario; one row per layout, tagged with its split."""
    with open(scenarios, encoding="utf-8") as stream:
        spec = json.load(stream)
    summary = evaluate(model=str(checkpoint), scenarios=str(scenarios), output=str(output))
    rows = []
    for layout in dict.fromkeys(s["layout"] for s in spec["scenarios"]):
        first = next(s for s in spec["scenarios"] if s["layout"] == layout)
        rows.append({**labels, "layout": layout, "split": first["split"],
                     "family": first.get("family", ""),
                     "scenario_version": spec["version"],
                     "checkpoint": str(checkpoint), **summary[f"layout={layout}"]})
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/final_comparison")
    p.add_argument("--track", choices=("model_free", "assisted", "safety_credit"), default="model_free")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--methods", nargs="+", default=None)
    p.add_argument("--expert-updates", type=int, default=2000)
    p.add_argument("--expert-weight", type=float, default=0.3)
    p.add_argument("--par-schedule", choices=("relative", "absolute"), default="absolute")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--scenarios", default=str(V2_PATH),
                   help="Frozen scenario file for shield-OFF evaluation (default: v2)")
    run_comparison(**vars(p.parse_args()))
