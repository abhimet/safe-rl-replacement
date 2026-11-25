"""Stage 2: 10x10 positive-control calibration, not a final comparison.

Usage: python run_10x10_calibration.py [--episodes 300] [--seed 42]

Train on rooms/zigzag, full-map observation and uniformly balanced starts.
Do not evaluate clutter until protocol/hyperparameters are frozen.
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(script, *args):
    command = [sys.executable, str(HERE / script), *map(str, args)]
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=HERE, check=True)


def results_for(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No evaluation rows in {path}")
    return {
        "starts": len(rows),
        "successes": sum(int(row["success"]) for row in rows),
        "collisions": sum(int(row["collision"]) for row in rows),
        "success_rate": round(sum(int(r["success"]) for r in rows)/len(rows), 4),
        "collision_rate": round(sum(int(r["collision"]) for r in rows)/len(rows), 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be at least 1")

    folder = HERE / "results" / "10x10_calibration" / f"neutral_seed{args.seed}_ep{args.episodes}"
    folder.mkdir(parents=True, exist_ok=True)
    # Fixed beta 0 is the neutral distribution control. This tests if the
    # 104-input DQN and training setup can learn before testing adaptivity.
    run("train.py", "--method", "par", "--storage-mode", "both",
        "--backup-mode", "standard", "--size", "10",
        "--observation-mode", "map", "--train-layouts", "rooms", "zigzag",
        "--train-start-policy", "all_reachable", "--episodes", args.episodes,
        "--window", "100", "--seed", args.seed, "--par-fixed-beta", "0",
        "--output", folder)

    model = folder / f"par_seed{args.seed}_both_allstarts_fixed_beta_pos0.pt"
    if not model.is_file():
        raise RuntimeError(f"Model missing: {model}")

    train_csv = model.with_suffix(".csv")
    with train_csv.open(newline="") as handle:
        records = list(csv.DictReader(handle))
    distinct_starts = {layout: len({r["train_start"] for r in records if r["layout"] == layout})
                       for layout in ("rooms", "zigzag")}

    comparisons = {}
    for layout in ("rooms", "zigzag"):
        report_csv = folder / f"{layout}_shield_off.csv"
        run("evaluate.py", "--model", model, "--size", "10",
            "--observation-mode", "map", "--layout", layout,
            "--all-starts", "--output", report_csv)
        comparisons[layout] = results_for(report_csv)
        comparisons[layout]["distinct_training_starts"] = distinct_starts[layout]

    result = {"experiment": "10x10 calibration (fixed beta 0)",
              "training_seed": args.seed, "training_episodes": args.episodes,
              "training_start_policy": "all_reachable",
              "training_layouts": ["rooms", "zigzag"],
              "holdout_test_layout": "clutter (NOT evaluated)",
              "results": comparisons}
    (folder / "calibration_report.json").write_text(json.dumps(result, indent=2))
    print("\nCALIBRATION FINISHED (NOT A FINAL RESEARCH RESULT)")
    print(json.dumps(result, indent=2))
    print("Interpret these as in-training-layout diagnostic results; do NOT claim generalization.")
    print("Upload calibration_report.json or paste the final summary.")


if __name__ == "__main__":
    main()
