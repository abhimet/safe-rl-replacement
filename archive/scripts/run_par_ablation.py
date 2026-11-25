"""Run the four PAR beta ablations and evaluate every reachable start.

Put this file next to train.py and evaluate.py, then run:
    python run_par_ablation.py

It stops at the FIRST training/evaluation error instead of generating
misleading FileNotFoundError messages for models that were never saved.
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


CONDITIONS = (
    ("adaptive", None, ""),
    ("fixed_neg4", -4.0, "_fixed_beta_neg4"),
    ("fixed_zero", 0.0, "_fixed_beta_pos0"),
    ("fixed_pos4", 4.0, "_fixed_beta_pos4"),
)


def run(command):
    print("\n$ " + " ".join(map(str, command)), flush=True)
    subprocess.run(command, check=True)


def matching_summary(path, seed, episodes, beta, size, window):
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return (
            data.get("method") == "par"
            and data.get("storage_mode") == "both"
            and data.get("backup_mode") == "standard"
            and data.get("seed") == seed
            and data.get("episodes") == episodes
            and data.get("environment", {}).get("size") == size
            and data.get("window") == window
            and data.get("par_fixed_beta") == beta
        )
    except (OSError, ValueError, TypeError):
        return False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--size", type=int, default=6)
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--output", default="results/par_ablation")
    p.add_argument("--force", action="store_true", help="Retrain even if a matching model exists")
    args = p.parse_args()

    project = Path(__file__).resolve().parent
    train = project / "train.py"
    evaluate = project / "evaluate.py"
    if not train.is_file() or not evaluate.is_file():
        p.error("Put run_par_ablation.py in the same folder as train.py and evaluate.py")

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    comparison = []

    for label, beta, suffix in CONDITIONS:
        prefix = f"par_seed{args.seed}_both{suffix}"
        model = out / f"{prefix}.pt"
        summary = out / f"{prefix}_summary.json"
        evaluation = out / f"{prefix}_off.csv"
        print(f"\n=== {label} ===", flush=True)

        if not args.force and model.is_file() and matching_summary(
            summary, args.seed, args.episodes, beta, args.size, args.window
        ):
            print(f"Reusing completed model: {model}", flush=True)
        else:
            cmd = [
                sys.executable, str(train),
                "--method", "par",
                "--storage-mode", "both",
                "--backup-mode", "standard",
                "--episodes", str(args.episodes),
                "--window", str(args.window),
                "--seed", str(args.seed),
                "--size", str(args.size),
                "--output", str(out),
            ]
            if beta is not None:
                cmd += ["--par-fixed-beta", str(beta)]
            run(cmd)

        if not model.is_file():
            raise RuntimeError(
                f"Training completed without creating expected model: {model}\n"
                "Check the saved model filename printed by train.py."
            )

        run([
            sys.executable, str(evaluate),
            "--model", str(model),
            "--size", str(args.size),
            "--all-starts",
            "--output", str(evaluation),
        ])
        if not evaluation.is_file():
            raise RuntimeError(f"Evaluation did not create {evaluation}")

        with evaluation.open(newline="", encoding="utf-8") as f:
            records = list(csv.DictReader(f))
        if not records:
            raise RuntimeError(f"No evaluation episodes in {evaluation}")

        n = len(records)
        successes = sum(int(row["success"]) for row in records)
        collisions = sum(int(row["collision"]) for row in records)
        comparison.append({
            "condition": label,
            "seed": args.seed,
            "episodes_trained": args.episodes,
            "evaluation_starts": n,
            "success_rate": round(successes / n, 4),
            "collision_rate": round(collisions / n, 4),
            "model": str(model),
        })

    comparison_path = out / f"par_ablation_seed{args.seed}_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=comparison[0].keys())
        writer.writeheader()
        writer.writerows(comparison)

    print("\n=== Completed comparison ===", flush=True)
    for row in comparison:
        print(
            f"{row['condition']:>11}: "
            f"success={row['success_rate']:.1%}, "
            f"collision={row['collision_rate']:.1%} "
            f"({row['evaluation_starts']} starts)"
        )
    print(f"Saved: {comparison_path}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(
            f"\nSTOPPED: a command failed (exit code {exc.returncode}). "
            "Fix the error above; no further experiments were attempted.",
            file=sys.stderr,
        )
        sys.exit(exc.returncode)
