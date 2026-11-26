"""Test whether giving rejected unsafe proposals their known GridWorld collision
outcome improves independent policy safety.

Run from the project directory:
    python run_10x10_safety_signal.py

This runs ONLY the counterfactual-collision control; existing models reused.
NEVER evaluates the held-out clutter layout. One seed, diagnostic only.
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAYOUTS = ("rooms", "zigzag")
EPISODES = 300
SEED = 42


def run(script, *args):
    command = [sys.executable, str(HERE / script), *map(str, args)]
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=HERE, check=True)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def verified_run(model, storage):
    summary = model.with_name(model.stem + "_summary.json")
    training_csv = model.with_suffix(".csv")
    for p in (model, summary, training_csv):
        if not p.is_file():
            raise FileNotFoundError(f"Expected prior model or metadata missing: {p}")
    meta = json.loads(summary.read_text())
    must = {"method": "par", "seed": SEED, "episodes": EPISODES,
            "storage_mode": storage, "backup_mode": "standard", "par_fixed_beta": 0.0,
            "input_dim": 104, "train_start_policy": "all_reachable"}
    for k, value in must.items():
        if meta.get(k) != value:
            raise RuntimeError(f"Reference mismatch for {model}: {k}={meta.get(k)!r}, expected {value!r}")
    env = meta.get("environment", {})
    if env.get("size") != 10 or env.get("observation_mode") != "map" or env.get("train_layouts") != list(LAYOUTS):
        raise RuntimeError(f"Wrong training environment in {summary}")
    rows = read_csv(training_csv)
    if len(rows) != EPISODES:
        raise RuntimeError(f"Training CSV has {len(rows)} rows; need {EPISODES}: {training_csv}")
    return meta, rows


def main():
    both = HERE / "results" / "10x10_calibration" / "neutral_seed42_ep300" / "par_seed42_both_allstarts_fixed_beta_pos0.pt"
    executed = HERE / "results" / "10x10_replay_diagnostic" / "seed42_ep300" / "executed" / "par_seed42_allstarts_fixed_beta_pos0.pt"
    output = HERE / "results" / "10x10_safety_signal" / "seed42_ep300"
    cf = output / "par_seed42_cf_collision_allstarts_fixed_beta_pos0.pt"

    print("SAFETY-SIGNAL CONTROL — ONE SEED, NOT FINAL RESEARCH EVIDENCE.", flush=True)
    print("Train layouts: rooms and zigzag. Holdout clutter: NOT touched.", flush=True)
    verified_run(both, "both")
    verified_run(executed, "executed")
    output.mkdir(parents=True, exist_ok=True)

    if not cf.is_file():
        run("train.py", "--method", "par", "--storage-mode", "counterfactual",
            "--backup-mode", "standard", "--size", 10, "--observation-mode", "map",
            "--train-layouts", *LAYOUTS, "--train-start-policy", "all_reachable",
            "--episodes", EPISODES, "--window", 100, "--seed", SEED,
            "--par-fixed-beta", 0, "--output", output)
    else:
        print(f"Reusing existing counterfactual model: {cf}", flush=True)
    verified_run(cf, "counterfactual")

    records = []
    for mode, model in (("both", both), ("executed", executed), ("counterfactual", cf)):
        meta, rows = verified_run(model, mode)
        steps = sum(int(r["steps"]) for r in rows)
        for layout in LAYOUTS:
            result = output / f"{mode}_{layout}_off.csv"
            run("evaluate.py", "--model", model, "--size", 10,
                "--observation-mode", "map", "--layout", layout,
                "--all-starts", "--output", result)
            data = read_csv(result)
            n = len(data)
            s = sum(int(row["success"]) for row in data)
            collisions = sum(int(row["collision"]) for row in data)
            record = {"mode": mode, "layout": layout, "successes": s,
                      "starts": n, "success_pct": round(100*s/n, 2),
                      "collision_pct": round(100*collisions/n, 2),
                      "training_steps": steps,
                      "credited_proposals": meta.get("credited_proposals", 0),
                      "counterfactual_samples": meta.get("counterfactual_collision_samples", 0)}
            records.append(record)
            print(f"{mode:>14} | {layout:<6} | shield OFF {s}/{n} ({100*s/n:.1f}%) | train steps {steps}", flush=True)

    with (output / "safety_signal_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    report = {
        "experimental_status": "one seed, diagnostic not final statistical comparison",
        "holdout_tested": False,
        "arms": ["both replay = simulated safe outcomes credited to rejected actions",
                 "executed replay = only real executed transitions",
                 "counterfactual replay = real executed transitions + synthetic true collision labels for rejected actions"],
        "limitations": ["different training steps and replay contents", "only 300 episodes per run", "only one seed", "oracle labels enabled by deterministic GridWorld"],
        "results": records}
    (output / "safety_signal_report.json").write_text(json.dumps(report, indent=2))
    print("\nSAFETY SIGNAL CHECK COMPLETE", json.dumps(records, indent=2), flush=True)
    print("Send the comparison table. Do not evaluate clutter yet.")


if __name__ == "__main__":
    main()
