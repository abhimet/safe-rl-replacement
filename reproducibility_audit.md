# Reproducibility audit

The manifest inventories completed training runs by their matching CSV, summary, and final checkpoint. It references intermediate checkpoints and raw evaluation files where present. Legacy standard DQN, corrected Double DQN smoke runs, and provisional 300-episode runs have distinct statuses. The original root-level 6×6 adaptive run and the November 8 rerun both use seed 42 and represent separate historical runs, not independent seeds. The saved checkpoints store online weights for inference.

The audit script recomputes training metrics and step totals from raw CSVs, checks optimizer counts for current runs against the replay warmup rule, and reloads and checks finite outputs for the new checkpoints. For the provisional evaluation it forces each named `rooms` start once using the same checkpoint and greedy evaluator as the CLI. It writes episode-level scenario and model metadata and aggregates those rows into `main_results_all_seeds.csv`. The earlier three-episode seeded CSVs in the same directory retain their original filenames. Those default seeds select only two distinct starts and are excluded from the consolidated metrics.

Local runs used Python 3.14.0, PyTorch 2.13.0+cpu, NumPy 2.5.1, Matplotlib 3.10.7, and pytest 9.2.0. The training source was the local Double DQN change on base commit `e11061f6b473b0d96d1aee233992d583a2f7918a`. The manifest has SHA-256 hashes of the two modified training files for the new runs. Historical version and optimizer-count fields are left blank where the source artifacts did not record them.

## Reproduce

From the repository root in an environment with `requirements.txt` installed:

```powershell
python -m pytest -q -p no:cacheprovider
python validate_training.py --output results/ddqn_validation

foreach ($Method in 'unshielded', 'random_safe', 'value_based', 'value_penalty', 'adaptive') {
    python train.py --method $Method --episodes 5 --seed 42 --env-config configs/integration_10x10.json --output results/ddqn_smoke_repro
}

foreach ($Method in 'unshielded', 'random_safe', 'value_penalty') {
    python train.py --method $Method --episodes 300 --seed 42 --env-config results/ddqn_provisional/environment.json --output results/ddqn_provisional_repro
}

python audit_training_artifacts.py
```

The audit reads the saved provisional run directory. To compare weights or logs from a new training repetition, use the separate output directory in the example. The legacy comparison table was independently recomputed against its 12 source CSVs during the repository audit. The local checkout has no frozen final configuration or evaluation scenarios, FABR output, official benchmark results, or additional completed training seeds for the final comparison.


The manifest contains 23 completed training records: 15 legacy, five corrected smoke runs, and three provisional supporting runs. The legacy shield-on smoke CSV has no verified checkpoint association and is listed here separately. [Legacy shield-on smoke CSV](results/checkpoint_eval/shield_on_smoke.csv)
