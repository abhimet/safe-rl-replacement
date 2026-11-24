# MSML635: Learning-Aware Action Replacement for Reducing Shield Dependence in RL

## New controlled FABR improvement (2026-11-22)

**Count-aware FABR** is available as `--method par --par-selector count_aware --par-novelty-weight 3`. It uses training-only, feedback-weighted action visitation counts to diversify replacement among shield-approved actions. Original Boltzmann FABR is unchanged (`--par-selector boltzmann`).

A distinct, nine-run **safety-credit** comparison (5k env steps, seeds 42–44) includes Highest-Q, original FABR, and count-aware FABR. See [FABR_V3_STUDY.md](FABR_V3_STUDY.md), [SOTA_COMPARISON_BRIEF.md](SOTA_COMPARISON_BRIEF.md), and [results/fabr_safety_credit_5k/aggregate.csv](results/fabr_safety_credit_5k/aggregate.csv). The count-aware variant averages 41.7% shield-OFF success on Rooms+Zigzag versus 34.0% original FABR and 17.8% Highest-Q, but lags Highest-Q on the development-evaluated Clutter map (16.0% versus 20.2%); on three fresh fixed holdouts it reaches only 9.1% versus 7.2% Highest-Q and 5.4% original FABR. Seed variance is high. **No SOTA superiority is claimed.**

```bash
python run_comparison.py --track safety_credit --methods value_based par par_count --steps 5000 --seeds 42 43 44 --output results/reproduction_5k
```

**Evaluation (2026-11-23):** every method is evaluated shield-OFF on the frozen
`evaluation_scenarios_v2.json` (581 scenarios: train rooms/zigzag, development-inspected
clutter, held-out pillars/spiral/heldout_1..3). `run_comparison.py` does this automatically.
See [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md). **Note:** the checkpoints and CSVs behind
the FABR v3 numbers above (`results/fabr_safety_credit_5k/`) were never committed, so those
numbers must be regenerated with the command above before they can be verified.

`--track safety_credit` uses model-derived collision counterfactuals and safe-action Double DQN backups for all three shielded conditions. This **does not have the same knowledge assumptions as the original model-free or shortest-path-teacher tracks**. Existing results/checkpoints are preserved.

---


**Audited engineering edition — Nov 22, 2026.** Start with [AUDIT_REPORT.md](AUDIT_REPORT.md) for tested results, important limits, changed training code, and the one-command comparison runner. The preserved sections below describe the earlier model-free project and historical experiments.

**Key distinction:** `--architecture spatial --expert-pretrain-updates 2000 --expert-loss-weight 0.3` is a **new model-assisted extension**, not a baseline feature or proof FABR improves replacement. It learns safe routes on the two training layouts but does not yet generalize reliably to Clutter. All published results must label these tracks separately. The audited ZIP contains evaluated seed-42 checkpoints under `results/audit_spatial_teacher_seed42/`.

**Team:** Abhiram Metuku (adaptive/evaluation/metrics/plots), Sai Malkireddy (environment/shield/tests), Sriram Vema (DQN/training). November 8 DQN/training validation is documented below; final comparative experiments and full environment integration remain pending.

## Setup

Python 3.10+ recommended. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
python -m pytest -q
```

Gymnasium is optional in this starter because `environment.py` exposes a compatible reset/step API even when Gymnasium is not installed. To use actual Gymnasium `Env`/`spaces` classes, install `gymnasium>=0.29,<2`.

## Train and evaluate

Sriram's November 8 implementation and validation are documented in
[sriram_nov08.md](sriram_nov08.md). Reproduce the twelve baseline smoke runs,
continuous 300-episode adaptive run, six checkpoint reload evaluations, and curve:

```bash
python validate_training.py --output results/nov08_sriram
```

Training and shield-off evaluation accept `--env-config` with a JSON object of
GridWorld constructor settings (`size`, `obstacles`, `start`, `goal`, `max_steps`).
JSON settings override `--size`. For initial 10x10 integration:

```bash
python train.py --method adaptive --episodes 300 --env-config configs/integration_10x10.json --output results/integration10
python evaluate.py --model results/integration10/adaptive_seed42.pt --env-config configs/integration_10x10.json --output results/integration10/shield_off.csv
```

Each 50th episode saves a state dictionary named `<method>_seed<seed>_ep<N>.pt`.
The final `<method>_seed<seed>.pt` uses the same loading format. Summaries record
the resolved environment, seed, versions, DQN settings, and CPU thread count.
The default single CPU thread avoids overhead for these small networks.

```bash
python train.py --method unshielded --episodes 300 --seed 42
python train.py --method random_safe --episodes 300 --seed 42
python train.py --method value_based --episodes 300 --seed 42
python train.py --method value_penalty --episodes 300 --seed 42 --penalty 1.0
python train.py --method fixed_mix --episodes 300 --seed 42
python train.py --method adaptive --episodes 300 --seed 42 --window 100
python evaluate.py --model results/adaptive_seed42.pt --episodes 100 --output results/adaptive_shield_off.csv
python plot_results.py --csv results/adaptive_seed42.csv
```

**Meaning of methods:** `unshielded` has no safety intervention; `random_safe` picks a random safe action; `value_based` picks the highest-Q safe action; `value_penalty` uses value-based replacement and subtracts a fixed training reward penalty when intervening; `fixed_mix` uses 50/50 random-safe vs value-based; `adaptive` uses the previous 100 unsafe-proposal indicators as the random-safe probability. The adaptive probability starts at 0.5 before any proposals.

**Logging:** Training CSV contains environment reward, steps, success, collision, unsafe proposals, interventions, counts of random/value replacement and recent unsafe rate. The replay buffer stores the **executed** action, while the unsafe proposal is tracked for aggregate statistics. `evaluate.py` runs greedy actions with **no shield**.

## Known limitations

- **Not a research result:** 300 episodes is a smoke-test setting, not evidence of convergence. Run multiple seeds and compare shield-off success AND collisions.
- The default 6x6 map is fixed and starts at (0,0). Repeating shield-off evaluation episodes on this deterministic map does not provide independent evaluation environments. The 10x10 layouts have multiple seeded starts; see `environment_settings.md`. Historical position-only observation did not encode the map. The newer `--observation-mode map` does encode obstacle geometry; see AUDIT_REPORT.md for limitations.
- With executed-action replay, the original unsafe proposal is **not directly penalized** by a transition. A penalty on the executed transition may not directly discourage the original proposal. Treat this as an explicit limitation/hypothesis, not a proven solution.
- The shield can only replace an unsafe action if at least one safe action exists. The no-safe-action case falls back to the original proposal.
- No ADVICE implementation; it is literature comparison only.
- `--size 6` uses the default 6×6 obstacle layout.
- `--size 10` defaults to the `rooms` layout.
- `configs/integration_10x10.json` provides a fixed obstacle configuration for preliminary training.
- The `rooms`, `zigzag`, and `clutter` layouts are selectable with the current CLI. Keep `clutter` held out when running the stated generalization evaluation.
