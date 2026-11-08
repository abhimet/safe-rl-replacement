# MSML635: Learning-Aware Action Replacement for Reducing Shield Dependence in RL

**Team:** Abhiram Metuku (adaptive/evaluation/metrics/plots), Sai Malkireddy (environment/shield/tests), Sriram Vema (DQN/training). This is a working **starter implementation**, not validated experimental results or the team's final integration.

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
- The default 6x6 map is fixed and starts at (0,0). Repeating shield-off evaluation episodes on this deterministic map does not provide independent evaluation environments. The 10x10 layouts have multiple seeded starts; see `environment_settings.md`. The observation does not encode the map, so each run should stay on one layout.
- With executed-action replay, the original unsafe proposal is **not directly penalized** by a transition. A penalty on the executed transition may not directly discourage the original proposal. Treat this as an explicit limitation/hypothesis, not a proven solution.
- The shield can only replace an unsafe action if at least one safe action exists. The no-safe-action case falls back to the original proposal.
- No ADVICE implementation; it is literature comparison only.
- `--size 6` and `--size 10` have default obstacle layouts (`default_6x6` and `rooms`). Other sizes have no default obstacles. The training and evaluation scripts cannot select the `zigzag` or `clutter` layouts yet.
- Team members should merge their actual working code carefully; don't overwrite their changes with these starter versions.
