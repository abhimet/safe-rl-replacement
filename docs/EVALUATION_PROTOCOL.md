# Evaluation protocol (v2, frozen 2026-11-23)

The single reference for how every final checkpoint (baselines, FABR, ablations) is evaluated.
Owner: Sai Malkireddy. Scenario file: [`evaluation_scenarios_v2.json`](evaluation_scenarios_v2.json).

## Protocol

- **Shield OFF**, greedy argmax, no exploration, no replacement. The interventions column is always 0.
- One episode per scenario, with a forced start. There is one scenario per reachable non-goal cell. GridWorld and the policy are deterministic, so repeating a scenario adds nothing.
- `max_steps = 100`, map observations (104 inputs, 10×10).
- Every episode ends in exactly one of: success, collision, timeout.

## Layout groups

| Split | Family | Layout | Scenarios | Source |
|---|---|---|---:|---|
| `train` | designed | rooms | 84 | `environment.py` |
| `train` | designed | zigzag | 71 | `environment.py` |
| `dev_inspected` | designed | clutter | 71 | `environment.py` |
| `heldout` | designed | pillars | 69 | `environment.py` |
| `heldout` | designed | spiral | 59 | `environment.py` |
| `heldout` | random | heldout_1 | 89 | `configs/holdout_suite/heldout_1.json` (sha256 checked) |
| `heldout` | random | heldout_2 | 77 | `configs/holdout_suite/heldout_2.json` (sha256 checked) |
| `heldout` | random | heldout_3 | 61 | `configs/holdout_suite/heldout_3.json` (sha256 checked) |

Totals: 155 train, 71 dev_inspected, 355 held-out, 581 in all.

## Reporting rules

1. **Report every layout separately.** Never pool splits.
2. **Clutter is not held-out evidence.** It was consulted during FABR development (`archive/docs/FABR_V3_STUDY.md`). Report it as "development-inspected".
3. **Report Spiral on its own.** Its paths are about 3× longer than on the training maps, so a failure there mixes unfamiliarity with difficulty.
4. Any combined held-out number must name exactly which layouts it pools, for example "held-out excl. Spiral (pillars + heldout_1..3, 296 scenarios)".
5. Only final, frozen checkpoints are evaluated on held-out maps. Never train on `dev_inspected` or `heldout` layouts, and never select models or hyperparameters with `heldout` layouts.
6. **Tuning uses development maps only** (`tune.py`): the `train` split (rooms, zigzag) and `dev_inspected` (clutter), with seeds 0–1 (disjoint from the final seeds 42–51). `tune.dev_evaluate` refuses any other split. In every result table, *familiar* and *clutter* are therefore marked as used for tuning, and **unseen** (pillars + heldout_1..3) is the headline, with spiral shown next to it.
7. Final runs read every setting from the frozen `configs/final_protocol.json`, which also pins this scenario file's sha256.

## Why v2 exists (what changed from v1)

On 2026-11-22 two held-out definitions were committed minutes apart: v1 `evaluation_scenarios.json` (clutter, spiral, pillars; commit `85a6d97`) and `configs/holdout_suite` (heldout_1..3; commit `5271a81`). They disagreed about whether Clutter was untouched, and `run_comparison.py` evaluated only rooms, zigzag and clutter, so Spiral and Pillars were never evaluated for any method. v2 merges both:

- v1 is kept **byte-for-byte**, and v2 pins its sha256 (`parent` field).
- rooms, zigzag, pillars and spiral scenarios are identical to v1. A test enforces this.
- clutter: `heldout` → `dev_inspected`.
- heldout_1..3 were added. Their files are verified against `configs/holdout_suite/manifest.json`, and evaluation refuses a modified file.
- v2 reproduces the earlier evaluation paths exactly. Per-episode rooms/zigzag results match saved CSVs for two existing checkpoints, and the heldout_1..3 summaries equal `evaluate_holdout_suite.py`'s output (this check was run when v2 was frozen; the legacy script is now in `archive/scripts/`).

## Commands

```bash
# Final comparison: every config of a protocol tier, evaluated on v2 automatically
python run_experiments.py --protocol configs/final_protocol.json --tier core --jobs 6
#   -> <mode>/runs/<method>_seed<k>/shield_off_episodes.csv   (581 per-episode rows)
#   -> per_seed.csv, aggregate.csv, paired.csv                (per layout and per pooled group)

# A single checkpoint
python evaluate.py --model <checkpoint.pt> --output results/eval.csv
python evaluate.py --model <checkpoint.pt> --split heldout --family random

# Confirm the scenario files are unchanged
python make_eval_scenarios.py --check
python make_eval_scenarios.py --version 2 --check
```

Per-episode columns: `scenario_id, split, family, layout, start, marked_start, shortest_path, reward, steps, success, collision, timeout, unsafe_proposals, interventions, start_unsafe`.

`start_unsafe` is 1 if the greedy action in the start state is unsafe. Scenarios cover every reachable non-goal cell, so its mean over a layout is the **greedy unsafe-state rate**: the fraction of reachable states in which the shield-off greedy action would hit a wall or leave the grid.

Pooled groups used in `run_experiments.py`, in reporting order: **unseen** (pillars + heldout_1..3, 296 scenarios; the headline), **unseen_spiral** (spiral, reported on its own), **familiar** (rooms + zigzag; used for tuning) and **dev_clutter** (clutter; used for tuning).

## Training maps vs. evaluation maps

Training uses rooms, zigzag and 100 generated maps (`environment.generate_training_pool`, seed 7).
A generated map is rejected if its walls overlap any dev or held-out evaluation map by more than
50% (Jaccard); `tests/test_training.py` enforces this. Historical v1/v2 provenance notes
(e.g. the FABR v3 study) are in `archive/docs/`.
