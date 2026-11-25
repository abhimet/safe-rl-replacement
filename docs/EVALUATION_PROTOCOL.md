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
2. **Clutter is not held-out evidence.** It was consulted during FABR development (`FABR_V3_STUDY.md`). Report it as "development-inspected".
3. **Report Spiral on its own.** Its paths are about 3× longer than on the training maps, so a failure there mixes unfamiliarity with difficulty.
4. Any combined held-out number must name exactly which layouts it pools, for example "held-out excl. Spiral (pillars + heldout_1..3, 296 scenarios)".
5. Only final, frozen checkpoints are evaluated on held-out maps. Never train on, or select models with, `dev_inspected` or `heldout` layouts.

## Why v2 exists (what changed from v1)

On 2026-11-22 two held-out definitions were committed minutes apart: v1 `evaluation_scenarios.json` (clutter, spiral, pillars; commit `85a6d97`) and `configs/holdout_suite` (heldout_1..3; commit `5271a81`). They disagreed about whether Clutter was untouched, and `run_comparison.py` evaluated only rooms, zigzag and clutter, so Spiral and Pillars were never evaluated for any method. v2 merges both:

- v1 is kept **byte-for-byte**, and v2 pins its sha256 (`parent` field).
- rooms, zigzag, pillars and spiral scenarios are identical to v1. A test enforces this.
- clutter: `heldout` → `dev_inspected`.
- heldout_1..3 were added. Their files are verified against `configs/holdout_suite/manifest.json`, and evaluation refuses a modified file.
- v2 reproduces the earlier evaluation paths exactly. Per-episode rooms/zigzag results match saved CSVs for two existing checkpoints, and the heldout_1..3 summaries equal `evaluate_holdout_suite.py`'s output (`test_v2_random_family_equals_legacy_holdout_evaluation`).

## Commands

```bash
# Full matched comparison: trains and evaluates every checkpoint on v2 automatically
python run_comparison.py --track safety_credit --methods value_based par par_count --steps 5000 --seeds 42 43 44 --output results/<name>
#   -> <method>_seed<k>_steps5000/shield_off_scenarios.csv   (581 per-episode rows)
#   -> per_seed_layout.csv, aggregate.csv                    (one row per layout, with split)

# A single existing checkpoint
python evaluate.py --model <checkpoint.pt> --scenarios evaluation_scenarios_v2.json --output results/eval_v2/<method>_seed<k>.csv
python evaluate.py --model <checkpoint.pt> --scenarios evaluation_scenarios_v2.json --split heldout --family random --output ...

# Confirm the scenario files are unchanged
python make_eval_scenarios.py --check
python make_eval_scenarios.py --version 2 --check
```

Per-episode columns: `scenario_id, split, family, layout, start, marked_start, shortest_path, reward, steps, success, collision, timeout, unsafe_proposals, interventions`. The printed summary breaks results down by split, family and layout.

## Known gap

The FABR v3 numbers in `FABR_V3_STUDY.md` and the README (17.8% / 34.0% / 41.7%) come from `results/fabr_safety_credit_5k/`. That folder has never been committed to any branch. Until it is regenerated with the `run_comparison.py` command above, those numbers can't be verified, and they have no Pillars or Spiral results.
