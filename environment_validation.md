# Environment and shield validation (Sai, tasks 1–2)

Validated on 2026-11-22, branch `sai/env-shield-scenarios`, base commit `6ab7adc`. Run with `python -m pytest -q test_environment.py test_shield.py test_scenarios.py`, which gives **192 passed**. The full suite gives 218 passed and 1 failed. That failure is `test_training.py::test_training_stores_executed_action_and_penalized_reward`, caused by the outdated `remember()` mock that Sriram owns as his task 1; this branch does not touch it. The baseline before any changes is saved in `results/env_validation/test_baseline.txt`.

**Result: no environment or shield defects found, so no fixes were made to `GridWorld`, `shield.py` or `train.py`.**

## Task 1: environment checklist

| Item | Evidence (tests in `test_environment.py` unless noted) | Result |
|---|---|---|
| Map dimensions and obstacles | `test_layout_configuration_and_paths` (obstacle counts checked against an independent BFS), `test_malformed_layouts_are_rejected` | ✅ |
| Start and goal locations | `test_layout_configuration_and_paths`, `test_invalid_configuration_is_rejected` | ✅ |
| Goal reachability | Every free cell reaches the goal (`len(distances) == free cells`). `test_shortest_path_reaches_goal_from_every_start` walks each path | ✅ |
| Movement | `test_rewards_and_flags_are_exact_for_every_cell_and_action` (every cell × action on all 5 10×10 maps), `test_movement_and_goal` | ✅ |
| Boundaries and collisions | Same exhaustive test, plus `test_wall_collision_*` and `test_obstacle_collision_*`. A collision gives −10, terminates, and leaves the position unchanged | ✅ |
| Goal termination | `test_goal_on_10x10`, `test_goal_on_new_heldout_layouts`. Success gives +10 and terminates | ✅ |
| Timeout | `test_truncation_on_10x10`. Success or collision on the last step is `terminated`, not `truncated` (`test_*_on_final_step_*`) | ✅ |
| Deterministic resets and seeds | `test_same_seed_gives_same_start`, `test_seeds_cover_all_candidate_starts`, `test_unseeded_resets_continue_seeded_stream`, `test_forced_start_ignores_seed` | ✅ |
| Map observations (104 dims) | `test_map_observation_encodes_position_goal_and_obstacles`, `..._distinguishes_layouts...`, `..._tracks_movement` | ✅ |
| Invalid actions | `test_step_rejects_invalid_actions` | ✅ |

**Documented behavior, not a defect:** `step()` after termination does not raise; it keeps moving the agent. This is the standard Gymnasium convention. `train.py` (L428/508) and `evaluate.py` both stop at `terminated or truncated`, so no results are affected.

## Task 2: shield checklist

**Safety definition:** an action is unsafe if and only if its target cell is outside the grid or is an obstacle. `GridWorld.safe_actions()` applies exactly the rule `step()` uses to decide collisions. There is no multi-step lookahead, because a single-step check is enough to prevent every collision in this environment.

| Item | Evidence (`test_shield.py`) | Result |
|---|---|---|
| Safe set matches the collision rule | `test_safe_actions_match_brute_force` (an independent checker, every free cell) | ✅ |
| Walls, boundaries and corners | `test_corner_and_edge_safe_sets` | ✅ |
| Shield prevents every collision | `test_shield_prevents_every_collision` (every cell × action × method in `shield.py`) | ✅ |
| Safe proposals are never replaced | `test_safe_proposal_is_not_replaced` | ✅ |
| Exactly one safe action | `test_single_safe_action_is_always_chosen`. The action is chosen even when it has the worst Q-value. Such cells exist in zigzag (1), clutter (2) and spiral (1) | ✅ |
| No safe action | `test_enclosed_cell_falls_back_to_proposal` (synthetic). `test_named_layouts_have_no_enclosed_cells` shows the case cannot occur on any real map | ✅ |
| `par` (FABR) and `worst_safe` in `train.py` | `test_par_replacements_are_safe_for_each_beta` (adaptive β and fixed β = −4, 0, +4) and `test_training_executes_only_safe_actions...`. A guard on `GridWorld.step` records zero unsafe executed actions | ✅ |
| Intervention and unsafe-proposal counts (training) | With the shield on, `interventions == unsafe_proposals` in every episode for all 7 shielded methods, and there are 0 collisions. Unshielded runs have `interventions == 0` | ✅ |
| Unsafe-proposal counts (evaluation) | `test_scenario_evaluation_rows_are_consistent` (`test_scenarios.py`). Shield-off rows have `interventions == 0` and exactly one of success, collision or timeout | ✅ |

### For Abhiram (FABR owner): a no-safe-action inconsistency
`shield.choose_action` handles a cell with no safe action by executing the proposal and reporting `no_safe_action`. The inline `par` and `worst_safe` branches in `train.py` (around L290 and L350) instead `raise RuntimeError("No safe replacement is available")`. The case **cannot occur on any named layout**, because every free cell has a safe action (tested). So no results are affected, but the documented FABR algorithm should state which behavior it intends. I have left the code unchanged.

## Files changed
- `environment.py`: added the held-out layouts `spiral` and `pillars`, plus the `TRAIN_LAYOUTS` and `HELDOUT_LAYOUTS` constants. Existing layouts are unchanged.
- `evaluate.py`: added `--scenarios` and `--split` (frozen-scenario evaluation that also records `timeout`). `--layout` choices now come from `LAYOUTS`. The legacy modes are unchanged: their output matches the scenario mode row for row on `rooms`, 84 of 84 rows.
- New files: `make_eval_scenarios.py`, `evaluation_scenarios.json`, `test_scenarios.py`, `layouts.md`.
- Tests added to `test_environment.py` and `test_shield.py`.
