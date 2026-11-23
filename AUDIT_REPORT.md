# MSML635 Safe-RL Repository Engineering Audit — 22 Nov 2026

## Scope and provenance

This working copy is based on the user-uploaded `safe-rl-replacement-main (5).zip`. The uploaded archive predates the `--max-env-steps` and safety-feedback patches. The original steps patch and the safety-feedback/FABR v2 patch were restored before further fixes were made. Previous experiment files in `results/` have been preserved; new reproducible pilot files are in `results/audit_spatial_teacher_seed42/`.

**What is verified:** code integration, navigation on the two training layouts with model-assisted pretraining, full start-position shield-off evaluation, 141 unit/integration/regression tests, and successful comparison-script smoke runs. **What is not verified:** a FABR-specific benefit over Highest-Q, training across all intended seeds/budgets, or reliable held-out navigation. Do not claim those results.

## Root-cause findings and code changes

| Finding | Correction | How verified |
|---|---|---|
| Old repo version did not contain recent training flags | Restored fixed environment-step budget, safety-feedback option and absolute FABR schedule | CLI smoke and training logs; 141 tests |
| Executed-only replay gives no explicit credit to rejected unsafe proposals | Retain a clearly labeled deterministic collision-counterfactual mode; optional ranking feedback remains, but is **off by default** after its failed pilot | Regression tests; comparison runner defaults to `executed` |
| Invalid `both` replay credits a rejected action with the executed replacement's outcome | **Now rejected with ValueError** | `test_fabricated_replay_transitions_forbidden` |
| A time-limit truncation was erroneously treated as a terminal Bellman state | Bootstrap at truncation; terminate only on collision/goal; potential shaping also uses this MDP convention | `test_time_limit_is_not_bellman_terminal` |
| Simple 104-input MLP poorly represents position and local walls | Added `SpatialQNetwork`, with agent/goal one-hot planes, egocentric 5×5 occupancy patch, goal-relative vector and original map | Standalone action-value fitting; deterministic-eval sweeps |
| Sparse RL experience alone doesn't teach routes through the two complex mazes | Added **optional** train-only optimal-Q teacher from known training map dynamics, pretraining, and auxiliary rehearsal; all source states are from Rooms and Zigzag only | 2,000-step integrated pilots below |
| Historical shield-on evaluator could not load the map model and did not support the FABR schedule | Updated `evaluate_shield_on.py` for map/network inference, full starts, FABR and worst-safe | Regression tests and held-out shield-on evaluation |
| Unclear experiment provenance and repetitive manual commands | Added `run_comparison.py`, separate `model_free` and `assisted` tracks, and a raw/aggregate CSV output | Both tracks executed in an 8-step smoke run |
| Possibility an empty/hollow obstacle configuration is accepted | Added out-of-grid obstacle validation | Regression test |
| Need to establish map solvability independent of learned policy | Added `evaluate_planner_reference.py` as an **explicit non-learning, full-map planning upper bound** | 100% on all reachable starts, all three layouts |

## Actual integrated experiments (not extrapolated)

Training budget **2,000 environment steps**, seed **42**, 10×10, map observations, Rooms + Zigzag training, exact all-starts shield-off evaluation, Clutter held out:

| Algorithm variant | Rooms (84) | Zigzag (71) | Clutter (71) |
|---|---:|---:|---:|
| Spatial Highest-Q, *without* planner teacher | 4/84 | 16/71 | 5/71 |
| **Spatial Highest-Q + planner teacher** (`2000` pretraining updates, `0.3` rehearsal weight) | **84/84** | **71/71** | 5/71 |
| **Spatial FABR absolute schedule + same teacher** | **84/84** | **71/71** | 5/71 |
| Exact full-map shortest path planner (non-RL upper bound) | 84/84 | 71/71 | 71/71 |

For **both model-assisted learned policies**, the shield-off collision count was zero on all three maps. On Clutter almost all episodes nevertheless *timed out*: local safety did not solve route finding. The teacher has access to full training-map geometry and shortest-path action values; the student does not query either the teacher or shield during `evaluate.py`. This **is not model-free DQN** and the gains cannot be credited to the replacement algorithm alone.

The previous user-run 20,000-step pure-model-free results were Highest-Q ~22.8% and FABR ~7.7% mean across seeds on Rooms+Zigzag, and the safety-feedback-weight-3 model failed 0/155 on seed 42. Those are valuable **historical experiments**, but do not directly compare to the new 2,000-step model-assisted results, or to runs after the truncation correction; training budgets, supervision and code differ.

## Open scientific limitations — important

1. **Held-out generalization is still poor.** Clutter 5/71 ≈ 7%, despite perfect success on the two teacher-seen layouts. Claims of generalization or SOTA performance are not supported.
2. **FABR has not shown an independent improvement over Highest-Q.** When both receive the teacher, both score 155/155 on training maps. The teacher dominates. Further comparisons must apply identical assistance to both methods, and should evaluate learning efficiency and intervention frequency, not only ceilinged success rates.
3. The planner teacher is **privileged model-based training**; optimal Q targets are generated using exact known training-map geometry/dynamics. The model-assisted track is not a direct reproduction of the team's original model-free hypothesis and must be labeled as an extension.
4. **No full three-seed, 20,000-step final study was run in this audit.** `run_comparison.py` provides a controlled setup for that work. No statistical-significance or improvement claim is justified by one seed.
5. The external MiniGridSafe / published SOTA replication has **not** been implemented here. The separate planning reference is not a published SOTA comparison.
6. The `value_penalty` baseline still applies the penalty to the *safe executed action*, as its original definition/code specified. It can discourage good interventions and is **not equivalent to penalizing the rejected unsafe proposal**. The assisted track forbids this incompatible objective rather than silently modifying the baseline.
7. The old relative FABR schedule remains for reproduction. The absolute `0..4` schedule avoids the unstable initial-reference ratio; its independent benefit remains unverified.
8. `safety_loss_weight=3` has already failed in prior experiments and is not recommended. It remains an optional historical ablation only.

## Reproducing the pilot, no manual patch needed

From the unzipped project folder:

```bash
python -m pip install -r requirements.txt
python -m pytest -q

python train.py --method value_based --max-env-steps 2000 --seed 42 --size 10 --observation-mode map --train-layouts rooms zigzag --train-start-policy all_reachable --storage-mode executed --reward-shaping potential --shaping-scale 4 --architecture spatial --expert-pretrain-updates 2000 --expert-loss-weight 0.3 --output results/my_pilot

python evaluate.py --model results/audit_spatial_teacher_seed42/value_based_seed42_steps2000_spatial_teacher2000_expertw0p3_allstarts_phi4.pt --size 10 --layout rooms --observation-mode map --all-starts --output results/my_room_eval.csv

python evaluate.py --model results/audit_spatial_teacher_seed42/value_based_seed42_steps2000_spatial_teacher2000_expertw0p3_allstarts_phi4.pt --size 10 --layout clutter --observation-mode map --all-starts --output results/my_heldout_eval.csv
```

The final output model is stored as a plain PyTorch `state_dict` and is reloadable by `evaluate.py`. Historic MLP checkpoints also remain loadable. Use matching model observation modes.

## Comparisons without command-by-command micromanagement

Original model-free comparisons (six methods, 20,000 env steps each, paired seeds, all 3 layouts):

```bash
python run_comparison.py --track model_free --steps 20000 --seeds 42 43 44 --par-schedule relative --output results/comparison_old_fabr
```

Model-free comparison with corrected *absolute* FABR schedule (a separate experiment, not historical reproduction):

```bash
python run_comparison.py --track model_free --steps 20000 --seeds 42 43 44 --par-schedule absolute --output results/comparison_new_fabr
```

Model-assisted comparison **must be reported separately** because it changes the learning setup:

```bash
python run_comparison.py --track assisted --steps 20000 --seeds 42 43 44 --output results/comparison_assisted
```

Each command saves per-seed/per-layout and aggregate CSVs. These complete comparisons are **not included as completed results** in this ZIP, and may be computationally expensive.

## Integrity instructions

- The original user ZIP remains untouched. Archive its historical results; do not overwrite or call them new experiments.
- The safety shield in `evaluate_shield_on.py` is a diagnostic. `evaluate.py` **does not perform intervention**.
- `run_comparison.py` separates tracks to prevent conflating model-assisted improvement with the FABR replacement hypothesis.
- An invalid attempt to use rejected-action relabeling with the executed replacement's next state now raises an exception.
