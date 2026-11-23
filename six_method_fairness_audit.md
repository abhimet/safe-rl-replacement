# Six-method fairness audit

The final six methods are unshielded, random-safe, highest-Q, highest-Q with fixed penalty, original adaptive mixture, and FABR. All five available baselines share the Double DQN learner, 2–64–64–4 network, normalized coordinates, replay capacity and batch size, Adam settings, discount, target synchronization, and epsilon schedule. For a common map configuration, environment rewards match. The intended training-reward change is the fixed penalty on an intervened executed-action transition. Training summaries capture resolved map settings and actual environment and optimizer steps.

The CLI budget is episodes. Methods that collide or finish early consume fewer transitions than methods that reach the cap. Report actual step and optimizer counts for any episode-matched comparison. If the final protocol requires a fixed transition budget, the current entry point does not enforce one. Training resets use training seed plus episode index. The current shield-off evaluator starts at seed zero, so its default seed range can overlap training resets and it does not consume a frozen scenario list.

The provisional runs match one another on episode budget, training seed, map, rewards, and learner settings. Final sign-off remains open until the team freezes the shared training configuration and evaluation scenarios, FABR is available, and all six method summaries and raw evaluations can be compared. Legacy standard DQN results and five-episode smoke results have separate provenance.


## Provisional seed-42 results

All three runs used 300 episodes on the named rooms layout. Unshielded collected 6,868 transitions, random-safe collected 26,466, and highest-Q with penalty collected 22,709. The three deterministic shield-off starts yielded no successes. Unshielded timed out on every start, random-safe collided on every start, and highest-Q with penalty collided once and timed out twice. These are observations from one seed and one layout. [Raw results and counts](main_results_all_seeds.csv)
