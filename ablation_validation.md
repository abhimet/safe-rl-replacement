# FABR ablation validation

Status **pending implementation and experiment evidence**. This checkout has the original adaptive mixture and fixed 50/50 mixture. It has no FABR selector, β controls, or β diagnostics.

For final validation, check that adaptive β, fixed β=0, and the prespecified fixed positive β share the same safe-action Boltzmann selector. Confirm that zero yields a uniform distribution over safe actions, fixed β stays fixed, and only the adaptive arm updates β from proposal feedback. Verify matched training seeds, configuration, episode or step budget, exploration, action/replay semantics, and evaluation scenario order. Compare configuration records and actual step and optimizer counts before aggregating success and collision by independent training seed. Final validation requires the implementation, matched run summaries, β and Q-gap traces, and raw evaluations.
