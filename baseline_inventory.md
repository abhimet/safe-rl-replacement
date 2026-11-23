# Baseline inventory

| Final baseline | CLI | Intervention behavior |
|---|---|---|
| Unshielded | `unshielded` | Execute every proposal |
| Random safe | `random_safe` | Uniform sample from safe actions |
| Highest Q | `value_based` | Maximize the online Q-value over safe actions |
| Highest Q with fixed penalty | `value_penalty` | Same replacement, then subtract the configured penalty from the replay reward on intervention |
| Original adaptive mixture | `adaptive` | Choose random-safe with probability equal to the preceding rolling unsafe-proposal rate. Otherwise choose highest Q. The empty-window probability is 0.5 |

All shielded methods pass safe proposals through and fall back to the proposal if no safe action exists. `fixed_mix` is the earlier 50/50 comparison and remains selectable. FABR belongs to the final comparison and is absent in this checkout.

All five assigned baseline methods completed corrected Double DQN smoke runs and loaded their final weights for shield-off evaluation. The runs used seed 42 and five episodes each on [the integration map](configs/integration_10x10.json). [Smoke results](results/ddqn_smoke)
