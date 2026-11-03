"""Shield safety rules are fixed across all shielded methods."""
def choose_action(env, proposed_action, method, q_values=None, rng=None, random_probability=None):
    if method == 'unshielded':
        return proposed_action, False, 'none'
    safe = env.safe_actions()
    if proposed_action in safe:
        return proposed_action, False, 'none'
    if not safe:
        return proposed_action, False, 'no_safe_action'
    if rng is None:
        import random
        rng = random.Random()
    strategy = method
    if method in ('adaptive', 'fixed_mix'):
        probability = 0.5 if method == 'fixed_mix' else random_probability
        if probability is None or not 0 <= probability <= 1:
            raise ValueError('Provide a probability between 0 and 1')
        strategy = 'random_safe' if rng.random() < probability else 'value_based'
    if strategy == 'random_safe':
        action = rng.choice(safe)
    elif strategy in ('value_based', 'value_penalty'):
        if q_values is None:
            raise ValueError('Q-values required for value-based replacement')
        action = max(safe, key=lambda a: float(q_values[a]))
    else:
        raise ValueError(f'Unknown method: {method}')
    return action, True, strategy
