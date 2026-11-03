
def choose_action(env, proposed_action, rng, enabled=True):
    """Return (executed_action, intervened). Week 1 uses random-safe replacement."""
    if not enabled:
        return proposed_action, False
    safe = env.safe_actions()
    if proposed_action in safe:
        return proposed_action, False
    if not safe:
        # No safe action exists; let the environment record the collision.
        return proposed_action, False
    return int(rng.choice(safe)), True
