"""Episode-level and aggregate statistics."""
def summarize(episodes):
    rows = list(episodes)
    if not rows:
        raise ValueError('No episodes to summarize')
    n = len(rows)
    total_steps = sum(int(row['steps']) for row in rows)
    return {
        'episodes': n,
        'success_rate': sum(int(row['success']) for row in rows) / n,
        'collision_rate': sum(int(row['collision']) for row in rows) / n,
        'average_reward': sum(float(row['reward']) for row in rows) / n,
        'average_steps': total_steps / n,
        'unsafe_proposal_rate': sum(int(row['unsafe_proposals']) for row in rows) / total_steps if total_steps else 0.0,
        'intervention_rate': sum(int(row['interventions']) for row in rows) / total_steps if total_steps else 0.0,
    }
