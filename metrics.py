"""Episode and run metrics for the MSML635 shield-dependence project."""


def summarize_episodes(episodes):
    """Summarize dictionaries with reward, success, collision, steps, unsafe_proposals."""
    if not episodes:
        raise ValueError('No episodes provided')
    n = len(episodes)
    total_steps = sum(int(e['steps']) for e in episodes)
    return {
        'episodes': n,
        'success_rate': sum(int(e['success']) for e in episodes) / n,
        'collision_rate': sum(int(e['collision']) for e in episodes) / n,
        'average_reward': sum(float(e['reward']) for e in episodes) / n,
        'average_steps': total_steps / n,
        'unsafe_proposal_rate': (sum(int(e.get('unsafe_proposals', 0)) for e in episodes) / total_steps if total_steps else 0.0),
        'intervention_rate': (sum(int(e.get('interventions', 0)) for e in episodes) / total_steps if total_steps else 0.0),
    }
