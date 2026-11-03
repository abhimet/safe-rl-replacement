"""Standalone adaptive replacement selection; integrate into training in Week 3."""
from collections import deque


class AdaptiveReplacement:
    def __init__(self, window=100, min_random_probability=0.1, max_random_probability=0.9):
        if window < 1:
            raise ValueError('window must be positive')
        if not 0 <= min_random_probability <= max_random_probability <= 1:
            raise ValueError('probabilities must be between 0 and 1')
        self.history = deque(maxlen=window)
        self.minimum = min_random_probability
        self.maximum = max_random_probability

    def record_proposal(self, was_unsafe):
        """Call once for every proposed action, not just unsafe ones."""
        self.history.append(bool(was_unsafe))

    def random_probability(self):
        """More recent unsafe proposals => more random-safe replacements."""
        rate = sum(self.history) / len(self.history) if self.history else 1.0
        return self.minimum + (self.maximum - self.minimum) * rate

    def replace(self, safe_actions, q_values, rng):
        """Choose safe replacement; q_values must contain one value per action."""
        if not safe_actions:
            raise ValueError('No safe actions available')
        if rng.random() < self.random_probability():
            return int(rng.choice(safe_actions))
        return int(max(safe_actions, key=lambda action: float(q_values[action])))
