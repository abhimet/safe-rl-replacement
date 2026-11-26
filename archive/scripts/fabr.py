"""Feedback-adaptive count-aware safe replacement (training-time only).

Does not use a planner, shortest-path map, reward oracle, or hidden transition.
The shield supplies safe actions; Q comes from the current DQN. The auxiliary
state-action visit table is used ONLY when training an unsafe proposal.

This is an experimental new selector, not a validated SOTA algorithm.
"""
from collections import defaultdict
import math

import numpy as np


class CountAwareReplacement:
    """Softmax of normalized safe action values and an adaptive count bonus.

    At states where the greedy policy proposes unsafe moves frequently,
    upweight safe actions the agent has less experience executing. The bonus
    decreases as the policy learns to propose safe actions independently.
    """

    def __init__(self, rng, novelty_weight=2.0):
        if not math.isfinite(novelty_weight) or novelty_weight < 0:
            raise ValueError("novelty_weight must be finite and nonnegative")
        self.rng = rng
        self.novelty_weight = float(novelty_weight)
        self.counts = defaultdict(int)

    def observe(self, state_key, executed_action):
        if not 0 <= int(executed_action) < 4:
            raise ValueError("invalid action")
        self.counts[(state_key, int(executed_action))] += 1

    def select(self, state_key, safe_actions, q_values, beta, unsafe_rate):
        safe = tuple(safe_actions)
        if not safe or len(set(safe)) != len(safe) or any(not 0 <= a < 4 for a in safe):
            raise ValueError("safe actions must be distinct nonempty action ids")
        if not math.isfinite(beta) or not 0 <= unsafe_rate <= 1:
            raise ValueError("invalid beta or unsafe rate")
        q = np.asarray(q_values, dtype=np.float64)
        if q.shape != (4,) or not np.isfinite(q).all():
            raise ValueError("q_values must be four finite action values")
        vals = q[list(safe)]
        std = float(vals.std())
        normalized = np.zeros_like(vals) if std < 1e-6 else (vals - vals.mean()) / std
        # Counts include ALL previously executed actions in this state,
        # not merely shield replacements. Encourages genuine coverage.
        novelty = np.array([1.0 / math.sqrt(1.0 + self.counts[(state_key, a)])
                            for a in safe], dtype=np.float64)
        # The progress signal is the recent *greedy* unsafe proposal rate.
        # Low unsafe rate -> low exploration bonus; high -> more exploration.
        scores = beta * normalized + self.novelty_weight * unsafe_rate * novelty
        scores -= float(scores.max())
        weights = np.exp(scores)
        return self.rng.choices(safe, weights=weights.tolist(), k=1)[0]
