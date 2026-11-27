"""Shield replacement strategies: what to execute when the agent proposes an unsafe action.

The shield itself is fixed and identical for every method: an action is unsafe
if it would leave the grid or enter a wall (``GridWorld.safe_actions``). The
methods differ only in WHICH safe action replaces an unsafe proposal.

    unshielded     no shield during training (lower bound)
    random_safe    uniform random safe action                     (= FABR with beta = 0)
    highest_q      safe action with the highest Q-value           (= FABR with beta -> inf)
    value_penalty  highest_q, plus a training-reward penalty when the shield intervenes
    adaptive_mix   random_safe with probability = recent unsafe-proposal rate, else highest_q
    fabr           Feedback-Adaptive Boltzmann Replacement (our method, see below)
    fabr_fixed     ablation: FABR's Boltzmann choice with a constant beta (no feedback)
    fabr_v2        highest_q execution; learning adds a large-margin loss on the blocked
                   proposal (feedback mode ``margin``, see dqn.margin_loss)
    fabr_v2_fixedtarget  ablation of fabr_v2: the same margin target, but a two-sided squared
                   loss that never switches off

FABR samples a safe action from a softmax over its z-scored Q-values with
inverse temperature beta, and sets beta from the recent rate r at which the
agent's GREEDY action is unsafe:

    beta = beta_min + (beta_max - beta_min) * (1 - r) ** 2

While the policy still prefers unsafe moves (r high), beta is low and
replacements explore the safe alternatives. As it learns (r -> 0), beta rises
and replacement concentrates on the highest-valued safe action.
"""
from collections import deque
import math

import numpy as np

METHODS = ("unshielded", "random_safe", "highest_q", "value_penalty",
           "adaptive_mix", "fabr", "fabr_fixed", "fabr_v2", "fabr_v2_fixedtarget")
# Margin-loss methods and the loss kind each one uses (dqn.margin_loss).
MARGIN_METHODS = {"fabr_v2": "hinge", "fabr_v2_fixedtarget": "fixed_target"}
HIGHEST_Q_METHODS = ("highest_q", "value_penalty", *MARGIN_METHODS)
# What a shielded agent stores about a blocked proposal (train.py):
#   none    only the executed transition (s, a_exec, r, s')
#   cf      also (s, a_prop, -10, terminal): ORACLE, the outcome of an action never executed
#   credit  (s, a_prop, r, s') INSTEAD of the executed transition (r, s' from the replacement)
#   margin  the executed transition, plus (s, a_prop) for the margin loss (fabr_v2 only)
FEEDBACK_MODES = ("none", "cf", "credit", "margin")


def fabr_beta(unsafe_rate, beta_min=0.0, beta_max=4.0):
    if not 0.0 <= unsafe_rate <= 1.0:
        raise ValueError("unsafe_rate must be in [0, 1]")
    return beta_min + (beta_max - beta_min) * (1.0 - unsafe_rate) ** 2


def boltzmann_choice(safe, q_values, beta, rng):
    """Sample from softmax(beta * z-scored Q) over the safe actions."""
    values = np.asarray([q_values[a] for a in safe], dtype=np.float64)
    std = float(values.std())
    z = np.zeros_like(values) if std < 1e-6 else (values - values.mean()) / std
    logits = beta * z
    weights = np.exp(logits - logits.max())
    return rng.choices(list(safe), weights=weights.tolist(), k=1)[0]


def highest_q(safe, q_values):
    return max(safe, key=lambda a: float(q_values[a]))


class Replacer:
    """Chooses the executed action for one training run.

    ``choose`` is called once per environment step, BEFORE the step is
    recorded, so feedback rates use only past proposals.
    """

    def __init__(self, method, rng, window=100, beta_min=0.0, beta_max=4.0, fixed_beta=None):
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        if window < 1:
            raise ValueError("window must be positive")
        if beta_min >= beta_max:
            raise ValueError("beta_min must be smaller than beta_max")
        if (method == "fabr_fixed") != (fixed_beta is not None):
            raise ValueError("fixed_beta is required for fabr_fixed and only allowed there")
        if fixed_beta is not None and not math.isfinite(fixed_beta):
            raise ValueError("fixed_beta must be finite")
        self.method = method
        self.rng = rng
        self.beta_min, self.beta_max, self.fixed_beta = beta_min, beta_max, fixed_beta
        self.proposal_unsafe = deque(maxlen=window)  # adaptive_mix feedback (includes exploration)
        self.greedy_unsafe = deque(maxlen=window)    # FABR feedback (policy only)

    @property
    def mix_probability(self):
        h = self.proposal_unsafe
        return sum(h) / len(h) if h else 0.5

    @property
    def greedy_unsafe_rate(self):
        h = self.greedy_unsafe
        return sum(h) / len(h) if h else 1.0

    @property
    def beta(self):
        if self.method == "fabr_fixed":
            return self.fixed_beta
        return fabr_beta(self.greedy_unsafe_rate, self.beta_min, self.beta_max)

    def choose(self, proposed, safe, q_values):
        """Return (executed_action, intervened)."""
        if self.method == "unshielded" or proposed in safe:
            executed, intervened = proposed, False
        elif not safe:
            raise RuntimeError("No safe action exists in this state")
        elif self.method == "random_safe":
            executed, intervened = self.rng.choice(list(safe)), True
        elif self.method in HIGHEST_Q_METHODS:
            executed, intervened = highest_q(safe, q_values), True
        elif self.method == "adaptive_mix":
            use_random = self.rng.random() < self.mix_probability
            executed = self.rng.choice(list(safe)) if use_random else highest_q(safe, q_values)
            intervened = True
        else:  # fabr, fabr_fixed
            executed, intervened = boltzmann_choice(safe, q_values, self.beta, self.rng), True
        self.proposal_unsafe.append(proposed not in safe)
        self.greedy_unsafe.append(int(np.argmax(q_values)) not in safe)
        return executed, intervened
