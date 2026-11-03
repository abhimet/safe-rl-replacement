import random
import pytest
from adaptive import UnsafeRateTracker
from environment import GridWorld
from metrics import summarize
from shield import choose_action

def test_metrics():
    result = summarize([
        {'steps': 10, 'reward': 5, 'success': 1, 'collision': 0, 'unsafe_proposals': 2, 'interventions': 2},
        {'steps': 5, 'reward': -10, 'success': 0, 'collision': 1, 'unsafe_proposals': 3, 'interventions': 0},
    ])
    assert result['success_rate'] == 0.5
    assert result['collision_rate'] == 0.5
    assert result['unsafe_proposal_rate'] == pytest.approx(1/3)
    assert result['intervention_rate'] == pytest.approx(2/15)

def test_adaptive_rate_window():
    tracker = UnsafeRateTracker(window=2)
    assert tracker.rate == 0.5
    tracker.record(True)
    tracker.record(False)
    assert tracker.rate == 0.5
    tracker.record(False)
    assert tracker.rate == 0.0

def test_fixed_mix_and_adaptive_safe():
    env = GridWorld(size=2, obstacles=set())
    env.reset()
    for method in ('adaptive', 'fixed_mix'):
        executed, intervened, strategy = choose_action(env, 0, method, q_values=[0, 1, 2, 0],
                                                       rng=random.Random(0), random_probability=0.7)
        assert intervened and executed in env.safe_actions()
        assert strategy in ('random_safe', 'value_based')
