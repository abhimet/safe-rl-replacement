import pytest
from adaptive import UnsafeRateTracker
import random
from environment import GridWorld
from shield import choose_action

def test_initial_rate():
    tracker = UnsafeRateTracker(window=5)
    assert tracker.rate == 0.5


def test_all_safe():
    tracker = UnsafeRateTracker(window=5)

    for _ in range(5):
        tracker.record(False)

    assert tracker.rate == 0.0


def test_all_unsafe():
    tracker = UnsafeRateTracker(window=5)

    for _ in range(5):
        tracker.record(True)

    assert tracker.rate == 1.0


def test_mixed_actions():
    tracker = UnsafeRateTracker(window=4)

    for unsafe in [True, False, True, False]:
        tracker.record(unsafe)

    assert tracker.rate == 0.5


def test_rolling_window():
    tracker = UnsafeRateTracker(window=3)

    for unsafe in [True, True, True, False]:
        tracker.record(unsafe)

    assert tracker.rate == pytest.approx(2 / 3)
  

def test_adaptive_random_safe():
    env = GridWorld(size=2, obstacles=set())
    env.reset()

    action, intervened, strategy = choose_action(
        env,
        proposed_action=0,
        method="adaptive",
        q_values=[0, 1, 2, 0],
        rng=random.Random(42),
        random_probability=1.0,
    )

    assert intervened
    assert strategy == "random_safe"
    assert action in env.safe_actions()


def test_adaptive_value_based():
    env = GridWorld(size=2, obstacles=set())
    env.reset()

    q_values = [0, 1, 10, 0]

    action, intervened, strategy = choose_action(
        env,
        proposed_action=0,
        method="adaptive",
        q_values=q_values,
        rng=random.Random(42),
        random_probability=0.0,
    )

    assert intervened
    assert strategy == "value_based"
    assert action == max(
        env.safe_actions(),
        key=lambda a: q_values[a],
    )


def test_adaptive_safe_passthrough():
    env = GridWorld(size=2, obstacles=set())
    env.reset()

    proposed = env.safe_actions()[0]

    action, intervened, strategy = choose_action(
        env,
        proposed_action=proposed,
        method="adaptive",
        q_values=[0, 1, 2, 0],
        rng=random.Random(42),
        random_probability=1.0,
    )

    assert action == proposed
    assert not intervened
    assert strategy == "none"