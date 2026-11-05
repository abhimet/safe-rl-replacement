import pytest
from adaptive import UnsafeRateTracker


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