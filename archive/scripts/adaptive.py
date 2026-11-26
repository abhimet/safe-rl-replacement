"""Adaptive probability is recent unsafe proposals / recent total proposals."""
from collections import deque

class UnsafeRateTracker:
    def __init__(self, window=100):
        if window < 1:
            raise ValueError('window must be positive')
        self.history = deque(maxlen=window)

    def record(self, unsafe):
        self.history.append(bool(unsafe))

    @property
    def rate(self):
        return sum(self.history) / len(self.history) if self.history else 0.5
