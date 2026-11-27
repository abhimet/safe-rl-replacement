import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def step_guard(monkeypatch):
    """Count the steps, and the unsafe actions, actually executed in the environment."""
    import environment
    original = environment.GridWorld.step
    calls = {"unsafe": 0, "steps": 0}

    def guarded(self, action):
        calls["steps"] += 1
        calls["unsafe"] += int(int(action) not in self.safe_actions())
        return original(self, action)

    monkeypatch.setattr(environment.GridWorld, "step", guarded)
    return calls
