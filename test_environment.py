import numpy as np
from environment import GridWorld
from shield import choose_action


def test_environment():
    env = GridWorld()
    state, _ = env.reset()
    assert state.shape == (2,)
    assert 0 not in env.safe_actions()  # top boundary
    assert 3 not in env.safe_actions()  # left boundary
    assert 1 in env.safe_actions()
    action, changed = choose_action(env, 0, np.random.default_rng(1), enabled=True)
    assert changed and action in env.safe_actions()
    _, _, terminated, _, info = env.step(0)
    assert terminated and info['collision']
    env.reset()
    _, _, terminated, _, info = env.step(1)
    assert not terminated and not info['collision']
    env.reset()
    assert choose_action(env, 0, np.random.default_rng(1), enabled=False) == (0, False)
    timeout_env = GridWorld(max_steps=1)
    timeout_env.reset()
    _, _, terminated, truncated, _ = timeout_env.step(1)
    assert truncated and not terminated
    goal_env = GridWorld(size=2, obstacles=set())
    goal_env.reset()
    goal_env.step(1)
    _, _, terminated, _, info = goal_env.step(2)
    assert terminated and info['success']
    print('All environment and shield tests passed.')


if __name__ == '__main__':
    test_environment()
