from environment import GridWorld
from shield import choose_action
import random

def test_movement_and_goal():
    env = GridWorld(size=2, obstacles=set())
    obs, _ = env.reset()
    assert obs.tolist() == [0.0, 0.0]
    obs, _, done, _, info = env.step(1)
    assert not done and not info['collision']
    _, reward, done, _, info = env.step(2)
    assert done and info['success'] and reward == 10

def test_collision_and_shield():
    env = GridWorld(size=2, obstacles=set())
    env.reset()
    assert 0 not in env.safe_actions()
    executed, intervened, _ = choose_action(env, 0, 'random_safe', rng=random.Random(0))
    assert intervened and executed in env.safe_actions()
    executed, intervened, _ = choose_action(env, 0, 'unshielded')
    assert executed == 0 and not intervened
    _, reward, done, _, info = env.step(0)
    assert done and info['collision'] and reward == -10

def test_timeout():
    env = GridWorld(size=3, obstacles=set(), max_steps=1)
    env.reset()
    _, _, done, truncated, _ = env.step(1)
    assert not done and truncated

def test_value_replacement():
    env = GridWorld(size=2, obstacles=set())
    env.reset()
    action, intervention, strategy = choose_action(env, 0, 'value_based', q_values=[100, 2, 3, 0])
    assert (action, intervention, strategy) == (2, True, 'value_based')
