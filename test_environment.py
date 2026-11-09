from environment import LAYOUTS, GridWorld, parse_layout
from shield import choose_action
import random
import pytest

ORIGINAL_6X6_OBSTACLES = {(1, 1), (1, 2), (2, 4), (3, 1), (3, 3), (4, 3)}
# Expected values were computed by a BFS written separately from environment.py.
EXPECTED_LAYOUTS = {
    'rooms': (15, {(0, 0): 18, (0, 9): 15, (6, 0): 12}),
    'zigzag': (28, {(0, 0): 38, (4, 9): 19, (8, 0): 10}),
    'clutter': (28, {(0, 0): 18, (4, 9): 5, (7, 0): 13}),
}

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

def test_default_6x6_is_unchanged():
    env = GridWorld(size=6)
    assert env.layout == 'default_6x6'
    assert env.obstacles == ORIGINAL_6X6_OBSTACLES
    assert env.goal == (5, 5) and env.max_steps == 100
    for seed in range(20):
        obs, info = env.reset(seed=seed)
        assert obs.tolist() == [0.0, 0.0] and info['start'] == (0, 0)

def test_size_10_defaults_to_layout_with_obstacles():
    env = GridWorld(size=10)
    assert env.layout == 'rooms' and env.size == 10 and env.obstacles

@pytest.mark.parametrize('name', sorted(EXPECTED_LAYOUTS))
def test_layout_configuration_and_paths(name):
    obstacle_count, start_distances = EXPECTED_LAYOUTS[name]
    env = GridWorld(layout=name)
    distances = env.distances_to_goal()
    assert env.size == 10 and env.goal == (9, 9)
    assert len(env.obstacles) == obstacle_count
    assert len(distances) == 100 - obstacle_count  # every free cell reaches the goal
    assert set(env.starts) == set(start_distances)
    assert {start: distances[start] for start in env.starts} == start_distances
    assert max(start_distances.values()) < env.max_steps

@pytest.mark.parametrize('name', sorted(LAYOUTS))
def test_shortest_path_reaches_goal_from_every_start(name):
    env = GridWorld(layout=name)
    distances = env.distances_to_goal()
    for start in env.starts:
        env.reset(options={'start': start})
        for expected_steps in range(1, distances[start] + 1):
            action = min(env.safe_actions(), key=lambda a: distances[
                (env.position[0] + env.ACTIONS[a][0], env.position[1] + env.ACTIONS[a][1])])
            _, reward, terminated, truncated, info = env.step(action)
            assert not info['collision'] and not truncated
        assert terminated and info['success'] and reward == 10
        assert env.steps == expected_steps

def test_wall_collision_on_10x10_keeps_position():
    env = GridWorld(layout='zigzag')
    env.reset(options={'start': (0, 0)})
    _, reward, terminated, truncated, info = env.step(0)
    assert env.position == (0, 0)
    assert reward == -10 and terminated and not truncated and info['collision'] and not info['success']

def test_obstacle_collision_on_10x10_keeps_position():
    env = GridWorld(layout='zigzag')
    env.reset(options={'start': (0, 0)})
    _, reward, terminated, _, info = env.step(2)
    assert env.position == (0, 0)
    assert reward == -10 and terminated and info['collision']

def test_goal_on_10x10():
    env = GridWorld(layout='rooms')
    env.reset(options={'start': (9, 8)})
    obs, reward, terminated, truncated, info = env.step(1)
    assert obs.tolist() == [1.0, 1.0]
    assert reward == 10 and terminated and not truncated and info['success']

def test_truncation_on_10x10():
    env = GridWorld(layout='rooms', max_steps=2)
    env.reset(options={'start': (0, 1)})
    assert env.step(1)[3] is False
    _, reward, terminated, truncated, info = env.step(3)
    assert reward == -0.1 and not terminated and truncated
    assert not info['collision'] and not info['success']

def test_reset_clears_episode_state():
    env = GridWorld(layout='rooms')
    env.reset(options={'start': (0, 1)})
    env.step(1)
    obs, _ = env.reset(options={'start': (0, 1)})
    assert env.steps == 0 and env.position == (0, 1)
    assert obs.tolist() == pytest.approx([0.0, 1 / 9])

@pytest.mark.parametrize('name', sorted(EXPECTED_LAYOUTS))
def test_same_seed_gives_same_start(name):
    first, second = GridWorld(layout=name), GridWorld(layout=name)
    for seed in range(50):
        obs_first, info_first = first.reset(seed=seed)
        obs_second, info_second = second.reset(seed=seed)
        assert info_first == info_second and obs_first.tolist() == obs_second.tolist()
        assert first.reset(seed=seed)[1] == info_first

@pytest.mark.parametrize('name', sorted(EXPECTED_LAYOUTS))
def test_seeds_cover_all_candidate_starts(name):
    env = GridWorld(layout=name)
    assert {env.reset(seed=seed)[1]['start'] for seed in range(100)} == set(env.starts)

def test_unseeded_resets_continue_seeded_stream():
    env = GridWorld(layout='rooms')
    def sequence():
        env.reset(seed=7)
        return [env.reset()[1]['start'] for _ in range(20)]
    first = sequence()
    assert first == sequence()
    assert len(set(first)) > 1

def test_custom_and_overridden_starts():
    env = GridWorld(size=3, obstacles=set(), starts=[(0, 0), (1, 1)])
    assert {env.reset(seed=seed)[1]['start'] for seed in range(30)} == {(0, 0), (1, 1)}
    env = GridWorld(layout='zigzag', start=(8, 0))
    assert {env.reset(seed=seed)[1]['start'] for seed in range(30)} == {(8, 0)}

@pytest.mark.parametrize('bad_start', [(1, 0), (10, 0), (-1, 0), (9, 9), (0, 0, 0)])
def test_invalid_reset_start_is_rejected(bad_start):
    env = GridWorld(layout='zigzag')
    with pytest.raises(ValueError):
        env.reset(options={'start': bad_start})

@pytest.mark.parametrize('kwargs', [
    {'size': 3, 'obstacles': {(1, 1)}, 'start': (1, 1)},  # on an obstacle
    {'size': 3, 'obstacles': set(), 'start': (2, 2)},  # on the goal
    {'size': 3, 'obstacles': set(), 'start': (3, 0)},  # outside
    {'size': 3, 'obstacles': {(1, 0), (1, 1), (1, 2)}, 'start': (0, 0)},  # unreachable goal
    {'size': 3, 'obstacles': set(), 'goal': (5, 5)},
    {'size': 3, 'obstacles': {(2, 2)}, 'goal': (2, 2)},
    {'size': 3, 'obstacles': set(), 'starts': []},
    {'size': 3, 'obstacles': set(), 'start': (0, 0), 'starts': [(0, 1)]},
    {'size': 1, 'obstacles': set()},
    {'size': 3, 'obstacles': set(), 'max_steps': 0},
    {'layout': 'missing'},
    {'layout': 'rooms', 'obstacles': set()},
    {'layout': 'rooms', 'goal': (0, 1)},
])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        GridWorld(**kwargs)

@pytest.mark.parametrize('rows', [
    ('S.', '.G', '..'),  # not square
    ('S.', '..'),  # no goal
    ('..', '.G'),  # no start
    ('SG', 'G.'),  # two goals
    ('S?', '.G'),  # unknown symbol
])
def test_malformed_layouts_are_rejected(rows):
    with pytest.raises(ValueError):
        parse_layout(rows)
