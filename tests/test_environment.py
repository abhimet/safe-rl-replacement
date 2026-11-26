from environment import LAYOUTS, GridWorld, parse_layout
import random
import pytest

ORIGINAL_6X6_OBSTACLES = {(1, 1), (1, 2), (2, 4), (3, 1), (3, 3), (4, 3)}
# Expected values were computed by a BFS written separately from environment.py.
EXPECTED_LAYOUTS = {
    'rooms': (15, {(0, 0): 18, (0, 9): 15, (6, 0): 12}),
    'zigzag': (28, {(0, 0): 38, (4, 9): 19, (8, 0): 10}),
    'clutter': (28, {(0, 0): 18, (4, 9): 5, (7, 0): 13}),
    'spiral': (40, {(0, 0): 40, (9, 0): 31, (9, 9): 58}),
    'pillars': (30, {(9, 0): 18, (9, 4): 14, (9, 9): 13}),
}
EXPECTED_GOALS = {'spiral': (4, 4), 'pillars': (0, 9)}

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
    _, reward, done, _, info = env.step(0)
    assert done and info['collision'] and reward == -10

def test_timeout():
    env = GridWorld(size=3, obstacles=set(), max_steps=1)
    env.reset()
    _, _, done, truncated, _ = env.step(1)
    assert not done and truncated

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
    assert env.size == 10 and env.goal == EXPECTED_GOALS.get(name, (9, 9))
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


def test_train_and_heldout_split_is_disjoint_and_complete():
    from environment import HELDOUT_LAYOUTS, TRAIN_LAYOUTS
    assert set(TRAIN_LAYOUTS).isdisjoint(HELDOUT_LAYOUTS)
    assert set(TRAIN_LAYOUTS) | set(HELDOUT_LAYOUTS) == set(TEN_BY_TEN)


def test_goal_on_new_heldout_layouts():
    env = GridWorld(layout='pillars')
    env.reset(options={'start': (0, 8)})
    obs, reward, terminated, truncated, info = env.step(1)
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


# --- Sai: environment validation additions (Task 1) ---
import numpy as np

TEN_BY_TEN = [name for name in sorted(LAYOUTS) if GridWorld(layout=name).size == 10]


@pytest.mark.parametrize('name', TEN_BY_TEN)
def test_map_observation_encodes_position_goal_and_obstacles(name):
    env = GridWorld(layout=name, observation_mode='map')
    for start in env.starts:
        obs, _ = env.reset(options={'start': start})
        assert obs.shape == (4 + 100,) and obs.dtype == np.float32
        assert obs.min() >= 0.0 and obs.max() <= 1.0
        assert obs[:2].tolist() == pytest.approx([start[0] / 9, start[1] / 9])
        assert obs[2:4].tolist() == pytest.approx([env.goal[0] / 9, env.goal[1] / 9])
        grid = obs[4:].reshape(10, 10)
        assert {(r, c) for r in range(10) for c in range(10) if grid[r, c] == 1.0} == env.obstacles
        assert set(np.unique(grid).tolist()) <= {0.0, 1.0}


def test_map_observation_distinguishes_layouts_with_equal_dimension():
    observations = {name: GridWorld(layout=name, observation_mode='map').reset(
        options={'start': (0, 0)})[0] for name in TEN_BY_TEN}
    assert len({obs.shape for obs in observations.values()}) == 1  # train.py multi-layout requirement
    names = list(observations)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not np.array_equal(observations[a], observations[b])


def test_map_observation_tracks_movement():
    env = GridWorld(layout='rooms', observation_mode='map')
    first, _ = env.reset(options={'start': (0, 0)})
    moved, *_ = env.step(1)
    assert moved[:2].tolist() == pytest.approx([0.0, 1 / 9])
    assert np.array_equal(first[2:], moved[2:])  # goal and map are static


@pytest.mark.parametrize('name', TEN_BY_TEN)
def test_rewards_and_flags_are_exact_for_every_cell_and_action(name):
    env = GridWorld(layout=name)
    for cell in env.distances_to_goal():
        if cell == env.goal:
            continue
        for action, (dr, dc) in enumerate(env.ACTIONS):
            env.reset(options={'start': cell})
            target = (cell[0] + dr, cell[1] + dc)
            hits = not (0 <= target[0] < 10 and 0 <= target[1] < 10) or target in env.obstacles
            _, reward, terminated, truncated, info = env.step(action)
            if hits:
                assert (reward, terminated, info['collision'], env.position) == (-10.0, True, True, cell)
            elif target == env.goal:
                assert (reward, terminated, info['success'], env.position) == (10.0, True, True, target)
            else:
                assert (reward, terminated, info['collision'], info['success']) == (-0.1, False, False, False)
                assert env.position == target
            assert not truncated


def test_success_on_final_step_is_terminal_not_truncated():
    env = GridWorld(layout='rooms', max_steps=1)
    env.reset(options={'start': (9, 8)})
    _, reward, terminated, truncated, info = env.step(1)
    assert reward == 10 and terminated and not truncated and info['success']


def test_collision_on_final_step_is_terminal_not_truncated():
    env = GridWorld(layout='zigzag', max_steps=1)
    env.reset(options={'start': (0, 0)})
    _, reward, terminated, truncated, info = env.step(0)
    assert reward == -10 and terminated and not truncated and info['collision']


def test_step_rejects_invalid_actions():
    env = GridWorld(layout='rooms')
    env.reset()
    for bad in (-1, 4):
        with pytest.raises(ValueError):
            env.step(bad)


@pytest.mark.parametrize('name', TEN_BY_TEN)
def test_forced_start_ignores_seed(name):
    env = GridWorld(layout=name)
    for seed in range(5):
        for start in env.starts:
            assert env.reset(seed=seed, options={'start': start})[1]['start'] == start
