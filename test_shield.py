import random
import pytest
from environment import LAYOUTS, GridWorld
from shield import choose_action

SHIELDED_METHODS = ('random_safe', 'value_based', 'value_penalty', 'fixed_mix', 'adaptive')
MOVES = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}


def free_cells(env):
    return [(r, c) for r in range(env.size) for c in range(env.size) if (r, c) not in env.obstacles]


def is_safe(env, cell, action):
    """Brute-force safety check written separately from GridWorld.safe_actions."""
    row, col = cell[0] + MOVES[action][0], cell[1] + MOVES[action][1]
    return 0 <= row < env.size and 0 <= col < env.size and (row, col) not in env.obstacles


@pytest.mark.parametrize('name', sorted(LAYOUTS))
def test_safe_actions_match_brute_force(name):
    env = GridWorld(layout=name)
    for cell in free_cells(env):
        env.position = cell
        assert env.safe_actions() == [a for a in MOVES if is_safe(env, cell, a)]


@pytest.mark.parametrize('method', SHIELDED_METHODS)
@pytest.mark.parametrize('name', sorted(LAYOUTS))
def test_shield_prevents_every_collision(name, method):
    env = GridWorld(layout=name)
    rng = random.Random(0)
    for cell in free_cells(env):
        for action in MOVES:
            env.position = cell
            q_values = [1.0, 2.0, 3.0, 4.0]
            q_values[action] = 100.0  # the proposal is always the most attractive action
            safe = is_safe(env, cell, action)
            executed, intervened, _ = choose_action(
                env, action, method, q_values=q_values, rng=rng, random_probability=0.5)
            assert intervened == (not safe)
            if safe:
                assert executed == action
            *_, info = env.step(executed)
            assert not info['collision']


@pytest.mark.parametrize('name', sorted(LAYOUTS))
def test_unshielded_passes_unsafe_actions_through(name):
    env = GridWorld(layout=name)
    for cell in free_cells(env):
        for action in MOVES:
            env.position = cell
            assert choose_action(env, action, 'unshielded') == (action, False, 'none')


@pytest.mark.parametrize('method', SHIELDED_METHODS)
def test_safe_proposal_is_not_replaced(method):
    env = GridWorld(size=3, obstacles=set())
    env.reset()
    result = choose_action(env, 1, method, q_values=[0, 0, 9, 0], rng=random.Random(0), random_probability=1.0)
    assert result == (1, False, 'none')


@pytest.mark.parametrize('method', ('value_based', 'value_penalty'))
def test_value_replacement_picks_best_safe_action(method):
    env = GridWorld(size=3, obstacles=set())
    env.reset()  # at (0, 0), only right and down are safe
    assert choose_action(env, 0, method, q_values=[100, 5, 7, 100]) == (2, True, method)
    assert choose_action(env, 3, method, q_values=[100, 7, 5, 100]) == (1, True, method)


def test_value_replacement_respects_obstacles():
    env = GridWorld(layout='zigzag')
    env.reset(options={'start': (0, 0)})  # down is an obstacle and up/left are walls
    assert choose_action(env, 2, 'value_based', q_values=[50, 1, 100, 50]) == (1, True, 'value_based')


def test_value_replacement_requires_q_values():
    env = GridWorld(size=3, obstacles=set())
    env.reset()
    with pytest.raises(ValueError):
        choose_action(env, 0, 'value_based')


def test_random_replacement_is_reproducible_and_covers_safe_actions():
    env = GridWorld(size=3, obstacles=set())
    env.reset()
    def draws(seed):
        rng = random.Random(seed)
        return [choose_action(env, 0, 'random_safe', rng=rng)[0] for _ in range(50)]
    assert draws(3) == draws(3)
    assert set(draws(3)) == set(env.safe_actions())


def test_fixed_mix_uses_both_strategies():
    env = GridWorld(size=3, obstacles=set())
    env.reset()
    rng = random.Random(0)
    strategies = {choose_action(env, 0, 'fixed_mix', q_values=[0, 1, 2, 0], rng=rng)[2] for _ in range(50)}
    assert strategies == {'random_safe', 'value_based'}


@pytest.mark.parametrize('method', SHIELDED_METHODS)
def test_enclosed_cell_falls_back_to_proposal(method):
    env = GridWorld(size=3, obstacles={(1, 0), (2, 1)}, start=(0, 0), goal=(0, 2))
    env.position = (2, 0)
    assert env.safe_actions() == []
    result = choose_action(env, 0, method, q_values=[0, 0, 0, 0], rng=random.Random(0), random_probability=0.5)
    assert result == (0, False, 'no_safe_action')


@pytest.mark.parametrize('action', (0, 1))  # unsafe and safe proposals at (0, 0)
def test_unknown_method_is_rejected(action):
    env = GridWorld(size=3, obstacles=set())
    env.reset()
    with pytest.raises(ValueError):
        choose_action(env, action, 'typo', q_values=[0, 0, 0, 0], rng=random.Random(0))


@pytest.mark.parametrize('probability', (None, -0.1, 1.1))
def test_adaptive_rejects_invalid_probability(probability):
    env = GridWorld(size=3, obstacles=set())
    env.reset()
    with pytest.raises(ValueError):
        choose_action(env, 0, 'adaptive', q_values=[0, 0, 0, 0], rng=random.Random(0),
                      random_probability=probability)
