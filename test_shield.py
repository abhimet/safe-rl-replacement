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


# --- Sai: shield validation additions (Task 2) ---
import environment
from train import PILOT_METHODS, train

TEN_BY_TEN = [name for name in sorted(LAYOUTS) if GridWorld(layout=name).size == 10]
ALL_SHIELDED = tuple(m for m in PILOT_METHODS if m != 'unshielded')


@pytest.mark.parametrize('name', TEN_BY_TEN)
def test_corner_and_edge_safe_sets(name):
    env = GridWorld(layout=name)
    for cell in free_cells(env):
        env.position = cell
        safe = set(env.safe_actions())
        row, col = cell
        if row == 0:
            assert 0 not in safe
        if row == env.size - 1:
            assert 2 not in safe
        if col == 0:
            assert 3 not in safe
        if col == env.size - 1:
            assert 1 not in safe
    env.position = (0, 0)
    assert set(env.safe_actions()) <= {1, 2}


def one_safe_cells(env):
    cells = []
    for cell in free_cells(env):
        env.position = cell
        if len(env.safe_actions()) == 1:
            cells.append(cell)
    return cells


def test_layouts_contain_one_safe_action_cells():
    # Dead ends exist, so the single-safe-action case is exercised on real maps.
    assert any(one_safe_cells(GridWorld(layout=name)) for name in TEN_BY_TEN)


@pytest.mark.parametrize('method', SHIELDED_METHODS)
@pytest.mark.parametrize('name', TEN_BY_TEN)
def test_single_safe_action_is_always_chosen(name, method):
    env = GridWorld(layout=name)
    rng = random.Random(1)
    for cell in one_safe_cells(env):
        env.position = cell
        (only,) = env.safe_actions()
        for action in MOVES:
            if action == only:
                continue
            q_values = [0.0, 0.0, 0.0, 0.0]
            q_values[action] = 100.0
            q_values[only] = -100.0  # even the worst Q-value must be chosen
            executed, intervened, _ = choose_action(
                env, action, method, q_values=q_values, rng=rng, random_probability=0.5)
            assert (executed, intervened) == (only, True)


def test_named_layouts_have_no_enclosed_cells():
    for name in LAYOUTS:
        env = GridWorld(layout=name)
        for cell in free_cells(env):
            env.position = cell
            assert env.safe_actions(), (name, cell)


@pytest.fixture
def executed_action_guard(monkeypatch):
    """Fail if any shielded training step executes an action outside safe_actions()."""
    original = environment.GridWorld.step
    calls = {'unsafe_executed': 0, 'steps': 0}

    def guarded(self, action):
        calls['steps'] += 1
        calls['unsafe_executed'] += int(int(action) not in self.safe_actions())
        return original(self, action)

    monkeypatch.setattr(environment.GridWorld, 'step', guarded)
    return calls


@pytest.mark.parametrize('method', ALL_SHIELDED)
def test_training_executes_only_safe_actions_and_counts_interventions(method, tmp_path, executed_action_guard):
    rows = train(method=method, episodes=3, seed=0, output=tmp_path, size=10, observation_mode='map',
                 train_layouts=['rooms', 'zigzag'], train_start_policy='all_reachable')
    assert executed_action_guard['steps'] > 0
    assert executed_action_guard['unsafe_executed'] == 0
    for row in rows:
        assert row['collision'] == 0
        assert row['interventions'] == row['unsafe_proposals']  # every unsafe proposal is replaced
        assert row['unsafe_proposals'] <= row['steps']


def test_unshielded_training_never_intervenes(tmp_path):
    rows = train(method='unshielded', episodes=3, seed=0, output=tmp_path, size=10, observation_mode='map',
                 train_layouts=['rooms', 'zigzag'], train_start_policy='all_reachable')
    assert all(row['interventions'] == 0 for row in rows)
    # An unshielded episode ending in collision must have proposed (and executed) an unsafe action.
    assert all(row['unsafe_proposals'] >= row['collision'] for row in rows)


@pytest.mark.parametrize('fixed_beta', (None, -4.0, 0.0, 4.0))
def test_par_replacements_are_safe_for_each_beta(fixed_beta, tmp_path, executed_action_guard):
    rows = train(method='par', episodes=3, seed=1, output=tmp_path, size=10, observation_mode='map',
                 train_layouts=['rooms', 'zigzag'], par_fixed_beta=fixed_beta)
    assert executed_action_guard['unsafe_executed'] == 0
    assert all(row['collision'] == 0 for row in rows)
