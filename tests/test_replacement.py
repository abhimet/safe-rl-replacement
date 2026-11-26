"""Shield replacement strategies (replacement.py) on the real layouts."""
from collections import Counter
import random

import pytest

from environment import LAYOUTS, GridWorld
from replacement import METHODS, Replacer, boltzmann_choice, fabr_beta

SHIELDED = tuple(m for m in METHODS if m != "unshielded")
TEN_BY_TEN = [name for name in sorted(LAYOUTS) if GridWorld(layout=name).size == 10]


def replacer(method, seed=0, **kwargs):
    if method == "fabr_fixed":
        kwargs.setdefault("fixed_beta", 2.0)
    return Replacer(method, random.Random(seed), **kwargs)


def free_cells(env):
    return [(r, c) for r in range(env.size) for c in range(env.size) if (r, c) not in env.obstacles]


@pytest.mark.parametrize("name", TEN_BY_TEN)
def test_safe_actions_match_brute_force(name):
    env = GridWorld(layout=name)
    for cell in free_cells(env):
        env.position = cell
        expected = [a for a, (dr, dc) in enumerate(env.ACTIONS)
                    if 0 <= cell[0] + dr < 10 and 0 <= cell[1] + dc < 10
                    and (cell[0] + dr, cell[1] + dc) not in env.obstacles]
        assert env.safe_actions() == expected


def test_named_layouts_have_no_enclosed_cells():
    for name in LAYOUTS:
        env = GridWorld(layout=name)
        for cell in free_cells(env):
            env.position = cell
            assert env.safe_actions(), (name, cell)


@pytest.mark.parametrize("method", SHIELDED)
@pytest.mark.parametrize("name", TEN_BY_TEN)
def test_shielded_methods_never_execute_unsafe_actions(name, method):
    env = GridWorld(layout=name)
    rep = replacer(method)
    rng = random.Random(1)
    for cell in free_cells(env):
        env.position = cell
        safe = env.safe_actions()
        for proposed in range(4):
            q = [rng.uniform(-5, 5) for _ in range(4)]
            executed, intervened = rep.choose(proposed, safe, q)
            assert executed in safe
            assert intervened == (proposed not in safe)
            if proposed in safe:
                assert executed == proposed


def test_unshielded_passes_unsafe_actions_through():
    rep = replacer("unshielded")
    assert rep.choose(0, [1, 2], [9, 0, 0, 0]) == (0, False)


def test_single_safe_action_is_always_chosen():
    for method in SHIELDED:
        rep = replacer(method)
        for _ in range(20):
            assert rep.choose(0, [3], [100, 0, 0, -100]) == (3, True)


@pytest.mark.parametrize("method", ("highest_q", "value_penalty"))
def test_highest_q_picks_best_safe_action(method):
    assert replacer(method).choose(0, [1, 2, 3], [100, 2, 3, 0]) == (2, True)


def test_random_safe_covers_all_safe_actions():
    rep = replacer("random_safe")
    picks = Counter(rep.choose(0, [1, 2, 3], [100, 50, 0, -50])[0] for _ in range(600))
    assert set(picks) == {1, 2, 3} and min(picks.values()) > 150


def test_boltzmann_limits():
    rng = random.Random(0)
    q = [0.0, 3.0, 1.0, -2.0]
    uniform = Counter(boltzmann_choice([1, 2, 3], q, 0.0, rng) for _ in range(900))
    assert min(uniform.values()) > 250  # beta = 0 is uniform random
    greedy = Counter(boltzmann_choice([1, 2, 3], q, 50.0, rng) for _ in range(200))
    assert greedy == {1: 200}  # large beta is argmax
    tied = Counter(boltzmann_choice([1, 2], [0, 5, 5, 0], 50.0, rng) for _ in range(400))
    assert set(tied) == {1, 2}  # equal Q-values stay uniform


def test_fabr_beta_schedule():
    assert fabr_beta(1.0) == 0.0 and fabr_beta(0.0) == 4.0
    assert fabr_beta(0.5, 1.0, 5.0) == pytest.approx(2.0)
    rates = [i / 10 for i in range(11)]
    betas = [fabr_beta(r) for r in rates]
    assert betas == sorted(betas, reverse=True)
    with pytest.raises(ValueError):
        fabr_beta(1.5)


def test_fabr_beta_follows_greedy_feedback_and_fixed_does_not():
    fabr, fixed = replacer("fabr", window=10), replacer("fabr_fixed", window=10, fixed_beta=1.5)
    assert fabr.beta == 0.0  # no history yet: treat the policy as unsafe and explore
    unsafe_greedy_q = [9, 0, 0, 0]  # argmax 0 is unsafe below
    for _ in range(10):
        fabr.choose(1, [1, 2], unsafe_greedy_q)
        fixed.choose(1, [1, 2], unsafe_greedy_q)
    assert fabr.greedy_unsafe_rate == 1.0 and fabr.beta == 0.0
    safe_greedy_q = [0, 9, 0, 0]
    for _ in range(10):
        fabr.choose(1, [1, 2], safe_greedy_q)
        fixed.choose(1, [1, 2], safe_greedy_q)
    assert fabr.greedy_unsafe_rate == 0.0 and fabr.beta == 4.0
    assert fixed.beta == 1.5


def test_adaptive_mix_probability_tracks_unsafe_proposals():
    rep = replacer("adaptive_mix", window=4)
    assert rep.mix_probability == 0.5
    for proposed in (0, 0, 1, 1):
        rep.choose(proposed, [1, 2], [0, 0, 0, 0])
    assert rep.mix_probability == 0.5
    for _ in range(4):
        rep.choose(0, [1, 2], [0, 0, 0, 0])
    assert rep.mix_probability == 1.0


@pytest.mark.parametrize("kwargs", [dict(method="nope"), dict(method="fabr", fixed_beta=1.0),
                                    dict(method="fabr_fixed"), dict(method="fabr", window=0),
                                    dict(method="fabr", beta_min=4, beta_max=4)])
def test_invalid_configuration_is_rejected(kwargs):
    method = kwargs.pop("method")
    with pytest.raises(ValueError):
        Replacer(method, random.Random(0), **kwargs)
