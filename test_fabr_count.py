"""Count-aware FABR tests: training-only safe selection and state feedback."""
import random

import numpy as np
import pytest

from fabr import CountAwareReplacement


def test_never_selects_an_unsafe_action():
    chooser = CountAwareReplacement(random.Random(7), novelty_weight=3)
    for _ in range(100):
        assert chooser.select(('rooms', (2, 2)), [0, 2],
                              [5, 100, 1, 1000], beta=2, unsafe_rate=.5) in (0, 2)


def test_counts_explore_unvisited_safe_actions_when_greedy_unsafe():
    chooser = CountAwareReplacement(random.Random(2), novelty_weight=8)
    for _ in range(100):
        chooser.observe(('rooms', (2, 2)), 0)
    choices = [chooser.select(('rooms', (2, 2)), [0, 2], [0, 0, 0, 0],
                              beta=0, unsafe_rate=1) for _ in range(100)]
    assert choices.count(2) > 90


def test_feedback_nearly_turns_off_exploration_at_zero_greedy_unsafe_rate():
    a = CountAwareReplacement(random.Random(4), novelty_weight=100)
    b = CountAwareReplacement(random.Random(4), novelty_weight=0)
    for _ in range(40):
        a.observe(('rooms', (1, 1)), 0)
    for _ in range(60):
        assert a.select(('rooms', (1, 1)), [0, 1], [0, 0, 0, 0], 2, 0) == \
               b.select(('rooms', (1, 1)), [0, 1], [0, 0, 0, 0], 2, 0)


def test_counts_do_not_leak_between_maps():
    chooser = CountAwareReplacement(random.Random(1), 1)
    chooser.observe((0, (0, 0)), 2)
    assert chooser.counts[((1, (0, 0)), 2)] == 0


@pytest.mark.parametrize('bad', [-1, float('nan'), float('inf')])
def test_rejects_bad_bonus(bad):
    with pytest.raises(ValueError):
        CountAwareReplacement(random.Random(1), bad)


def test_rejects_missing_or_invalid_safe_actions():
    chooser = CountAwareReplacement(random.Random(1))
    for safe in ([], [0, 0], [-1], [4]):
        with pytest.raises(ValueError):
            chooser.select(('rooms', 0), safe, [0,0,0,0], 1, .5)
    with pytest.raises(ValueError):
        chooser.select(('rooms', 0), [0], [float('nan'),0,0,0], 1, .5)


def test_training_saves_new_selector_metadata(tmp_path):
    import json
    from train import train
    train(method='par', seed=43, size=10, observation_mode='map',
          train_layouts=('rooms', 'zigzag'), train_start_policy='all_reachable',
          max_env_steps=8, architecture='spatial', par_schedule='absolute',
          par_beta_min=0, par_beta_max=4, par_selector='count_aware',
          par_novelty_weight=3, storage_mode='counterfactual',
          backup_mode='shield_safe', reward_shaping='potential',
          output=str(tmp_path))
    summaries = list(tmp_path.glob('*summary.json'))
    assert len(summaries) == 1
    config = json.loads(summaries[0].read_text())
    assert config['environment_steps'] == 8
    assert config['par_selector'] == 'count_aware'
    assert config['par_novelty_weight'] == 3
    assert config['architecture'] == 'spatial'
    assert config['backup_mode'] == 'shield_safe'
    assert config['storage_mode'] == 'counterfactual'
