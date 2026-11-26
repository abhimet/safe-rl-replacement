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


def _train_count_aware(tmp_path, **extra):
    import json
    from train import train
    train(method='par', seed=42, size=10, observation_mode='map',
          train_layouts=('rooms', 'zigzag'), train_start_policy='all_reachable',
          max_env_steps=150, architecture='spatial', par_schedule='absolute',
          par_beta_min=0, par_beta_max=4, par_selector='count_aware',
          par_novelty_weight=3, storage_mode='counterfactual',
          backup_mode='shield_safe', reward_shaping='potential',
          output=str(tmp_path), **extra)
    return json.loads(next(tmp_path.glob('*summary.json')).read_text())


def test_adaptive_run_records_selector_averages(tmp_path):
    summary = _train_count_aware(tmp_path)
    assert summary['count_aware_replacements'] > 0
    assert 0 <= summary['count_aware_mean_beta'] <= 4
    assert 0 <= summary['count_aware_mean_unsafe_rate'] <= 1
    assert summary['par_fixed_unsafe_rate'] is None


def test_fixed_ablation_ignores_live_feedback(tmp_path):
    summary = _train_count_aware(tmp_path, par_fixed_beta=1.25, par_fixed_unsafe_rate=0.4)
    assert summary['count_aware_replacements'] > 0
    assert summary['count_aware_mean_beta'] == pytest.approx(1.25)
    assert summary['count_aware_mean_unsafe_rate'] == pytest.approx(0.4)
    assert summary['par_fixed_unsafe_rate'] == 0.4


@pytest.mark.parametrize('extra', [dict(par_fixed_unsafe_rate=1.5),
                                   dict(par_fixed_unsafe_rate=0.5, par_selector='boltzmann')])
def test_rejects_bad_fixed_unsafe_rate(tmp_path, extra):
    from train import train
    with pytest.raises(ValueError):
        train(method='par', max_env_steps=5, output=str(tmp_path),
              **{'par_selector': 'count_aware', **extra})


def test_fixed_constants_average_par_count_seeds(tmp_path):
    import json
    from run_comparison import fixed_ablation_constants
    with pytest.raises(FileNotFoundError):
        fixed_ablation_constants(tmp_path, 5000)
    for seed, beta, rate in ((42, 1.0, 0.2), (43, 2.0, 0.4)):
        run = tmp_path / f'par_count_seed{seed}_steps5000'
        run.mkdir()
        (run / 'x_summary.json').write_text(json.dumps(
            {'count_aware_mean_beta': beta, 'count_aware_mean_unsafe_rate': rate}))
    # Different budgets and the fixed ablation's own runs are never used.
    other = tmp_path / 'par_count_fixed_seed42_steps5000'
    other.mkdir()
    (other / 'x_summary.json').write_text(json.dumps(
        {'count_aware_mean_beta': 9.0, 'count_aware_mean_unsafe_rate': 0.9}))
    constants = fixed_ablation_constants(tmp_path, 5000)
    assert constants['par_fixed_beta'] == pytest.approx(1.5)
    assert constants['par_fixed_unsafe_rate'] == pytest.approx(0.3)
    assert len(constants['derived_from']) == 2
