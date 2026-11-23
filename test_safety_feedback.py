"""Tests for optional train-only supervision of unsafe action Q-values."""

import numpy as np
import pytest

from dqn import DQNAgent
from train import train


def test_safety_feedback_requires_current_action_labels():
    agent = DQNAgent(seed=5, safety_loss_weight=1.0, batch_size=2)
    with pytest.raises(ValueError, match="current_safe_actions"):
        agent.remember([0, 0], 1, 1, [0, 0], True)


def test_ranking_feedback_teaches_greedy_policy_to_avoid_known_unsafe_actions():
    agent = DQNAgent(seed=5, lr=0.003, batch_size=16,
                     safety_loss_weight=3.0, safety_margin=1.0)
    state = np.asarray([0.35, 0.65], dtype=np.float32)
    # This training state has exactly two safe actions (right/down).
    for _ in range(64):
        agent.remember(state, 1, 0.0, state, True,
                       current_safe_actions=[1, 2])
    for _ in range(150):
        agent.learn()
    q = agent.q_values(state)
    assert int(np.argmax(q)) in (1, 2)
    assert max(q[0], q[3]) <= max(q[1], q[2]) - 0.8
    assert agent.last_stats["safety_loss"] < 0.05


def test_safety_feedback_train_smoke(tmp_path):
    rows = train(method="value_based", episodes=3,
                 output=tmp_path, safety_loss_weight=1.0,
                 safety_margin=1.0)
    assert rows
    assert (tmp_path / "value_based_seed42_safetyw1.pt").exists()


def test_safety_feedback_rejects_fake_replay(tmp_path):
    with pytest.raises(ValueError, match="executed-only"):
        train(method="par", episodes=1, output=tmp_path,
              safety_loss_weight=1.0, storage_mode="both")


def test_absolute_schedule_avoids_negative_beta_and_uses_own_filename(tmp_path):
    rows = train(method="par", episodes=3, output=tmp_path,
                 par_schedule="absolute", par_beta_min=0.0,
                 par_beta_max=4.0)
    assert all(0.0 <= row["par_beta"] <= 4.0 for row in rows)
    assert (tmp_path / "par_seed42_absolute.pt").exists()


def test_safety_feedback_with_masked_double_dqn_bootstrap(tmp_path):
    rows = train(method="value_based", episodes=3, output=tmp_path,
                 backup_mode="shield_safe", safety_loss_weight=3.0)
    assert rows
    assert (tmp_path / "value_based_seed42_safetyw3.pt").exists()
