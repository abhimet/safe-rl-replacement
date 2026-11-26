"""Independent regression checks for the audited navigation pipeline."""

import numpy as np
import pytest
import torch

from dqn import QNetwork, SpatialQNetwork, DQNAgent, network_from_weights
from environment import GridWorld
from evaluate import evaluate
from evaluate_shield_on import evaluate as evaluate_on
from planner_teacher import training_teacher
from train import train


def test_spatial_encoder_dimensions_and_checkpoint_reloading():
    model = SpatialQNetwork(input_dim=104)
    input_obs, _ = GridWorld(layout="rooms", observation_mode="map").reset()
    values = model(torch.tensor(input_obs, dtype=torch.float32))
    assert values.shape == (4,)
    restored = network_from_weights(model.state_dict())
    assert isinstance(restored, SpatialQNetwork)
    assert torch.allclose(restored(torch.tensor(input_obs)), values)
    assert isinstance(network_from_weights(QNetwork(104).state_dict()), QNetwork)
    with pytest.raises(ValueError):
        SpatialQNetwork(input_dim=105)


def test_teacher_uses_only_training_layouts_and_ranks_collision_below_goal_progress():
    env = GridWorld(layout="rooms", observation_mode="map")
    observations, labels = training_teacher([env], reward_shaping="potential")
    assert len(observations) == len(env.distances_to_goal()) - 1 == 84
    assert labels.shape == (84, 4)
    for obs, q in zip(observations, labels):
        position = tuple(np.round(obs[:2] * 9).astype(int))
        env.position = position
        safe = env.safe_actions()
        assert int(np.argmax(q)) in safe
        if len(safe) < 4:
            assert max(q[a] for a in range(4) if a not in safe) < max(q[a] for a in safe)


def test_spatial_agent_pretraining_syncs_target_network():
    agent = DQNAgent(input_dim=104, architecture="spatial", seed=42, batch_size=8)
    obs, q = training_teacher([GridWorld(layout="rooms", observation_mode="map")])
    agent.set_expert_dataset(obs, q)
    loss = agent.pretrain_expert(5)
    assert np.isfinite(loss)
    assert all(torch.equal(a, b) for a, b in zip(agent.online.parameters(), agent.target.parameters()))


def test_bad_obstacles_rejected():
    with pytest.raises(ValueError, match="Obstacle coordinates"):
        GridWorld(size=3, obstacles=[(100, 4)])


def test_time_limit_is_not_bellman_terminal(tmp_path, monkeypatch):
    import train as training
    seen = []
    class Inspection(DQNAgent):
        def act(self, observation, epsilon=0):
            return 1
        def remember(self, state, executed_action, reward, next_state, done, next_safe_actions=None):
            seen.append(done)
            return super().remember(state, executed_action, reward, next_state, done, next_safe_actions)
    monkeypatch.setattr(training, "DQNAgent", Inspection)
    config = tmp_path / "env.json"
    config.write_text('{"size": 2, "max_steps": 1, "obstacles": []}')
    rows = train(method="unshielded", episodes=1, env_config=config, output=tmp_path)
    assert rows[0]["timeout"] == 1
    assert seen == [False]


def test_shield_off_really_executes_unsafe_policy_and_on_intercepts(tmp_path):
    model = QNetwork(input_dim=2)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.net[-1].bias[0] = 100  # greedy UP is unsafe from the top-left
        model.net[-1].bias[1] = 1
        model.net[-1].bias[2] = 1
    path = tmp_path / "constant_up.pt"
    torch.save(model.state_dict(), path)
    config = tmp_path / "env.json"
    config.write_text('{"size": 2, "max_steps": 5, "obstacles": [], "start": [0, 0]}')
    off = evaluate(str(path), env_config=str(config), output=tmp_path / "off.csv", episodes=1)
    on = evaluate_on(str(path), env_config=str(config), output=tmp_path / "on.csv", episodes=1)
    assert off["collision_rate"] == 1.0 and off["intervention_rate"] == 0.0
    assert on["collision_rate"] == 0.0 and on["intervention_rate"] > 0.0


def test_recommended_fixed_step_spatial_training_smoke(tmp_path):
    rows = train(method="par", max_env_steps=8, seed=5,
                 size=10, observation_mode="map", train_layouts=("rooms", "zigzag"),
                 train_start_policy="all_reachable", architecture="spatial",
                 expert_pretrain_updates=2, expert_loss_weight=0.1,
                 par_schedule="absolute", par_beta_min=0, par_beta_max=4,
                 output=tmp_path)
    assert sum(row["steps"] for row in rows) == 8
    assert list(tmp_path.glob("*.pt"))


def test_expert_is_explicitly_rejected_for_penalty_objective(tmp_path):
    with pytest.raises(ValueError, match="incompatible"):
        train(method="value_penalty", observation_mode="map",
              expert_pretrain_updates=5, output=tmp_path)



def test_fabricated_replay_transitions_forbidden(tmp_path):
    with pytest.raises(ValueError, match="fabricates an invalid transition"):
        train(method="value_based", episodes=1, storage_mode="both", output=tmp_path)
