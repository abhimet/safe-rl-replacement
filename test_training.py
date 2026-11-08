"""DQN mechanics and training integration acceptance checks."""
import csv
import json

import numpy as np
import pytest
import torch

from dqn import DQNAgent, QNetwork
from evaluate import evaluate
from train import METHODS, train


def test_exploration_and_replay_copy():
    agent = DQNAgent(seed=4, capacity=2)
    observation = np.array([0., 0.], dtype=np.float32)
    assert set(agent.act(observation, epsilon=1) for _ in range(100)) == set(range(4))
    assert agent.act(observation) == int(agent.q_values(observation).argmax())
    agent.remember(observation, 2, 1, observation, True)
    observation[:] = 1
    assert np.array_equal(agent.memory[0][0], [0, 0])
    assert agent.memory[0][1] == 2
    for _ in range(3):
        agent.remember(observation, 1, 0, observation, False)
    assert len(agent.memory) == 2


def test_learning_terminal_target_and_target_sync():
    torch.set_num_threads(1)
    agent = DQNAgent(batch_size=2)
    assert agent.learn() is None
    # Terminal targets must ignore a large target-network prediction.
    for network in (agent.online, agent.target):
        for parameter in network.parameters():
            parameter.data.zero_()
    agent.target.net[-1].bias.data.fill_(100)
    for _ in range(2):
        agent.remember([0, 0], 1, 2, [1, 1], True)
    assert agent.learn() == pytest.approx(1.5)
    assert agent.online.net[-1].bias[1].item() > 0
    assert agent.target.net[-1].bias[1].item() == 100
    for _ in range(99):
        assert np.isfinite(agent.learn())
    assert agent.updates == 100
    for online, target in zip(agent.online.parameters(), agent.target.parameters()):
        assert torch.equal(online, target)


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('size', [6, 10])
def test_all_methods_and_sizes(tmp_path, method, size):
    config = tmp_path / 'env.json'
    config.write_text(json.dumps({'size': size, 'max_steps': 40, 'obstacles': [[1, 1]]}))
    rows = train(method=method, episodes=4, output=tmp_path, env_config=config)
    assert len(rows) == 4
    assert all(0 <= r['interventions'] <= r['unsafe_proposals'] <= r['steps'] <= 40 for r in rows)
    if method == 'unshielded':
        assert all(r['interventions'] == 0 for r in rows)
    else:
        assert all(r['collision'] == 0 for r in rows)
        assert all(r['random_replacements'] + r['value_replacements'] == r['interventions'] for r in rows)
    model = tmp_path / f'{method}_seed42.pt'
    summary = evaluate(model, episodes=2, output=tmp_path / 'evaluation.csv', env_config=config)
    assert summary['episodes'] == 2


def test_checkpoints_and_reproducibility(tmp_path):
    config = tmp_path / 'env.json'
    config.write_text(json.dumps({'size': 2, 'max_steps': 4, 'obstacles': []}))
    first = train('adaptive', episodes=100, output=tmp_path / 'first', env_config=config)
    second = train('adaptive', episodes=100, output=tmp_path / 'second', env_config=config)
    assert first == second
    for episode in (50, 100):
        filename = f'adaptive_seed42_ep{episode}.pt'
        a = torch.load(tmp_path / 'first' / filename, weights_only=True)
        b = torch.load(tmp_path / 'second' / filename, weights_only=True)
        network = QNetwork()
        network.load_state_dict(a)
        assert all(torch.equal(a[key], b[key]) for key in a)
        assert torch.isfinite(network(torch.zeros(2))).all()
    with (tmp_path / 'first' / 'adaptive_seed42.csv').open(newline='') as f:
        assert len(list(csv.DictReader(f))) == 100


def test_training_stores_executed_action_and_penalized_reward(tmp_path, monkeypatch):
    import train as training

    transitions = []

    class InspectAgent(DQNAgent):
        def act(self, observation, epsilon=0):
            return 0  # up is unsafe from the configured start

        def remember(self, state, executed_action, reward, next_state, done):
            transitions.append((executed_action, reward, done))
            super().remember(state, executed_action, reward, next_state, done)

    monkeypatch.setattr(training, 'DQNAgent', InspectAgent)
    config = tmp_path / 'env.json'
    config.write_text(json.dumps({'size': 2, 'max_steps': 1, 'obstacles': []}))
    rows = train('value_penalty', episodes=1, output=tmp_path, penalty=2, env_config=config)
    assert len(transitions) == 1
    action, reward, done = transitions[0]
    assert action in (1, 2)
    assert reward == pytest.approx(-2.1)
    assert done
    assert rows[0]['reward'] == pytest.approx(-0.1)
    assert rows[0]['unsafe_proposals'] == rows[0]['interventions'] == rows[0]['value_replacements'] == 1
