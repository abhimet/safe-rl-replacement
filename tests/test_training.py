"""Double DQN, the training pool and the training loop."""
import csv
import json

import numpy as np
import pytest
import torch

import environment
from dqn import DQNAgent, network_from_weights
from environment import GridWorld, generate_training_pool, wall_overlap
from make_eval_scenarios import evaluation_maps
from replacement import METHODS
from train import shaping_bonus, train, training_maps

SHIELDED = tuple(m for m in METHODS if m != "unshielded")


def tiny(method, tmp_path, **kwargs):
    kwargs.setdefault("fixed_beta", 2.0 if method == "fabr_fixed" else None)
    return train(method=method, steps=150, seed=0, output=tmp_path, pool_size=3, verbose=False, **kwargs)


# ---- training pool ----

def test_pool_is_deterministic_and_valid():
    a, b = generate_training_pool(n=20, seed=3), generate_training_pool(n=20, seed=3)
    assert [(n, e.obstacles, e.goal) for n, e in a] == [(n, e.obstacles, e.goal) for n, e in b]
    assert generate_training_pool(n=20, seed=4)[0][1].obstacles != a[0][1].obstacles
    for _, env in a:
        assert env.size == 10 and env.observation_dim == 104
        assert len(env.starts) >= 30
        assert set(env.starts) == set(env.distances_to_goal()) - {env.goal}
        for cell in env.starts:
            env.position = cell
            assert env.safe_actions()


def test_training_maps_do_not_resemble_evaluation_maps():
    maps = training_maps(pool_size=100)
    assert [n for n, _ in maps[:2]] == ["rooms", "zigzag"] and len(maps) == 102
    evaluation = evaluation_maps()
    assert set(evaluation) == {"clutter", "pillars", "spiral", "heldout_1", "heldout_2", "heldout_3"}
    for _, env in maps[2:]:
        for other in evaluation.values():
            assert wall_overlap(env.obstacles, other.obstacles) <= 0.5


# ---- DQN ----

def test_terminal_target_ignores_bootstrap_and_target_syncs():
    agent = DQNAgent(input_dim=104, seed=0, batch_size=1, target_sync=1)
    state = np.zeros(104, dtype=np.float32)
    agent.remember(state, 1, -10.0, state, True)
    agent.learn()
    assert agent.updates == 1
    for p, t in zip(agent.online.parameters(), agent.target.parameters()):
        assert torch.equal(p, t)


def test_shield_safe_backup_requires_next_safe_actions():
    agent = DQNAgent(input_dim=104, backup_mode="shield_safe")
    state = np.zeros(104, dtype=np.float32)
    with pytest.raises(ValueError):
        agent.remember(state, 0, 0.0, state, False, None)
    agent.remember(state, 0, 0.0, state, True, None)  # terminal transitions need no mask


def test_checkpoint_round_trip(tmp_path):
    agent = DQNAgent(input_dim=104)
    model = network_from_weights(agent.online.state_dict())
    x = torch.rand(104)
    assert torch.allclose(model(x), agent.online(x))


# ---- training loop ----

@pytest.mark.parametrize("method", METHODS)
def test_every_method_trains_and_writes_outputs(method, tmp_path):
    summary = tiny(method, tmp_path)
    assert summary["steps"] == 150 and summary["method"] == method
    rows = list(csv.DictReader((tmp_path / f"{method}_seed0.csv").open()))
    assert int(rows[-1]["env_steps"]) == 150
    assert sum(int(r["steps"]) for r in rows) == 150
    assert (tmp_path / f"{method}_seed0.pt").exists()
    saved = json.loads((tmp_path / f"{method}_seed0_summary.json").read_text())
    assert saved["config"]["training_maps"] == 5
    for r in rows:
        if not int(r["budget_cut"]):
            assert int(r["success"]) + int(r["collision"]) + int(r["timeout"]) == 1
    if method == "unshielded":
        assert all(int(r["interventions"]) == 0 for r in rows)
    if method in ("fabr", "fabr_fixed"):
        assert summary["mean_beta"] is not None
    if method == "fabr_fixed":
        assert {float(r["beta"]) for r in rows} == {2.0}


@pytest.fixture
def step_guard(monkeypatch):
    """Count unsafe actions actually executed in the environment."""
    original = environment.GridWorld.step
    calls = {"unsafe": 0, "steps": 0}

    def guarded(self, action):
        calls["steps"] += 1
        calls["unsafe"] += int(int(action) not in self.safe_actions())
        return original(self, action)

    monkeypatch.setattr(environment.GridWorld, "step", guarded)
    return calls


@pytest.mark.parametrize("method", SHIELDED)
def test_shielded_training_never_collides(method, tmp_path, step_guard):
    summary = tiny(method, tmp_path)
    assert step_guard["steps"] == 150 and step_guard["unsafe"] == 0
    assert summary["training"]["collision_rate"] == 0.0
    assert summary["training"]["intervention_rate"] == summary["training"]["unsafe_proposal_rate"]


def test_counterfactual_collision_is_stored_for_each_intervention(tmp_path, monkeypatch):
    stored = []
    original = DQNAgent.remember

    def spy(self, state, action, reward, next_state, done, next_safe=None, **kwargs):
        stored.append((action, reward, done, np.array_equal(state, next_state)))
        return original(self, state, action, reward, next_state, done, next_safe, **kwargs)

    monkeypatch.setattr(DQNAgent, "remember", spy)
    tiny("highest_q", tmp_path)
    rows = list(csv.DictReader((tmp_path / "highest_q_seed0.csv").open()))
    interventions = sum(int(r["interventions"]) for r in rows)
    counterfactual = [s for s in stored if s[2] and s[3] and -10 <= s[1] <= -6]  # -10 + shaping in [0, 4]
    assert interventions > 0 and len(stored) == 150 + interventions
    assert len(counterfactual) >= interventions


def test_unshielded_stores_no_counterfactuals(tmp_path, monkeypatch):
    count = {"n": 0}
    original = DQNAgent.remember

    def spy(self, *args, **kwargs):
        count["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DQNAgent, "remember", spy)
    tiny("unshielded", tmp_path)
    assert count["n"] == 150


def test_training_is_reproducible(tmp_path):
    tiny("fabr", tmp_path / "a")
    tiny("fabr", tmp_path / "b")
    assert (tmp_path / "a" / "fabr_seed0.csv").read_text() == (tmp_path / "b" / "fabr_seed0.csv").read_text()
    wa = torch.load(tmp_path / "a" / "fabr_seed0.pt")
    wb = torch.load(tmp_path / "b" / "fabr_seed0.pt")
    assert all(torch.equal(wa[k], wb[k]) for k in wa)


def test_shaping_telescopes_and_vanishes_at_terminal():
    env = GridWorld(layout="rooms")
    d = env.distances_to_goal()
    start = max(d, key=d.get)
    # Terminal potential is zero, so the bonus into a terminal state is just -Phi(s).
    assert shaping_bonus(d, start, start, True, 0.99, 4.0) == pytest.approx(4.0)
    # Moving one step closer gives a positive bonus.
    closer = next(n for n in d if d[n] == d[start] - 1 and abs(n[0] - start[0]) + abs(n[1] - start[1]) == 1)
    assert shaping_bonus(d, start, closer, False, 0.99, 4.0) > 0


def test_no_collision_feedback_stores_only_executed_transitions(tmp_path, monkeypatch):
    count = {"n": 0}
    original = DQNAgent.remember

    def spy(self, *args, **kwargs):
        count["n"] += 1
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DQNAgent, "remember", spy)
    summary = tiny("highest_q", tmp_path, collision_feedback=False)
    assert summary["config"]["collision_feedback"] is False
    assert summary["training"]["intervention_rate"] > 0 and count["n"] == 150
