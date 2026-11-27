"""Feedback modes, n-step returns, replay buffer, FABR v2's margin loss, networks, metrics."""
import csv
import random
from collections import deque
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import environment
import replacement
from dqn import DQNAgent, ReplayBuffer, margin_loss
from environment import GridWorld
from evaluate import rollout
from metrics import summarize
from replacement import MARGIN_METHODS, Replacer
from run_experiments import LEGACY_METHODS, run_matches
from train import NStep, resolve_feedback_mode
from helpers import tiny

LEGACY_SHIELDED = tuple(m for m in LEGACY_METHODS if m != "unshielded")
VALID = ([(m, mode) for m in LEGACY_SHIELDED for mode in ("none", "cf", "credit")]
         + [(m, "margin") for m in MARGIN_METHODS])



# ---- feedback modes ----

def test_feedback_mode_resolution_and_legacy_aliases():
    assert resolve_feedback_mode("highest_q") == "cf"  # historical default
    assert resolve_feedback_mode("highest_q", collision_feedback=False) == "none"
    assert resolve_feedback_mode("highest_q", "credit") == "credit"
    assert resolve_feedback_mode("fabr_v2") == "margin"
    assert resolve_feedback_mode("fabr_v2", collision_feedback=True) == "margin"  # legacy flag ignored
    assert resolve_feedback_mode("unshielded", "cf") == "none"
    for bad in [("highest_q", "margin", None), ("fabr_v2", "cf", None), ("highest_q", "none", True),
                ("unshielded", "margin", None), ("highest_q", "oracle", None)]:
        with pytest.raises(ValueError):
            resolve_feedback_mode(*bad)


@pytest.mark.parametrize("mode,flag", [("none", False), ("cf", True)])
def test_legacy_alias_gives_identical_training(tmp_path, mode, flag):
    tiny("highest_q", tmp_path / "flag", collision_feedback=flag)
    tiny("highest_q", tmp_path / "mode", feedback_mode=mode)
    a, b = tmp_path / "flag" / "highest_q_seed0", tmp_path / "mode" / "highest_q_seed0"
    assert Path(f"{a}.csv").read_text() == Path(f"{b}.csv").read_text()
    wa, wb = torch.load(f"{a}.pt"), torch.load(f"{b}.pt")
    assert all(torch.equal(wa[k], wb[k]) for k in wa)


def test_credit_stores_proposal_with_the_replacements_outcome(tmp_path, monkeypatch):
    stored, steps, choices = [], [], []
    remember, step, choose = DQNAgent.remember, environment.GridWorld.step, Replacer.choose

    def spy_remember(self, state, action, reward, next_state, done, *args, **kwargs):
        stored.append((np.array(state), action, reward, np.array(next_state), done))
        return remember(self, state, action, reward, next_state, done, *args, **kwargs)

    def spy_step(self, action):
        out = step(self, action)
        steps.append((np.array(out[0]), out[1]))
        return out

    def spy_choose(self, proposed, safe, q):
        executed, intervened = choose(self, proposed, safe, q)
        choices.append((proposed, executed, intervened, list(safe)))
        return executed, intervened

    monkeypatch.setattr(DQNAgent, "remember", spy_remember)
    monkeypatch.setattr(environment.GridWorld, "step", spy_step)
    monkeypatch.setattr(Replacer, "choose", spy_choose)
    tiny("highest_q", tmp_path, feedback_mode="credit")
    assert len(stored) == len(steps) == len(choices) == 150  # no executed (or cf) transition added
    interventions = 0
    for (state, action, reward, next_state, _), (obs, env_reward), (proposed, executed, hit, safe) in zip(
            stored, steps, choices):
        assert action == proposed                      # credited to the blocked proposal ...
        assert np.array_equal(next_state, obs)         # ... with the replacement's next state ...
        assert abs(reward - env_reward) <= 4.0 + 1e-6  # ... and its reward (+ shaping in [-4, 4])
        if hit:
            interventions += 1
            assert proposed not in safe and executed in safe and env_reward > -10
    assert interventions > 0


def test_margin_mode_marks_interventions_and_stores_no_counterfactual(tmp_path, monkeypatch):
    calls = []
    remember = DQNAgent.remember

    def spy(self, *args, **kwargs):
        calls.append(kwargs)
        return remember(self, *args, **kwargs)

    monkeypatch.setattr(DQNAgent, "remember", spy)
    summary = tiny("fabr_v2", tmp_path)
    assert summary["config"]["feedback_mode"] == "margin" and len(calls) == 150
    marked = [c for c in calls if c["intervened"]]
    rows = list(csv.DictReader((tmp_path / "fabr_v2_seed0.csv").open()))
    assert len(marked) == sum(int(r["interventions"]) for r in rows) > 0
    assert all(c["proposed"] not in c["safe_actions"] for c in marked)


@pytest.mark.parametrize("method,mode", VALID)
def test_shielded_training_never_executes_unsafe_actions_in_any_mode(method, mode, tmp_path, step_guard):
    summary = tiny(method, tmp_path, feedback_mode=mode, n_step=3 if mode == "credit" else 1)
    assert step_guard["steps"] == 150 and step_guard["unsafe"] == 0
    assert summary["training"]["collision_rate"] == 0.0
    assert summary["training"]["intervention_rate"] == summary["training"]["unsafe_proposal_rate"]


@pytest.mark.parametrize("method", MARGIN_METHODS)
def test_fabr_v2_executes_the_highest_q_safe_action(method):
    rep = Replacer(method, random.Random(0))
    assert rep.choose(0, [1, 2, 3], [100, 2, 3, 0]) == (2, True)
    assert method in replacement.HIGHEST_Q_METHODS


# ---- n-step returns ----

def test_nstep_returns_discounts_and_flushes():
    n = NStep(3, 0.5)
    assert n.add("s0", 0, 1.0, "s1", False, [0], None, False) == []
    assert n.add("s1", 1, 2.0, "s2", False, [0], None, False) == []
    out = n.add("s2", 2, 4.0, "s3", False, [1], "x", False)
    assert out == [("s0", 0, 1 + 0.5 * 2 + 0.25 * 4, "s3", False, [1], 0.125, None)]
    out = n.add("s3", 3, 8.0, "s4", True, None, None, True)  # terminal: flush everything
    assert [(o[0], o[2], o[3], o[4], o[6]) for o in out] == [
        ("s1", 2 + 0.5 * 4 + 0.25 * 8, "s4", True, 0.125), ("s2", 4 + 0.5 * 8, "s4", True, 0.25),
        ("s3", 8.0, "s4", True, 0.5)]
    assert not n.pending
    m = NStep(3, 0.5)  # time limit: flushed, but still bootstraps (done=False)
    m.add("a", 0, 1.0, "b", False, [0], None, False)
    assert [(o[2], o[4], o[6]) for o in m.add("b", 0, 1.0, "c", False, [2], None, True)] == [
        (1.5, False, 0.25), (1.0, False, 0.5)]
    one = NStep(1, 0.99)
    assert one.add("s", 1, -0.1, "t", False, [0], None, False) == [("s", 1, -0.1, "t", False, [0], 0.99, None)]


def test_nstep_training_stores_every_transition_once(tmp_path, monkeypatch):
    count = {"n": 0}
    remember = DQNAgent.remember

    def spy(self, *args, **kwargs):
        count["n"] += 1
        assert kwargs["discount"] in (0.99, 0.99 ** 2, 0.99 ** 3)
        return remember(self, *args, **kwargs)

    monkeypatch.setattr(DQNAgent, "remember", spy)
    summary = tiny("highest_q", tmp_path, feedback_mode="none", n_step=3)
    assert count["n"] == 150 and summary["config"]["n_step"] == 3


# ---- replay buffer ----

def test_ring_buffer_samples_exactly_like_a_deque():
    old, new = deque(maxlen=5), ReplayBuffer(5, 2)
    for i in range(8):
        old.append(i)
        new.append(np.full(2, i), i, 0.0, np.zeros(2), False, np.ones(4, bool), 0.99, i, False,
                   np.ones(4, bool))
    for seed in range(5):
        expected = random.Random(seed).sample(old, 3)
        got = new.sample(random.Random(seed), 3)["actions"].tolist()
        assert got == expected


# ---- margin loss ----

def q_batch(q_prop, best=2.0):
    q = torch.tensor([[q_prop, best, 1.0, -5.0]], requires_grad=True)
    return q, torch.tensor([0]), torch.tensor([[False, True, True, False]]), torch.tensor([True])


def test_hinge_margin_loss_is_zero_iff_proposal_is_margin_below_best_safe():
    for q_prop, expected in [(0.0, 0.0), (1.0, 0.0), (1.5, 0.5), (3.0, 2.0)]:
        q, a, safe, hit = q_batch(q_prop)
        assert margin_loss(q, a, safe, hit, 1.0, "hinge").item() == pytest.approx(expected)


def test_fixed_target_loss_is_two_sided_and_pulls_up_from_below():
    for q_prop, expected in [(1.0, 0.0), (0.0, 1.0), (2.0, 1.0)]:
        q, a, safe, hit = q_batch(q_prop)
        assert margin_loss(q, a, safe, hit, 1.0, "fixed_target").item() == pytest.approx(expected)
    q, a, safe, hit = q_batch(0.0)  # below the target: descent pushes Q(s, a_prop) up
    margin_loss(q, a, safe, hit, 1.0, "fixed_target").backward()
    assert q.grad[0, 0] < 0


def test_stop_grad_blocks_the_gradient_on_the_best_safe_action():
    for stop, nonzero in [(False, True), (True, False)]:
        q, a, safe, hit = q_batch(1.5)
        margin_loss(q, a, safe, hit, 1.0, "hinge", stop_grad=stop).backward()
        assert (q.grad[0, 1] != 0) == nonzero and q.grad[0, 0] > 0


def test_margin_loss_only_averages_intervened_samples():
    q = torch.tensor([[1.5, 2.0, 0, 0], [100.0, 0, 0, 0]])
    a = torch.tensor([0, 0])
    safe = torch.tensor([[False, True, True, True], [False, True, True, True]])
    assert margin_loss(q, a, safe, torch.tensor([True, False]), 1.0).item() == pytest.approx(0.5)
    assert margin_loss(q, a, safe, torch.tensor([False, False]), 1.0).item() == 0.0
    with pytest.raises(ValueError):
        margin_loss(q, a, safe, torch.tensor([True, True]), 1.0, "nohinge")


def test_agent_adds_the_margin_loss_to_td():
    agent = DQNAgent(input_dim=104, seed=0, batch_size=4,
                     margin_config=dict(kind="hinge", margin=1.0, weight=0.5, stop_grad=False))
    state = np.zeros(104, dtype=np.float32)
    for _ in range(4):
        agent.remember(state, 1, -0.1, state, True, None, proposed=0, intervened=True, safe_actions=[1, 2])
    agent.learn()
    assert agent.last_margin_loss is not None and np.isfinite(agent.last_loss)
    with pytest.raises(ValueError):
        agent.remember(state, 1, 0.0, state, True, None, intervened=True)


# ---- greedy unsafe-state metric ----

class AlwaysUp(nn.Module):
    input_dim = 104

    def forward(self, x):
        return torch.tensor([1.0, 0.0, 0.0, 0.0])


def test_start_unsafe_measures_the_greedy_action_in_every_start_state():
    env = GridWorld(layout="rooms", observation_mode="map")
    starts = sorted(c for c in env.distances_to_goal() if c != env.goal)
    rows = [rollout(AlwaysUp(), env, s)[0] for s in starts]
    expected = [int(r == 0 or (r - 1, c) in env.obstacles) for r, c in starts]
    assert [row["start_unsafe"] for row in rows] == expected
    assert summarize(rows)["greedy_unsafe_state_rate"] == pytest.approx(sum(expected) / len(starts))
    assert "greedy_unsafe_state_rate" not in summarize([{**rows[0], "start_unsafe": ""}])


def test_distance_potential_uses_bfs_distance():
    from train import potential, shaping_bonus
    env = GridWorld(layout="rooms")
    d = env.distances_to_goal()
    far = max(d, key=d.get)
    assert potential(d, far, 4.0, "distance") == -d[far]
    assert potential(d, far, 4.0) == pytest.approx(-4.0)
    closer = next(n for n in d if d[n] == d[far] - 1 and abs(n[0] - far[0]) + abs(n[1] - far[1]) == 1)
    assert shaping_bonus(d, far, closer, False, 0.99, 4.0, "distance") == pytest.approx(
        0.99 * -(d[far] - 1) + d[far])
    with pytest.raises(ValueError):
        potential(d, far, 4.0, "euclidean")


# ---- convolutional Q-network and training goal-reached rate ----

def test_conv_network_shapes_checkpoint_and_rollout(tmp_path):
    from dqn import ConvQNetwork, create_network, network_from_weights
    net = create_network(104, "conv")
    assert isinstance(net, ConvQNetwork) and net.input_dim == 104
    env = GridWorld(layout="rooms", observation_mode="map")
    obs, _ = env.reset(options={"start": (0, 0)})
    x = torch.as_tensor(obs)
    assert net(x).shape == (4,) and net(torch.stack([x, x])).shape == (2, 4)
    planes = net.planes(x.unsqueeze(0))
    assert planes.shape == (1, 3, 12, 12)
    assert planes[0, 0, 0].sum() == 12 and planes[0, 1, 1, 1] == 1 and planes[0, 2, 10, 10] == 1  # border, agent, goal
    rebuilt = network_from_weights(net.state_dict())
    assert isinstance(rebuilt, ConvQNetwork) and torch.allclose(rebuilt(x), net(x))
    row, _ = rollout(rebuilt, env, (0, 0))
    assert row["success"] + row["collision"] + row["timeout"] == 1


def test_conv_training_and_final_goal_rate(tmp_path):
    summary = tiny("fabr_v2", tmp_path, architecture="conv")
    assert summary["config"]["architecture"] == "conv"
    weights = torch.load(tmp_path / "fabr_v2_seed0.pt")
    assert "conv.0.weight" in weights
    training = summary["training"]
    assert "final_success_rate" in training
    assert training["final_success_rate"] is None or 0 <= training["final_success_rate"] <= 1
    assert run_matches(summary, "fabr_v2", {"steps": 150, "pool_size": 3, "architecture": "conv",
                                            "feedback_mode": "margin"})
    assert not run_matches(summary, "fabr_v2", {"steps": 150, "pool_size": 3, "feedback_mode": "margin"})


