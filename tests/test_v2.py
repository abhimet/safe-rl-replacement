"""Feedback modes, n-step returns, FABR v2's margin loss, the frozen protocol and tuning guards."""
import csv
import inspect
import json
import random
import shutil
import subprocess
import sys
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
from evaluate import load_scenarios, rollout
from make_eval_scenarios import ROOT, V2_PATH, sha256
from metrics import summarize
from replacement import FEEDBACK_MODES, MARGIN_METHODS, METHODS, Replacer
from run_experiments import (LEGACY_METHODS, load_protocol, parse_config, run_matches,
                             run_protocol)
from train import NStep, resolve_feedback_mode, train

LEGACY_SHIELDED = tuple(m for m in LEGACY_METHODS if m != "unshielded")
VALID = ([(m, mode) for m in LEGACY_SHIELDED for mode in ("none", "cf", "credit")]
         + [(m, "margin") for m in MARGIN_METHODS])


def tiny(method, out, **kwargs):
    kwargs.setdefault("fixed_beta", 2.0 if method == "fabr_fixed" else None)
    return train(method=method, steps=150, seed=0, output=out, pool_size=3, verbose=False, **kwargs)


@pytest.fixture
def step_guard(monkeypatch):
    original = environment.GridWorld.step
    calls = {"unsafe": 0, "steps": 0}

    def guarded(self, action):
        calls["steps"] += 1
        calls["unsafe"] += int(int(action) not in self.safe_actions())
        return original(self, action)

    monkeypatch.setattr(environment.GridWorld, "step", guarded)
    return calls


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


# ---- tuning only sees development maps ----

def test_tuning_refuses_held_out_maps(tmp_path):
    import tune
    agent = DQNAgent(input_dim=104)
    torch.save(agent.online.state_dict(), tmp_path / "m.pt")
    for bad in [("heldout",), ("train", "heldout"), ()]:
        with pytest.raises(ValueError):
            tune.dev_evaluate(tmp_path / "m.pt", splits=bad)
    _, chosen = load_scenarios(V2_PATH, splits=tune.DEV_SPLITS)
    assert {s["split"] for s in chosen} == {"train", "dev_inspected"}
    assert {s["layout"] for s in chosen} == {"rooms", "zigzag", "clutter"}
    assert tune.TUNING_SEEDS and not set(tune.TUNING_SEEDS) & set(range(42, 52))


def test_tuning_selection_rules(tmp_path, monkeypatch):
    """The fixed rules applied to fake results (no training)."""
    import tune

    def fake(directory, method, seed, kw):
        familiar = 0.30 + 0.10 * (kw["n_step"] == 3) + 0.01 * (kw["lr"] == 3e-4)  # lr gain < 2 pp
        familiar += {50000: 0.0, 100000: 0.05, 150000: 0.12}[kw["steps"]]
        score = 0.1
        if method == "value_penalty":
            score = {2.0: 0.4, 5.0: 0.4}.get(kw["penalty"], 0.1)  # tie: the earlier one wins
        if method == "fabr_v2":
            score = 0.2 + 0.1 * (kw["margin_lambda"] == 0.5) + 0.1 * (kw["margin"] == 2.0) \
                - 0.05 * kw["margin_stop_grad"]
        split = {"success_rate": familiar, "collision_rate": 0.0}
        return {"split=train": split, "split=dev_inspected": split,
                "overall": {"success_rate": score, "collision_rate": 0.0},
                "training": {"success_rate": 0.5, "final_success_rate": 0.6}}

    monkeypatch.setattr(tune, "run_trial", fake)
    tuner = tune.Tuner(jobs=1, root=tmp_path / "tuning", report=tmp_path / "TUNING.md")
    for stage in tune.STAGES:
        getattr(tuner, stage)()
    sel = tuner.selected
    assert sel["train"]["n_step"] == 3 and sel["train"]["lr"] == 1e-4
    assert sel["steps"] == 150000  # 100k < 50% familiar and 150k adds >= 5 pp
    assert sel["value_penalty"] == {"penalty": 2.0}
    assert sel["fabr_v2"] == {"margin": 2.0, "margin_lambda": 0.5, "margin_stop_grad": False}
    trials = [r for r in tuner.log if r["stage"] == "fabr_v2"]
    assert len(trials) == 6 and len([r for r in tuner.log if r["stage"] == "value_penalty"]) == 6
    assert sum(r.get("chosen", False) for r in trials) == 1
    assert "lambda=0.5,m=2" in (tmp_path / "TUNING.md").read_text()


@pytest.mark.parametrize("gain,kept", [(0.03, True), (0.01, False)])
def test_round2_keeps_conv_only_for_a_2pp_dev_gain(tmp_path, monkeypatch, gain, kept):
    import tune
    round1 = tmp_path / "round1"
    round1.mkdir()
    first = {"train": {"lr": 1e-4, "n_step": 3, "capacity": 10000, "epsilon_fraction": 0.8, "pool_size": 100,
                       "pool_seed": 7, "shaping_scale": 4.0, "window": 100}, "steps": 100000,
             "value_penalty": {"penalty": 10.0}, "fabr_v2": {"margin": 1.0, "margin_lambda": 0.1,
                                                             "margin_stop_grad": False}}
    (round1 / "selected.json").write_text(json.dumps(first))
    (round1 / "log.json").write_text(json.dumps([{"stage": "nav", "trial": "baseline", "familiar_success": 0,
                                                  "clutter_success": 0, "dev_success": 0, "dev_collision": 0,
                                                  "dev_score": 0}]))
    seen = []

    def fake(directory, method, seed, kw):
        seen.append((method, kw))
        dev = 0.20 + gain * (kw.get("architecture") == "conv") + 0.01 * (kw.get("shaping") == "distance")
        split = {"success_rate": dev, "collision_rate": 0.0}
        return {"split=train": split, "split=dev_inspected": split,
                "overall": {"success_rate": dev, "collision_rate": 0.0},
                "training": {"success_rate": 0.5, "final_success_rate": 0.7}}

    monkeypatch.setattr(tune, "run_trial", fake)
    report = tmp_path / "TUNING.md"
    tuner = tune.Tuner(jobs=1, root=tmp_path / "r2", report=report, round=2, round1=round1)
    for stage in tune.ROUND_STAGES[2]:
        getattr(tuner, stage)()
    assert (tuner.selected["train"].get("architecture") == "conv") == kept
    assert "shaping" not in tuner.selected["train"]  # +1 pp is below the 2 pp rule
    phi = [kw for m, kw in seen if m == "highest_q" and kw.get("shaping") == "distance"]
    assert phi and all((kw.get("architecture") == "conv") == kept for kw in phi)  # tested on the winner
    assert tuner.selected["steps"] == 100000 and tuner.selected["train"]["n_step"] == 3
    # the method grids run at the chosen navigation settings, with the same sizes as round 1
    for method in ("value_penalty", "fabr_v2"):
        runs = [kw for m, kw in seen if m == method]
        assert len(runs) == 12 and all((kw.get("architecture") == "conv") == kept for kw in runs)
    assert all(kw["feedback_mode"] == "cf" for m, kw in seen if m == "highest_q")
    text = report.read_text()
    assert "# Round 1" in text and "# Round 2" in text and "0.500 / 0.700" in text
    proposed = tmp_path / "proposed.json"
    tune.write_protocol(tuner.selected, proposed, source=ROOT / "configs" / "final_protocol.json", version=2)
    new = json.loads(proposed.read_text())
    assert new["frozen"] is False and new["version"] == 2
    assert (new["train"].get("architecture") == "conv") == kept
    load_protocol(proposed)


def test_round2_refuses_to_overwrite_a_frozen_protocol(tmp_path):
    import tune
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps({**json.loads((ROOT / "configs" / "final_protocol.json").read_text()),
                                  "frozen": True}))
    selected = json.loads((ROOT / "docs" / "tuning" / "selected.json").read_text())
    with pytest.raises(RuntimeError):
        tune.write_protocol(selected, frozen)


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


# ---- frozen protocol ----

def test_repository_protocol_is_valid():
    protocol, _ = load_protocol(ROOT / "configs" / "final_protocol.json")
    tiers = protocol["tiers"]
    assert len(tiers["core"]["configs"]) == 11 and len(tiers["full"]["configs"]) == 21
    assert set(tiers["quick"]["configs"]) <= set(tiers["core"]["configs"]) <= set(tiers["full"]["configs"])
    assert tiers["core"]["seeds"] == list(range(42, 52))
    configs = [parse_config(c) for c in tiers["full"]["configs"]]
    assert {m for m, _ in configs} == set(METHODS)
    assert {mode for _, mode in configs} == set(FEEDBACK_MODES)
    assert "fabr_v2_fixedtarget" not in protocol["methods"]  # always shares fabr_v2's settings


def test_run_protocol_takes_no_training_settings():
    assert list(inspect.signature(run_protocol).parameters) == ["protocol_path", "tier", "output",
                                                                 "seeds", "jobs"]


def test_cli_rejects_training_flags_with_a_protocol():
    result = subprocess.run([sys.executable, "run_experiments.py", "--protocol",
                             "configs/final_protocol.json", "--steps", "10"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0 and "only source of training settings" in result.stderr


@pytest.fixture(scope="module")
def small_protocol(tmp_path_factory):
    """A protocol over a reduced scenario file (2 starts per layout) to keep the test fast."""
    root = tmp_path_factory.mktemp("protocol")
    shutil.copytree(ROOT / "configs" / "holdout_suite", root / "configs" / "holdout_suite")
    spec = json.loads(V2_PATH.read_text())
    kept, per_layout = [], {}
    for s in spec["scenarios"]:
        if per_layout.get(s["layout"], 0) < 2:
            per_layout[s["layout"]] = per_layout.get(s["layout"], 0) + 1
            kept.append(s)
    spec["scenarios"] = kept
    scenarios = root / "scenarios.json"
    scenarios.write_text(json.dumps(spec))
    protocol = json.loads((ROOT / "configs" / "final_protocol.json").read_text())
    protocol["scenarios"] = {"file": str(scenarios), "sha256": sha256(scenarios)}
    protocol["train"].update(steps=120, pool_size=2, n_step=2)
    protocol["methods"]["fabr_v2"]["margin_lambda"] = 0.3
    protocol["tiers"] = {"t": {"seeds": [1, 2], "configs": ["unshielded", "fabr@credit",
                                                            "fabr_fixed@credit", "fabr_v2@margin"]}}
    path = root / "protocol.json"
    path.write_text(json.dumps(protocol))
    return root, path, protocol


def test_protocol_is_the_only_source_of_run_settings(small_protocol):
    root, path, protocol = small_protocol
    out = root / "out"
    run_protocol(path, "t", out, seeds=[1])
    assert not (out / "credit" / "runs" / "fabr_fixed_seed1").exists()  # waits for every fabr seed
    run_protocol(path, "t", out, seeds=[2])
    beta = json.loads((out / "credit" / "fabr_fixed_beta.json").read_text())["fixed_beta"]
    _, digest = load_protocol(path)
    for mode, method in [("none", "unshielded"), ("credit", "fabr"), ("credit", "fabr_fixed"),
                         ("margin", "fabr_v2")]:
        for seed in (1, 2):
            summary = json.loads((out / mode / "runs" / f"{method}_seed{seed}" /
                                  f"{method}_seed{seed}_summary.json").read_text())
            assert summary["protocol_sha256"] == digest and summary["steps"] == 120
            config = summary["config"]
            assert config["feedback_mode"] == mode and config["n_step"] == 2 and config["pool_size"] == 2
            if method == "fabr_v2":
                assert config["margin_lambda"] == 0.3
            if method == "fabr_fixed":
                assert config["fixed_beta"] == beta
    agg = list(csv.DictReader((out / "aggregate.csv").open()))
    assert {r["config"] for r in agg} == {"unshielded", "fabr@credit", "fabr_fixed@credit", "fabr_v2@margin"}
    assert {r["seeds"] for r in agg} == {"2"}
    comparisons = {r["comparison"] for r in csv.DictReader((out / "paired.csv").open())}
    assert "fabr_v2@margin - unshielded" in comparisons and "fabr@credit - fabr_fixed@credit" in comparisons
    metrics = {r["metric"] for r in csv.DictReader((out / "paired.csv").open())}
    assert {"greedy_unsafe_state_rate", "average_reward", "timeout_rate"} <= metrics


def test_determinism_check_retrains_identically(small_protocol, tmp_path, monkeypatch):
    import importlib.util
    root, path, protocol = small_protocol
    out = root / "det"
    run_protocol(path, "t", out, seeds=[1])
    spec = importlib.util.spec_from_file_location("check_determinism", ROOT / "cluster" / "check_determinism.py")
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    assert check.main(["--protocol", str(path), "--results", str(out), "--tier", "t",
                       "--runs", "fabr_v2@margin:1", "unshielded:1"]) == 0


def test_changed_protocol_refuses_to_reuse_runs(small_protocol):
    root, path, protocol = small_protocol
    changed = root / "changed.json"
    changed.write_text(json.dumps({**protocol, "note": "edited"}))
    with pytest.raises(FileExistsError):
        run_protocol(changed, "t", root / "out", seeds=[1])
    bad = root / "bad_scenarios.json"
    bad.write_text(json.dumps({**protocol, "scenarios": {**protocol["scenarios"], "sha256": "0" * 64}}))
    with pytest.raises(ValueError):
        load_protocol(bad)
    unknown = root / "unknown.json"
    unknown.write_text(json.dumps({**protocol, "train": {**protocol["train"], "learning_rate": 1}}))
    with pytest.raises(ValueError):
        load_protocol(unknown)


def test_legacy_summaries_still_match():
    legacy = {"steps": 50000, "config": {"pool_size": 100, "fixed_beta": None, "collision_feedback": False,
                                         "lr": 1e-4, "penalty": 1.0}}
    kwargs = {"steps": 50000, "pool_size": 100, "fixed_beta": None, "feedback_mode": "none"}
    assert run_matches(legacy, "highest_q", kwargs)
    assert not run_matches(legacy, "highest_q", {**kwargs, "feedback_mode": "cf"})
    assert not run_matches(legacy, "highest_q", {**kwargs, "n_step": 3})


# ---- reproduce.py ----

def test_verify_is_exact_except_last_bits_of_float_averages():
    import reproduce
    stored = {"model": "a", "scenario_version": 2, "overall": {"success_rate": 0.25, "average_reward": -9.981755593803792,
                                        "average_steps": 40.0}}
    same = {"model": "b", "scenario_version": 2,
            "overall": {**stored["overall"], "average_reward": -9.981755593803786}}
    assert reproduce.summary_mismatches(stored, same) == []          # ~6e-15: summation order only
    worse = {"scenario_version": 2, "overall": {**same["overall"], "success_rate": 0.2525}}
    assert reproduce.summary_mismatches(stored, worse) == ["overall:success_rate"]
    far = {"scenario_version": 3, "overall": {**stored["overall"], "average_reward": -9.98}}
    assert reproduce.summary_mismatches(stored, far) == ["scenario_version", "overall:average_reward"]


def test_reproduce_refuses_a_draft_protocol_and_unknown_commands():
    import reproduce
    protocol = json.loads((ROOT / "configs" / "final_protocol.json").read_text())
    if not protocol["frozen"]:
        with pytest.raises(SystemExit):
            reproduce.main(["quick"])
    with pytest.raises(SystemExit):
        reproduce.main(["nonsense"])
