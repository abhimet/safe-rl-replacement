import json
import random

import pytest
import torch

from dqn import QNetwork
from environment import HELDOUT_LAYOUTS, TRAIN_LAYOUTS, GridWorld
from evaluate import evaluate, load_scenarios
from make_eval_scenarios import DEFAULT_PATH, build, render


@pytest.fixture(scope="module")
def spec():
    return json.loads(DEFAULT_PATH.read_text())


def test_committed_scenarios_match_fresh_generation():
    assert DEFAULT_PATH.read_text() == render(build())


def test_scenarios_are_unique_reachable_and_split_correctly(spec):
    ids = [s["id"] for s in spec["scenarios"]]
    assert len(ids) == len(set(ids))
    assert set(spec["train_layouts"]).isdisjoint(spec["heldout_layouts"])
    for s in spec["scenarios"]:
        env = GridWorld(layout=s["layout"], max_steps=spec["max_steps"], observation_mode=spec["observation_mode"])
        start = tuple(s["start"])
        assert env.reset(options={"start": start})[1]["start"] == start
        assert env.distances_to_goal()[start] == s["shortest_path"] < spec["max_steps"]
        assert s["marked_start"] == (start in env.starts)
        assert s["split"] == ("train" if s["layout"] in TRAIN_LAYOUTS else "heldout")


def test_every_reachable_cell_and_marked_start_is_covered(spec):
    for name in (*TRAIN_LAYOUTS, *HELDOUT_LAYOUTS):
        env = GridWorld(layout=name)
        cells = {tuple(s["start"]) for s in spec["scenarios"] if s["layout"] == name}
        assert cells == set(env.distances_to_goal()) - {env.goal}
        assert set(env.starts) <= cells


def test_load_scenarios_filters_split():
    _, train = load_scenarios(DEFAULT_PATH, "train")
    _, heldout = load_scenarios(DEFAULT_PATH, "heldout")
    assert {s["split"] for s in train} == {"train"} and {s["split"] for s in heldout} == {"heldout"}
    with pytest.raises(ValueError):
        load_scenarios(DEFAULT_PATH, "test")


def test_scenario_evaluation_rows_are_consistent(tmp_path):
    torch.manual_seed(0)
    model = tmp_path / "random.pt"
    torch.save(QNetwork(input_dim=104).state_dict(), model)
    out = tmp_path / "eval.csv"
    summary = evaluate(str(model), output=str(out), scenarios=str(DEFAULT_PATH), split="train")
    import csv
    rows = list(csv.DictReader(out.open()))
    _, expected = load_scenarios(DEFAULT_PATH, "train")
    assert [r["scenario_id"] for r in rows] == [s["id"] for s in expected]
    for r in rows:
        outcome = int(r["success"]) + int(r["collision"]) + int(r["timeout"])
        assert outcome == 1  # exactly one way to end an episode
        assert int(r["interventions"]) == 0  # shield is off
        assert int(r["unsafe_proposals"]) >= int(r["collision"])
        assert int(r["steps"]) <= 100
    assert summary["overall"]["episodes"] == len(expected)
    assert summary["overall"]["success_rate"] == pytest.approx(
        sum(int(r["success"]) for r in rows) / len(rows))


def test_scenario_evaluation_rejects_position_mode_model(tmp_path):
    model = tmp_path / "pos.pt"
    torch.save(QNetwork(input_dim=2).state_dict(), model)
    with pytest.raises(ValueError):
        evaluate(str(model), output=str(tmp_path / "x.csv"), scenarios=str(DEFAULT_PATH))
