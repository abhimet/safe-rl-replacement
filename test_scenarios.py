import json
from pathlib import Path
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


# ---- Version 2: official evaluation set (v1 above stays frozen) ----
import csv
import hashlib
import shutil
from collections import Counter

from make_eval_scenarios import V2_PATH, holdout_suite
from run_comparison import evaluate_checkpoint



@pytest.fixture(scope="module")
def spec_v2():
    return json.loads(V2_PATH.read_text())


def test_v1_is_unchanged_and_pinned_by_v2(spec_v2):
    assert spec_v2["parent"]["sha256"] == hashlib.sha256(DEFAULT_PATH.read_bytes()).hexdigest()
    assert spec_v2["parent"]["version"] == 1 and spec_v2["parent"]["frozen"] == "2026-11-22"


def test_committed_v2_matches_fresh_generation():
    assert V2_PATH.read_text() == render(build(2))


def test_v2_splits_and_counts(spec_v2):
    assert spec_v2["train_layouts"] == ["rooms", "zigzag"]
    assert spec_v2["dev_inspected_layouts"] == ["clutter"]
    assert spec_v2["heldout_layouts"] == ["pillars", "spiral", "heldout_1", "heldout_2", "heldout_3"]
    groups = [spec_v2["train_layouts"], spec_v2["dev_inspected_layouts"], spec_v2["heldout_layouts"]]
    assert sum(map(len, groups)) == len(set().union(*groups)) == len(spec_v2["layouts"])
    counts = Counter(s["layout"] for s in spec_v2["scenarios"])
    assert counts == {"rooms": 84, "zigzag": 71, "clutter": 71, "pillars": 69, "spiral": 59,
                      "heldout_1": 89, "heldout_2": 77, "heldout_3": 61}
    assert Counter(s["split"] for s in spec_v2["scenarios"]) == {"train": 155, "dev_inspected": 71, "heldout": 355}
    for s in spec_v2["scenarios"]:
        info = spec_v2["layouts"][s["layout"]]
        assert (s["split"], s["family"]) == (info["split"], info["family"])
    assert len({s["id"] for s in spec_v2["scenarios"]}) == 581


def test_v2_matches_v1_on_shared_layouts(spec, spec_v2):
    key = lambda s: (s["id"], tuple(s["start"]), s["shortest_path"], s["marked_start"])
    for name in ("rooms", "zigzag", "clutter", "pillars", "spiral"):
        assert [key(s) for s in spec["scenarios"] if s["layout"] == name] == \
               [key(s) for s in spec_v2["scenarios"] if s["layout"] == name]


def test_v2_holdout_suite_matches_manifest_and_covers_every_reachable_cell(spec_v2):
    manifest = {m["layout"]: m for m in json.loads(Path("configs/holdout_suite/manifest.json").read_text())}
    for name, relative, config in holdout_suite():
        assert spec_v2["layouts"][name]["sha256"] == manifest[name]["sha256"]
        env = GridWorld(**config)
        cells = {tuple(s["start"]) for s in spec_v2["scenarios"] if s["layout"] == name}
        assert cells == set(env.distances_to_goal()) - {env.goal}
        assert len(cells) == manifest[name]["reachable_starts"]


def test_tampered_holdout_config_is_rejected(tmp_path, spec_v2):
    shutil.copytree("configs", tmp_path / "configs")
    shutil.copy(V2_PATH, tmp_path / V2_PATH.name)
    target = tmp_path / "configs/holdout_suite/heldout_1.json"
    target.write_text(target.read_text().replace('"goal"', '"goal" '))
    torch.save(QNetwork(input_dim=104).state_dict(), tmp_path / "m.pt")
    with pytest.raises(ValueError, match="sha256"):
        evaluate(str(tmp_path / "m.pt"), output=str(tmp_path / "x.csv"),
                 scenarios=str(tmp_path / V2_PATH.name), family="random")


def test_v2_filters():
    _, dev = load_scenarios(V2_PATH, "dev_inspected")
    _, rnd = load_scenarios(V2_PATH, "heldout", "random")
    assert {s["layout"] for s in dev} == {"clutter"}
    assert {s["layout"] for s in rnd} == {"heldout_1", "heldout_2", "heldout_3"}
    with pytest.raises(ValueError):
        load_scenarios(V2_PATH, family="mixed")


def test_v2_random_family_equals_legacy_holdout_evaluation(tmp_path):
    """v2 must reproduce evaluate_holdout_suite.py's env_config/all_starts path exactly."""
    torch.manual_seed(3)
    model = tmp_path / "random.pt"
    torch.save(QNetwork(input_dim=104).state_dict(), model)
    summary = evaluate(str(model), output=str(tmp_path / "v2.csv"), scenarios=str(V2_PATH), family="random")
    for i in (1, 2, 3):
        legacy = evaluate(str(model), env_config=f"configs/holdout_suite/heldout_{i}.json", size=10,
                          observation_mode="map", all_starts=True, output=str(tmp_path / f"l{i}.csv"))
        assert summary[f"layout=heldout_{i}"] == legacy


def test_v2_evaluation_rows_and_runner_rows(tmp_path):
    torch.manual_seed(1)
    model = tmp_path / "random.pt"
    torch.save(QNetwork(input_dim=104).state_dict(), model)
    rows = evaluate_checkpoint(model, V2_PATH, tmp_path / "all.csv", method="m", seed=0)
    episodes = list(csv.DictReader((tmp_path / "all.csv").open()))
    assert len(episodes) == 581
    for r in episodes:
        assert int(r["success"]) + int(r["collision"]) + int(r["timeout"]) == 1
        assert int(r["interventions"]) == 0
    assert [r["layout"] for r in rows] == ["rooms", "zigzag", "clutter", "pillars", "spiral",
                                           "heldout_1", "heldout_2", "heldout_3"]
    assert {r["layout"]: r["split"] for r in rows}["clutter"] == "dev_inspected"
    assert sum(r["episodes"] for r in rows) == 581
