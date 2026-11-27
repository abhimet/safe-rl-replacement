"""Tuning (tune.py): dev maps only, fixed selection rules, round 2, protocol proposals."""
import json

import pytest
import torch

from dqn import DQNAgent
from evaluate import load_scenarios
from make_eval_scenarios import ROOT, V2_PATH
from run_experiments import load_protocol


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
    selected = json.loads((ROOT / "docs" / "tuning" / "round1" / "selected.json").read_text())
    with pytest.raises(RuntimeError):
        tune.write_protocol(selected, frozen)


