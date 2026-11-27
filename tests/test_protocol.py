"""The frozen protocol, the protocol runner, determinism and reproduce.py."""
import csv
import inspect
import json
import shutil
import subprocess
import sys

import pytest

from make_eval_scenarios import ROOT, V2_PATH, sha256
from replacement import FEEDBACK_MODES, METHODS
from run_experiments import load_protocol, parse_config, run_matches, run_protocol


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
