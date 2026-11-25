"""End-to-end: run_experiments -> plots -> demo, on a tiny budget."""
import csv
import json

import pytest

import plots
from demo import make_demo
from evaluate import load_network
from run_experiments import GROUPS, run_experiments


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("exp")
    run_experiments(output=root, steps=200, seeds=(1, 2), methods=("fabr_fixed", "fabr", "highest_q"),
                    pool_size=3)
    return root


def test_outputs_and_fixed_beta_is_fabr_mean(experiment):
    agg = list(csv.DictReader((experiment / "aggregate.csv").open()))
    assert {r["method"] for r in agg} == {"fabr", "fabr_fixed", "highest_q"}
    assert set(GROUPS) <= {r["group"] for r in agg}
    fabr_betas = [json.loads((experiment / "runs" / f"fabr_seed{s}" / f"fabr_seed{s}_summary.json")
                             .read_text())["mean_beta"] for s in (1, 2)]
    fixed = json.loads((experiment / "fabr_fixed_beta.json").read_text())["fixed_beta"]
    assert fixed == pytest.approx(sum(fabr_betas) / 2, abs=1e-4)
    paired = list(csv.DictReader((experiment / "paired.csv").open()))
    assert {r["comparison"] for r in paired} == {"fabr - fabr_fixed", "fabr - highest_q"}


def test_rerun_reuses_finished_runs(experiment, capsys):
    run_experiments(output=experiment, steps=200, seeds=(1, 2), methods=("fabr", "highest_q", "fabr_fixed"),
                    pool_size=3)
    assert capsys.readouterr().out.count("reusing finished run") == 6
    with pytest.raises(FileExistsError):
        run_experiments(output=experiment, steps=300, seeds=(1,), methods=("fabr",), pool_size=3)


def test_plots_and_demo(experiment):
    plots.main(experiment)
    figures = experiment / "figures"
    for name in ("success_by_group", "outcomes", "training_curves", "fabr_beta"):
        assert (figures / f"{name}.png").stat().st_size > 1000
    agents = [("fabr", 1, load_network(experiment / "runs" / "fabr_seed1" / "fabr_seed1.pt"))]
    runs = make_demo(agents, "heldout_1", (9, 8), output=figures / "demo", fps=20)
    assert (figures / "demo.gif").exists() and (figures / "demo.png").exists()
    shield_on, shield_off = runs[0][3], runs[1][3]
    assert shield_on["collision"] == 0
    assert shield_off["interventions"] == 0


def test_parallel_jobs_match_serial(tmp_path):
    kwargs = dict(steps=120, seeds=(5,), methods=("fabr", "fabr_fixed"), pool_size=2,
                  collision_feedback=False)
    serial = run_experiments(output=tmp_path / "serial", **kwargs)
    parallel = run_experiments(output=tmp_path / "parallel", jobs=2, **kwargs)
    assert serial == parallel
    config = json.loads((tmp_path / "parallel" / "experiment_config.json").read_text())
    assert config["collision_feedback"] is False
