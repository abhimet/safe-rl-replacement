"""Figures for a finished run_experiments.py output directory.

    python plots.py --results results/final            # writes results/final/figures/*.png
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from replacement import METHODS  # noqa: E402
from run_experiments import GROUPS  # noqa: E402

# Fixed colour per method (validated categorical palette; colour follows the method).
COLORS = {"fabr": "#257de0", "highest_q": "#f47230", "random_safe": "#00b894",
          "adaptive_mix": "#f2a600", "value_penalty": "#ee7fac", "unshielded": "#008a17",
          "fabr_fixed": "#6a4ce2"}
LABELS = {"fabr": "FABR (ours)", "fabr_fixed": "FABR fixed-β (ablation)", "highest_q": "Highest-Q",
          "random_safe": "Random safe", "adaptive_mix": "Adaptive mix",
          "value_penalty": "Value + penalty", "unshielded": "Unshielded"}
GROUP_TITLES = {"familiar": "Familiar (rooms, zigzag)", "dev_clutter": "Clutter (dev-inspected)",
                "unseen": "Unseen (pillars, heldout 1–3)", "unseen_spiral": "Unseen: spiral"}
OUTCOMES = (("success_rate", "Success", "#257de0"), ("collision_rate", "Collision", "#ec4647"),
            ("timeout_rate", "Timeout", "#c9c9c9"))
TEXT, GRID = "#333333", "#e5e5e5"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
                     "xtick.color": TEXT, "ytick.color": TEXT, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def present(methods_in_data):
    return [m for m in METHODS if m in methods_in_data]


def success_by_group(agg, out):
    """Small multiples: one panel per evaluation group, methods as labelled bars."""
    methods = present({r["method"] for r in agg})
    cells = {(r["method"], r["group"]): r for r in agg}
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(3.2 * len(GROUPS), 0.45 * len(methods) + 1.4),
                             sharey=True)
    y = np.arange(len(methods))[::-1]
    for ax, group in zip(axes, GROUPS):
        means = [float(cells[(m, group)]["success_rate_mean"]) for m in methods]
        cis = [float(cells[(m, group)]["success_rate_ci95"]) for m in methods]
        ax.barh(y, means, height=0.7, color=[COLORS[m] for m in methods], edgecolor="white", linewidth=2)
        ax.errorbar(means, y, xerr=cis, fmt="none", ecolor=TEXT, elinewidth=1, capsize=2)
        for yi, mean, ci in zip(y, means, cis):
            ax.text(min(mean + ci, 1) + 0.02, yi, f"{mean:.0%}", va="center", fontsize=8, color=TEXT)
        ax.set_xlim(0, 1.15)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0%", "50%", "100%"])
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(GROUP_TITLES[group], fontsize=10, color=TEXT)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LABELS[m] for m in methods])
    fig.suptitle("Shield-OFF success rate (mean ± 95% CI over seeds)", color=TEXT, x=0.02, ha="left")
    fig.savefig(out / "success_by_group.png")
    plt.close(fig)


def outcomes(agg, out):
    """Stacked success / collision / timeout per method, one panel per group."""
    methods = present({r["method"] for r in agg})
    cells = {(r["method"], r["group"]): r for r in agg}
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(3.2 * len(GROUPS), 0.45 * len(methods) + 1.6),
                             sharey=True)
    y = np.arange(len(methods))[::-1]
    for ax, group in zip(axes, GROUPS):
        left = np.zeros(len(methods))
        for key, label, color in OUTCOMES:
            values = np.array([float(cells[(m, group)][f"{key}_mean"]) for m in methods])
            ax.barh(y, values, left=left, height=0.7, color=color, edgecolor="white", linewidth=2, label=label)
            left += values
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0%", "50%", "100%"])
        ax.set_title(GROUP_TITLES[group], fontsize=10, color=TEXT)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LABELS[m] for m in methods])
    axes[-1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    fig.suptitle("How shield-OFF episodes end", color=TEXT, x=0.02, ha="left")
    fig.savefig(out / "outcomes.png")
    plt.close(fig)


def binned_training(results, bins=40):
    """{method: {field: (x, mean over seeds)}} for per-step rates over training."""
    curves = defaultdict(lambda: defaultdict(list))
    for path in sorted(Path(results, "runs").glob("*/*_seed*.csv")):
        rows = read_csv(path)
        method = path.stem.rsplit("_seed", 1)[0]
        steps = np.array([int(r["env_steps"]) for r in rows])
        edges = np.linspace(0, steps.max(), bins + 1)
        which = np.clip(np.searchsorted(edges, steps, side="right") - 1, 0, bins - 1)
        length = np.array([int(r["steps"]) for r in rows])
        for field in ("greedy_unsafe", "interventions", "collision"):
            values = np.array([float(r[field]) for r in rows])
            per_bin = [values[which == b].sum() / max(1, length[which == b].sum() if field != "collision"
                                                      else (which == b).sum()) for b in range(bins)]
            curves[method][field].append(per_bin)
        betas = [(int(r["env_steps"]), float(r["beta"])) for r in rows if r["beta"]]
        if betas:
            curves[method]["beta_raw"].append(betas)
        curves[method]["x"] = (edges[:-1] + edges[1:]) / 2
    return curves


def training_curves(results, out):
    curves = binned_training(results)
    panels = (("greedy_unsafe", "Greedy policy proposes an unsafe move\n(fraction of steps)"),
              ("interventions", "Shield interventions\n(fraction of steps)"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (field, title) in zip(axes, panels):
        for method in present(curves):
            if method == "unshielded" and field == "interventions":
                continue
            ax.plot(curves[method]["x"], np.mean(curves[method][field], axis=0), color=COLORS[method],
                    linewidth=2, label=LABELS[method])
        ax.set_title(title, fontsize=10, color=TEXT, loc="left")
        ax.set_xlabel("Environment steps")
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_ylim(bottom=0)
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(out / "training_curves.png")
    plt.close(fig)


def beta_trace(results, out):
    curves = binned_training(results)
    if "fabr" not in curves:
        return
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for seed_trace in curves["fabr"]["beta_raw"]:
        x, b = zip(*seed_trace)
        ax.plot(x, b, color=COLORS["fabr"], linewidth=1.2, alpha=0.6)
    ax.plot([], [], color=COLORS["fabr"], linewidth=2, label="FABR β per episode (one line per seed)")
    if "fabr_fixed" in curves and curves["fabr_fixed"]["beta_raw"]:
        fixed = curves["fabr_fixed"]["beta_raw"][0][0][1]
        ax.axhline(fixed, color=COLORS["fabr_fixed"], linewidth=2, linestyle="--",
                   label=f"Fixed-β ablation (β = {fixed:.2f})")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("β (inverse temperature)")
    ax.set_title("FABR adapts β to the agent's unsafe-proposal rate", fontsize=10, color=TEXT, loc="left")
    ax.grid(color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(out / "fabr_beta.png")
    plt.close(fig)


def main(results="results/final"):
    results = Path(results)
    out = results / "figures"
    out.mkdir(exist_ok=True)
    agg = [r for r in read_csv(results / "aggregate.csv") if r["group"] in GROUPS]
    success_by_group(agg, out)
    outcomes(agg, out)
    training_curves(results, out)
    beta_trace(results, out)
    print(f"Figures written to {out}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results/final")
    main(**vars(p.parse_args()))
