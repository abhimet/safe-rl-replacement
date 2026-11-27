"""Figures for a finished run_experiments.py output directory.

    python plots.py --results results/core             # writes results/core/figures/*.png

Works for protocol outputs (rows keyed by config "method@mode", runs under <mode>/runs/)
and for legacy single-setting outputs (rows keyed by method, runs under runs/).
"""
import argparse
import csv
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from replacement import FEEDBACK_MODES, METHODS  # noqa: E402
from run_experiments import GROUPS, TUNING_GROUPS  # noqa: E402

# Fixed colour per method (validated categorical palette; colour follows the method).
COLORS = {"fabr": "#257de0", "highest_q": "#f47230", "random_safe": "#00b894",
          "adaptive_mix": "#f2a600", "value_penalty": "#ee7fac", "unshielded": "#008a17",
          "fabr_fixed": "#6a4ce2", "fabr_v2": "#0b3c8c", "fabr_v2_fixedtarget": "#7fb2f0"}
LABELS = {"fabr": "FABR (ours)", "fabr_fixed": "FABR fixed-β (ablation)", "highest_q": "Highest-Q",
          "random_safe": "Random safe", "adaptive_mix": "Adaptive mix",
          "value_penalty": "Value + penalty", "unshielded": "Unshielded",
          "fabr_v2": "FABR v2 (margin)", "fabr_v2_fixedtarget": "FABR v2 fixed-target (ablation)"}
MODE_LABELS = {"none": "no feedback", "cf": "cf ORACLE", "credit": "credit", "margin": "margin"}
MODE_HATCH = {"none": "", "cf": "///", "credit": "...", "margin": ""}
GROUP_TITLES = {g: t + ("*" if g in TUNING_GROUPS else "") for g, t in {
    "unseen": "Unseen (pillars, heldout 1–3)", "unseen_spiral": "Unseen: spiral",
    "familiar": "Familiar (rooms, zigzag)", "dev_clutter": "Clutter (dev)"}.items()}
TUNING_NOTE = "* used for tuning (not held-out evidence)"
OUTCOMES = (("success_rate", "Success", "#257de0"), ("collision_rate", "Collision", "#ec4647"),
            ("timeout_rate", "Timeout", "#c9c9c9"))
TEXT, GRID = "#333333", "#e5e5e5"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
                     "xtick.color": TEXT, "ytick.color": TEXT, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight"})


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def split_name(name):
    """'method@mode' or 'method' -> (method, mode or None)."""
    method, _, mode = name.partition("@")
    return method, mode or None


def present(names):
    """Names in a stable order: by method, then by feedback mode."""
    order = {m: i for i, m in enumerate(METHODS)}
    modes = {m: i for i, m in enumerate((None, *FEEDBACK_MODES))}
    return sorted(set(names), key=lambda n: (order[split_name(n)[0]], modes[split_name(n)[1]]))


def color(name):
    return COLORS[split_name(name)[0]]


def label(name):
    method, mode = split_name(name)
    return LABELS[method] + (f" · {MODE_LABELS[mode]}" if mode else "")


def hatch(name):
    return MODE_HATCH.get(split_name(name)[1], "")


def row_key(rows):
    return "config" if "config" in rows[0] else "method"


def success_by_group(agg, out):
    """Small multiples: one panel per evaluation group, methods as labelled bars."""
    key = row_key(agg)
    methods = present({r[key] for r in agg})
    cells = {(r[key], r["group"]): r for r in agg}
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(3.2 * len(GROUPS), 0.45 * len(methods) + 1.6),
                             sharey=True)
    y = np.arange(len(methods))[::-1]
    for ax, group in zip(axes, GROUPS):
        means = [float(cells[(m, group)]["success_rate_mean"]) for m in methods]
        cis = [float(cells[(m, group)]["success_rate_ci95"]) for m in methods]
        bars = ax.barh(y, means, height=0.7, color=[color(m) for m in methods], edgecolor="white",
                       linewidth=2)
        for bar, m in zip(bars, methods):
            bar.set_hatch(hatch(m))
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
    axes[0].set_yticklabels([label(m) for m in methods])
    fig.suptitle("Shield-OFF success rate (mean ± 95% CI over seeds)", color=TEXT, x=0.02, ha="left")
    fig.text(0.02, -0.02, TUNING_NOTE, fontsize=8, color=TEXT)
    fig.savefig(out / "success_by_group.png")
    plt.close(fig)


def outcomes(agg, out):
    """Stacked success / collision / timeout per method, one panel per group."""
    key = row_key(agg)
    methods = present({r[key] for r in agg})
    cells = {(r[key], r["group"]): r for r in agg}
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(3.2 * len(GROUPS), 0.45 * len(methods) + 1.6),
                             sharey=True)
    y = np.arange(len(methods))[::-1]
    for ax, group in zip(axes, GROUPS):
        left = np.zeros(len(methods))
        for metric, name, shade in OUTCOMES:
            values = np.array([float(cells[(m, group)][f"{metric}_mean"]) for m in methods])
            ax.barh(y, values, left=left, height=0.7, color=shade, edgecolor="white", linewidth=2, label=name)
            left += values
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0%", "50%", "100%"])
        ax.set_title(GROUP_TITLES[group], fontsize=10, color=TEXT)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([label(m) for m in methods])
    axes[-1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)
    fig.suptitle("How shield-OFF episodes end", color=TEXT, x=0.02, ha="left")
    fig.text(0.02, -0.02, TUNING_NOTE, fontsize=8, color=TEXT)
    fig.savefig(out / "outcomes.png")
    plt.close(fig)


def binned_training(results, bins=40):
    """{name: {field: (x, mean over seeds)}} for per-step rates over training.

    name is the method (legacy layout runs/) or "method@mode" (protocol layout <mode>/runs/).
    """
    curves = defaultdict(lambda: defaultdict(list))
    paths = [(p, None) for p in Path(results, "runs").glob("*/*_seed*.csv")]
    paths += [(p, p.parents[2].name) for p in Path(results).glob("*/runs/*/*_seed*.csv")]
    for path, mode in sorted(paths):
        rows = read_csv(path)
        method = path.stem.rsplit("_seed", 1)[0]
        if mode is not None and method != "unshielded":
            method = f"{method}@{mode}"
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
        curves[method]["edges"] = edges
    return curves


def training_curves(results, out):
    curves = binned_training(results)
    panels = (("greedy_unsafe", "Greedy policy proposes an unsafe move\n(fraction of steps)"),
              ("interventions", "Shield interventions\n(fraction of steps)"))
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, (field, title) in zip(axes, panels):
        for method in present(curves):
            if split_name(method)[0] == "unshielded" and field == "interventions":
                continue
            ax.plot(curves[method]["x"], np.mean(curves[method][field], axis=0), color=color(method),
                    linewidth=2, label=label(method),
                    linestyle={"cf": "--", "credit": ":"}.get(split_name(method)[1], "-"))
        ax.set_title(title, fontsize=10, color=TEXT, loc="left")
        ax.set_xlabel("Environment steps")
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_ylim(bottom=0)
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(out / "training_curves.png")
    plt.close(fig)


def beta_trace(results, out):
    curves = binned_training(results)
    fabr = [n for n in present(curves) if split_name(n)[0] == "fabr"]
    if not fabr:
        return
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for name, style in zip(fabr, ("-", "--", ":", "-.")):
        mode = split_name(name)[1]
        x, edges = curves[name]["x"], curves[name]["edges"]
        binned = []
        for seed_trace in curves[name]["beta_raw"]:
            steps, b = (np.array(v) for v in zip(*seed_trace))
            which = np.clip(np.searchsorted(edges, steps, side="right") - 1, 0, len(x) - 1)
            per_bin = [b[which == i].mean() if (which == i).any() else np.nan for i in range(len(x))]
            binned.append(per_bin)
            ax.plot(x, per_bin, color=COLORS["fabr"], linewidth=0.8, alpha=0.3, linestyle=style)
        if binned:
            with warnings.catch_warnings():  # bins no seed reached are NaN by design
                warnings.simplefilter("ignore", RuntimeWarning)
                ax.plot(x, np.nanmean(binned, axis=0), color=COLORS["fabr"], linewidth=2.2, linestyle=style,
                        label=f"{label(name)} β (mean over seeds)")
        fixed_name = "fabr_fixed" + (f"@{mode}" if mode else "")
        if fixed_name in curves and curves[fixed_name]["beta_raw"]:
            fixed = curves[fixed_name]["beta_raw"][0][0][1]
            ax.axhline(fixed, color=COLORS["fabr_fixed"], linewidth=2, linestyle=style,
                       label=f"{label(fixed_name)} (β = {fixed:.2f})")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("β (inverse temperature)")
    ax.set_title("FABR adapts β to the agent's unsafe-proposal rate", fontsize=10, color=TEXT, loc="left")
    ax.grid(color=GRID, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.savefig(out / "fabr_beta.png")
    plt.close(fig)


def main(results="results/core"):
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
    p.add_argument("--results", default="results/core")
    main(**vars(p.parse_args()))
