"""Animate trained agents on the same map and start, with the shield ON vs OFF.

    python demo.py --results results/final --methods fabr unshielded --layout pillars --start 9 0

Each row is one trained agent (method, seed). The left column shows it with the
shield on (red ✕ = the shield blocked the proposed move and substituted a safe
one); the right column shows it with the shield off, executing its own choices.
Writes <output>.gif (animation) and <output>.png (final frame).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from evaluate import load_network, rollout, scenario_env  # noqa: E402
from make_eval_scenarios import V2_PATH  # noqa: E402
from plots import COLORS, LABELS, TEXT  # noqa: E402

WALL, FLOOR, GOAL = "#3a3a3a", "#f7f7f7", "#f2a600"
ARROWS = {0: (0, -0.35), 1: (0.35, 0), 2: (0, 0.35), 3: (-0.35, 0)}  # (dx, dy) in plot coords


def outcome(row):
    return "reached goal" if row["success"] else "COLLISION" if row["collision"] else "timed out"


def draw_map(ax, env):
    ax.set_xlim(-0.5, env.size - 0.5)
    ax.set_ylim(env.size - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.add_patch(plt.Rectangle((-0.5, -0.5), env.size, env.size, color=FLOOR, zorder=0))
    for r, c in env.obstacles:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=WALL, zorder=1))
    ax.scatter([env.goal[1]], [env.goal[0]], marker="*", s=320, color=GOAL, edgecolor="white",
               linewidth=1.5, zorder=3)


def make_demo(agents, layout, start, output="results/demo", fps=4):
    spec = json.loads(Path(V2_PATH).read_text())
    env = scenario_env(spec, layout, V2_PATH)
    start = tuple(start)
    if start not in env.distances_to_goal() or start == env.goal:
        raise ValueError(f"{start} is not a reachable start on {layout}")

    runs = []  # (method, seed, shield, row, trajectory)
    for method, seed, network in agents:
        for shield in (True, False):
            row, trajectory = rollout(network, env, start, shield=shield)
            runs.append((method, seed, shield, row, trajectory))

    fig, axes = plt.subplots(len(agents), 2, figsize=(7, 3.7 * len(agents)), squeeze=False)
    artists = []
    for ax, (method, seed, shield, row, trajectory) in zip(axes.flat, runs):
        draw_map(ax, env)
        color = COLORS.get(method, "#257de0")
        ax.set_title(f"{LABELS.get(method, method)} (seed {seed})\nshield {'ON' if shield else 'OFF'}: "
                     f"{outcome(row)} in {row['steps']} steps", fontsize=9, color=TEXT)
        trail, = ax.plot([], [], color=color, linewidth=2, alpha=0.6, zorder=2)
        agent = ax.scatter([], [], s=140, color=color, edgecolor="white", linewidth=2, zorder=5)
        blocked = ax.scatter([], [], marker="x", s=60, color="#ec4647", linewidth=2, zorder=4)
        arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0), zorder=6,
                            arrowprops=dict(arrowstyle="->", color="#ec4647", linewidth=2))
        artists.append((trajectory, row, trail, agent, blocked, arrow))

    frames = max(len(a[0]) for a in artists) + fps  # hold the last frame for a second

    def update(frame):
        for trajectory, row, trail, agent, blocked, arrow in artists:
            t = min(frame, len(trajectory) - 1)
            cols = [p[1] for p, *_ in trajectory[:t + 1]]
            rows = [p[0] for p, *_ in trajectory[:t + 1]]
            trail.set_data(cols, rows)
            agent.set_offsets([[cols[-1], rows[-1]]])
            marks = [[p[1], p[0]] for p, _, _, hit in trajectory[:t] if hit]
            blocked.set_offsets(marks if marks else [[float("nan"), float("nan")]])
            # Show the unsafe move the shield-off agent crashes with, as a red arrow.
            last = t == len(trajectory) - 1 and row["collision"]
            if last:
                (r, c), proposed, *_ = trajectory[-2]
                dx, dy = ARROWS[proposed]
                arrow.set_position((c, r))
                arrow.xy = (c + 2 * dx, r + 2 * dy)
            arrow.set_visible(bool(last))
        return []

    animation = FuncAnimation(fig, update, frames=frames, interval=1000 // fps)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output.with_suffix(".gif"), writer=PillowWriter(fps=fps))
    update(frames - 1)
    fig.savefig(output.with_suffix(".png"))
    plt.close(fig)
    for method, seed, shield, row, _ in runs:
        print(f"{method:<14} seed {seed} shield {'ON ' if shield else 'OFF'}: {outcome(row)}, "
              f"{row['steps']} steps, {row['interventions']} interventions")
    print(f"Saved {output.with_suffix('.gif')} and {output.with_suffix('.png')}")
    return runs


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results/final", help="run_experiments.py output directory")
    p.add_argument("--methods", nargs="+", default=["fabr", "unshielded"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--layout", default="pillars", help="Any layout in evaluation_scenarios_v2.json")
    p.add_argument("--start", type=int, nargs=2, default=[9, 0], metavar=("ROW", "COL"))
    p.add_argument("--output", default=None, help="Output path without extension")
    args = p.parse_args()
    agents = [(m, args.seed, load_network(Path(args.results, "runs", f"{m}_seed{args.seed}",
                                                f"{m}_seed{args.seed}.pt"))) for m in args.methods]
    make_demo(agents, args.layout, args.start,
              output=args.output or Path(args.results, "figures", f"demo_{args.layout}"))
