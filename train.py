
"""Train Safe RL methods, including Pessimism-Annealed Replacement (PAR)."""

import argparse
import csv
import json
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch

from adaptive import UnsafeRateTracker
from dqn import DQNAgent
from environment import GridWorld
from metrics import summarize
from shield import choose_action, METHODS


PILOT_METHODS = tuple(dict.fromkeys((*METHODS, "worst_safe", "par")))


def train(
    method="adaptive",
    episodes=300,
    seed=42,
    output="results",
    size=6,
    window=100,
    penalty=1.0,
    env_config=None,
    threads=1,
    lr=1e-4,
    backup_mode="standard",
    storage_mode="executed",
    par_beta_min=-4.0,
    par_beta_max=4.0,
):
    if method not in PILOT_METHODS or episodes < 1:
        raise ValueError("Invalid method or episode count")

    if storage_mode not in ("executed", "both"):
        raise ValueError("Invalid storage mode")

    if method == "value_penalty" and storage_mode == "both":
        raise ValueError("Penalty baseline needs separate handling")

    if backup_mode == "shield_safe" and method == "unshielded":
        raise ValueError("shield_safe backup is only for shielded methods")

    if threads < 1 or window < 1:
        raise ValueError("threads and window must be positive")

    if par_beta_min >= par_beta_max:
        raise ValueError("par_beta_min must be smaller than par_beta_max")

    torch.set_num_threads(threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    rng = random.Random(seed + 1)

    settings = {"size": size}

    if env_config:
        with open(env_config, encoding="utf-8") as f:
            settings.update(json.load(f))

    env = GridWorld(**settings)

    agent = DQNAgent(
        seed=seed,
        lr=lr,
        backup_mode=backup_mode,
    )

    tracker = UnsafeRateTracker(window=window)

    # PAR tracks greedy unsafe actions separately from
    # epsilon-random exploration.
    greedy_history = deque(maxlen=window)
    initial_unsafe_rate = None

    rows = []
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    prefix = f"{method}_seed{seed}"

    if storage_mode == "both":
        prefix += "_both"

    for episode in range(episodes):

        obs, _ = env.reset(seed=seed + episode)

        reward_sum = 0.0
        unsafe_count = 0
        intervention_count = 0
        random_count = 0
        value_count = 0
        worst_count = 0
        credited_count = 0
        greedy_unsafe_count = 0

        q_gap_sum = 0.0
        q_gap_count = 0
        positive_q_gap_count = 0

        beta_values = []

        info = {
            "success": False,
            "collision": False,
        }

        for _ in range(env.max_steps):

            epsilon = max(
                0.05,
                1.0 - 0.95 * episode
                / max(1, int(episodes * 0.8)),
            )

            proposed = agent.act(obs, epsilon)

            safe = list(env.safe_actions())
            unsafe = proposed not in safe

            probability = tracker.rate
            q = agent.q_values(obs)

            # Measure whether the greedy policy itself
            # prefers an unsafe action.
            greedy_unsafe = int(
                int(np.argmax(q)) not in safe
            )

            greedy_unsafe_count += greedy_unsafe
            greedy_history.append(greedy_unsafe)

            # Establish initial reference rate after W checks.
            if (
                initial_unsafe_rate is None
                and len(greedy_history) == window
            ):
                initial_unsafe_rate = float(
                    np.mean(greedy_history)
                )

            # Q-gap:
            # best unsafe Q-value minus best safe Q-value.
            if safe:
                unsafe_actions = [
                    a for a in range(4)
                    if a not in safe
                ]

                if unsafe_actions:
                    gap = float(
                        max(q[a] for a in unsafe_actions)
                        - max(q[a] for a in safe)
                    )

                    q_gap_sum += gap
                    q_gap_count += 1
                    positive_q_gap_count += int(gap > 0)

            # -----------------------------------------
            # WORST-SAFE BASELINE
            # -----------------------------------------

            if method == "worst_safe":

                if unsafe:
                    if not safe:
                        raise RuntimeError(
                            "No safe replacement is available"
                        )

                    executed = min(
                        safe,
                        key=lambda a: float(q[a]),
                    )

                    intervened = True
                    strategy = "worst_safe"

                else:
                    executed = proposed
                    intervened = False
                    strategy = "none"

            # -----------------------------------------
            # NEW METHOD: PAR
            # -----------------------------------------

            elif method == "par":

                current_rate = float(
                    np.mean(greedy_history)
                )

                # Begin pessimistically, then move toward
                # value-based replacement as greedy
                # unsafe preferences decrease.
                if initial_unsafe_rate is None:
                    beta = par_beta_min

                elif initial_unsafe_rate == 0.0:
                    beta = (
                        par_beta_max
                        if current_rate == 0.0
                        else par_beta_min
                    )

                else:
                    improvement = (
                        1.0
                        - min(
                            current_rate / initial_unsafe_rate,
                            1.0,
                        )
                    )

                    beta = (
                        par_beta_min
                        + (par_beta_max - par_beta_min)
                        * improvement
                    )

                beta_values.append(beta)

                if unsafe:

                    if not safe:
                        raise RuntimeError(
                            "No safe replacement is available"
                        )

                    safe_q = np.asarray(
                        [q[a] for a in safe],
                        dtype=np.float64,
                    )

                    # Normalize Q-values across safe actions.
                    std = float(safe_q.std())

                    if std < 1e-6:
                        normalized = np.zeros_like(safe_q)
                    else:
                        normalized = (
                            (safe_q - safe_q.mean()) / std
                        )

                    # Boltzmann replacement distribution.
                    logits = beta * normalized
                    logits -= logits.max()

                    weights = np.exp(logits)

                    executed = rng.choices(
                        safe,
                        weights=weights.tolist(),
                        k=1,
                    )[0]

                    intervened = True
                    strategy = "par"

                else:
                    executed = proposed
                    intervened = False
                    strategy = "none"

            # -----------------------------------------
            # EXISTING METHODS
            # -----------------------------------------

            else:
                executed, intervened, strategy = (
                    choose_action(
                        env,
                        proposed,
                        method,
                        q_values=q,
                        rng=rng,
                        random_probability=probability,
                    )
                )

            tracker.record(unsafe)

            random_count += int(
                strategy == "random_safe"
            )

            value_count += int(
                strategy in ("value_based", "value_penalty")
            )

            worst_count += int(
                strategy == "worst_safe"
            )

            # Execute selected action.
            next_obs, reward, terminated, truncated, info = (
                env.step(executed)
            )

            done = terminated or truncated

            # Preserve original value_penalty baseline.
            training_reward = (
                reward - penalty
                if method == "value_penalty" and intervened
                else reward
            )

            next_safe = (
                env.safe_actions()
                if backup_mode == "shield_safe" and not done
                else None
            )

            # Real executed transition.
            agent.remember(
                obs,
                executed,
                training_reward,
                next_obs,
                done,
                next_safe,
            )

            # Optional shield-mediated action relabeling.
            # This is NOT the true unshielded transition.
            if storage_mode == "both" and intervened:

                agent.remember(
                    obs,
                    proposed,
                    reward,
                    next_obs,
                    done,
                    next_safe,
                )

                credited_count += 1

            # One learning update per environment step.
            agent.learn()

            unsafe_count += int(unsafe)
            intervention_count += int(intervened)
            reward_sum += reward

            obs = next_obs

            if done:
                break

        steps = env.steps

        rows.append({
            "episode": episode + 1,
            "reward": round(reward_sum, 4),
            "steps": steps,
            "success": int(info["success"]),
            "collision": int(info["collision"]),
            "unsafe_proposals": unsafe_count,
            "interventions": intervention_count,
            "random_replacements": random_count,
            "value_replacements": value_count,
            "worst_replacements": worst_count,
            "credited_proposals": credited_count,

            "adaptive_rate": round(tracker.rate, 4),

            "greedy_unsafe_rate": round(
                greedy_unsafe_count / max(1, steps), 4
            ),

            "greedy_history_rate": round(
                float(np.mean(greedy_history)), 4
            ),

            "mean_unsafe_q_gap": (
                round(q_gap_sum / q_gap_count, 4)
                if q_gap_count else None
            ),

            "positive_q_gap_rate": (
                round(
                    positive_q_gap_count / q_gap_count,
                    4,
                )
                if q_gap_count else None
            ),

            "par_beta": (
                round(beta_values[-1], 4)
                if beta_values else None
            ),

            "par_beta_mean": (
                round(float(np.mean(beta_values)), 4)
                if beta_values else None
            ),

            "q_max_batch": round(
                agent.last_stats.get("q_max", 0.0), 4
            ),

            "target_max_batch": round(
                agent.last_stats.get("target_max", 0.0), 4
            ),

            "loss_last_batch": round(
                agent.last_stats.get("loss", 0.0), 6
            ),
        })

        # Save checkpoints every 50 episodes.
        if (episode + 1) % 50 == 0:

            path = (
                out / f"{prefix}_ep{episode+1}.pt"
            )

            torch.save(
                agent.online.state_dict(),
                path,
            )

            print(
                f"Saved checkpoint: {path}; "
                f"Q={rows[-1]['q_max_batch']:.2f}; "
                f"target={rows[-1]['target_max_batch']:.2f}; "
                f"beta={rows[-1]['par_beta']}"
            )

    # Save episode metrics.
    with (out / f"{prefix}.csv").open(
        "w", newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)

    # Save model.
    torch.save(
        agent.online.state_dict(),
        out / f"{prefix}.pt",
    )

    # Save summary.
    summary = {
        "method": method,
        "storage_mode": storage_mode,
        "seed": seed,
        "episodes": episodes,
        "environment": settings,
        "window": window,
        "penalty": penalty,
        "lr": lr,
        "backup_mode": backup_mode,
        "par_beta_min": par_beta_min,
        "par_beta_max": par_beta_max,
        "initial_greedy_unsafe_rate": initial_unsafe_rate,
        "target": "double_dqn",
        "torch_version": torch.__version__,

        "credited_proposals": sum(
            row["credited_proposals"]
            for row in rows
        ),

        **summarize(rows),
    }

    with (
        out / f"{prefix}_summary.json"
    ).open("w") as f:

        json.dump(summary, f, indent=2)

    print(
        f"Trained {method} ({storage_mode}); "
        f"saved to {out}/{prefix}.*"
    )

    print(json.dumps(summary, indent=2))

    return rows


if __name__ == "__main__":

    p = argparse.ArgumentParser()

    p.add_argument(
        "--method",
        choices=PILOT_METHODS,
        default="adaptive",
    )

    p.add_argument(
        "--episodes",
        type=int,
        default=300,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    p.add_argument(
        "--output",
        default="results",
    )

    p.add_argument(
        "--size",
        type=int,
        default=6,
    )

    p.add_argument(
        "--window",
        type=int,
        default=100,
    )

    p.add_argument(
        "--penalty",
        type=float,
        default=1.0,
    )

    p.add_argument("--env-config")

    p.add_argument(
        "--threads",
        type=int,
        default=1,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    p.add_argument(
        "--backup-mode",
        choices=("standard", "shield_safe"),
        default="standard",
    )

    p.add_argument(
        "--storage-mode",
        choices=("executed", "both"),
        default="executed",
    )

    p.add_argument(
        "--par-beta-min",
        type=float,
        default=-4.0,
    )

    p.add_argument(
        "--par-beta-max",
        type=float,
        default=4.0,
    )

    train(**vars(p.parse_args()))
