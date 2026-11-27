"""Train one method with Double DQN on the 10x10 training maps.

Every method uses the same configuration; only the shield's replacement rule
(replacement.py) differs:

* Training maps: the designed ``rooms`` and ``zigzag`` layouts plus a seeded
  pool of random-obstacle maps (``environment.generate_training_pool``) that
  cannot resemble any evaluation map. A random map is drawn each episode, and
  starts cycle through every reachable cell of that map.
* Observation: agent position, goal position and the 10x10 wall grid (104 inputs).
* Network: SpatialQNetwork, Double DQN, one gradient update per environment step.
* Shielded methods bootstrap only through safe actions (the target's next
  action is chosen among the actions the shield allows).
* --feedback-mode decides what a shielded agent stores about a blocked proposal
  (replacement.FEEDBACK_MODES):
    none    only the executed transition (s, a_exec, r, s').
    cf      also (s, a_prop, -10, terminal): exactly what would have happened in
            this deterministic grid. An ORACLE: it uses the outcome of an action
            that was never executed (Alshiekh et al. 2018, post-posed option 1).
    credit  (s, a_prop, r, s') INSTEAD of the executed transition, where r and s'
            come from executing the replacement (Alshiekh et al. option 2).
    margin  the executed transition plus (s, a_prop) for the FABR v2 margin loss.
  Default: cf (the historical default) for the original methods, margin for
  fabr_v2*, and none for unshielded (which never intervenes). The old
  --no-collision-feedback flag and collision_feedback= argument still work as
  aliases for none / cf.
* --n-step n uses n-step returns (default 1). The counterfactual stays 1-step.
* Potential-based reward shaping from the map's shortest-path distances.
  Shaping preserves the optimal policy and is used for training only.

Usage:
    python train.py --method fabr --steps 50000 --seed 42 --output results/runs/fabr_seed42
    python train.py --method fabr_v2 --n-step 3 --output results/one     # margin feedback
    python train.py --method highest_q --feedback-mode credit --output results/one
"""
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from collections import deque

from dqn import DQNAgent
from environment import GridWorld, generate_training_pool
from metrics import summarize
from replacement import FEEDBACK_MODES, MARGIN_METHODS, METHODS, Replacer

TRAIN_LAYOUTS = ("rooms", "zigzag")
COLLISION_REWARD = -10.0


def potential(distances, position, scale):
    """Phi(s) = -scale * distance_to_goal / max_distance, in [-scale, 0]."""
    return -float(scale) * distances[tuple(position)] / max(distances.values())


def shaping_bonus(distances, position, next_position, done, gamma, scale):
    """F = gamma * Phi(s') - Phi(s), with Phi(terminal) = 0."""
    after = 0.0 if done else potential(distances, next_position, scale)
    return gamma * after - potential(distances, position, scale)


def resolve_feedback_mode(method, feedback_mode=None, collision_feedback=None):
    """The feedback mode for a method, accepting the legacy collision_feedback flag.

    The legacy flag predates the margin mode: unshielded and fabr_v2* ignore it.
    """
    if feedback_mode is not None and feedback_mode not in FEEDBACK_MODES:
        raise ValueError(f"feedback_mode must be one of {FEEDBACK_MODES}")
    if method == "unshielded":
        if feedback_mode == "margin":
            raise ValueError("margin feedback needs a fabr_v2 method")
        return "none"  # no shield, no interventions: every mode is the same
    if method in MARGIN_METHODS:
        if feedback_mode not in (None, "margin"):
            raise ValueError(f"{method} needs feedback_mode='margin'")
        return "margin"
    legacy = None if collision_feedback is None else ("cf" if collision_feedback else "none")
    if feedback_mode is not None and legacy is not None and feedback_mode != legacy:
        raise ValueError(f"collision_feedback={collision_feedback} contradicts feedback_mode={feedback_mode!r}")
    mode = feedback_mode or legacy or "cf"
    if mode == "margin":
        raise ValueError(f"margin feedback is only for {tuple(MARGIN_METHODS)}")
    return mode


class NStep:
    """Turns 1-step transitions into n-step ones within an episode.

    ``add`` returns the transitions ready for replay as tuples
    (state, action, n-step reward, next_state, done, next_safe, discount, extras),
    where discount = gamma ** k for the k rewards summed. With n = 1 every
    transition is returned immediately and unchanged (discount = gamma).
    """

    def __init__(self, n, gamma):
        if n < 1:
            raise ValueError("n_step must be at least 1")
        self.n, self.gamma, self.pending = int(n), gamma, deque()

    def add(self, state, action, reward, next_state, done, next_safe, extras, episode_end):
        self.pending.append((state, action, reward, extras))
        ready = []
        if len(self.pending) == self.n:
            ready.append(self._pop(next_state, done, next_safe))
        if episode_end:  # terminal, time limit or end of the step budget: flush
            while self.pending:
                ready.append(self._pop(next_state, done, next_safe))
        return ready

    def _pop(self, next_state, done, next_safe):
        k = len(self.pending)
        total = sum(self.gamma ** i * item[2] for i, item in enumerate(self.pending))
        state, action, _, extras = self.pending.popleft()
        return state, action, total, next_state, done, next_safe, self.gamma ** k, extras


def training_maps(pool_size=100, pool_seed=7):
    """[(name, GridWorld)]: rooms, zigzag and the generated pool."""
    from make_eval_scenarios import evaluation_maps
    avoid = [env.obstacles for env in evaluation_maps().values()]
    maps = [(name, GridWorld(layout=name, observation_mode="map")) for name in TRAIN_LAYOUTS]
    if pool_size:
        maps += generate_training_pool(n=pool_size, seed=pool_seed, avoid=avoid)
    return maps


def train(method="fabr", steps=50000, seed=42, output="results/run", pool_size=100, pool_seed=7,
          window=100, penalty=1.0, lr=1e-4, beta_min=0.0, beta_max=4.0, fixed_beta=None,
          shaping_scale=4.0, epsilon_fraction=0.8, collision_feedback=None, feedback_mode=None,
          n_step=1, capacity=10000, margin=1.0, margin_lambda=0.5, margin_stop_grad=False,
          protocol_sha256=None, verbose=True):
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {METHODS}")
    if steps < 1:
        raise ValueError("steps must be positive")
    if not 0 < epsilon_fraction <= 1:
        raise ValueError("epsilon_fraction must be in (0, 1]")
    mode = resolve_feedback_mode(method, feedback_mode, collision_feedback)
    shielded = method != "unshielded"

    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    replace_rng = random.Random(seed + 1)
    map_rng = random.Random(seed + 2)
    start_rng = random.Random(seed + 3)

    maps = training_maps(pool_size, pool_seed)
    distances = [env.distances_to_goal() for _, env in maps]
    start_queues = [[] for _ in maps]

    margin_config = (dict(kind=MARGIN_METHODS[method], margin=margin, weight=margin_lambda,
                          stop_grad=margin_stop_grad) if method in MARGIN_METHODS else None)
    agent = DQNAgent(input_dim=maps[0][1].observation_dim, seed=seed, lr=lr, capacity=capacity,
                     backup_mode="shield_safe" if shielded else "standard",
                     margin_config=margin_config)
    nstep = NStep(n_step, agent.gamma)
    replacer = Replacer(method, replace_rng, window=window, beta_min=beta_min,
                        beta_max=beta_max, fixed_beta=fixed_beta)

    rows, betas = [], []
    total = 0
    while total < steps:
        index = map_rng.randrange(len(maps))
        name, env = maps[index]
        if not start_queues[index]:
            start_queues[index] = sorted(c for c in distances[index] if c != env.goal)
            start_rng.shuffle(start_queues[index])
        obs, _ = env.reset(options={"start": start_queues[index].pop()})

        ep = dict(reward=0.0, unsafe_proposals=0, greedy_unsafe=0, interventions=0)
        ep_betas = []
        info, terminated, truncated = {"success": False, "collision": False}, False, False
        while not (terminated or truncated) and total < steps:
            epsilon = max(0.05, 1.0 - 0.95 * total / max(1, int(steps * epsilon_fraction)))
            proposed, q = agent.act_with_q(obs, epsilon)
            safe = env.safe_actions()
            if method in ("fabr", "fabr_fixed"):
                ep_betas.append(replacer.beta)
            executed, intervened = replacer.choose(proposed, safe, q)

            position = env.position
            next_obs, reward, terminated, truncated, info = env.step(executed)
            total += 1
            # Truncation (time limit) is not a terminal MDP state: keep bootstrapping.
            train_reward = reward + shaping_bonus(distances[index], position, env.position,
                                                  terminated, agent.gamma, shaping_scale)
            if method == "value_penalty" and intervened:
                train_reward -= penalty
            next_safe = env.safe_actions() if shielded and not terminated else None
            # credit: the blocked proposal is credited with the replacement's outcome.
            stored_action = proposed if mode == "credit" else executed
            episode_end = terminated or truncated or total >= steps
            for item in nstep.add(obs, stored_action, train_reward, next_obs, terminated, next_safe,
                                  (proposed, intervened, safe), episode_end):
                s0, a0, r0, s1, done, safe1, discount, (prop0, inter0, safe0) = item
                agent.remember(s0, a0, r0, s1, done, safe1, discount=discount, proposed=prop0,
                               intervened=inter0 and mode == "margin", safe_actions=safe0)
            if intervened and mode == "cf":
                # Known outcome of the blocked proposal in this deterministic grid (ORACLE).
                cf_reward = COLLISION_REWARD + shaping_bonus(
                    distances[index], position, position, True, agent.gamma, shaping_scale)
                agent.remember(obs, proposed, cf_reward, obs, True, None)
            agent.learn()

            ep["reward"] += reward
            ep["unsafe_proposals"] += int(proposed not in safe)
            ep["greedy_unsafe"] += int(int(np.argmax(q)) not in safe)
            ep["interventions"] += int(intervened)
            obs = next_obs

        betas += ep_betas
        rows.append({
            "episode": len(rows) + 1, "env_steps": total, "map": name, "start": str(env.start),
            "reward": round(ep["reward"], 4), "steps": env.steps,
            "success": int(info["success"]), "collision": int(info["collision"]),
            "timeout": int(truncated), "budget_cut": int(not (terminated or truncated)),
            "unsafe_proposals": ep["unsafe_proposals"], "greedy_unsafe": ep["greedy_unsafe"],
            "interventions": ep["interventions"], "epsilon": round(epsilon, 4),
            "beta": round(float(np.mean(ep_betas)), 4) if ep_betas else "",
        })

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    prefix = f"{method}_seed{seed}"
    with (out / f"{prefix}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    torch.save(agent.online.state_dict(), out / f"{prefix}.pt")

    completed = [r for r in rows if not r["budget_cut"]]
    summary = {
        "method": method, "seed": seed, "steps": total, "episodes": len(completed),
        "config": dict(pool_size=pool_size, pool_seed=pool_seed, window=window, penalty=penalty,
                       lr=lr, beta_min=beta_min, beta_max=beta_max, fixed_beta=fixed_beta,
                       shaping_scale=shaping_scale, epsilon_fraction=epsilon_fraction,
                       feedback_mode=mode, collision_feedback=mode == "cf",
                       n_step=n_step, capacity=capacity, margin=margin,
                       margin_lambda=margin_lambda, margin_stop_grad=margin_stop_grad,
                       training_maps=len(maps)),
        "protocol_sha256": protocol_sha256,
        "mean_beta": float(np.mean(betas)) if betas else None,
        "training": summarize(completed) if completed else None,
        "torch_version": torch.__version__,
    }
    (out / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    if verbose:
        t = summary["training"] or {}
        print(f"[{method}@{mode} seed={seed}] {total} steps, {len(completed)} episodes, "
              f"train success={t.get('success_rate', 0):.3f} collision={t.get('collision_rate', 0):.3f} "
              f"interventions/step={t.get('intervention_rate', 0):.3f}"
              + (f" mean beta={summary['mean_beta']:.2f}" if betas else ""), flush=True)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=METHODS, default="fabr")
    p.add_argument("--steps", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results/run")
    p.add_argument("--pool-size", type=int, default=100)
    p.add_argument("--fixed-beta", type=float, default=None, help="Required for fabr_fixed")
    p.add_argument("--beta-max", type=float, default=4.0)
    p.add_argument("--penalty", type=float, default=1.0, help="value_penalty intervention penalty")
    p.add_argument("--feedback-mode", choices=FEEDBACK_MODES, default=None,
                   help="What a shielded agent stores about a blocked proposal (see above)")
    p.add_argument("--no-collision-feedback", dest="collision_feedback", action="store_const",
                   const=False, default=None, help="Legacy alias for --feedback-mode none")
    p.add_argument("--n-step", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epsilon-fraction", type=float, default=0.8)
    p.add_argument("--capacity", type=int, default=10000, help="Replay buffer size")
    p.add_argument("--margin", type=float, default=1.0, help="fabr_v2 margin m")
    p.add_argument("--margin-lambda", type=float, default=0.5, help="fabr_v2 margin-loss weight")
    p.add_argument("--margin-stop-grad", action="store_true",
                   help="fabr_v2: no gradient through the max over safe actions")
    train(**vars(p.parse_args()))
