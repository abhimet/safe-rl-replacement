"""Train a DQN with any shield strategy and save metrics/model."""
import argparse
import csv
import json
from pathlib import Path
import random
import numpy as np
import torch
from adaptive import UnsafeRateTracker
from dqn import DQNAgent
from environment import GridWorld
from metrics import summarize
from shield import choose_action

METHODS = ('unshielded', 'random_safe', 'value_based', 'value_penalty', 'fixed_mix', 'adaptive')

def train(method='random_safe', episodes=300, seed=42, output='results', size=6, window=100, penalty=1.0,
          env_config=None, threads=1):
    if method not in METHODS or episodes < 1:
        raise ValueError('Invalid method or episode count')
    if threads < 1:
        raise ValueError('threads must be positive')
    torch.set_num_threads(threads)
    settings = {'size': size}
    if env_config is not None:
        with open(env_config, encoding='utf-8') as f:
            settings.update(json.load(f))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed + 1)
    env = GridWorld(**settings)
    agent = DQNAgent(seed=seed)
    tracker = UnsafeRateTracker(window=window)
    rows = []
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        unsafe_count = intervention_count = 0
        selected_random = selected_value = 0
        for step in range(env.max_steps):
            epsilon = max(0.05, 1.0 - 0.95 * episode / max(1, int(episodes * 0.8)))
            proposed = agent.act(obs, epsilon)
            unsafe = proposed not in env.safe_actions()
            probability = tracker.rate  # use history BEFORE observing current proposal
            executed, intervened, strategy = choose_action(
                env, proposed, method, q_values=agent.q_values(obs), rng=rng, random_probability=probability)
            tracker.record(unsafe)
            if strategy == 'random_safe':
                selected_random += 1
            elif strategy in ('value_based', 'value_penalty'):
                selected_value += 1
            next_obs, reward, terminated, truncated, info = env.step(executed)
            adjusted_reward = reward - penalty if method == 'value_penalty' and intervened else reward
            agent.remember(obs, executed, adjusted_reward, next_obs, terminated or truncated)
            agent.learn()
            unsafe_count += int(unsafe)
            intervention_count += int(intervened)
            total_reward += reward  # report environment reward, not modified training reward
            obs = next_obs
            if terminated or truncated:
                break
        rows.append({
            'episode': episode + 1,
            'reward': round(total_reward, 4),
            'steps': env.steps,
            'success': int(info['success']),
            'collision': int(info['collision']),
            'unsafe_proposals': unsafe_count,
            'interventions': intervention_count,
            'random_replacements': selected_random,
            'value_replacements': selected_value,
            'adaptive_rate': round(tracker.rate, 4)
        })
        if (episode + 1) % 50 == 0:
            checkpoint_path = (
                out / f"{method}_seed{seed}_ep{episode + 1}.pt"
            )
            torch.save(agent.online.state_dict(), checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    prefix = f'{method}_seed{seed}'
    with (out / f'{prefix}.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    torch.save(agent.online.state_dict(), out / f'{prefix}.pt')
    with (out / f'{prefix}_summary.json').open('w') as f:
        json.dump({'method': method, 'seed': seed, 'size': env.size, 'episodes': episodes,
                   'environment': {'size': env.size, 'start': env.start, 'goal': env.goal,
                                   'obstacles': sorted(env.obstacles), 'max_steps': env.max_steps},
                   'window': window, 'penalty': penalty, 'threads': threads,
                   'torch_version': torch.__version__, 'numpy_version': np.__version__,
                   'dqn': {'lr': 0.001, 'gamma': 0.99, 'batch_size': 64,
                           'capacity': 10000, 'target_update_interval': 100,
                           'epsilon_start': 1.0, 'epsilon_end': 0.05,
                           'epsilon_decay_episodes': max(1, int(episodes * 0.8)),
                           'replay_action': 'executed', 'checkpoint_interval': 50},
                   **summarize(rows)}, f, indent=2)
    print(f'Trained {method} seed={seed}; saved to {out}/{prefix}.*')
    print(json.dumps(summarize(rows), indent=2))
    return rows

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=METHODS, default='random_safe')
    p.add_argument('--episodes', type=int, default=300)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output', default='results')
    p.add_argument('--size', type=int, default=6)
    p.add_argument('--window', type=int, default=100)
    p.add_argument('--penalty', type=float, default=1.0)
    p.add_argument('--env-config', help='JSON GridWorld constructor settings; overrides --size')
    p.add_argument('--threads', type=int, default=1)
    args = p.parse_args()
    train(**vars(args))
