"""Evaluate a saved DQN policy with no shield and greedy action selection."""
import argparse
import csv
from pathlib import Path
import torch
from dqn import DQNAgent
from environment import GridWorld
from metrics import summarize_episodes


def evaluate(model_path, episodes=100, seed=1000, output=None):
    if episodes < 1:
        raise ValueError('episodes must be positive')
    env = GridWorld()
    agent = DQNAgent(seed=seed)
    agent.online.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    agent.online.eval()
    rows = []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        reward_total = 0.0
        unsafe = 0
        for step in range(env.max_steps):
            action = agent.act(state, epsilon=0.0)
            unsafe += int(action not in env.safe_actions())
            state, reward, terminated, truncated, info = env.step(action)  # NO SHIELD
            reward_total += reward
            if terminated or truncated:
                break
        rows.append({'episode': episode + 1, 'reward': reward_total,
                     'steps': step + 1, 'success': int(info['success']),
                     'collision': int(info['collision']),
                     'unsafe_proposals': unsafe, 'interventions': 0})
    summary = summarize_episodes(rows)
    for key, value in summary.items():
        print(f'{key}: {value:.4f}' if isinstance(value, float) else f'{key}: {value}')
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1000)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    evaluate(args.model, args.episodes, args.seed, args.output)
