"""Greedy policy evaluation with NO shield interventions."""
import argparse
import csv
import json
from pathlib import Path
import torch
from dqn import QNetwork
from environment import GridWorld
from metrics import summarize

def evaluate(model, episodes=100, output='results/shield_off.csv', size=6, env_config=None):
    if episodes < 1:
        raise ValueError('episodes must be positive')
    network = QNetwork()
    network.load_state_dict(torch.load(model, map_location='cpu', weights_only=True))
    network.eval()
    settings = {'size': size}
    if env_config is not None:
        with open(env_config, encoding='utf-8') as f:
            settings.update(json.load(f))
    env = GridWorld(**settings)
    rows = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=episode)
        reward_sum = 0.0
        unsafe = 0
        for _ in range(env.max_steps):
            with torch.no_grad():
                action = int(network(torch.as_tensor(obs, dtype=torch.float32)).argmax().item())
            unsafe += int(action not in env.safe_actions())
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += reward
            if terminated or truncated:
                break
        rows.append({'episode': episode + 1, 'reward': round(reward_sum, 4),
                     'steps': env.steps, 'success': int(info['success']), 'collision': int(info['collision']),
                     'unsafe_proposals': unsafe, 'interventions': 0})
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    print(json.dumps(summary, indent=2))
    print(f'Shield OFF results saved to {out}')
    return summary

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--output', default='results/shield_off.csv')
    p.add_argument('--size', type=int, default=6)
    p.add_argument('--env-config', help='Same JSON environment settings used in training')
    evaluate(**vars(p.parse_args()))
