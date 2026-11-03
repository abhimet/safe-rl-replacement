import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from environment import GridWorld
from shield import choose_action
from dqn import DQNAgent


def train(method, episodes, seed, output):
    rng = np.random.default_rng(seed)
    env = GridWorld()
    agent = DQNAgent(seed=seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        epsilon = max(0.05, 1.0 - 0.95 * episode / max(1, episodes - 1))
        total_reward = 0.0
        proposals_unsafe = 0
        interventions = 0
        success = False
        collision = False
        for step in range(env.max_steps):
            proposed = agent.act(state, epsilon)
            if proposed not in env.safe_actions():
                proposals_unsafe += 1
            executed, intervened = choose_action(env, proposed, rng, enabled=(method == 'random_safe'))
            interventions += int(intervened)
            next_state, reward, terminated, truncated, info = env.step(executed)
            done = terminated or truncated
            agent.remember(state, executed, reward, next_state, done)
            agent.learn()
            total_reward += reward
            state = next_state
            success = info['success']
            collision = info['collision']
            if done:
                break
        rows.append({
            'episode': episode + 1, 'reward': round(total_reward, 3),
            'steps': step + 1, 'success': int(success), 'collision': int(collision),
            'unsafe_proposals': proposals_unsafe, 'interventions': interventions,
            'epsilon': round(epsilon, 4),
        })
        if (episode + 1) % 50 == 0:
            recent = rows[-50:]
            print(f"Episode {episode+1}: success(last 50)={sum(r['success'] for r in recent)/len(recent):.2f}")
    with (output / f'{method}_seed{seed}.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    torch.save(agent.online.state_dict(), output / f'{method}_seed{seed}.pt')
    print(f'Saved results in {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=['unshielded', 'random_safe'], default='random_safe')
    parser.add_argument('--episodes', type=int, default=300)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='results')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be positive')
    train(args.method, args.episodes, args.seed, args.output)
