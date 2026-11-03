"""Plot training CSV produced by train.py."""
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt


def moving_average(values, window):
    return [sum(values[max(0, i - window + 1):i + 1]) / min(i + 1, window)
            for i in range(len(values))]


def plot(csv_path, output='results/training_plot.png', window=25):
    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('Training CSV is empty')
    episodes = [int(row['episode']) for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    for ax, field, label in zip(axes, ['reward', 'success', 'unsafe_proposals'],
                                ['Episode reward', 'Success (0 or 1)', 'Unsafe proposals per episode']):
        values = [float(row[field]) for row in rows]
        ax.plot(episodes, values, alpha=0.2, label='Episode')
        ax.plot(episodes, moving_average(values, window), label=f'{window}-episode average')
        ax.set_ylabel(label)
        ax.legend()
    axes[-1].set_xlabel('Training episode')
    fig.tight_layout()
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f'Saved {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--output', default='results/training_plot.png')
    parser.add_argument('--window', type=int, default=25)
    args = parser.parse_args()
    if args.window < 1:
        parser.error('--window must be positive')
    plot(args.csv, args.output, args.window)
