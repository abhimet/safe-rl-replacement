"""Create training learning curves from a CSV."""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot(csv_path, output=None, window=20):
    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('CSV has no rows')
    def moving_average(values):
        return [sum(values[max(0, i-window+1):i+1]) / min(i+1, window) for i in range(len(values))]
    episodes = [int(r['episode']) for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for ax, key, title in zip(axes, ('reward', 'success', 'unsafe_proposals'),
                              ('Environment reward', 'Success (moving average)', 'Unsafe proposals per episode')):
        values = [float(r[key]) for r in rows]
        ax.plot(episodes, moving_average(values), label=f'{window}-episode moving average')
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel('Training episode')
    fig.tight_layout()
    destination = Path(output) if output else Path(csv_path).with_suffix('.png')
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=140)
    plt.close(fig)
    print(f'Saved {destination}')
    return destination

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--csv', required=True)
    p.add_argument('--output')
    p.add_argument('--window', type=int, default=20)
    args = p.parse_args()
    plot(args.csv, output=args.output, window=args.window)
