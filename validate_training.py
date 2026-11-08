"""Reproduce Sriram's November 8 training artifacts from one command."""
import argparse
import json
from pathlib import Path

import torch

from evaluate import evaluate
from plot_results import plot
from train import METHODS, train


def validate(output='results/nov08_sriram'):
    root = Path(output)
    config = Path(__file__).parent / 'configs' / 'integration_10x10.json'
    smoke = []
    for size in (6, 10):
        for method in METHODS:
            destination = root / 'smoke' / f'{size}x{size}'
            rows = train(method, episodes=5, output=destination, size=size,
                         env_config=config if size == 10 else None)
            smoke.append({'method': method, 'size': size, 'episodes': len(rows),
                          'total_steps': sum(r['steps'] for r in rows)})
    train('adaptive', episodes=300, seed=42, output=root)
    plot(root / 'adaptive_seed42.csv')
    evaluations = {}
    for episode in range(50, 301, 50):
        name = f'adaptive_seed42_ep{episode}'
        evaluations[name] = evaluate(root / f'{name}.pt', episodes=10,
                                     output=root / 'reload_evaluation' / f'{name}.csv')
    final = torch.load(root / 'adaptive_seed42.pt', weights_only=True)
    checkpoint = torch.load(root / 'adaptive_seed42_ep300.pt', weights_only=True)
    assert all(torch.equal(final[key], checkpoint[key]) for key in final)
    report = {'smoke_runs': smoke, 'checkpoint_reload_evaluations': evaluations,
              'final_matches_episode300': True,
              'note': 'Deterministic fixed-start evaluation; repeated episodes are not independent trials.'}
    (root / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Validation complete: {root / "validation.json"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/nov08_sriram')
    validate(**vars(parser.parse_args()))
