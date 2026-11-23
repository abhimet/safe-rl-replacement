"""Evaluate the provisional runs and rebuild local evidence tables."""
import csv
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from dqn import QNetwork
from environment import GridWorld
from evaluate import evaluate
from metrics import summarize

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
PROVISIONAL = RESULTS / 'ddqn_provisional'


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def provisional_evaluation():
    settings = json.loads((PROVISIONAL / 'environment.json').read_text())
    starts = GridWorld(**settings).starts
    methods = ('unshielded', 'random_safe', 'value_penalty')
    rows = []
    with TemporaryDirectory() as directory:
        scratch = Path(directory)
        for method in methods:
            prefix = f'{method}_seed42'
            checkpoint = PROVISIONAL / f'{prefix}.pt'
            training = PROVISIONAL / f'{prefix}_summary.json'
            status = 'provisional_complete' if checkpoint.is_file() and training.is_file() else 'missing'
            entry = {'method': method, 'training_seed': 42, 'status': status,
                     'scenario_set': 'provisional_rooms_all_named_starts'}
            if status == 'missing':
                rows.append(entry)
                continue
            run = json.loads(training.read_text())
            episodes = []
            for index, start in enumerate(starts, start=1):
                config = scratch / 'scenario.json'
                raw = scratch / 'episode.csv'
                config.write_text(json.dumps({**settings, 'start': start}))
                evaluate(checkpoint, episodes=1, output=raw, env_config=config)
                episode, = read_csv(raw)
                episodes.append({**episode, 'episode': index, 'scenario_id': f'rooms_{index}',
                                 'layout': settings['layout'], 'start_row': start[0],
                                 'start_col': start[1], 'reset_seed': 0,
                                 'method': method, 'training_seed': 42,
                                 'checkpoint': checkpoint.relative_to(ROOT).as_posix()})
            destination = PROVISIONAL / f'{prefix}_all_starts_shield_off.csv'
            write_csv(destination, episodes)
            rows.append({**entry, 'training_episodes': run['episodes'],
                         'training_steps': run['training_steps'],
                         'optimizer_updates': run['optimizer_updates'],
                         'checkpoint': checkpoint.relative_to(ROOT).as_posix(),
                         'evaluation_csv': destination.relative_to(ROOT).as_posix(),
                         **summarize(episodes)})
    write_csv(ROOT / 'main_results_all_seeds.csv', rows)


def inventory():
    rows = []
    groups = ('ddqn_smoke', 'ddqn_provisional')
    for summary in sorted(RESULTS.rglob('*_summary.json')):
        config = json.loads(summary.read_text())
        prefix = summary.name.removesuffix('_summary.json')
        folder = summary.parent
        training_csv = folder / f'{prefix}.csv'
        training_rows = read_csv(training_csv)
        steps = sum(int(row['steps']) for row in training_rows)
        metrics = summarize(training_rows)
        for key, value in metrics.items():
            if key in config and not math.isclose(value, config[key], abs_tol=1e-8):
                raise ValueError(f'{summary}: {key} disagrees with raw data')
        if config.get('training_steps', steps) != steps:
            raise ValueError(f'{summary}: training step count disagrees with raw data')
        if 'optimizer_updates' in config:
            expected = max(0, steps - config['dqn']['batch_size'] + 1)
            if config['optimizer_updates'] != expected:
                raise ValueError(f'{summary}: optimizer update count disagrees with raw data')
        checkpoint = folder / f'{prefix}.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        matching = sorted(folder.glob(f'{prefix}_ep*.pt'))
        local_ddqn = folder.name in groups
        provenance = (json.loads((folder / 'source_provenance.json').read_text(encoding='utf-8-sig'))
                      if local_ddqn else {})
        evaluation = sorted(folder.glob(f'{prefix}*shield_off.csv'))
        if folder == RESULTS and prefix == 'adaptive_seed42':
            evaluation += sorted((folder / 'checkpoint_eval').glob('shield_*_ep*.csv'))
            evaluation += [folder / 'adaptive_shield_off.csv']
        if folder.name == 'nov08_training':
            evaluation += sorted((folder / 'reload_evaluation').glob(f'{prefix}_ep*.csv'))
        if local_ddqn:
            network = QNetwork()
            for path in [checkpoint, *matching]:
                network.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
                if not torch.isfinite(network(torch.zeros(2))).all():
                    raise ValueError(f'{path}: nonfinite prediction')
        def relative(path):
            return path.relative_to(ROOT).as_posix()
        rows.append({
            'run_id': relative(folder / prefix),
            'status': ('provisional_complete' if folder == PROVISIONAL else
                       'smoke_complete' if local_ddqn else 'legacy_preliminary'),
            'method': config['method'], 'seed': config['seed'],
            'algorithm': config.get('dqn', {}).get('algorithm', 'legacy_dqn'),
            'episodes': len(training_rows), 'training_steps': steps,
            'optimizer_updates': config.get('optimizer_updates', ''),
            'training_summary': relative(summary), 'training_csv': relative(training_csv),
            'final_checkpoint': relative(checkpoint),
            'intermediate_checkpoints': '|'.join(map(relative, matching)),
            'evaluation_csvs': '|'.join(map(relative, evaluation)),
            'torch_version': config.get('torch_version', ''),
            'numpy_version': config.get('numpy_version', ''),
            'python_version': provenance.get('python_version', ''),
            'base_commit': provenance.get('base_commit', ''),
            'dqn_source_sha256': provenance.get('dqn_source_sha256', ''),
            'training_source_sha256': provenance.get('training_source_sha256', ''),
        })
    write_csv(ROOT / 'experiment_manifest.csv', rows)


if __name__ == '__main__':
    torch.set_num_threads(1)
    provisional_evaluation()
    inventory()
