"""Compare single uploads and re-uploading with matched Ansatz blocks."""
import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
import cudaq
from reuploading import ROOT, COUNTS, np, load_data, split_data, fit, forward, representation, metrics, resources
from articles.day18.benchmark import search

OUT = ROOT / 'results/day22'


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + '\n', encoding='utf-8')


def read(name):
    return json.loads((OUT / f'{name}.json').read_text())


def fingerprint():
    return hashlib.sha256(b''.join((OUT / f'{name}.json').read_bytes()
                                  for name in ('protocol', 'datasets', 'selected'))).hexdigest()


def train():
    rows, source = load_data()
    protocol = {'source': source, 'task': 'deduplicated binary Iris, 99 rows, Day20 splits',
                'split_seeds': [2028, 2029], 'initialization_seeds': [42, 43], 'models': COUNTS,
                'circuit_resources': {m: resources(m) for m in COUNTS},
                'budget': 73, 'optimizer': 'Day18 coordinate search; initial step .4',
                'loss': 'Brier', 'threshold': .5, 'selection': 'final validation Brier then lower seed, per model',
                'preprocessing': 'train-only PCA2 minmax or raw4 minmax; logistic4 train standardization',
                'quantum': '2 qubits, positive_half RY, 2 or 4 Ansatz blocks, matched once/repeat controls; (1-ZZ)/2',
                'training_engine': 'NumPy CPU exact reference; zero CUDA-Q training calls',
                'shots': -1, 'noise': 'none'}
    write(OUT / 'protocol.json', protocol)
    datasets, candidates, selected = {}, [], []
    for split_seed in protocol['split_seeds']:
        splits = split_data(rows, split_seed)
        raw = {k: [r['features'] for r in v] for k, v in splits.items()}
        labels = {k: [r['label'] for r in v] for k, v in splits.items()}
        prep = fit(raw['train'], labels['train'])
        datasets[str(split_seed)] = {'splits': splits, 'preprocessor': prep}
        for model, count in COUNTS.items():
            group = []
            for seed in protocol['initialization_seeds']:
                initial = np.random.default_rng(seed).normal(0, .2, count)
                def objective(w):
                    return metrics(labels['train'], forward(model, raw['train'], prep, w))['brier']
                start = time.perf_counter()
                weights, history, sweeps = search(objective, initial, protocol['budget'])
                record = {'split_seed': split_seed, 'model': model, 'seed': seed,
                          'initial_weights': initial.tolist(), 'weights': weights.tolist(), 'history': history,
                          'fit_seconds_numpy': time.perf_counter() - start, 'full_sweeps': sweeps,
                          'objective_evaluations': history[-1]['evaluations'], 'training_examples_evaluated': 73 * len(raw['train']),
                          'validation': metrics(labels['validation'], forward(model, raw['validation'], prep, weights))}
                candidates.append(record)
                group.append(record)
                print('fit', split_seed, model, seed, history[-1]['loss'], flush=True)
            winner = min(group, key=lambda r: (r['validation']['brier'], r['seed']))
            scores = {}
            for partition in splits:
                p = forward(model, raw[partition], prep, winner['weights'])
                x, flags = representation(model, raw[partition], prep, winner['weights'])
                scores[partition] = {'probabilities': p.tolist(), 'metrics': metrics(labels[partition], p),
                                     'clipped_values': int(flags.sum()),
                                     
                                     'constant_baseline': metrics(labels[partition], np.full(len(p), np.mean(labels['train'])))}
            selected.append({'split_seed': split_seed, 'model': model, 'seed': winner['seed'],
                             'weights': winner['weights'], 'scores': scores})
    write(OUT / 'datasets.json', datasets)
    write(OUT / 'candidates.json', candidates)
    write(OUT / 'selected.json', selected)
    summary = {'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
               'numpy': np.__version__, 'platform': platform.platform(), 'OMP_NUM_THREADS': os.getenv('OMP_NUM_THREADS'),
               'fits': len(candidates), 'cudaq_training_calls': 0, 'artifact_sha256': fingerprint(),
               'passed': all(c['objective_evaluations'] == 73 and c['history'][-1]['loss'] <= c['history'][0]['loss'] for c in candidates)}
    write(OUT / 'training_summary.json', summary)
    return summary


def verify(backend):
    cudaq.set_target(backend)
    datasets, selected, records = read('datasets'), read('selected'), []
    for model in selected:
        data = datasets[str(model['split_seed'])]
        for partition, rows in data['splits'].items():
            start = time.perf_counter()
            p = forward(model['model'], [r['features'] for r in rows], data['preprocessor'], model['weights'], 'cudaq')
            elapsed = time.perf_counter() - start
            error = float(np.max(abs(p - model['scores'][partition]['probabilities'])))
            records.append({'split_seed': model['split_seed'], 'model': model['model'], 'split': partition,
                            'probabilities': p.tolist(), 'metrics': metrics([r['label'] for r in rows], p),
                            'max_reference_error': error, 'seconds_including_compilation': elapsed,
                            'observe_calls': 0 if model['model'] == 'logistic4' else len(rows), 'passed': error < 1e-5})
    summary = {'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'backend': backend,
               'precision': str(cudaq.get_target().get_precision()), 'cudaq': version('cuda-quantum-cu12'),
               'python': platform.python_version(), 'numpy': np.__version__, 'OMP_NUM_THREADS': os.getenv('OMP_NUM_THREADS'),
               'artifact_sha256': fingerprint(), 'records': len(records), 'shots': -1, 'noise': 'none',
               'observe_calls': sum(r['observe_calls'] for r in records),
               'max_reference_error': max(r['max_reference_error'] for r in records),
               'scope': 'frozen inference verification, not GPU training or isolated performance benchmark',
               'passed': all(r['passed'] for r in records)}
    write(OUT / backend / 'verification.json', records)
    write(OUT / backend / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('train', 'verify'))
    parser.add_argument('--backend', choices=('qpp-cpu', 'nvidia'), default='qpp-cpu')
    args = parser.parse_args()
    result = train() if args.phase == 'train' else verify(args.backend)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
