"""A deterministic, derivative-free hybrid loop using the Day 09 PQC."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import argparse
import csv
import json
import os
import platform
from datetime import datetime, timezone
from importlib.metadata import version
import cudaq
import numpy as np
from articles.day09.pqc import initialize, predict, reference_prediction, sample_prediction


def dataset():
    train = np.array([(a, b) for a in (-0.6, 0., 0.6) for b in (-0.5, 0., 0.5)])
    holdout = np.array([[-0.8, -0.3], [-0.2, 0.7], [0.3, -0.7], [0.8, 0.2]])
    return train, np.cos(np.pi * train[:, 1] + 0.6), holdout, np.cos(np.pi * holdout[:, 1] + 0.6)


def mse(predictions, labels):
    p, y = np.asarray(predictions, dtype=float), np.asarray(labels, dtype=float)
    if p.ndim != 1 or p.size == 0 or p.shape != y.shape or not (np.isfinite(p).all() and np.isfinite(y).all()):
        raise ValueError('MSE requires equal, nonempty, finite vectors')
    return float(np.mean((p - y)**2))


def coordinate_search(objective, initial, sweeps=40, step=0.4, tolerance=1e-4):
    """Accept only strict improvements; halve step after a sweep with no move."""
    w = np.asarray(initial, dtype=float).copy()
    if w.ndim != 1 or not w.size or not np.isfinite(w).all():
        raise ValueError('initial must be a nonempty finite vector')
    if type(sweeps) is not int or sweeps < 1 or not np.isfinite([step, tolerance]).all() or not 0 < tolerance < step:
        raise ValueError('positive sweeps and 0 < tolerance < step required')
    evaluations = 0
    def evaluate(candidate):
        nonlocal evaluations
        value = float(objective(candidate.copy()))
        evaluations += 1
        if not np.isfinite(value):
            raise ValueError('objective must be finite')
        return value
    loss = evaluate(w)
    history = [{'sweep': 0, 'loss': loss, 'step': step, 'evaluations': evaluations, 'weights': w.tolist()}]
    reason = 'max_sweeps'
    for sweep in range(1, sweeps + 1):
        improved = False
        used_step = step
        for index in range(len(w)):
            candidates = []
            for direction in (-1, 1):
                candidate = w.copy()
                candidate[index] += direction * step
                candidates.append((evaluate(candidate), candidate))
            candidate_loss, candidate = min(candidates, key=lambda pair: pair[0])
            if candidate_loss < loss:
                loss, w, improved = candidate_loss, candidate, True
        history.append({'sweep': sweep, 'loss': loss, 'step': used_step, 'evaluations': evaluations, 'weights': w.tolist()})
        if not improved:
            step *= 0.5
        if step < tolerance:
            reason = 'step_tolerance'
            break
    return w, history, reason


def run(backend='qpp-cpu', seed=42, sweeps=40):
    cudaq.set_target(backend)
    x, y, held, held_y = dataset()
    initial = initialize(1, seed)
    def batch(features, weights, forward=predict):
        return np.array([forward(row, weights, 1) for row in features])
    def objective(weights):
        return mse(batch(x, weights), y)
    weights, history, reason = coordinate_search(objective, initial, sweeps=sweeps)
    rows = []
    for split, features, labels in [('train', x, y), ('holdout', held, held_y)]:
        for index, (feature, label) in enumerate(zip(features, labels)):
            exact = predict(feature, weights)
            reference = reference_prediction(feature, weights)
            sampled = sample_prediction(feature, weights, shots=1000, seed=1000+index)
            rows.append({'split': split, 'features': feature.tolist(), 'label': float(label),
                         'initial': predict(feature, initial), 'final': exact,
                         'reference': reference, 'sampled': sampled})
    metrics = {}
    for split in ('train', 'holdout'):
        records = [r for r in rows if r['split'] == split]
        labels = [r['label'] for r in records]
        metrics[split] = {key+'_mse': mse([r[key] for r in records], labels) for key in ('initial', 'final')}
        metrics[split]['constant_baseline_mse'] = mse([float(y.mean())]*len(records), labels)
    max_error = max(abs(r['final']-r['reference']) for r in rows)
    passed = (metrics['train']['final_mse'] < metrics['train']['initial_mse'] and
              max_error < 1e-5 and all(b['loss'] <= a['loss'] for a,b in zip(history,history[1:])))
    summary = {'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'backend': backend,
               'precision': str(cudaq.get_target().get_precision()), 'python': platform.python_version(),
               'cudaq': version('cuda-quantum-cu12'), 'numpy': np.__version__, 'platform': platform.platform(),
               'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'), 'seed': seed,
               'layers': 1, 'initial_weights': initial.tolist(), 'final_weights': weights.tolist(),
               'optimizer': 'sequential coordinate search', 'initial_step': 0.4, 'step_tolerance': 1e-4,
               'max_sweeps': sweeps, 'completed_sweeps': len(history)-1, 'stop_reason': reason,
               'objective_evaluations': history[-1]['evaluations'],
               'training_observe_calls': 9*history[-1]['evaluations'],
               'training_shots': -1, 'noise_model': 'none', 'metrics': metrics,
               'max_reference_error': max_error, 'passed': bool(passed)}
    return summary, history, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('qpp-cpu','nvidia'), default='qpp-cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sweeps', type=int, default=40)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    summary, history, rows = run(args.backend, args.seed, args.sweeps)
    out = args.output_dir or ROOT/'results/day10'/args.backend/f'seed{args.seed}'
    out.mkdir(parents=True, exist_ok=True)
    for name, value in [('summary',summary),('history',history),('predictions',rows)]:
        (out/f'{name}.json').write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    with (out/'training_curve.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['sweep','loss','step','evaluations','weights'])
        writer.writeheader()
        writer.writerows({**row, 'weights':json.dumps(row['weights'])} for row in history)
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary['passed'] else 1)

if __name__ == '__main__':
    main()
