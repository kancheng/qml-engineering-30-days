"""Compare counts, per-shot return values and exact/finite-shot expectations."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import time

import cudaq
import numpy as np

from kernels import return_kernel, sample_kernel, state_kernel


def z0_from_counts(counts: dict[str, int]) -> float:
    if not counts or any(k not in ('00', '01', '10', '11') for k in counts):
        raise ValueError('expected nonempty two-bit counts')
    if any(type(v) is not int or v < 0 for v in counts.values()) or sum(counts.values()) == 0:
        raise ValueError('expected nonnegative integer counts and positive total')
    return sum((1 if k[0] == '0' else -1) * v for k, v in counts.items()) / sum(counts.values())


def counts_from_returns(values: list[int]) -> dict[str, int]:
    if not values or any(type(v) is not int or not 0 <= v <= 3 for v in values):
        raise ValueError('expected nonempty two-bit integer returns')
    return dict(Counter(format(v, '02b') for v in values))


def compare(theta: float, shots: int = 256, seed: int = 42) -> dict:
    if not np.isfinite(theta) or type(shots) is not int or shots <= 0:
        raise ValueError('finite theta and positive integer shots required')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be a nonnegative 32-bit integer')
    timings = {}
    tick = time.perf_counter()
    cudaq.set_random_seed(seed)
    counts = {str(k): int(v) for k,v in cudaq.sample(sample_kernel, theta, shots_count=shots,
                                                     explicit_measurements=True).items()}
    timings['sample'] = time.perf_counter() - tick
    tick = time.perf_counter()
    cudaq.set_random_seed(seed)
    returns = [int(v) for v in cudaq.run(return_kernel, theta, shots_count=shots)]
    timings['run'] = time.perf_counter() - tick
    histogram = counts_from_returns(returns)
    tick = time.perf_counter()
    exact = float(cudaq.observe(state_kernel, cudaq.spin.z(0), theta, shots_count=-1).expectation())
    timings['observe_exact'] = time.perf_counter() - tick
    tick = time.perf_counter()
    cudaq.set_random_seed(seed)
    sampled = float(cudaq.observe(state_kernel, cudaq.spin.z(0), theta, shots_count=shots).expectation())
    timings['observe_shots'] = time.perf_counter() - tick
    tick = time.perf_counter()
    exact_xx = float(cudaq.observe(state_kernel, cudaq.spin.x(0) * cudaq.spin.x(1),
                                  theta, shots_count=-1).expectation())
    timings['observe_exact_xx'] = time.perf_counter() - tick
    reference = float(np.cos(theta))
    estimates = {'sample': z0_from_counts(counts), 'run': z0_from_counts(histogram),
                 'observe_shots': sampled, 'observe_exact': exact}
    checks = {'sample_shots': sum(counts.values()) == shots, 'run_length': len(returns) == shots,
              'sample_support': set(counts) <= {'00','11'}, 'run_support': set(returns) <= {0,3},
              'exact_matches_reference': bool(np.isclose(exact, reference, rtol=0, atol=1e-5)),
              'xx_matches_reference': bool(np.isclose(exact_xx, np.sin(theta), rtol=0, atol=1e-5)),
              'bounded_estimates': all(abs(v) <= 1 + 1e-5 for v in estimates.values())}
    return {'theta_radians': theta, 'shots': shots, 'seed': seed, 'counts': counts,
            'run_values': returns, 'run_histogram': histogram, 'estimates': estimates,
            'observe_exact_xx': exact_xx, 'reference_xx': float(np.sin(theta)),
            'reference_z0': reference, 'sampling_standard_error': float(np.sqrt(max(0,1-reference**2)/shots)),
            'absolute_errors': {k: abs(v-reference) for k,v in estimates.items()},
            'timings_seconds': timings, 'checks': checks, 'passed': all(checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('qpp-cpu','nvidia'), default='qpp-cpu')
    parser.add_argument('--demo', action='store_true', help='one theta=pi/3 case with 32 shots')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    cudaq.set_target(args.backend)
    if args.demo:
        result = compare(float(np.pi/3), shots=32)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result['passed'] else 1)
    rows = [compare(float(theta), shots, seed) for theta in (0, np.pi/3, np.pi/2, np.pi)
            for shots in (32,256) for seed in (42,43)]
    summary = {'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'backend': args.backend,
               'target': cudaq.get_target().name, 'python': platform.python_version(),
               'numpy': np.__version__, 'cudaq': version('cuda-quantum-cu12'),
               'platform': platform.platform(), 'precision': str(cudaq.get_target().get_precision()),
               'OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS'), 'qubits': 2,
               'state': 'cos(theta/2)|00> + sin(theta/2)|11>', 'observable': 'Z0',
               'noise_model': 'none', 'shots': [32,256], 'seeds': [42,43], 'record_count': len(rows),
               'max_exact_error': max(r['absolute_errors']['observe_exact'] for r in rows),
               'passed': all(r['passed'] for r in rows),
               'scope': 'API semantics and correctness; no training or performance benchmark',
               'shot_note': 'each finite-shot API call has its own N shots; single Pauli observable'}
    output = args.output_dir or Path(__file__).resolve().parents[2] / 'results/day08' / args.backend
    output.mkdir(parents=True, exist_ok=True)
    (output/'raw_results.json').write_text(json.dumps(rows, indent=2)+'\n', encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    flat = [{'theta_radians': r['theta_radians'], 'shots': r['shots'], 'seed':r['seed'],
             'reference_z0':r['reference_z0'], **r['estimates'], 'passed':r['passed']} for r in rows]
    with (output/'api_comparison.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(flat[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps(summary,indent=2))
    raise SystemExit(0 if summary['passed'] else 1)


if __name__ == '__main__':
    main()
