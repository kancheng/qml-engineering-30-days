"""Run one explicitly local backend and save every finite-shot observation."""
import argparse
import hashlib
import json
import math
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from execution import (ROOT, MODES, SHOTS, SEEDS, THETA, NOISE_P, OBSERVABLES,
                       configure, ordering_probe, reference, sample_counts,
                       estimate, state, noise_kwargs, cudaq)
OUT = ROOT / 'results/day29'


def run(mode):
    configure(mode)
    probe = {str(k): int(v) for k, v in cudaq.sample(ordering_probe, shots_count=32).items()}
    if probe != {'10': 32}:
        raise ValueError(f'Unexpected bit ordering: {probe}')
    p = NOISE_P if mode == 'density-bitflip' else 0.
    ref = reference(THETA, p)
    exact = {}
    if mode != 'ionq-emulate':
        operators = {'Z0': cudaq.spin.z(0), 'ZZ': cudaq.spin.z(0)*cudaq.spin.z(1),
                     'X0': cudaq.spin.x(0), 'XX': cudaq.spin.x(0)*cudaq.spin.x(1)}
        for name, operator in operators.items():
            value = float(cudaq.observe(state, operator, THETA, shots_count=-1,
                                        **noise_kwargs(mode)).expectation())
            if not math.isfinite(value) or abs(value-ref[name]) > 1e-10:
                raise ValueError('Exact simulator disagrees with analytic reference')
            exact[name] = value
    rows = []
    # Union bound over all 180 expectation checks across three modes.
    family_checks = len(MODES)*len(SHOTS)*len(SEEDS)*4
    for shots in SHOTS:
        for seed in SEEDS:
            for basis, names in OBSERVABLES.items():
                start = time.perf_counter()
                counts = sample_counts(mode, basis, shots, seed)
                elapsed = time.perf_counter()-start
                stats = {}
                for name in names:
                    result = estimate(counts, shots, name)
                    result.update(reference=ref[name], error=result['value']-ref[name])
                    stats[name] = result
                bound = math.sqrt(2*math.log(2*family_checks/.01)/shots)
                rows.append(dict(shots=shots, seed=seed, basis='Z' if basis == 0 else 'X',
                                 counts=counts, statistics=stats, sample_wall_seconds=elapsed,
                                 hoeffding_family99_bound=bound,
                                 passed=all(abs(s['error']) <= bound for s in stats.values())))
    sources = ['articles/day29/execution.py', 'articles/day08/kernels.py', 'articles/day26/noise.py']
    result = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(), mode=mode,
                  execution_kind='local_noisy_simulator' if p else 'local_ideal_emulation' if mode == 'ionq-emulate' else 'local_ideal_simulator',
                  cudaq=version('cuda-quantum-cu12'), python=platform.python_version(),
                  OMP_NUM_THREADS=os.getenv('OMP_NUM_THREADS'),
                  CUDAQ_DEFAULT_SIMULATOR=os.getenv('CUDAQ_DEFAULT_SIMULATOR'),
                  target=str(cudaq.get_target()), theta=THETA, noise_p=p,
                  noise_location='one bit flip on q0 at RZ(0) after preparation' if p else 'none',
                  ordering_probe=probe, analytic_reference=ref, exact_simulator=exact,
                  source_sha256={s: hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
                  remote_submitted=False, job_ids=[], queue_seconds=None,
                  sample_calls=len(rows), requested_shots=sum(r['shots'] for r in rows),
                  validation_probe_shots=32, exact_observe_calls=len(exact),
                  passed=all(r['passed'] for r in rows), records=rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'{mode}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(f"{mode}: {len(rows)} records, {result['requested_shots']} shots, passed={result['passed']}")
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=MODES, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if run(args.mode)['passed'] else 1)
