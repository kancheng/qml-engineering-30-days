"""Sweep input and weight settings without fitting any model parameters."""

import argparse
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

from pqc import initialize, inputs, model, parameter_count, predict, reference_prediction, reference_state, sample_prediction


def run_experiment(backend: str) -> tuple[list[dict], dict]:
    if backend not in ('qpp-cpu','nvidia'):
        raise ValueError('unsupported backend')
    cudaq.set_target(backend)
    started = time.perf_counter()
    rows = []
    for layers in (1,2):
        for kind in ('zero','seed42','seed43'):
            w = np.zeros(parameter_count(layers)) if kind == 'zero' else initialize(layers,int(kind[4:]))
            for features in ([0.0,0.0],[0.25,-0.4],[0.5,0.5],[-0.75,0.2]):
                angles, weights = inputs(features,w,layers)
                state = cudaq.get_state(model,angles,weights,layers)
                actual = np.array([state.amplitude(label) for label in ('00','01','10','11')])
                reference = reference_state(features,w,layers)
                fidelity = float(abs(np.vdot(reference,actual))**2)
                exact = predict(features,w,layers)
                expected = reference_prediction(features,w,layers)
                sampled = sample_prediction(features,w,layers)
                checks = {'state_fidelity': bool(np.isclose(fidelity,1,rtol=0,atol=1e-5)),
                          'exact_matches_numpy': abs(exact-expected) <= 1e-5,
                          'shot_count': sum(sampled['counts'].values()) == 1000,
                          'bounded_output': abs(exact) <= 1+1e-5 and abs(sampled['prediction']) <= 1}
                rows.append({'features':list(features),'layers':layers,'weight_setting':kind,'weights':weights,
                             'parameter_count':len(weights),'exact':exact,'numpy_reference':expected,
                             'fidelity':fidelity,'sampled':sampled,'checks':checks,'passed':all(checks.values())})
    # A local response scan: vary one weight at a time, keeping x and others fixed.
    features = [0.25,-0.4]
    initial = initialize(1)
    response = []
    for index in range(4):
        for delta in (-0.5,0.0,0.5):
            candidate = initial.copy()
            candidate[index] += delta
            value = predict(features,candidate)
            expected = reference_prediction(features,candidate)
            response.append({'weight_index':index,'delta':delta,'weights':candidate.tolist(),
                             'prediction':value,'numpy_reference':expected,'passed':abs(value-expected) <= 1e-5})
    summary = {'generated_at_utc':datetime.now(timezone.utc).isoformat(),'backend':backend,
               'target':cudaq.get_target().name,'python':platform.python_version(),'numpy':np.__version__,
               'cudaq':version('cuda-quantum-cu12'),'platform':platform.platform(),
               'precision':str(cudaq.get_target().get_precision()),'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS'),
               'qubits':2,'encoding':'RY(pi*x0), RY(pi*x1)','ansatz':'[RY on both, CNOT(0,1), RY on both] repeated L times',
               'observable':'Z0 Z1','parameters_per_layer':4,'logical_preparation_depth':'1 + 3*L before compiler optimization',
               'noise_model':'none','initialization':'normal(mean=0,std=0.2); seeds 42 and 43 plus zeros',
               'sweep_records':len(rows),'response_records':len(response),'shots':1000,'sampling_seed':42,
               'max_exact_error':max(abs(r['exact']-r['numpy_reference']) for r in rows),
               'max_fidelity_error':max(abs(r['fidelity']-1) for r in rows),
               'response_features':features,'response_base_weights':initial.tolist(),'response':response,
               'passed':all(r['passed'] for r in rows+response),
               'execution_seconds':time.perf_counter()-started,
               'timing_note':'includes compilation, reference and sampling; not a benchmark',
               'scope':'PQC structure and parameter response, no training, gradient or advantage claim'}
    return rows,summary


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=('qpp-cpu','nvidia'),default='qpp-cpu')
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    rows,summary=run_experiment(args.backend)
    output=args.output_dir or Path(__file__).resolve().parents[2]/'results/day09'/args.backend
    output.mkdir(parents=True,exist_ok=True)
    (output/'raw_results.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    flat=[{'layers':r['layers'],'features':json.dumps(r['features']),'weight_setting':r['weight_setting'],
           'weights':json.dumps(r['weights']),'exact':r['exact'],'numpy_reference':r['numpy_reference'],
           'sampled':r['sampled']['prediction'],'fidelity':r['fidelity'],'passed':r['passed']} for r in rows]
    with (output/'pqc_sweep.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(flat[0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(flat)
    angles,w=inputs([0.25,-0.4],initialize(1),1)
    (output/'circuit.txt').write_text(cudaq.draw(model,angles,w,1),encoding='utf-8')
    print(f"{args.backend}: {len(rows)} cases + {len(summary['response'])} weight responses; passed={summary['passed']}")
    print(f"max exact error: {summary['max_exact_error']:.3g}")
    raise SystemExit(0 if summary['passed'] else 1)


if __name__ == '__main__':
    main()
