"""Validate signed amplitude preparation and record logical gate costs."""
import argparse
import json
import os
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from amplitude_encoding import cudaq, np, normalize, gate_angles, gates, sampled_gates, loaded, simulator_state, extract

CASES=[('three',[3,4,0]),('scaled',[30,40,0]),('global_sign',[-3,-4,0]),
       ('relative_sign',[3,-4,0]),('two',[3,4]),('negative_two',[-3,4]),
       ('basis01',[0,1,0,0]),('basis10',[0,0,1,0]),('right_only',[0,0,-3,4]),
       ('left_only',[-3,4,0,0]),('bell',[1,0,0,1]),('signed_four',[1,-2,3,-4])]


def run(backend):
    cudaq.set_target(backend)
    rows=[]
    for index,(name,values) in enumerate(CASES):
        a,meta=normalize(values);n=meta['qubits'];angles=gate_angles(a)
        explicit=extract(cudaq.get_state(gates,angles,n),n)
        state=simulator_state(a)
        direct=extract(cudaq.get_state(loaded,state),n)
        fidelities={key:float(abs(np.vdot(a,v))**2) for key,v in [('gates',explicit),('loaded',direct)]}
        # X on the last logical qubit detects the [3,+/-4,0,0] relative sign.
        op=cudaq.spin.x(n-1)
        x=float(cudaq.observe(gates,op,angles,n,shots_count=-1).expectation())
        expected_x=float(sum(a[i]*a[i^1] for i in range(len(a))))
        seed=42+index;cudaq.set_random_seed(seed)
        counts={str(k):int(v) for k,v in cudaq.sample(sampled_gates,angles,n,shots_count=1000,explicit_measurements=True).items()}
        error=abs(x-expected_x)
        rows.append({'name':name,**meta,'amplitudes':a.tolist(),'angles':angles,'fidelities':fidelities,
                     'gate_probabilities':(abs(explicit)**2).tolist(),'loaded_probabilities':(abs(direct)**2).tolist(),
                     'x_last':x,'reference_x_last':expected_x,'observable_error':error,
                     'counts':counts,'shots':1000,'sampling_seed':seed,
                     'logical_ry':1 if n==1 else 3,'logical_cnot':0 if n==1 else 2,
                     'passed':all(abs(f-1)<1e-5 for f in fidelities.values()) and error<1e-5 and sum(counts.values())==1000})
    summary={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'backend':backend,
             'precision':str(cudaq.get_target().get_precision()),'python':platform.python_version(),
             'numpy':np.__version__,'cudaq':version('cuda-quantum-cu12'),'platform':platform.platform(),
             'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS'),'cases':len(rows),'noise_model':'none',
             'max_fidelity_error':max(abs(f-1) for r in rows for f in r['fidelities'].values()),
             'max_observable_error':max(r['observable_error'] for r in rows),
             'basis_convention':'logical |q0 q1>; reverse bits for State.from_data buffer',
             'cost_scope':'source-level gates before optimization, not hardware depth or benchmark',
             'passed':all(r['passed'] for r in rows)}
    return rows,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=('qpp-cpu','nvidia'),default='qpp-cpu')
    p.add_argument('--output-dir',type=Path)
    args=p.parse_args();rows,summary=run(args.backend)
    out=args.output_dir or Path(__file__).resolve().parents[2]/'results/day13'/args.backend
    out.mkdir(parents=True,exist_ok=True)
    for name,obj in [('predictions',rows),('summary',summary)]:
        (out/f'{name}.json').write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
    a,meta=normalize([1,-2,3,-4])
    (out/'circuit.txt').write_text(cudaq.draw(gates,gate_angles(a),meta['qubits']),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    raise SystemExit(0 if summary['passed'] else 1)

if __name__=='__main__':main()
