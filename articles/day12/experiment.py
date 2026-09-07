"""Compare three angle ranges and four state-preparation choices."""
import argparse
import csv
import json
import os
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from angle_encoding import (cudaq, np, MAPS, AXES, LABELS, fit_scaler, transform,
                            to_angles, validate, reference, state_vector, analytic_bloch,
                            product_fidelity, encoded, measured)


def run(backend):
    cudaq.set_target(backend)
    train = [[10,100],[20,160],[30,120],[40,200]]
    holdout = [[25,150],[5,250],[50,80],[10,200]]
    scaler = fit_scaler(train)
    rows=[]
    for split, raw in [('train',train),('holdout',holdout)]:
        scaled, flags = transform(raw, scaler)
        for index, features in enumerate(scaled):
            for mapping in MAPS:
                angles=to_angles(features, mapping)
                for axis in AXES:
                    _, code=validate(angles, axis)
                    actual=state_vector(angles, axis)
                    expected=reference(angles, axis)
                    fidelity=float(abs(np.vdot(expected,actual))**2)
                    bloch=[float(cudaq.observe(encoded,op,angles,code,shots_count=-1).expectation())
                           for op in (cudaq.spin.x(0),cudaq.spin.y(0),cudaq.spin.z(0))]
                    analytic=analytic_bloch(angles[0],axis)
                    error=float(np.max(np.abs(np.array(bloch)-analytic)))
                    row={'split':split,'index':index,'raw':raw[index],'scaled':features.tolist(),
                         'clipped':flags[index].tolist(),'mapping':mapping,'axis':axis,'angles':angles,
                         'bloch_q0':bloch,'analytic_bloch_q0':analytic,'observable_error':error,
                         'probabilities':(abs(actual)**2).tolist(),'reference_probabilities':(abs(expected)**2).tolist(),
                         'fidelity':fidelity,'passed':bool(abs(fidelity-1)<1e-5 and error<1e-5)}
                    if mapping=='positive_half' and axis=='ry':
                        seed=42+index+(100 if split=='holdout' else 0)
                        cudaq.set_random_seed(seed)
                        counts={str(k):int(v) for k,v in cudaq.sample(measured,angles,code,shots_count=1000,explicit_measurements=True).items()}
                        row['sampling']={'seed':seed,'shots':1000,'counts':counts}
                        row['passed'] &= sum(counts.values())==1000
                    rows.append(row)
    pairs=[]
    for name,a,b in [('endpoints',[-1,0],[1,0]),('opposite_signs',[-.25,0],[.25,0])]:
        for mapping in MAPS:
            first,second=to_angles(a,mapping),to_angles(b,mapping)
            actual=float(abs(np.vdot(state_vector(first),state_vector(second)))**2)
            expected=product_fidelity(first,second)
            pairs.append({'pair':name,'features_a':a,'features_b':b,'mapping':mapping,'axis':'ry',
                          'fidelity':actual,'analytic_fidelity':expected,'passed':abs(actual-expected)<1e-5})
    summary={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'backend':backend,
             'precision':str(cudaq.get_target().get_precision()),'python':platform.python_version(),
             'numpy':np.__version__,'cudaq':version('cuda-quantum-cu12'),'platform':platform.platform(),
             'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS'),'scaler':scaler,'cases':len(rows),
             'pair_cases':len(pairs),'sampled_cases':8,'basis_order':LABELS,'noise_model':'none',
             'max_observable_error':max(r['observable_error'] for r in rows),
             'max_fidelity_error':max(abs(r['fidelity']-1) for r in rows),
             'passed':all(r['passed'] for r in rows+pairs)}
    return rows,pairs,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=('qpp-cpu','nvidia'),default='qpp-cpu')
    p.add_argument('--output-dir',type=Path)
    args=p.parse_args()
    rows,pairs,summary=run(args.backend)
    out=args.output_dir or Path(__file__).resolve().parents[2]/'results/day12'/args.backend
    out.mkdir(parents=True,exist_ok=True)
    for name,data in [('predictions',rows),('pairs',pairs),('summary',summary)]:
        (out/f'{name}.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    with (out/'angle_sweep.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['split','index','mapping','axis','angles','bloch_q0','fidelity','observable_error','passed']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:json.dumps(r[k]) if isinstance(r[k],list) else r[k] for k in fields} for r in rows)
    (out/'circuit.txt').write_text(cudaq.draw(encoded,to_angles([.25,-.4]),0),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    raise SystemExit(0 if summary['passed'] else 1)

if __name__=='__main__': main()
