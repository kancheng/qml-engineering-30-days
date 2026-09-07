"""Raw features -> fitted scaler -> angles -> state -> measurement."""
import argparse
import json
import os
import platform
from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version
from encoding import cudaq, np, fit_scaler, transform, angle_reference, encoded, measured


def run(backend):
    cudaq.set_target(backend)
    train = [[10,100],[20,160],[30,120],[40,200]]
    holdout = [[25,150],[5,250],[50,80],[10,200]]
    scaler = fit_scaler(train)
    rows=[]
    labels=['00','01','10','11']  # Logical q0 then q1, not raw state buffer order.
    for split, raw in [('train',train),('holdout',holdout)]:
        scaled, outside = transform(raw, scaler)
        for index, (values, features, flags) in enumerate(zip(raw, scaled, outside)):
            angles=(np.pi*features).tolist()
            state=cudaq.get_state(encoded,angles)
            actual=np.array([state.amplitude(label) for label in labels])
            expected=angle_reference(features)
            fidelity=float(abs(np.vdot(expected,actual))**2)
            z=float(cudaq.observe(encoded,cudaq.spin.z(0),angles,shots_count=-1).expectation())
            x=float(cudaq.observe(encoded,cudaq.spin.x(0),angles,shots_count=-1).expectation())
            seed=42+index+(100 if split=='holdout' else 0)
            cudaq.set_random_seed(seed)
            counts={str(k):int(v) for k,v in cudaq.sample(measured,angles,shots_count=1000,explicit_measurements=True).items()}
            error=max(abs(z-np.cos(angles[0])),abs(x-np.sin(angles[0])))
            rows.append({'split':split,'raw':values,'scaled':features.tolist(),'clipped':flags.tolist(),
                         'angles':angles,'probabilities':(abs(actual)**2).tolist(),
                         'reference_probabilities':(expected**2).tolist(),'fidelity':fidelity,
                         'z0':z,'x0':x,'observable_error':float(error),'counts':counts,'shots':1000,'seed':seed,
                         'passed': bool(abs(fidelity-1)<1e-5 and error<1e-5 and sum(counts.values())==1000)})
    # Scaler leakage counterexample is diagnostic only; never used for the circuit rows.
    leaked=fit_scaler(train+holdout)
    correct,_=transform([[25,150]],scaler)
    contaminated,_=transform([[25,150]],leaked)
    summary={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'backend':backend,
             'precision':str(cudaq.get_target().get_precision()),'python':platform.python_version(),
             'numpy':np.__version__,'cudaq':version('cuda-quantum-cu12'),'platform':platform.platform(),
             'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS'),'scaler':scaler,'cases':len(rows),
             'clipped_values':sum(sum(r['clipped']) for r in rows),'basis_order':labels,
             'max_observable_error':max(r['observable_error'] for r in rows),
             'max_fidelity_error':max(abs(r['fidelity']-1) for r in rows),
             'leakage_example':{'raw':[25,150],'train_only':correct.tolist(),'fit_all_incorrect':contaminated.tolist()},
             'noise_model':'none','passed':all(r['passed'] for r in rows)}
    return rows,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=('qpp-cpu','nvidia'),default='qpp-cpu')
    p.add_argument('--output-dir',type=Path)
    args=p.parse_args()
    rows,summary=run(args.backend)
    out=args.output_dir or Path(__file__).resolve().parents[2]/'results/day11'/args.backend
    out.mkdir(parents=True,exist_ok=True)
    for name,obj in [('predictions',rows),('summary',summary)]:
        (out/f'{name}.json').write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
    raise SystemExit(0 if summary['passed'] else 1)

if __name__=='__main__': main()
