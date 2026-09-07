"""Classical ridge solves on fixed quantum / RBF / linear Gram matrices."""
import argparse, json, hashlib, time, platform, os
from datetime import datetime, timezone
from importlib.metadata import version
import cudaq
from kernels import ROOT, np, MODELS, load_data, split_data, fit_scaler, transform, kernel, solve, metrics, diagnostics
OUT = ROOT/'results/day23'
LAMBDAS = [.01, .1, 1.]


def write(name, value):
    p = OUT/f'{name}.json'; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def read(name):
    return json.loads((OUT/f'{name}.json').read_text())


def fingerprint():
    return hashlib.sha256(b''.join((OUT/f'{n}.json').read_bytes() for n in ('protocol','datasets','selected'))).hexdigest()


def train():
    rows, source = load_data()
    write('protocol', {'source': source, 'split_seeds':[2028,2029], 'models':list(MODELS), 'lambdas':LAMBDAS,
                      'preprocessing':'train-only raw4 minmax [-1,1], clipping; all models same inputs',
                      'selection':'minimum validation Brier of clipped ridge score, then smaller lambda',
                      'ridge':'(K + lambda I) alpha = y; no intercept; no n factor',
                      'quantum':'fixed RY01 CNOT RY23; squared overlap, 2 qubits, no trainable circuit parameters',
                      'rbf_gamma':1., 'linear':'1 + x dot z / 4; constant feature is regularized', 'training_engine':'NumPy reference kernels and classical solves',
                      'score':'clip(Kquery,train @ alpha, 0, 1); not calibrated probability', 'threshold':.5})
    datasets, selected, candidates, matrices = {}, [], [], {}
    for seed in (2028,2029):
        splits = split_data(rows,seed)
        raw = {s:[r['features'] for r in rows] for s,rows in splits.items()}
        y = {s:[r['label'] for r in rows] for s,rows in splits.items()}
        prep = fit_scaler(raw['train']); x = {s:transform(v,prep)[0] for s,v in raw.items()}
        datasets[str(seed)] = {'splits':splits, 'preprocessor':prep}
        for model in MODELS:
            start = time.perf_counter()
            ks = {s:kernel(model,v,x['train']) for s,v in x.items()}
            elapsed = time.perf_counter()-start
            for s,k in ks.items(): matrices[f'{seed}_{model}_{s}'] = k
            group = []
            for lam in LAMBDAS:
                start = time.perf_counter(); alpha = solve(ks['train'],y['train'],lam)
                seconds = time.perf_counter()-start
                val = metrics(y['validation'],np.clip(ks['validation']@alpha,0,1))
                r = {'split_seed':seed, 'model':model, 'lambda':lam, 'alpha':alpha.tolist(),
                     'validation':val, 'solve_seconds_numpy':seconds}
                candidates.append(r); group.append(r)
            win = min(group,key=lambda r:(r['validation']['brier'],r['lambda']))
            scores = {}
            for s,k in ks.items():
                raw_score = k@win['alpha']; p = np.clip(raw_score,0,1)
                scores[s] = {'raw_scores':raw_score.tolist(), 'clipped_scores':p.tolist(), 'metrics':metrics(y[s],p),
                             'clipped_scores_count':int(((raw_score<0)|(raw_score>1)).sum()),
                             'clipped_input_values':int(transform(raw[s],prep)[1].sum()),
                             'constant_baseline':metrics(y[s],np.full(len(y[s]),np.mean(y['train'])))}
            selected.append({'split_seed':seed, 'model':model, 'lambda':win['lambda'], 'alpha':win['alpha'],
                             'scores':scores, 'gram_diagnostics':diagnostics(ks['train']), 'kernel_seconds_numpy':elapsed})
            print(seed,model,'lambda',win['lambda'],'test',scores['test']['metrics'],flush=True)
    write('datasets',datasets); write('candidates',candidates); write('selected',selected)
    np.savez_compressed(OUT/'matrices.npz',**matrices)
    write('training_summary',{'generated_at_utc':datetime.now(timezone.utc).isoformat(), 'python':platform.python_version(),
                             'numpy':np.__version__, 'platform':platform.platform(), 'ridge_solves':len(candidates),
                             'cudaq_training_calls':0, 'artifact_sha256':fingerprint(), 'passed':all(np.isfinite(c['alpha']).all() for c in candidates)})


def verify(backend):
    cudaq.set_target(backend); data=read('datasets'); refs=np.load(OUT/'matrices.npz'); records=[]; arrays={}
    for s in read('selected'):
        if s['model']!='quantum': continue
        seed=s['split_seed']; d=data[str(seed)]
        x={name:transform([r['features'] for r in rows],d['preprocessor'])[0] for name,rows in d['splits'].items()}
        for name,v in x.items():
            start=time.perf_counter(); k=kernel('quantum',v,x['train'],'cudaq'); elapsed=time.perf_counter()-start
            ref=refs[f'{seed}_quantum_{name}']; error=float(np.max(abs(k-ref)))
            p=np.clip(k@s['alpha'],0,1); score_error=float(np.max(abs(p-s['scores'][name]['clipped_scores'])))
            records.append({'split_seed':seed,'split':name,'matrix_shape':list(k.shape),'observe_calls':int(k.size),
                            'max_kernel_error':error,'max_score_error':score_error,'clipped_scores':p.tolist(),
                            'metrics':metrics([r['label'] for r in d['splits'][name]],p),'seconds_including_compilation':elapsed,
                            'gram_diagnostics':diagnostics(k) if name=='train' else None,
                            'passed':error<1e-5 and score_error<1e-4})
            arrays[f'{seed}_{name}']=k
            print(backend,seed,name,'error',error,flush=True)
    write(f'{backend}/verification',records); np.savez_compressed(OUT/backend/'matrices.npz',**arrays)
    summary={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'backend':backend,'cudaq':version('cuda-quantum-cu12'),
             'precision':str(cudaq.get_target().get_precision()),'OMP_NUM_THREADS':os.getenv('OMP_NUM_THREADS'),
             'shots':-1,'noise':'none','artifact_sha256':fingerprint(),'records':len(records),
             'observe_calls':sum(r['observe_calls'] for r in records),'max_kernel_error':max(r['max_kernel_error'] for r in records),
             'max_score_error':max(r['max_score_error'] for r in records),'passed':all(r['passed'] for r in records),
             'scope':'complete quantum matrices, frozen NumPy alpha; no CUDA-Q training or speedup benchmark'}
    write(f'{backend}/summary',summary); print(json.dumps(summary,indent=2)); return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('phase',choices=('train','verify'))
    p.add_argument('--backend',choices=('qpp-cpu','nvidia'),default='qpp-cpu'); a=p.parse_args()
    if a.phase=='train': train()
    else: raise SystemExit(0 if verify(a.backend)['passed'] else 1)
