"""Fixed-protocol XOR benchmark: logistic link, MLP and VQC."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import argparse,json,time,os,platform,hashlib
from datetime import datetime,timezone
from importlib.metadata import version
import numpy as np
import cudaq
from articles.day15.classifier import dataset,metrics,probabilities
from articles.day11.encoding import fit_scaler,transform
from articles.day16.models import mlp_forward
from articles.day14.model import predict_batch,reference_prediction,Config

COUNTS={'logistic':3,'mlp':9,'vqc':4}
BUDGET=109


def forward(model,x,w,reference=False):
    x=np.asarray(x,dtype=float);w=np.asarray(w,dtype=float)
    if model not in COUNTS or w.shape!=(COUNTS[model],) or not np.isfinite(w).all():raise ValueError('invalid model weights')
    if x.ndim!=2 or not len(x) or x.shape[1]!=2 or not np.isfinite(x).all():raise ValueError('invalid feature batch')
    if model=='logistic':return .5*(1+np.tanh((x@w[:2]+w[2])/2))
    if model=='mlp':return mlp_forward(x,w)
    z=[reference_prediction(a,w,Config()) for a in x] if reference else predict_batch(x,w,Config())
    return probabilities(z)


def search(objective,initial,budget=BUDGET):
    w=np.asarray(initial,dtype=float).copy()
    if w.ndim!=1 or not len(w) or not np.isfinite(w).all() or type(budget)is not int or budget<3 or budget%2!=1:raise ValueError('nonempty weights and odd budget >=3 required')
    calls=0
    def evaluate(v):
        nonlocal calls
        value=float(objective(v.copy()));calls+=1
        if not np.isfinite(value):raise ValueError('nonfinite objective')
        return value
    loss=evaluate(w);step=.4;history=[{'evaluations':calls,'loss':loss,'weights':w.tolist(),'step':step}];sweeps=0
    while calls+2<=budget:
        improved=False;complete=True
        for j in range(len(w)):
            if calls+2>budget:complete=False;break
            candidates=[]
            for sign in (-1,1):
                v=w.copy();v[j]+=sign*step;candidates.append((evaluate(v),v))
            value,v=min(candidates,key=lambda pair:pair[0])
            if value<loss:loss,w,improved=value,v,True
            history.append({'evaluations':calls,'loss':loss,'weights':w.tolist(),'step':step})
        if complete:
            sweeps+=1
            if not improved:step*=.5
    return w,history,sweeps


def run(backend,out):
    cudaq.set_target(backend)
    data=dataset(2027);scaler=fit_scaler(data['train']['raw'])
    scaled={name:transform(split['raw'],scaler)[0] for name,split in data.items()}
    protocol={'dataset_seed':2027,'initialization_seeds':[42,43],'models':COUNTS,'objective':'Brier',
              'budget_per_fit':BUDGET,'optimizer':'coordinate search, complete +/- pairs, step halved after full no-improvement sweep',
              'initial_step':.4,'threshold':.5,'selection':'minimum final validation Brier; lower seed breaks ties',
              'preprocessing':'train-only minmax [-1,1], clipping','vqc':'angle positive_half, one layer, p1=(1-ZZ)/2',
              'dataset_sha256':hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()}
    out.mkdir(parents=True,exist_ok=True)
    for name,obj in [('protocol',protocol),('dataset',data),('scaler',scaler)]:
        (out/f'{name}.json').write_text(json.dumps(obj,indent=2)+'\n')
    candidates=[];selected=[]
    for model,count in COUNTS.items():
        group=[]
        for seed in (42,43):
            initial=np.random.default_rng(seed).normal(0,.2,count)
            def objective(w):return metrics(data['train']['labels'],forward(model,scaled['train'],w))['brier']
            start=time.perf_counter();w,history,sweeps=search(objective,initial);elapsed=time.perf_counter()-start
            val=metrics(data['validation']['labels'],forward(model,scaled['validation'],w))
            record={'model':model,'seed':seed,'parameters':count,'initial_weights':initial.tolist(),'weights':w.tolist(),
                    'history':history,'validation':val,'full_sweeps':sweeps,'objective_evaluations':history[-1]['evaluations'],
                    'training_observe_calls':BUDGET*12 if model=='vqc' else 0,'fit_seconds_including_first_compilation':elapsed}
            group.append(record);candidates.append(record)
            print(backend,model,seed,'train',history[0]['loss'],'->',history[-1]['loss'],flush=True)
        winner=min(group,key=lambda r:(r['validation']['brier'],r['seed']))
        scores={}
        for split in data:
            p=forward(model,scaled[split],winner['weights'])
            scores[split]={'probabilities':p.tolist(),'metrics':metrics(data[split]['labels'],p)}
        error=0.
        if model=='vqc':
            for split in data:
                ref=forward(model,scaled[split],winner['weights'],reference=True)
                error=max(error,float(np.max(abs(ref-scores[split]['probabilities']))))
        selected.append({'model':model,'seed':winner['seed'],'weights':winner['weights'],'scores':scores,'max_reference_error':error})
    baseline={name:metrics(split['labels'],[float(np.mean(data['train']['labels']))]*len(split['labels'])) for name,split in data.items()}
    summary={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'backend':backend,'precision':str(cudaq.get_target().get_precision()),
             'python':platform.python_version(),'numpy':np.__version__,'cudaq':version('cuda-quantum-cu12'),'platform':platform.platform(),
             'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS'),'classical_execution':'NumPy CPU in both backend runs',
             'fits':len(candidates),'baseline':baseline,'noise_model':'none','shots':-1,
             'passed':all(r['objective_evaluations']==BUDGET and r['history'][-1]['loss']<=r['history'][0]['loss'] for r in candidates) and all(r['max_reference_error']<1e-5 for r in selected)}
    for name,obj in [('candidates',candidates),('selected',selected),('summary',summary)]:
        (out/f'{name}.json').write_text(json.dumps(obj,indent=2)+'\n')
    print(json.dumps(summary,indent=2));return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backend',choices=('qpp-cpu','nvidia'),default='qpp-cpu');p.add_argument('--output-dir',type=Path)
    a=p.parse_args();s=run(a.backend,a.output_dir or ROOT/'results/day18'/a.backend);raise SystemExit(0 if s['passed'] else 1)
