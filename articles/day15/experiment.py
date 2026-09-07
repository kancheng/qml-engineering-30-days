"""Train two seeds; select by validation Brier; evaluate test once per backend."""
import argparse
import csv
import json
import os
import platform
from datetime import datetime,timezone
from importlib.metadata import version
from pathlib import Path
from classifier import (np,ROOT,CONFIG,dataset,fit_scaler,transform,probabilities,metrics,predict_proba,load_checkpoint)
from articles.day14.model import cudaq,initialize,predict_batch,arguments,measured
from articles.day10.train import coordinate_search


def run(backend,out):
    cudaq.set_target(backend);splits=dataset();scaler=fit_scaler(splits['train']['raw'])
    train=splits['train'];validation=splits['validation'];candidates=[]
    scaled,_=transform(train['raw'],scaler)
    def objective(w):return metrics(train['labels'],probabilities(predict_batch(scaled,w,CONFIG)))['brier']
    for seed in (42,43):
        initial=initialize(1,seed)
        final,history,reason=coordinate_search(objective,initial,sweeps=16)
        # Only validation ranks seeds; test does not enter this loop.
        val,_=predict_proba(validation['raw'],final,scaler)
        for h in history:
            vp,_=predict_proba(validation['raw'],h['weights'],scaler,reference=True)
            h['validation_brier_reference']=metrics(validation['labels'],vp)['brier']
        candidates.append({'seed':seed,'weights':final.tolist(),'initial_weights':initial.tolist(),
                           'history':history,'stop_reason':reason,'validation':metrics(validation['labels'],val),
                           'training_observe_calls':len(scaled)*history[-1]['evaluations']})
        print(backend,'seed',seed,'train Brier',history[0]['loss'],'->',history[-1]['loss'],flush=True)
    selected=min(candidates,key=lambda c:(c['validation']['brier'],c['seed']))
    checkpoint={'schema_version':1,'config':{'encoding':'angle','layers':1},'mapping':'positive_half',
                'readout':'p1=(1-ZZ)/2','threshold':.5,'feature_order':['x0','x1'],
                'label_rule':'1 iff x0*x1<0','scaler':scaler,'weights':selected['weights'],'seed':selected['seed']}
    out.mkdir(parents=True,exist_ok=True)
    (out/'checkpoint.json').write_text(json.dumps(checkpoint,indent=2)+'\n')
    restored=load_checkpoint(out/'checkpoint.json');records={};max_error=0
    prior=float(np.mean(train['labels']))
    for name,data in splits.items():
        p,flags=predict_proba(data['raw'],restored['weights'],restored['scaler'])
        ref,_=predict_proba(data['raw'],restored['weights'],restored['scaler'],reference=True)
        error=float(np.max(abs(p-ref)));max_error=max(max_error,error)
        records[name]={**data,'probabilities':p.tolist(),'reference_probabilities':ref.tolist(),
                       'predictions':(p>=.5).astype(int).tolist(),'clipped':flags.tolist(),
                       'metrics':metrics(data['labels'],p),'constant_baseline':metrics(data['labels'],[prior]*len(p))}
    shot_records=[]
    test=splits['test'];test_scaled,_=transform(test['raw'],scaler)
    for i,x in enumerate(test_scaled):
        cudaq.set_random_seed(1000+i)
        counts={str(k):int(v) for k,v in cudaq.sample(measured,*arguments(x,restored['weights'],CONFIG),shots_count=1000,explicit_measurements=True).items()}
        p=sum(v for k,v in counts.items() if k[0]!=k[1])/1000
        shot_records.append({'index':i,'counts':counts,'shots':1000,'seed':1000+i,'p1':p})
    summary={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'backend':backend,
             'precision':str(cudaq.get_target().get_precision()),'python':platform.python_version(),
             'cudaq':version('cuda-quantum-cu12'),'numpy':np.__version__,'platform':platform.platform(),
             'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS'),'dataset_seed':2026,'split_sizes':{k:len(v['raw']) for k,v in splits.items()},
             'initialization_seeds':[42,43],'selected_seed':selected['seed'],'selection':'lowest final validation Brier; seed breaks ties',
             'sweeps':16,'step':.4,'step_tolerance':1e-4,'training_shots':-1,'noise_model':'none',
             'test':records['test']['metrics'],'test_sampled':metrics(test['labels'],[r['p1'] for r in shot_records]),
             'max_probability_reference_error':max_error,
             'passed':bool(max_error<1e-5 and all(c['history'][-1]['loss']<c['history'][0]['loss'] for c in candidates) and all(sum(r['counts'].values())==1000 for r in shot_records))}
    for name,obj in [('dataset',splits),('candidates',candidates),('predictions',records),('test_sampling',shot_records),('summary',summary)]:
        (out/f'{name}.json').write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
    with (out/'training_curve.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['seed','sweep','train_brier','validation_brier_reference','objective_evaluations'])
        for c in candidates:
            for h in c['history']:w.writerow([c['seed'],h['sweep'],h['loss'],h['validation_brier_reference'],h['evaluations']])
    print(json.dumps(summary,indent=2),flush=True)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backend',choices=('qpp-cpu','nvidia'),default='qpp-cpu');p.add_argument('--output-dir',type=Path)
    args=p.parse_args();out=args.output_dir or ROOT/'results/day15'/args.backend
    result=run(args.backend,out)
    raise SystemExit(0 if result['passed'] else 1)

if __name__=='__main__':main()
