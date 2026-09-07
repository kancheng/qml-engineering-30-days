"""Train reference models, then separately verify frozen models with CUDA-Q."""
import argparse,json,time,platform,os
from datetime import datetime,timezone
from pathlib import Path
from importlib.metadata import version
from iris_models import np,ROOT,COUNTS,load_data,split_data,fit_preprocessor,preprocess,initialize,forward,metrics
from articles.day18.benchmark import search
import cudaq

RESULTS=ROOT/'results/day20'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def train():
    rows,source=load_data()
    protocol={'task':'Iris versicolor vs virginica; exact duplicate rows removed before split','source':source,
              'split_seeds':[2028,2029],'initialization_seeds':[42,43],'models':COUNTS,'budget_per_fit':73,
              'loss':'Brier','optimizer':'Day18 coordinate search, step .4, full pairs only',
              'preprocessing':'train-only mean/std, PCA2, PC minmax [-1,1], clipping','selection':'final validation Brier, lower seed breaks ties',
              'threshold':.5,'training_engine':'NumPy CPU for all models, including exact statevector quantum reference',
              'quantum_scope':'CUDA-Q only verifies frozen selected models; no claim of CUDA-Q/GPU training',
              'input_features':['sepal_length_cm','sepal_width_cm','petal_length_cm','petal_width_cm']}
    write(RESULTS/'protocol.json',protocol);candidates=[];selected=[];datasets={}
    for split_seed in protocol['split_seeds']:
        splits=split_data(rows,split_seed);prep=fit_preprocessor([r['features'] for r in splits['train']]);datasets[str(split_seed)]={'splits':splits,'preprocessor':prep}
        x={k:preprocess([r['features'] for r in v],prep)[0] for k,v in splits.items()};y={k:[r['label'] for r in v] for k,v in splits.items()}
        for model in COUNTS:
            group=[]
            for seed in (42,43):
                v=initialize(model,seed)
                def objective(w):return metrics(y['train'],forward(model,x['train'],w))['brier']
                start=time.perf_counter();w,history,sweeps=search(objective,v,73);elapsed=time.perf_counter()-start
                validation=metrics(y['validation'],forward(model,x['validation'],w))
                record={'split_seed':split_seed,'model':model,'seed':seed,'weights':w.tolist(),'history':history,
                        'validation':validation,'fit_seconds_numpy':elapsed,'objective_evaluations':history[-1]['evaluations'],
                        'training_examples_evaluated':len(y['train'])*history[-1]['evaluations'],'full_sweeps':sweeps}
                candidates.append(record);group.append(record)
                print('fit',split_seed,model,seed,'train Brier',history[-1]['loss'],flush=True)
            winner=min(group,key=lambda r:(r['validation']['brier'],r['seed']));scores={}
            for split in splits:
                p=forward(model,x[split],winner['weights']);flags=preprocess([r['features'] for r in splits[split]],prep)[1]
                scores[split]={'probabilities':p.tolist(),'metrics':metrics(y[split],p),'clipped_values':int(flags.sum()),
                               'constant_baseline':metrics(y[split],[float(np.mean(y['train']))]*len(y[split]))}
            selected.append({'split_seed':split_seed,'model':model,'seed':winner['seed'],'weights':winner['weights'],'scores':scores})
    write(RESULTS/'datasets.json',datasets);write(RESULTS/'candidates.json',candidates);write(RESULTS/'selected.json',selected)
    write(RESULTS/'training_summary.json',{'generated_at_utc':datetime.now(timezone.utc).isoformat(),'python':platform.python_version(),'numpy':np.__version__,
                                         'platform':platform.platform(),'fits':len(candidates),'deduplicated_samples':len(rows),'cudaq_training_calls':0,
                                         'passed':all(c['objective_evaluations']==73 and c['history'][-1]['loss']<=c['history'][0]['loss'] for c in candidates)})


def verify(backend):
    cudaq.set_target(backend);datasets=json.loads((RESULTS/'datasets.json').read_text());selected=json.loads((RESULTS/'selected.json').read_text());records=[]
    for model in selected:
        data=datasets[str(model['split_seed'])]
        for split,rows in data['splits'].items():
            x,_=preprocess([r['features'] for r in rows],data['preprocessor']);start=time.perf_counter()
            p=forward(model['model'],x,model['weights'],engine='cudaq');elapsed=time.perf_counter()-start
            expected=np.array(model['scores'][split]['probabilities']);error=float(np.max(abs(p-expected)))
            records.append({'split_seed':model['split_seed'],'model':model['model'],'split':split,'probabilities':p.tolist(),
                            'metrics':metrics([r['label'] for r in rows],p),'max_reference_error':error,
                            'verification_seconds_including_compilation':elapsed,'cudaq_observe_calls':0 if model['model']=='mlp' else len(rows),'passed':error<1e-5})
    write(RESULTS/backend/'verification.json',records)
    summary={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'backend':backend,'precision':str(cudaq.get_target().get_precision()),
             'cudaq':version('cuda-quantum-cu12'),'numpy':np.__version__,'python':platform.python_version(),'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS'),
             'records':len(records),'max_reference_error':max(r['max_reference_error'] for r in records),
             'cudaq_observe_calls':sum(r['cudaq_observe_calls'] for r in records),'shots':-1,'noise_model':'none',
             'scope':'frozen-model inference validation, not training or isolated timing benchmark','passed':all(r['passed'] for r in records)}
    write(RESULTS/backend/'summary.json',summary);print(json.dumps(summary,indent=2));return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('train','verify'));p.add_argument('--backend',choices=('qpp-cpu','nvidia'),default='qpp-cpu');a=p.parse_args()
    if a.phase=='train':train()
    else:raise SystemExit(0 if verify(a.backend)['passed'] else 1)
