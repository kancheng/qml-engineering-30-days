"""Forward validation and a small optimizer integration check for both encodings."""
import argparse
import json
import os
import platform
from dataclasses import asdict
from datetime import datetime,timezone
from importlib.metadata import version
from pathlib import Path
from model import (cudaq,np,Config,initialize,arguments,circuit,measured,
                   predict,predict_batch,reference_state,reference_prediction)
from articles.day10.train import coordinate_search,mse

DATA={'angle':[[-.6,.2],[.25,-.4],[.8,.5]],
      'amplitude':[[3,4,0],[1,-2,3,-4],[0,0,-3,4]]}


def run(backend):
    cudaq.set_target(backend);rows=[];training=[]
    for encoding,features in DATA.items():
        for layers in (1,2):
            config=Config(encoding,layers)
            for kind,weights in [('zero',np.zeros(4*layers)),('seed42',initialize(layers,42))]:
                for index,x in enumerate(features):
                    args=arguments(x,weights,config)
                    state=cudaq.get_state(circuit,*args)
                    actual=np.array([state.amplitude(label) for label in ('00','01','10','11')])
                    expected=reference_state(x,weights,config)
                    fidelity=float(abs(np.vdot(expected,actual))**2)
                    value=predict(x,weights,config);ref=reference_prediction(x,weights,config)
                    seed=42+index;cudaq.set_random_seed(seed)
                    counts={str(k):int(v) for k,v in cudaq.sample(measured,*args,shots_count=1000,explicit_measurements=True).items()}
                    sampled=sum((1 if k[0]==k[1] else -1)*v for k,v in counts.items())/1000
                    rows.append({'config':asdict(config),'features':x,'weight_setting':kind,'weights':weights.tolist(),
                                 'exact':value,'reference':ref,'fidelity':fidelity,'counts':counts,'shots':1000,
                                 'sampling_seed':seed,'sampled_zz':sampled,
                                 'passed':abs(value-ref)<1e-5 and abs(fidelity-1)<1e-5 and sum(counts.values())==1000})
        config=Config(encoding,1)
        teacher=np.array([.2,.6,-.3,.4])
        labels=np.array([reference_prediction(x,teacher,config) for x in features])
        initial=initialize(1,42)
        def objective(w):return mse(predict_batch(features,w,config),labels)
        final,history,reason=coordinate_search(objective,initial,sweeps=4)
        reference_loss=mse([reference_prediction(x,final,config) for x in features],labels)
        training.append({'config':asdict(config),'features':features,'labels':labels.tolist(),
                         'label_source':'NumPy teacher, same model family; integration check only',
                         'teacher_weights':teacher.tolist(),'initial_weights':initial.tolist(),'final_weights':final.tolist(),
                         'history':history,'stop_reason':reason,'reference_final_loss':reference_loss,
                         'training_observe_calls':3*history[-1]['evaluations'],
                         'passed':history[-1]['loss']<history[0]['loss'] and abs(history[-1]['loss']-reference_loss)<1e-5})
    summary={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'backend':backend,
             'precision':str(cudaq.get_target().get_precision()),'python':platform.python_version(),
             'cudaq':version('cuda-quantum-cu12'),'numpy':np.__version__,'platform':platform.platform(),
             'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS'),'forward_cases':len(rows),'training_checks':len(training),
             'observable':'ZZ','noise_model':'none','training_shots':-1,
             'max_exact_error':max(abs(r['exact']-r['reference']) for r in rows),
             'max_fidelity_error':max(abs(r['fidelity']-1) for r in rows),
             'passed':all(r['passed'] for r in rows+training)}
    return rows,training,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=('qpp-cpu','nvidia'),default='qpp-cpu')
    p.add_argument('--output-dir',type=Path)
    args=p.parse_args();rows,training,summary=run(args.backend)
    out=args.output_dir or Path(__file__).resolve().parents[2]/'results/day14'/args.backend
    out.mkdir(parents=True,exist_ok=True)
    for name,value in [('predictions',rows),('training',training),('summary',summary)]:
        (out/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    for encoding,features in DATA.items():
        (out/f'{encoding}_circuit.txt').write_text(cudaq.draw(circuit,*arguments(features[0],initialize(),Config(encoding))),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    for record in training:print(record['config']['encoding'],record['history'][0]['loss'],'->',record['history'][-1]['loss'])
    raise SystemExit(0 if summary['passed'] else 1)

if __name__=='__main__':main()
