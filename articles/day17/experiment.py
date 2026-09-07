"""Compare exact gradients, finite-difference steps, and a chain-rule update."""
import argparse
import json
import os
import platform
from datetime import datetime,timezone
from importlib.metadata import version
from pathlib import Path
from gradients import np,ROOT,Config,predict,parameter_shift,matrix_gradient,difference,loss_and_gradient
from articles.day14.model import cudaq,initialize,reference_prediction,arguments,measured


def run(backend):
    cudaq.set_target(backend);rows=[];steps=[1e-1,1e-3,1e-5,1e-7]
    for layers in (1,2):
        config=Config('angle',layers);w=initialize(layers,42)
        for features in ([-.6,.2],[.25,-.4],[.8,.5]):
            exact=matrix_gradient(features,w,config);shift=parameter_shift(features,w,config)
            for step in steps:
                fd=difference(lambda v:predict(features,v,config),w,step)
                rows.append({'layers':layers,'features':features,'weights':w.tolist(),'step':step,
                             'matrix_gradient':exact.tolist(),'parameter_shift':shift.tolist(),'finite_difference':fd.tolist(),
                             'shift_max_error':float(np.max(abs(shift-exact))),'fd_max_error':float(np.max(abs(fd-exact))),
                             'passed':bool(np.max(abs(shift-exact))<1e-5)})
    x=[[.25,-.4],[-.6,.2]];y=[1,1];w=initialize();history=[]
    initial=w.copy()
    for iteration in range(5):
        loss,gradient=loss_and_gradient(x,y,w)
        ref=difference(lambda v:float(np.mean([((1-reference_prediction(a,v))/2-b)**2 for a,b in zip(x,y)])),w,1e-5)
        history.append({'iteration':iteration,'weights':w.tolist(),'loss':loss,'gradient':gradient.tolist(),
                        'numpy_loss_fd':ref.tolist(),'gradient_error':float(np.max(abs(gradient-ref)))})
        if iteration<4:w=w-.4*gradient
    sampling=[];features=[.25,-.4];w=initial;index=1;target=matrix_gradient(features,w)[index]
    for shots in (100,1000):
        for seed in range(10):
            estimates=[];counts_pair=[]
            for sign in (1,-1):
                shifted=w.copy();shifted[index]+=sign*np.pi/2
                cudaq.set_random_seed(1000+2*seed+(sign==-1))
                counts={str(k):int(v) for k,v in cudaq.sample(measured,*arguments(features,shifted,Config()),shots_count=shots,explicit_measurements=True).items()}
                estimates.append(sum((1 if k[0]==k[1] else -1)*v for k,v in counts.items())/shots);counts_pair.append(counts)
            sampling.append({'shots_per_shift':shots,'seeds':[1000+2*seed,1001+2*seed],
                             'counts_plus_minus':counts_pair,'gradient':(estimates[0]-estimates[1])/2,'reference':float(target)})
    summary={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'backend':backend,
             'precision':str(cudaq.get_target().get_precision()),'python':platform.python_version(),
             'cudaq':version('cuda-quantum-cu12'),'numpy':np.__version__,'platform':platform.platform(),
             'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS'),'gradient_cases':6,'fd_records':len(rows),
             'max_shift_error':max(r['shift_max_error'] for r in rows),'max_loss_gradient_error':max(h['gradient_error'] for h in history),
             'training_features':x,'training_labels':y,'learning_rate':.4,'updates':4,'history_evaluations':5,
             'training_observe_calls':5*2*(1+2*4),'sampling_features':features,'sampling_weights':initial.tolist(),'sampling_index':1,
             'sampling_records':len(sampling),'noise_model':'none','finite_difference_is_diagnostic':True,
             'passed':all(r['passed'] for r in rows) and all(h['gradient_error']<1e-5 for h in history) and history[-1]['loss']<history[0]['loss']}
    return rows,history,sampling,summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backend',choices=('qpp-cpu','nvidia'),default='qpp-cpu');p.add_argument('--output-dir',type=Path)
    args=p.parse_args();rows,history,sampling,summary=run(args.backend)
    out=args.output_dir or ROOT/'results/day17'/args.backend;out.mkdir(parents=True,exist_ok=True)
    for name,obj in [('gradients',rows),('training',history),('sampling',sampling),('summary',summary)]:
        (out/f'{name}.json').write_text(json.dumps(obj,indent=2)+'\n')
    print(json.dumps(summary,indent=2));raise SystemExit(0 if summary['passed'] else 1)

if __name__=='__main__':main()
