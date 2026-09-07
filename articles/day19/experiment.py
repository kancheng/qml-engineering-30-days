"""Audit all hybrid gradients and train a small fixed integration dataset."""
import argparse,json,os,platform
from datetime import datetime,timezone
from importlib.metadata import version
from pathlib import Path
from hybrid import np,cudaq,ROOT,initialize,forward,loss_and_gradient
from articles.day17.gradients import difference

FEATURES=[[-.6,.4],[.3,.5],[.7,-.2]]
LABELS=[1,0,1]


def run(backend,out):
    cudaq.set_target(backend);v=initialize();initial=v.copy();history=[]
    for iteration in range(9):
        loss,g=loss_and_gradient(FEATURES,LABELS,v)
        ref=difference(lambda w:float(np.mean((forward(FEATURES,w,True)[0]-LABELS)**2)),v,1e-5)
        history.append({'iteration':iteration,'parameters':v.tolist(),'loss':loss,'gradient':g.tolist(),
                        'reference_loss_fd':ref.tolist(),'max_gradient_error':float(np.max(abs(g-ref))),
                        'group_gradient_norms':{'encoder':float(np.linalg.norm(g[:6])),'quantum':float(np.linalg.norm(g[6:10])),'head':float(np.linalg.norm(g[10:]))}})
        if iteration<8:v=v-.4*g
    p,cache=forward(FEATURES,v);reference,_=forward(FEATURES,v,True)
    changes={'encoder':float(np.linalg.norm(v[:6]-initial[:6])),'quantum':float(np.linalg.norm(v[6:10]-initial[6:10])),'head':float(np.linalg.norm(v[10:]-initial[10:]))}
    checkpoint={'schema_version':1,'architecture':'2 -> affine+tanh(2) -> pi angles -> 2-qubit ZZ -> affine+sigmoid(1)',
                'parameter_order':{'W':[0,4],'b':[4,6],'quantum_weights':[6,10],'head_scale':[10,11],'head_bias':[11,12]},
                'parameters':v.tolist(),'input_contract':'two finite scaled features; no fitted scaler in this synthetic demo','seed':42}
    out.mkdir(parents=True,exist_ok=True)
    (out/'checkpoint.json').write_text(json.dumps(checkpoint,indent=2)+'\n')
    restored=json.loads((out/'checkpoint.json').read_text());reloaded,_=forward(FEATURES,restored['parameters'])
    summary={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'backend':backend,'precision':str(cudaq.get_target().get_precision()),
             'python':platform.python_version(),'numpy':np.__version__,'cudaq':version('cuda-quantum-cu12'),'platform':platform.platform(),
             'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS'),'features':FEATURES,'labels':LABELS,'seed':42,
             'parameters':12,'parameter_groups':{'encoder':6,'quantum':4,'head':2},'updates':8,'learning_rate':.4,
             'training_observe_calls':9*3*13,'max_gradient_error':max(r['max_gradient_error'] for r in history),
             'initial_loss':history[0]['loss'],'final_loss':history[-1]['loss'],'group_parameter_changes':changes,
             'probabilities':p.tolist(),'reference_probabilities':reference.tolist(),'cache':{k:a.tolist() for k,a in cache.items()},
             'reload_max_error':float(np.max(abs(p-reloaded))),'noise_model':'none','shots':-1,
             'passed':bool(max(r['max_gradient_error'] for r in history)<1e-5 and np.max(abs(p-reference))<1e-5 and np.max(abs(p-reloaded))<1e-5 and history[-1]['loss']<history[0]['loss'] and all(a>1e-6 for a in changes.values()))}
    for name,obj in [('history',history),('summary',summary)]:
        (out/f'{name}.json').write_text(json.dumps(obj,indent=2)+'\n')
    print(json.dumps(summary,indent=2));return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=('qpp-cpu','nvidia'),default='qpp-cpu');p.add_argument('--output-dir',type=Path)
    a=p.parse_args();s=run(a.backend,a.output_dir or ROOT/'results/day19'/a.backend);raise SystemExit(0 if s['passed'] else 1)
