"""Untrained forward slices and one-at-a-time parameter responses; no ranking."""
import argparse
import json
import os
import platform
from datetime import datetime,timezone
from importlib.metadata import version
from pathlib import Path
from models import np,cudaq,ROOT,initialize,mlp_initialize,mlp_forward,qnn_probability


def run(backend):
    cudaq.set_target(backend);rows=[];responses=[]
    for seed in (42,43):
        qw=initialize(1,seed);mw=mlp_initialize(seed)
        for x0 in np.linspace(-1,1,21):
            x=[float(x0),.25];q=qnn_probability(x,qw);ref=qnn_probability(x,qw,True)
            mlp=float(mlp_forward([x],mw)[0])
            rows.append({'seed':seed,'features':x,'qnn_weights':qw.tolist(),'mlp_weights':mw.tolist(),
                         'qnn_p1':q,'qnn_reference':ref,'mlp_p1':mlp,
                         'passed':abs(q-ref)<1e-5 and 0<=q<=1 and 0<=mlp<=1})
    x=[.25,-.4]
    for name,initial in [('qnn',initialize()),('mlp',mlp_initialize())]:
        for i in range(len(initial)):
            for delta in (-.5,0.,.5):
                candidate=initial.copy();candidate[i]+=delta
                value=qnn_probability(x,candidate) if name=='qnn' else float(mlp_forward([x],candidate)[0])
                ref=qnn_probability(x,candidate,True) if name=='qnn' else None
                responses.append({'model':name,'features':x,'seed':42,'index':i,'delta':delta,
                                  'weights':candidate.tolist(),'p1':value,'qnn_reference':ref,
                                  'passed':0<=value<=1 and (ref is None or abs(value-ref)<1e-5)})
    # A rotation is linear in the state even when cos(theta) is nonlinear in theta.
    from articles.day03.quantum_gates import ry,KET_ZERO,KET_ONE
    unitary=ry(.7);a=.3;b=-.4
    linear_error=float(np.max(abs(unitary@(a*KET_ZERO+b*KET_ONE)-(a*(unitary@KET_ZERO)+b*(unitary@KET_ONE)))))
    # Midpoint at pi/4 differs from the mean of outputs at 0 and pi/2.
    midpoint_gap=float(np.cos(np.pi/4)-.5*(np.cos(0)+np.cos(np.pi/2)))
    summary={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'backend':backend,
             'precision':str(cudaq.get_target().get_precision()),'python':platform.python_version(),
             'cudaq':version('cuda-quantum-cu12'),'numpy':np.__version__,'platform':platform.platform(),
             'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS'),'qnn_parameters':4,'mlp_parameters':9,
             'qnn_config':{'encoding':'angle','mapping':'positive_half','layers':1,'readout':'p1=(1-ZZ)/2'},
             'mlp_architecture':'2 -> tanh(2) -> sigmoid(1), biases included',
             'initialization':'normal(0,0.2), seeds 42 and 43; not trained or tuned',
             'slice_cases':len(rows),'parameter_response_cases':len(responses),
             'max_qnn_reference_error':max(abs(r['qnn_p1']-r['qnn_reference']) for r in rows),
             'state_linearity_error':linear_error,'cosine_midpoint_gap':midpoint_gap,
             'noise_model':'none','shots':-1,'mlp_execution':'NumPy host CPU on both runs',
             'passed':all(r['passed'] for r in rows+responses) and linear_error<1e-12 and abs(midpoint_gap)>.1}
    return rows,responses,summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backend',choices=('qpp-cpu','nvidia'),default='qpp-cpu');p.add_argument('--output-dir',type=Path)
    args=p.parse_args();rows,responses,summary=run(args.backend)
    out=args.output_dir or ROOT/'results/day16'/args.backend;out.mkdir(parents=True,exist_ok=True)
    for name,obj in [('predictions',rows),('parameter_response',responses),('summary',summary)]:
        (out/f'{name}.json').write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2))
    raise SystemExit(0 if summary['passed'] else 1)

if __name__=='__main__':main()
