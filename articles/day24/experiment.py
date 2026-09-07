"""Fixed gradient ensemble; no optimizer or classification dataset."""
import argparse,json,hashlib,platform,os,time
from datetime import datetime,timezone
from importlib.metadata import version
import cudaq
from landscape import ROOT,np,initialize,reference,shift,product_control
OUT=ROOT/'results/day24'


def write(name,obj):
    p=OUT/f'{name}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2)+'\n')


def read(name):return json.loads((OUT/f'{name}.json').read_text())


def fingerprint():return hashlib.sha256(b''.join((OUT/f'{n}.json').read_bytes() for n in ('protocol','samples'))).hexdigest()


def stats(values):
    a=np.asarray(values)
    return {'count':len(a),'mean':float(a.mean()),'sample_variance_ddof1':float(a.var(ddof=1)),
            'rms':float(np.sqrt(np.mean(a*a))),'median_abs':float(np.median(abs(a))),
            'fraction_abs_lt_001':float(np.mean(abs(a)<.01))}


def run():
    protocol={'qubits':[2,4,6,8],'depths':[1,4,8],'initializations':['uniform','small'],'seeds':list(range(32)),
              'uniform':'Uniform(-pi,pi)','small':'Normal(0,0.1), not identity-block initialization',
              'circuit':'each block RY then RZ per wire; sequential CZ chain (0,1),(1,2),...,(n-2,n-1)',
              'costs':{'local':'<Z0>','global':'<Z0 Z1 ... Z(n-1)>'},'derivative':'first block RY on qubit0, index0 only',
              'engine':'NumPy tangent-state derivative; CUDA-Q independently verifies parameter shift',
              'scope':'finite small-n ensemble, no training, no proof of asymptotic barren plateau',
              'shots':-1,'noise':'none','statistic':'sample variance ddof=1 across 32 initializations, not across parameters',
              'control':'classically tractable product RY states, uniform angles; exact variances local=1/2, global=2^-n'}
    write('protocol',protocol);samples=[];groups=[];start=time.perf_counter()
    for n in protocol['qubits']:
        for depth in protocol['depths']:
            for mode in protocol['initializations']:
                current=[]
                for seed in protocol['seeds']:
                    w=initialize(n,depth,mode,seed);cost,gradient=reference(n,depth,w)
                    r={'qubits':n,'depth':depth,'initialization':mode,'seed':seed,'weights':w.tolist(),'cost':cost,'gradient':gradient}
                    samples.append(r);current.append(r)
                groups.append({'qubits':n,'depth':depth,'initialization':mode,
                               'statistics':{c:stats([r['gradient'][c] for r in current]) for c in ('local','global')}})
                print(n,depth,mode,flush=True)
    controls=[{'qubits':n,'exact_variance':{'local':.5,'global':2.**(-n)},
               'samples':[{'seed':s,**product_control(n,s)} for s in protocol['seeds']]} for n in protocol['qubits']]
    write('samples',samples);write('statistics',groups);write('product_control',controls)
    write('summary',{'generated_at_utc':datetime.now(timezone.utc).isoformat(),'samples':len(samples),'gradient_values':2*len(samples),
                     'numpy':np.__version__,'python':platform.python_version(),'platform':platform.platform(),
                     'reference_seconds':time.perf_counter()-start,'artifact_sha256':fingerprint(),
                     'passed':all(np.isfinite(list(r['gradient'].values())).all() for r in samples)})


def verify(backend):
    cudaq.set_target(backend);records=[];start=time.perf_counter()
    for r in read('samples'):
        for cost in ('local','global'):
            g,pair=shift(r['qubits'],r['depth'],r['weights'],cost);error=abs(g-r['gradient'][cost])
            records.append({k:r[k] for k in ('qubits','depth','initialization','seed')} |
                           {'cost':cost,'gradient':g,'shifted_expectations':pair,'absolute_error':error,'passed':error<1e-5})
    write(f'{backend}/verification',records)
    summary={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'backend':backend,'cudaq':version('cuda-quantum-cu12'),
             'precision':str(cudaq.get_target().get_precision()),'OMP_NUM_THREADS':os.getenv('OMP_NUM_THREADS'),
             'records':len(records),'observe_calls':2*len(records),'max_gradient_error':max(r['absolute_error'] for r in records),
             'seconds_including_compilation':time.perf_counter()-start,'shots':-1,'noise':'none',
             'artifact_sha256':fingerprint(),'passed':all(r['passed'] for r in records)}
    write(f'{backend}/summary',summary);print(json.dumps(summary,indent=2));return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('run','verify'));p.add_argument('--backend',choices=('qpp-cpu','nvidia'),default='qpp-cpu');a=p.parse_args()
    if a.phase=='run':run()
    else:raise SystemExit(0 if verify(a.backend)['passed'] else 1)
