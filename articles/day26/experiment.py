"""Exact density matrices, finite-shot samples and frozen Wine VQC noise."""
import argparse,json,time,hashlib
from datetime import datetime,timezone
from importlib.metadata import version
import cudaq
from noise import ROOT,np,CHANNELS,OBS,density,exact,probabilities,sample,noise_model
from articles.day14.model import preparation,arguments,Config
from articles.day15.classifier import metrics
OUT=ROOT/'results/day26';PS=[0.,.1,.3,.75,1.];SHOTS=4096


def write(name,obj):
    p=OUT/f'{name}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2)+'\n')


def read(name):return json.loads((OUT/f'{name}.json').read_text())


@cudaq.kernel
def classifier(data:list[float],weights:list[float],encoding:int,layers:int):
    q=cudaq.qvector(2);preparation(q,data,weights,encoding,layers);rz(0.0,q[0])


def run():
    cudaq.set_target('density-matrix-cpu')
    protocol={'channels':list(CHANNELS),'probabilities':PS,'states':{'0':'|00>','1':'|+0>','2':'Bell Phi+'},
              'injection':'one channel on q0 after final rz(0); preparation and measurement basis gates ideal',
              'observables':list(OBS),'shots':SHOTS,'seeds':[42,43,44],
              'depolarization':'(1-p)rho + p/3(XrhoX+YrhoY+ZrhoZ); maximally mixed at p=.75',
              'sampling_check':'all 4 bin frequencies within .06 of exact probabilities; descriptive absolute threshold, not confidence interval',
              'wine':'frozen Day25 VQC test samples; terminal noise only, no retraining or noise-aware selection'}
    write('protocol',protocol);records=[]
    for kind in range(3):
        for channel in CHANNELS:
            for p in PS:
                rho=density(kind,channel,p)
                for name,o in OBS.items():
                    expected=float(np.trace(rho@o).real);actual=exact(kind,channel,p,name)
                    records.append({'kind':kind,'channel':channel,'p':p,'observable':name,'expected':expected,'actual':actual,
                                    'absolute_error':abs(actual-expected),'passed':abs(actual-expected)<1e-10})
    write('exact',records)
    # Wine examples reuse saved preprocessing and selected VQC; they do not refit.
    sys_path=str(ROOT/'articles/day25')
    import sys
    sys.path.insert(0,sys_path)
    from wine_models import representation
    selected_path=ROOT/'results/day25/selected.json';dataset_path=ROOT/'results/day25/datasets.json'
    selected=json.loads(selected_path.read_text());datasets=json.loads(dataset_path.read_text());wine=[]
    write('wine_source',{'selected_sha256':hashlib.sha256(selected_path.read_bytes()).hexdigest(),
                         'datasets_sha256':hashlib.sha256(dataset_path.read_bytes()).hexdigest()})
    for model in selected:
        if model['model']!='vqc':continue
        d=datasets[str(model['split_seed'])];rows=d['splits']['test'];x,_=representation('vqc',[r['features'] for r in rows],d['preprocessor'],model['weights'])
        baseline=np.array(model['scores']['test']['probabilities']);z=1-2*baseline
        for channel in CHANNELS:
            for p in (0.,.1,.3):
                factor={'bit_flip':1-2*p,'phase_flip':1.,'depolarizing':1-4*p/3}[channel]
                expected=(1-factor*z)/2
                actual=np.array([(1-float(cudaq.observe(classifier,cudaq.spin.z(0)*cudaq.spin.z(1),
                                  *arguments(row,model['weights'],Config()),noise_model=noise_model(channel,p),shots_count=-1).expectation()))/2 for row in x])
                error=float(np.max(abs(actual-expected)))
                wine.append({'split_seed':model['split_seed'],'channel':channel,'p':p,'probabilities':actual.tolist(),
                             'metrics':metrics([r['label'] for r in rows],actual),'max_analytic_error':error,
                             'constant_baseline':model['scores']['test']['constant_baseline'],'passed':error<1e-10})
    write('wine',wine)
    summary={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'backend':'density-matrix-cpu','cudaq':version('cuda-quantum-cu12'),
             'numpy':np.__version__,'exact_records':len(records),'wine_records':len(wine),'exact_observe_calls':len(records)+len(wine)*26,
             'max_exact_error':max(r['absolute_error'] for r in records),'max_wine_error':max(r['max_analytic_error'] for r in wine),
             'passed':all(r['passed'] for r in records+wine)}
    write('summary',summary);print(json.dumps(summary,indent=2));return summary


def sampled(backend):
    cudaq.set_target(backend);records=[];start=time.perf_counter()
    for kind in range(3):
        for channel in CHANNELS:
            for p in PS:
                for basis in (0,1):
                    expected=probabilities(kind,channel,p,basis)
                    for seed in (42,43,44):
                        counts=sample(kind,channel,p,basis,SHOTS,seed)
                        actual=np.array([counts.get(k,0)/SHOTS for k in ('00','01','10','11')]);error=float(np.max(abs(actual-expected)))
                        records.append({'kind':kind,'channel':channel,'p':p,'basis':'X' if basis else 'Z','seed':seed,
                                        'shots':SHOTS,'counts':counts,'expected_probabilities':expected.tolist(),
                                        'max_frequency_error':error,'passed':sum(counts.values())==SHOTS and error<.06})
    write(f'{backend}/samples',records)
    summary={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'backend':backend,'cudaq':version('cuda-quantum-cu12'),
             'precision':str(cudaq.get_target().get_precision()),'records':len(records),'shots_per_record':SHOTS,
             'total_shots':len(records)*SHOTS,'max_frequency_error':max(r['max_frequency_error'] for r in records),
             'seconds_including_compilation':time.perf_counter()-start,'passed':all(r['passed'] for r in records),
             'scope':'finite-shot noisy sampling; not exact GPU density-matrix or isolated speed benchmark'}
    write(f'{backend}/summary',summary);print(json.dumps(summary,indent=2));return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('run','sample'));p.add_argument('--backend',choices=('density-matrix-cpu','nvidia'),default='density-matrix-cpu');a=p.parse_args()
    s=run() if a.phase=='run' else sampled(a.backend);raise SystemExit(0 if s['passed'] else 1)
