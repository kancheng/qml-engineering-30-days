"""Serial exact-observe latency benchmark; cold-process cost kept separate."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import argparse,json,time,os,platform,subprocess,hashlib
from datetime import datetime,timezone
from importlib.metadata import version
import numpy as np
import cudaq
from articles.day24.landscape import circuit,reference
OUT=ROOT/'results/day27'
PROTOCOL={'qubits':[4,8,12,16],'blocks':[4,12],'seed':42,'warmups_after_first':2,'repeats':7,
          'targets':['qpp-cpu','nvidia-fp64','nvidia'],'threads':1,'shots':-1,'noise':'none',
          'circuit':'Day24 RY/RZ per wire followed by nearest-neighbor CZ chain; 2*n*blocks parameters',
          'observable':'Z0 Z1 ... Z(n-1)','timed_region':'blocking observe + expectation retrieval; excludes weights/observable setup',
          'order':'one process per backend, serial CPU then GPU fp64 then GPU fp32; cases increasing n then blocks',
          'scope':'host-visible end-to-end exact inference latency, not training or device kernel time'}


def write(name,obj):
    p=OUT/f'{name}.json';p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2)+'\n')


def read(name):return json.loads((OUT/f'{name}.json').read_text())


def weights(n,blocks):return np.random.default_rng(42).uniform(-np.pi,np.pi,2*n*blocks).tolist()


def observable(n):
    o=cudaq.spin.z(0)
    for j in range(1,n):o=o*cudaq.spin.z(j)
    return o


def measure(n,blocks,w,o):
    start=time.perf_counter();value=float(cudaq.observe(circuit,o,n,blocks,w,shots_count=-1).expectation())
    return time.perf_counter()-start,value


def run(target):
    if os.getenv('OMP_NUM_THREADS')!='1':raise ValueError('set OMP_NUM_THREADS=1 before starting Python')
    start=time.perf_counter()
    if target=='nvidia-fp64':cudaq.set_target('nvidia',option='fp64')
    else:cudaq.set_target(target)
    setup=time.perf_counter()-start
    records=[]
    for n in PROTOCOL['qubits']:
        for blocks in PROTOCOL['blocks']:
            w=weights(n,blocks);o=observable(n)
            first,value=measure(n,blocks,w,o)
            warmups=[measure(n,blocks,w,o) for _ in range(2)]
            runs=[measure(n,blocks,w,o) for _ in range(7)]
            values=[value]+[v for _,v in warmups+runs];times=[t for t,_ in runs]
            ref=reference(n,blocks,w)[0]['global'] if n<=8 and blocks<=8 else None
            record={'qubits':n,'blocks':blocks,'parameters':len(w),'weights_sha256':hashlib.sha256(np.array(w,dtype=np.float64).tobytes()).hexdigest(),
                    'ry_gates':n*blocks,'rz_gates':n*blocks,'cz_gates':(n-1)*blocks,
                    'statevector_bytes_lower_bound':(16 if target!='nvidia' else 8)*2**n,
                    'first_call_seconds':first,'first_value':value,'warmup_seconds':[t for t,_ in warmups],
                    'warmup_values':[v for _,v in warmups],'seconds':times,'values':[v for _,v in runs],
                    'median_seconds':float(np.median(times)),'q25_seconds':float(np.quantile(times,.25)),
                    'q75_seconds':float(np.quantile(times,.75)),'min_seconds':min(times),'max_seconds':max(times),
                    'repeat_spread':float(np.ptp(values)),'numpy_reference':ref,
                    'reference_error':abs(value-ref) if ref is not None else None,
                    'passed':np.isfinite(values).all().item() and np.ptp(values)<1e-5 and (ref is None or abs(value-ref)<1e-5)}
            records.append(record);print(target,n,blocks,'median',record['median_seconds'],flush=True)
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True)
    cpu='unknown'
    for line in Path('/proc/cpuinfo').read_text().splitlines():
        if line.startswith('model name'):cpu=line.split(':',1)[1].strip();break
    write(f'{target}/records',records)
    summary={'generated_at_utc':datetime.now(timezone.utc).isoformat(),'backend':target,'precision':str(cudaq.get_target().get_precision()),
             'cudaq':version('cuda-quantum-cu12'),'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform(),
             'cpu':cpu,'logical_cpus':os.cpu_count(),'OMP_NUM_THREADS':os.getenv('OMP_NUM_THREADS'),
             'gpu_snapshot_after_run':gpu.stdout.strip() or gpu.stderr.strip(),'target_setup_seconds':setup,
             'protocol':PROTOCOL,'records':len(records),'observe_calls':len(records)*10,'passed':all(r['passed'] for r in records)}
    write(f'{target}/summary',summary)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backend',choices=PROTOCOL['targets'],default='qpp-cpu');a=p.parse_args()
    s=run(a.backend);raise SystemExit(0 if s['passed'] else 1)
