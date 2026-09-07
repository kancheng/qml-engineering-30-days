"""Predict by comparing a new Iris sample with all saved training samples."""
import argparse
import cudaq
from experiment import read
from kernels import np, transform, kernel

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',nargs=4,type=float,default=[6.,2.9,4.5,1.5])
    p.add_argument('--split-seed',type=int,choices=(2028,2029),default=2028)
    p.add_argument('--backend',choices=('qpp-cpu','nvidia'),default='qpp-cpu')
    a=p.parse_args();cudaq.set_target(a.backend)
    d=read('datasets')[str(a.split_seed)];x,flags=transform([a.features],d['preprocessor'])
    train,_=transform([r['features'] for r in d['splits']['train']],d['preprocessor'])
    print('scaled=',x.tolist(),'clipped inputs=',int(flags.sum()))
    for s in read('selected'):
        if s['split_seed']!=a.split_seed:continue
        score=float((kernel(s['model'],x,train,'cudaq')@s['alpha'])[0]);bounded=float(np.clip(score,0,1))
        print(s['model'],'lambda=',s['lambda'],'raw score=',score,'clipped score=',bounded,
              'class=', 'virginica' if bounded>=.5 else 'versicolor')
