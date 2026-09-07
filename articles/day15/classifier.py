"""XOR classifier built on Day 14, with persisted preprocessing and threshold."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import json
import numpy as np
from articles.day14.model import Config,predict_batch,reference_prediction
from articles.day11.encoding import fit_scaler,transform

CONFIG=Config('angle',1)


def dataset(seed=2026):
    """Three samples per quadrant per split; labels are XOR of coordinate signs."""
    rng=np.random.default_rng(seed)
    splits={}
    for name in ('train','validation','test'):
        x=[]
        for sx,sy in [(-1,-1),(-1,1),(1,-1),(1,1)]:
            x.extend(rng.uniform(.25,.95,(3,2))*[sx,sy])
        x=np.array(x);order=rng.permutation(len(x));x=x[order]
        splits[name]={'raw':x.tolist(),'labels':(x[:,0]*x[:,1]<0).astype(int).tolist()}
    return splits


def probabilities(expectations):
    z=np.asarray(expectations,dtype=float)
    if z.ndim!=1 or not np.isfinite(z).all() or np.any(np.abs(z)>1+1e-5):
        raise ValueError('expected finite ZZ values within numerical tolerance of [-1,1]')
    return (1-np.clip(z,-1,1))/2


def metrics(labels,p):
    y=np.asarray(labels);p=np.asarray(p,dtype=float)
    if y.ndim!=1 or not y.size or p.shape!=y.shape or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('expected nonempty binary labels and matching probabilities')
    predicted=(p>=.5).astype(int)
    confusion=[[int(np.sum((y==a)&(predicted==b))) for b in (0,1)] for a in (0,1)]
    return {'brier':float(np.mean((p-y)**2)),'accuracy':float(np.mean(predicted==y)),
            'confusion_matrix':confusion}


def predict_proba(raw,weights,scaler,reference=False):
    scaled,flags=transform(raw,scaler)
    values=([reference_prediction(x,weights,CONFIG) for x in scaled] if reference else predict_batch(scaled,weights,CONFIG))
    return probabilities(values),flags


def load_checkpoint(path):
    checkpoint=json.loads(Path(path).read_text())
    if checkpoint['schema_version']!=1 or checkpoint['config']!={'encoding':'angle','layers':1} or checkpoint['mapping']!='positive_half' or checkpoint['readout']!='p1=(1-ZZ)/2' or checkpoint['threshold']!=.5:
        raise ValueError('unsupported classifier checkpoint')
    w=np.asarray(checkpoint['weights'],dtype=float)
    if w.shape!=(4,) or not np.isfinite(w).all():raise ValueError('invalid checkpoint weights')
    transform([[0.,0.]],checkpoint['scaler'])
    return checkpoint
