"""Iris binary task and train-only standardization/PCA/angle scaling."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import csv,hashlib
import numpy as np
from articles.day16.models import mlp_forward
from articles.day14.model import Config,predict_batch,reference_prediction
from articles.day19.hybrid import forward as hybrid_forward
from articles.day15.classifier import probabilities,metrics
from articles.day11.encoding import fit_scaler,transform

COUNTS={'mlp':9,'vqc':4,'hybrid':12}


def load_data():
    path=ROOT/'data/day20/iris.data';rows=[];seen=set();duplicates=[]
    for index,row in enumerate(csv.reader(path.read_text().splitlines())):
        if not row:continue
        if row[4] not in ('Iris-versicolor','Iris-virginica'):continue
        features=list(map(float,row[:4]));label=int(row[4]=='Iris-virginica');key=tuple(features+[label])
        if key in seen:duplicates.append(index);continue
        seen.add(key);rows.append({'id':index,'features':features,'label':label})
    return rows,{'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'dropped_duplicate_ids':duplicates,
                 'classes':{'0':'Iris-versicolor','1':'Iris-virginica'}}


def split_data(rows,seed):
    rng=np.random.default_rng(seed);splits={k:[] for k in ('train','validation','test')}
    for label in (0,1):
        group=[r for r in rows if r['label']==label];order=rng.permutation(len(group));a=int(.6*len(group));b=a+int(.2*len(group))
        for name,indices in zip(splits,(order[:a],order[a:b],order[b:])):splits[name].extend(group[i] for i in indices)
    return splits


def fit_preprocessor(raw):
    x=np.asarray(raw,dtype=float)
    if x.ndim!=2 or x.shape[1]!=4 or len(x)<3 or not np.isfinite(x).all():raise ValueError('finite Nx4 training data required')
    mean=x.mean(axis=0);std=x.std(axis=0)
    if np.any(std==0):raise ValueError('constant training feature')
    z=(x-mean)/std;_,s,vt=np.linalg.svd(z,full_matrices=False);components=vt[:2].copy()
    for row in components:
        if row[np.argmax(abs(row))]<0:row*=-1
    pcs=z@components.T
    return {'mean':mean.tolist(),'std':std.tolist(),'components':components.tolist(),
            'explained_variance_ratio':((s[:2]**2)/sum(s**2)).tolist(),'pc_scaler':fit_scaler(pcs)}


def preprocess(raw,prep):
    x=np.asarray(raw,dtype=float)
    if x.ndim!=2 or not len(x) or x.shape[1]!=4 or not np.isfinite(x).all():raise ValueError('finite nonempty Nx4 raw input required')
    z=(x-np.array(prep['mean']))/np.array(prep['std']);pcs=z@np.array(prep['components']).T
    return transform(pcs,prep['pc_scaler'])


def initialize(model,seed):
    v=np.random.default_rng(seed).normal(0,.2,COUNTS[model])
    if model=='hybrid':v[10]=.8
    return v


def forward(model,x,weights,engine='numpy'):
    if model not in COUNTS or engine not in ('numpy','cudaq'):raise ValueError('unknown model or engine')
    w=np.asarray(weights,dtype=float)
    if w.shape!=(COUNTS[model],) or not np.isfinite(w).all():raise ValueError('invalid parameter vector')
    if model=='mlp':return mlp_forward(x,w)
    if model=='hybrid':return hybrid_forward(x,w,reference=engine=='numpy')[0]
    values=([reference_prediction(a,w,Config()) for a in x] if engine=='numpy' else predict_batch(x,w,Config()))
    return probabilities(values)
