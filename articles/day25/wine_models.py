"""Wine cultivar 2/3 benchmark with train-only PCA and full-feature control."""
from pathlib import Path
import sys,csv,hashlib
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from articles.day20.iris_models import split_data,forward as iris_forward
from articles.day11.encoding import fit_scaler,transform
from articles.day15.classifier import metrics
COUNTS={'mlp':9,'vqc':4,'hybrid':12,'logistic13':14}
FEATURES=['alcohol','malic_acid','ash','alcalinity_of_ash','magnesium','total_phenols','flavanoids',
          'nonflavanoid_phenols','proanthocyanins','color_intensity','hue','od280_od315','proline']


def load_data():
    path=ROOT/'data/day25/wine.data';rows=[];seen=set();dropped=[]
    for index,row in enumerate(csv.reader(path.read_text().splitlines())):
        if not row:continue
        if len(row)!=14:raise ValueError('expected class plus 13 measurements')
        if row[0] not in ('2','3'):continue
        values=list(map(float,row[1:]));label=int(row[0]=='3');key=tuple(values+[label])
        if key in seen:dropped.append(index);continue
        seen.add(key);rows.append({'id':index,'features':values,'label':label})
    return rows,{'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'dropped_duplicate_ids':dropped,
                 'classes':{'0':'UCI cultivar 2','1':'UCI cultivar 3'},'features':FEATURES}


def matrix(raw):
    x=np.asarray(raw,dtype=float)
    if x.ndim!=2 or not len(x) or x.shape[1]!=13 or not np.isfinite(x).all():raise ValueError('finite nonempty Nx13 data required')
    return x


def fit(raw,labels):
    x=matrix(raw)
    if len(x)<3:raise ValueError('at least three train samples required')
    mean=x.mean(0);std=x.std(0)
    if np.any(std==0):raise ValueError('constant training column')
    z=(x-mean)/std;_,s,vt=np.linalg.svd(z,full_matrices=False);components=vt[:2].copy()
    for row in components:
        if row[np.argmax(abs(row))]<0:row*=-1
    return {'mean':mean.tolist(),'std':std.tolist(),'components':components.tolist(),
            'explained_variance_ratio':(s[:2]**2/np.sum(s**2)).tolist(),'pc_scaler':fit_scaler(z@components.T)}


def representation(model,raw,prep,weights):
    if model not in COUNTS:raise ValueError('unknown model')
    w=np.asarray(weights,dtype=float)
    if w.shape!=(COUNTS[model],) or not np.isfinite(w).all():raise ValueError('invalid weights')
    z=(matrix(raw)-prep['mean'])/prep['std']
    if model=='logistic13':return z,np.zeros_like(z,dtype=bool)
    return transform(z@np.array(prep['components']).T,prep['pc_scaler'])


def forward(model,raw,prep,weights,engine='numpy'):
    if engine not in ('numpy','cudaq'):raise ValueError('unknown engine')
    x,_=representation(model,raw,prep,weights);w=np.asarray(weights)
    if model=='logistic13':return .5*(1+np.tanh((x@w[:13]+w[13])/2))
    return iris_forward(model,x,w,engine)
