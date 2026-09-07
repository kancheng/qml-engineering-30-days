"""Parameter-shift on independent RY weights, with matrix derivative reference."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
from articles.day14.model import Config,arguments,predict
from articles.day03.quantum_gates import I,Y,Z,ry,KET_ZERO
from articles.day04.bell_state import CNOT


def difference(function,weights,step):
    w=np.asarray(weights,dtype=float)
    if w.ndim!=1 or not w.size or not np.isfinite(w).all() or not np.isfinite(step) or step<=0:
        raise ValueError('finite nonempty weights and positive finite step required')
    result=[]
    for i in range(len(w)):
        plus=w.copy();minus=w.copy();plus[i]+=step;minus[i]-=step
        result.append((function(plus)-function(minus))/(2*step))
    return np.array(result)


def parameter_shift(features,weights,config=Config()):
    arguments(features,weights,config)
    # Multiplying the central difference at pi/2 by pi/2 yields the 1/2 rule.
    return difference(lambda w:predict(features,w,config),weights,np.pi/2)*(np.pi/2)


def matrix_gradient(features,weights,config=Config()):
    _,w,_,layers=arguments(features,weights,config)
    # Build the feature state directly; zero weights still contain CNOTs.
    if config.encoding=='angle':
        data=arguments(features,weights,config)[0]
        state=np.kron(ry(data[0])@KET_ZERO,ry(data[1])@KET_ZERO)
    else:
        from articles.day13.amplitude_encoding import normalize
        state=normalize(features)[0].astype(complex)
    derivatives=np.zeros((len(w),4),dtype=complex)
    for layer in range(layers):
        for offset in range(4):
            index=4*layer+offset;gate=ry(w[index]);dg=(-.5j*Y)@gate
            op=np.kron(gate,I) if offset%2==0 else np.kron(I,gate)
            dop=np.kron(dg,I) if offset%2==0 else np.kron(I,dg)
            derivatives=derivatives@op.T
            derivatives[index]+=dop@state
            state=op@state
            if offset==1:
                state=CNOT@state;derivatives=derivatives@CNOT.T
    return np.array([2*np.vdot(d,np.kron(Z,Z)@state).real for d in derivatives])


def loss_and_gradient(features,labels,weights,config=Config()):
    rows=list(features);y=np.asarray(labels)
    if not rows or y.shape!=(len(rows),) or not np.isin(y,[0,1]).all():raise ValueError('matching nonempty batch and binary labels required')
    values=np.array([predict(x,weights,config) for x in rows]);p=(1-values)/2
    jac=np.array([parameter_shift(x,weights,config) for x in rows])
    # d (p-y)^2 / dw = -(p-y) * df/dw.
    return float(np.mean((p-y)**2)),np.mean(-(p-y)[:,None]*jac,axis=0)
