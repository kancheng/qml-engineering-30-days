"""Classical encoder -> independent-angle PQC -> classical sigmoid head."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import cudaq
from articles.day09.pqc import model as quantum_circuit
from articles.day03.quantum_gates import KET_ZERO,Z,ry
from articles.day04.bell_state import CNOT


def initialize(seed=42):
    parameters=np.random.default_rng(seed).normal(0,.2,12)
    parameters[10]=.8
    return parameters


def validate(features,parameters):
    x=np.asarray(features,dtype=float);v=np.asarray(parameters,dtype=float)
    if x.ndim!=2 or not len(x) or x.shape[1]!=2 or not np.isfinite(x).all():raise ValueError('nonempty finite Bx2 features required')
    if v.shape!=(12,) or not np.isfinite(v).all():raise ValueError('12 finite parameters required')
    return x,v


def quantum_value(angles,weights,reference=False):
    if reference:
        state=np.kron(ry(angles[0])@KET_ZERO,ry(angles[1])@KET_ZERO)
        state=np.kron(ry(weights[0]),ry(weights[1]))@state
        state=CNOT@state
        state=np.kron(ry(weights[2]),ry(weights[3]))@state
        return float(np.vdot(state,np.kron(Z,Z)@state).real)
    return float(cudaq.observe(quantum_circuit,cudaq.spin.z(0)*cudaq.spin.z(1),
                               list(angles),list(weights),1,shots_count=-1).expectation())


def forward(features,parameters,reference=False):
    x,v=validate(features,parameters)
    hidden=np.tanh(x@v[:4].reshape(2,2)+v[4:6]);angles=np.pi*hidden
    q=np.array([quantum_value(a,v[6:10],reference) for a in angles])
    logits=v[10]*q+v[11];p=.5*(1+np.tanh(logits/2))
    return p,{'hidden':hidden,'angles':angles,'quantum':q,'logits':logits}


def quantum_jacobian(angles,weights):
    combined=np.r_[angles,weights];result=[]
    for i in range(6):
        plus=combined.copy();minus=combined.copy();plus[i]+=np.pi/2;minus[i]-=np.pi/2
        result.append(.5*(quantum_value(plus[:2],plus[2:])-quantum_value(minus[:2],minus[2:])))
    return np.array(result)


def loss_and_gradient(features,labels,parameters):
    x,v=validate(features,parameters);y=np.asarray(labels)
    if y.shape!=(len(x),) or not np.isin(y,[0,1]).all():raise ValueError('matching binary labels required')
    p,cache=forward(x,v);gradient=np.zeros(12)
    delta=2*(p-y)*p*(1-p)/len(x)
    for i,row in enumerate(x):
        jac=quantum_jacobian(cache['angles'][i],v[6:10])
        # Keep the encoded angles fixed when shifting PQC weights, and vice versa.
        dz=delta[i]*v[10]*jac[:2]*np.pi*(1-cache['hidden'][i]**2)
        gradient[:4]+=np.outer(row,dz).ravel();gradient[4:6]+=dz
        gradient[6:10]+=delta[i]*v[10]*jac[2:]
    gradient[10]=delta@cache['quantum'];gradient[11]=delta.sum()
    return float(np.mean((p-y)**2)),gradient
